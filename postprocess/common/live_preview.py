"""Bounded postprocess previews for the desktop GUI.

Postprocess algorithms publish only source-space geometry.  A single worker
owns the video decoder, rendering and JPEG encoder, so preview work can never
block an algorithm or retain decoded full-resolution frames in its queue.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


PREVIEW_MARKER = "[live-preview]"
_PATH_ENV = "MASK_PIPELINE_PREVIEW_PATH"
_CONTROL_ENV = "MASK_PIPELINE_PREVIEW_CONTROL_PATH"
_WIDTH_ENV = "MASK_PIPELINE_PREVIEW_WIDTH"
_HEIGHT_ENV = "MASK_PIPELINE_PREVIEW_HEIGHT"
_QUALITY_ENV = "MASK_PIPELINE_PREVIEW_JPEG_QUALITY"
_FPS_ENV = "MASK_PIPELINE_POSTPROCESS_PREVIEW_FPS"


@dataclass(frozen=True, slots=True)
class PreviewGeometry:
    frame: int
    stage: str
    label: str
    polygons: tuple[tuple[tuple[float, float], ...], ...] = ()
    ellipses: tuple[tuple[float, float, float, float, float], ...] = ()
    boxes: tuple[tuple[float, float, float, float], ...] = ()
    points: tuple[tuple[float, float], ...] = ()
    track_id: str = ""
    status: str = "running"
    detail: str = ""
    is_keyframe: bool = False
    is_interpolated: bool = False
    is_removed: bool = False
    # Optional low-resolution algorithm frame (not a source-frame reference).
    # The queue holds one item per stage, so this remains strictly bounded.
    preview_image: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class _ArtifactPreview:
    stage: str
    label: str
    artifacts: Mapping[str, Path]
    metadata: Mapping[str, Any]


def _polygons(value: Any) -> tuple[tuple[tuple[float, float], ...], ...]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        return tuple(
            tuple((float(point[0]), float(point[1])) for point in polygon)
            for polygon in parsed
            if len(polygon) >= 3
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()


def geometry_from_detection_record(
    record: Mapping[str, Any],
    *,
    stage: str,
    label: str,
    detail: str = "",
) -> PreviewGeometry:
    polygons: list[tuple[tuple[float, float], ...]] = []
    boxes: list[tuple[float, float, float, float]] = []
    classes: set[str] = set()
    track = ""
    for detection in record.get("detections", []):
        if not isinstance(detection, Mapping):
            continue
        raw_class = str(detection.get("class_name", detection.get("label", "")))
        if raw_class:
            classes.add(
                {
                    "男性器": "male",
                    "女性器": "female",
                    "結合部分": "contact",
                }.get(raw_class, raw_class if raw_class.isascii() else "class")
            )
        candidate = detection.get("segmentation", detection.get("polygons", []))
        if isinstance(candidate, Mapping):
            candidate = candidate.get("polygons", [])
        polygons.extend(_polygons(candidate))
        bbox = detection.get("bbox_xyxy", detection.get("bbox"))
        if isinstance(bbox, Mapping):
            try:
                boxes.append(
                    tuple(
                        float(bbox[key]) for key in ("x1", "y1", "x2", "y2")
                    )
                )
            except (KeyError, TypeError, ValueError):
                pass
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            boxes.append(tuple(float(value) for value in bbox[:4]))
        track = str(detection.get("track_id", track))
    return PreviewGeometry(
        frame=int(record.get("frame_index", 0)),
        stage=stage,
        label=label,
        polygons=tuple(polygons),
        boxes=tuple(boxes),
        track_id=track,
        detail=(
            f"{detail} / classes {','.join(sorted(classes))}"
            if classes
            else detail
        ),
    )


def _fit_canvas(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, round(source_w * scale))
    resized_h = max(1, round(source_h * scale))
    resized = cv2.resize(
        image,
        (resized_w, resized_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.full((height, width, 3), 12, np.uint8)
    ox, oy = (width - resized_w) // 2, (height - resized_h) // 2
    canvas[oy : oy + resized_h, ox : ox + resized_w] = resized
    return canvas, scale, ox, oy


def _draw(canvas: np.ndarray, item: PreviewGeometry, scale: float, ox: int, oy: int) -> None:
    def point(value: tuple[float, float]) -> tuple[int, int]:
        return round(ox + value[0] * scale), round(oy + value[1] * scale)

    color = (80, 80, 255) if item.is_removed else (255, 110, 210)
    if item.is_interpolated:
        color = (60, 200, 255)
    fill = canvas.copy()
    rendered: list[np.ndarray] = []
    for polygon in item.polygons:
        points = np.asarray([point(value) for value in polygon], np.int32)
        if len(points) >= 3:
            rendered.append(points)
    if rendered:
        cv2.fillPoly(fill, rendered, color, lineType=cv2.LINE_AA)
        cv2.addWeighted(fill, 0.34, canvas, 0.66, 0, dst=canvas)
        cv2.polylines(canvas, rendered, True, color, 2, cv2.LINE_AA)
    for cx, cy, rx, ry, theta in item.ellipses:
        cv2.ellipse(
            canvas,
            point((cx, cy)),
            (max(1, round(rx * scale)), max(1, round(ry * scale))),
            float(theta),
            0,
            360,
            color,
            3 if item.is_keyframe else 2,
            cv2.LINE_AA,
        )
    for x1, y1, x2, y2 in item.boxes:
        cv2.rectangle(canvas, point((x1, y1)), point((x2, y2)), (70, 220, 160), 2, cv2.LINE_AA)
    for value in item.points:
        cv2.circle(canvas, point(value), 3, (70, 235, 255), -1, cv2.LINE_AA)
    if item.is_removed:
        for x1, y1, x2, y2 in item.boxes:
            cv2.line(canvas, point((x1, y1)), point((x2, y2)), color, 3, cv2.LINE_AA)
    text = item.label
    if item.track_id:
        text += f"  ID {item.track_id}"
    cv2.rectangle(canvas, (12, 12), (min(canvas.shape[1] - 12, 620), 64), (8, 12, 20), -1)
    cv2.putText(canvas, text[:72], (24, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 248, 255), 1, cv2.LINE_AA)
    flags = []
    if item.is_keyframe:
        flags.append("KEYFRAME")
    if item.is_interpolated:
        flags.append("INTERPOLATED")
    if item.is_removed:
        flags.append("REMOVED")
    detail = " / ".join(filter(None, [item.detail, *flags]))
    cv2.putText(canvas, detail[:92], (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


class PostprocessPreviewSink:
    """Stage-coalescing, bounded and rate-limited preview worker."""

    def __init__(
        self,
        path: Path,
        video: Path,
        *,
        width: int = 960,
        height: int = 540,
        quality: int = 85,
        max_fps: float = 5.0,
        control_path: Path | None = None,
    ) -> None:
        # Preview-only OpenCV operations must not borrow every host core from
        # tracking/approximation. OpenCV defaults to all 24 cores on the target
        # workstation, which caused visible desktop stalls for tiny JPEGs.
        cv2.setNumThreads(2)
        self.path = path.resolve()
        self.video = video.resolve()
        self.width = width
        self.height = height
        self.quality = quality
        self.max_fps = max(0.2, max_fps)
        self.control_path = control_path.resolve() if control_path else None
        self._condition = threading.Condition()
        # One newest value per stage.  There is no per-frame or per-track growth.
        self._pending: OrderedDict[str, PreviewGeometry | _ArtifactPreview] = OrderedDict()
        self._closed = False
        self._dropped = 0
        self._last_offer: dict[str, float] = {}
        self._thread = threading.Thread(target=self._run, name="postprocess-preview", daemon=True)
        self._thread.start()

    @classmethod
    def from_environment(cls, video: Path | None) -> "PostprocessPreviewSink | None":
        raw = os.environ.get(_PATH_ENV, "").strip()
        if not raw or video is None or not Path(video).is_file():
            return None
        try:
            return cls(
                Path(raw),
                Path(video),
                width=max(64, int(os.environ.get(_WIDTH_ENV, "960"))),
                height=max(64, int(os.environ.get(_HEIGHT_ENV, "540"))),
                quality=min(100, max(1, int(os.environ.get(_QUALITY_ENV, "85")))),
                max_fps=float(os.environ.get(_FPS_ENV, "5")),
                control_path=(Path(value) if (value := os.environ.get(_CONTROL_ENV, "").strip()) else None),
            )
        except ValueError:
            return None

    def enabled(self) -> bool:
        return self.control_path is None or self.control_path.is_file()

    def should_sample(self, stage: str) -> bool:
        """Cheap gate used before callers materialize geometry."""

        if not self.enabled():
            return False
        now = time.monotonic()
        previous = self._last_offer.get(stage, 0.0)
        if now - previous < 1.0 / self.max_fps:
            return False
        self._last_offer[stage] = now
        return True

    def submit(self, item: PreviewGeometry) -> None:
        if not self.enabled():
            return
        self._enqueue(item.stage, item)

    def _enqueue(
        self,
        key: str,
        item: PreviewGeometry | _ArtifactPreview,
    ) -> None:
        with self._condition:
            if self._closed:
                return
            if key in self._pending:
                self._pending.pop(key)
                self._dropped += 1
            elif len(self._pending) >= 12:
                self._pending.popitem(last=False)
                self._dropped += 1
            self._pending[key] = item
            self._condition.notify()

    def stage_started(self, stage: str, label: str) -> None:
        # A zero-geometry event still updates the GUI stage immediately.
        self.submit(PreviewGeometry(0, stage, label, detail="processing"))

    def stage_artifacts(
        self,
        stage: str,
        label: str,
        artifacts: Mapping[str, Path],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled():
            return
        # SQLite/JSON inspection is deliberately deferred to the same worker
        # that owns video decode and JPEG creation.
        self._enqueue(
            f"{stage}:artifacts",
            _ArtifactPreview(
                stage,
                label,
                dict(artifacts),
                dict(metadata or {}),
            ),
        )

    def _resolve_artifact_preview(self, task: _ArtifactPreview) -> None:
        artifacts = task.artifacts
        tracked = artifacts.get("tracked_sqlite")
        if tracked is not None:
            removed = self._sample_removed_mask(Path(tracked))
            if removed is not None:
                self.submit(removed)
        faces = artifacts.get("face_masks_sqlite")
        if faces is not None:
            for special in (
                self._sample_removed_face(Path(faces)),
                self._sample_interpolated_face(Path(faces)),
            ):
                if special is not None:
                    self.submit(special)
        item = self._sample_artifacts(task.stage, task.label, artifacts)
        if item is not None:
            summary = self._metadata_detail(task.metadata)
            if summary:
                item = PreviewGeometry(
                    frame=item.frame,
                    stage=item.stage,
                    label=item.label,
                    polygons=item.polygons,
                    ellipses=item.ellipses,
                    boxes=item.boxes,
                    points=item.points,
                    track_id=item.track_id,
                    status=item.status,
                    detail=summary,
                    is_keyframe=item.is_keyframe,
                    is_interpolated=item.is_interpolated,
                    is_removed=item.is_removed,
                    preview_image=item.preview_image,
                )
            self.submit(item)

    @staticmethod
    def _metadata_detail(metadata: Mapping[str, Any]) -> str:
        values: list[str] = []
        for key, label in (
            ("removed_short_tracks", "short tracks removed"),
            ("removed_face_tracks", "face tracks removed"),
            ("interpolated_rows", "interpolated"),
            ("cuts", "cuts"),
            ("rows_after_prune", "masks"),
        ):
            if key in metadata:
                values.append(f"{label} {metadata[key]}")
        return " / ".join(values[:3])

    def close(self) -> None:
        with self._condition:
            self._closed = True
            # The next workflow stage immediately replaces this preview. Do
            # not delay postprocess completion to render stale queued frames.
            self._pending.clear()
            self._condition.notify()
        self._thread.join(timeout=0.2)

    def _sample_artifacts(self, stage: str, label: str, artifacts: Mapping[str, Path]) -> PreviewGeometry | None:
        for name in (
            "face_masks_sqlite", "predictions_sqlite", "keyframes_sqlite",
            "approximated_sqlite", "tracked_sqlite",
        ):
            path = artifacts.get(name)
            if path is not None:
                item = self._sample_masks_sqlite(Path(path), stage, label)
                if item is not None:
                    return item
        cuts = artifacts.get("cuts_json")
        if cuts is not None:
            try:
                value = json.loads(Path(cuts).read_text(encoding="utf-8"))
                frames = value.get("frames", [])
                frame = int(frames[0]) if frames else 0
                return PreviewGeometry(frame, stage, label, detail=f"{len(frames)} cuts", status="complete")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        for name in ("interpolated_union_json", "filled_union_json", "keyframes_json"):
            path = artifacts.get(name)
            if path is not None:
                item = self._sample_ellipse_json(Path(path), stage, label)
                if item is not None:
                    return item
        return None

    @staticmethod
    def _sample_masks_sqlite(path: Path, stage: str, label: str) -> PreviewGeometry | None:
        try:
            with sqlite3.connect(
                f"file:{path}?mode=ro&immutable=1", uri=True
            ) as connection:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(masks)")}
                if not {"frame", "track_id", "polygons"}.issubset(columns):
                    return None
                first = connection.execute(
                    "SELECT frame FROM masks WHERE polygons IS NOT NULL AND polygons != '[]' ORDER BY frame LIMIT 1"
                ).fetchone()
                if first is None:
                    return None
                frame = int(first[0])
                rows = connection.execute(
                    "SELECT track_id, polygons, COALESCE(shape_type, 'polygon') FROM masks WHERE frame=? LIMIT 24",
                    (frame,),
                ).fetchall()
            polygons: list[tuple[tuple[float, float], ...]] = []
            track = ""
            shape = ""
            for track_id, raw, shape_type in rows:
                polygons.extend(_polygons(raw))
                track = str(track_id)
                shape = str(shape_type)
            return PreviewGeometry(frame, stage, label, tuple(polygons), track_id=track, detail=shape, status="complete")
        except (OSError, sqlite3.Error):
            return None

    @staticmethod
    def _sample_removed_mask(path: Path) -> PreviewGeometry | None:
        try:
            with sqlite3.connect(
                f"file:{path}?mode=ro&immutable=1", uri=True
            ) as connection:
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "raw_tracked_masks" not in tables:
                    return None
                row = connection.execute(
                    """
                    SELECT frame, raw_track_id, polygons, bbox_xyxy_json
                    FROM raw_tracked_masks
                    WHERE removed_by_short_track=1
                    ORDER BY frame LIMIT 1
                    """
                ).fetchone()
            if row is None:
                return None
            boxes: tuple[tuple[float, float, float, float], ...] = ()
            if row[3]:
                box = json.loads(str(row[3]))
                boxes = (tuple(float(value) for value in box[:4]),)
            return PreviewGeometry(
                int(row[0]), "short_track_filter", "short-track deletion",
                polygons=_polygons(row[2]), boxes=boxes, track_id=str(row[1]),
                detail="rejected transient detection", status="complete",
                is_removed=True,
            )
        except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _sample_removed_face(path: Path) -> PreviewGeometry | None:
        try:
            with sqlite3.connect(
                f"file:{path}?mode=ro&immutable=1", uri=True
            ) as connection:
                row = connection.execute(
                    """
                    SELECT frame, raw_track_id, head_x1, head_y1, head_x2, head_y2
                    FROM face_tracking_assignments
                    WHERE removed_by_short_track=1
                    ORDER BY frame LIMIT 1
                    """
                ).fetchone()
            if row is None:
                return None
            return PreviewGeometry(
                int(row[0]), "face_short_track_filter", "face short-track deletion",
                boxes=((float(row[2]), float(row[3]), float(row[4]), float(row[5])),),
                track_id=str(row[1]), detail="rejected transient face",
                status="complete", is_removed=True,
            )
        except (OSError, sqlite3.Error):
            return None

    @staticmethod
    def _sample_interpolated_face(path: Path) -> PreviewGeometry | None:
        try:
            with sqlite3.connect(
                f"file:{path}?mode=ro&immutable=1", uri=True
            ) as connection:
                row = connection.execute(
                    """
                    SELECT m.frame, m.track_id, m.polygons
                    FROM masks m
                    JOIN mask_provenance p
                      ON p.frame=m.frame AND p.track_id=m.track_id
                    WHERE p.is_interpolated=1
                    ORDER BY m.frame LIMIT 1
                    """
                ).fetchone()
            if row is None:
                return None
            return PreviewGeometry(
                int(row[0]), "face_interpolation", "face gap interpolation",
                polygons=_polygons(row[2]), track_id=str(row[1]),
                detail="linear between observations", status="complete",
                is_interpolated=True,
            )
        except (OSError, sqlite3.Error):
            return None

    @staticmethod
    def _sample_ellipse_json(path: Path, stage: str, label: str) -> PreviewGeometry | None:
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not rows:
                return None
            row = rows[0]
            values = row.get("ellipse_params")
            if values is None and "ellipse" in row:
                values = [row["ellipse"]]
            ellipses = tuple(tuple(float(value) for value in ellipse[:5]) for ellipse in values or [])
            return PreviewGeometry(
                int(row.get("frame", 0)), stage, label, ellipses=ellipses,
                track_id=str(row.get("track_id", "")), status="complete",
                detail=str(row.get("mode", "ellipse")),
                is_keyframe=bool(row.get("has_keyframe", "keyframe" in path.name)),
                is_interpolated=not bool(row.get("has_keyframe", 1)),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _run(self) -> None:
        capture: cv2.VideoCapture | None = None
        fps = 0.0
        last_write = 0.0
        last_frame: int | None = None
        last_image: np.ndarray | None = None
        try:
            while True:
                with self._condition:
                    while not self._pending and not self._closed:
                        self._condition.wait()
                    if not self._pending and self._closed:
                        return
                    _stage, item = self._pending.popitem(last=False)
                if isinstance(item, _ArtifactPreview):
                    self._resolve_artifact_preview(item)
                    continue
                delay = 1.0 / self.max_fps - (time.monotonic() - last_write)
                if delay > 0:
                    time.sleep(delay)
                if not self.enabled():
                    continue
                try:
                    if item.preview_image is not None:
                        image = item.preview_image
                    elif (
                        not item.polygons
                        and not item.ellipses
                        and not item.boxes
                        and not item.points
                    ):
                        # Stage-start/status events carry no source geometry.
                        # Re-seeking frame zero for every such event used most
                        # of the postprocess LIVE budget and made tracking look
                        # frozen. Keep the last visual context (or a neutral
                        # canvas before the first geometric event) instead.
                        image = (
                            last_image
                            if last_image is not None
                            else np.full(
                                (self.height, self.width, 3),
                                12,
                                dtype=np.uint8,
                            )
                        )
                    elif item.frame == last_frame and last_image is not None:
                        image = last_image
                    else:
                        if capture is None:
                            capture = cv2.VideoCapture(
                                str(self.video),
                                cv2.CAP_FFMPEG,
                                [cv2.CAP_PROP_N_THREADS, 1],
                            )
                            if not capture.isOpened():
                                capture.release()
                                capture = cv2.VideoCapture(str(self.video))
                            fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
                        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, item.frame))
                        ok, image = capture.read()
                        if not ok:
                            continue
                        last_frame = item.frame
                        last_image = image
                    canvas, scale, ox, oy = _fit_canvas(image, self.width, self.height)
                    _draw(canvas, item, scale, ox, oy)
                    self._write(canvas, item, fps)
                    last_write = time.monotonic()
                except Exception as exc:
                    print(f"[live-preview-warning] {exc}", flush=True)
        finally:
            if capture is not None:
                capture.release()

    def _write(self, canvas: np.ndarray, item: PreviewGeometry, fps: float) -> None:
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        slot = int(time.monotonic() * self.max_fps) % 3
        target = self.path.with_name(f"{self.path.stem}-{slot}{self.path.suffix or '.jpg'}")
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_bytes(encoded.tobytes())
        os.replace(temporary, target)
        payload = {
            "path": str(target), "phase": "postprocess", "stage": item.stage,
            "status": item.status, "detail": item.detail, "frame_index": item.frame,
            "timestamp_sec": item.frame / fps if fps > 0 else 0.0,
            "model": item.label, "width": self.width, "height": self.height,
            "generated_at_ms": int(time.time() * 1000), "dropped": self._dropped,
        }
        print(f"{PREVIEW_MARKER} {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}", flush=True)


_ACTIVE: PostprocessPreviewSink | None = None


def activate_postprocess_preview(video: Path | None) -> PostprocessPreviewSink | None:
    global _ACTIVE
    _ACTIVE = PostprocessPreviewSink.from_environment(video)
    return _ACTIVE


def active_postprocess_preview() -> PostprocessPreviewSink | None:
    return _ACTIVE


def close_postprocess_preview() -> None:
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.close()
        _ACTIVE = None


__all__ = [
    "PreviewGeometry", "PostprocessPreviewSink", "activate_postprocess_preview",
    "active_postprocess_preview", "close_postprocess_preview",
    "geometry_from_detection_record",
]
