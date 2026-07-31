"""Bounded, asynchronous inference-overlay previews for the desktop GUI."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from contracts import DetectionFrame, Frame, InferenceFrame, SegmentationFrame


PREVIEW_MARKER = "[live-preview]"
PATH_ENVIRONMENT = "MASK_PIPELINE_PREVIEW_PATH"
INTERVAL_ENVIRONMENT = "MASK_PIPELINE_PREVIEW_INTERVAL_FRAMES"
WIDTH_ENVIRONMENT = "MASK_PIPELINE_PREVIEW_WIDTH"
HEIGHT_ENVIRONMENT = "MASK_PIPELINE_PREVIEW_HEIGHT"
QUALITY_ENVIRONMENT = "MASK_PIPELINE_PREVIEW_JPEG_QUALITY"
CONTROL_ENVIRONMENT = "MASK_PIPELINE_PREVIEW_CONTROL_PATH"
MAX_FPS_ENVIRONMENT = "MASK_PIPELINE_INFERENCE_PREVIEW_FPS"


@dataclass(frozen=True, slots=True)
class _PendingPreview:
    frame_index: int
    timestamp_sec: float
    image: np.ndarray
    result: InferenceFrame


@dataclass(frozen=True, slots=True)
class _CanvasTransform:
    scale: float
    offset_x: int
    offset_y: int

    def point(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(round(self.offset_x + x * self.scale)),
            int(round(self.offset_y + y * self.scale)),
        )


def _fit_canvas(
    image: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, _CanvasTransform]:
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    canvas = np.full((height, width, 3), 12, dtype=np.uint8)
    offset_x = (width - resized_width) // 2
    offset_y = (height - resized_height) // 2
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return canvas, _CanvasTransform(scale, offset_x, offset_y)


def _label(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    cv2.putText(
        canvas,
        text,
        (max(3, x), max(13, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def _display_class_name(value: str, class_id: int) -> str:
    aliases = {
        "男性器": "male",
        "女性器": "female",
        "結合部分": "contact",
    }
    if value in aliases:
        return aliases[value]
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return f"class-{class_id}"
    return value


def _draw_segmentation(
    canvas: np.ndarray,
    transform: _CanvasTransform,
    result: SegmentationFrame,
) -> None:
    fill = canvas.copy()
    polygons: list[np.ndarray] = []
    for instance in result.instances:
        for polygon in instance.segmentation.polygons:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points *= transform.scale
            points += (transform.offset_x, transform.offset_y)
            points = np.rint(points).astype(np.int32)
            if len(points) >= 3:
                polygons.append(points)
    if polygons:
        cv2.fillPoly(fill, polygons, (190, 60, 238), lineType=cv2.LINE_AA)
        cv2.addWeighted(fill, 0.45, canvas, 0.55, 0.0, dst=canvas)
        cv2.polylines(
            canvas,
            polygons,
            True,
            (230, 120, 255),
            1,
            cv2.LINE_AA,
        )
    for instance in result.instances:
        detection = instance.detection
        p1 = transform.point(detection.bbox.x1, detection.bbox.y1)
        p2 = transform.point(detection.bbox.x2, detection.bbox.y2)
        cv2.rectangle(canvas, p1, p2, (72, 160, 255), 1, cv2.LINE_AA)
        classification = detection.classification
        class_name = _display_class_name(
            classification.class_name if classification else detection.class_name,
            classification.class_id if classification else detection.class_id,
        )
        score = classification.score if classification else detection.score
        _label(canvas, f"{class_name} {score:.2f}", p1, (120, 190, 255))


def _draw_face_mask(
    canvas: np.ndarray,
    transform: _CanvasTransform,
    observation,
) -> None:
    mask = observation.mask
    if mask is None:
        return
    x1, y1 = transform.point(mask.box_x1, mask.box_y1)
    x2, y2 = transform.point(mask.box_x2, mask.box_y2)
    x1 = min(canvas.shape[1], max(0, x1))
    x2 = min(canvas.shape[1], max(0, x2))
    y1 = min(canvas.shape[0], max(0, y1))
    y2 = min(canvas.shape[0], max(0, y2))
    if x2 <= x1 or y2 <= y1:
        return
    probability = np.frombuffer(mask.data, dtype=np.uint8).reshape(
        mask.height,
        mask.width,
    )
    probability = cv2.resize(
        probability,
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32) / 255.0
    alpha = (probability * 0.32)[..., None]
    region = canvas[y1:y2, x1:x2].astype(np.float32)
    color = np.empty_like(region)
    color[:] = (255, 105, 170)
    canvas[y1:y2, x1:x2] = np.clip(
        region * (1.0 - alpha) + color * alpha,
        0,
        255,
    ).astype(np.uint8)


def _draw_detections(
    canvas: np.ndarray,
    transform: _CanvasTransform,
    result: DetectionFrame,
) -> None:
    for detection in result.detections:
        observation = detection.face_observation
        if observation is not None:
            _draw_face_mask(canvas, transform, observation)
    for detection in result.detections:
        p1 = transform.point(detection.bbox.x1, detection.bbox.y1)
        p2 = transform.point(detection.bbox.x2, detection.bbox.y2)
        color = (83, 215, 154) if detection.class_id == 1 else (255, 184, 92)
        cv2.rectangle(canvas, p1, p2, color, 1, cv2.LINE_AA)
        _label(
            canvas,
            f"{detection.class_name} {detection.score:.2f}",
            p1,
            color,
        )
        observation = detection.face_observation
        if observation is None:
            continue
        ellipse = observation.ellipse
        if ellipse is not None:
            center = transform.point(ellipse.cx, ellipse.cy)
            axes = (
                max(1, int(round(ellipse.major_radius * transform.scale))),
                max(1, int(round(ellipse.minor_radius * transform.scale))),
            )
            cv2.ellipse(
                canvas,
                center,
                axes,
                math.degrees(ellipse.theta_radians),
                0,
                360,
                (255, 126, 209),
                2,
                cv2.LINE_AA,
            )
        for point in observation.keypoints:
            if not point.valid:
                continue
            cv2.circle(
                canvas,
                transform.point(point.x, point.y),
                3,
                (92, 235, 255),
                -1,
                cv2.LINE_AA,
            )


def render_preview(
    frame: Frame,
    result: InferenceFrame,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Render one contract result into an exact-size BGR preview."""

    canvas, transform = _fit_canvas(frame.image, width=width, height=height)
    if isinstance(result, SegmentationFrame):
        _draw_segmentation(canvas, transform, result)
    elif isinstance(result, DetectionFrame):
        _draw_detections(canvas, transform, result)
    else:
        raise TypeError(f"unsupported preview result: {type(result)!r}")
    return canvas


class LivePreviewSink:
    """Latest-wins worker: preview work can never backpressure inference."""

    def __init__(
        self,
        path: Path,
        *,
        phase: str,
        interval_frames: int,
        width: int,
        height: int,
        jpeg_quality: int,
        max_fps: float = 5.0,
        control_path: Path | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.phase = phase
        self.interval_frames = interval_frames
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self.max_fps = max(0.2, float(max_fps))
        self.control_path = (
            None if control_path is None else control_path.expanduser().resolve()
        )
        self._condition = threading.Condition()
        # At most two source images are retained. Resizing is deliberately
        # deferred to this worker so a 24-thread OpenCV resize cannot pause the
        # inference producer or the GUI progress stream.
        self._pending: deque[_PendingPreview] = deque(maxlen=2)
        self._last_submit = 0.0
        self._closed = False
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._run,
            name="live-preview",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def from_environment(cls) -> LivePreviewSink | None:
        raw_path = os.environ.get(PATH_ENVIRONMENT, "").strip()
        if not raw_path:
            return None
        try:
            interval_frames = max(
                1,
                int(os.environ.get(INTERVAL_ENVIRONMENT, "5")),
            )
            width = max(64, int(os.environ.get(WIDTH_ENVIRONMENT, "960")))
            height = max(64, int(os.environ.get(HEIGHT_ENVIRONMENT, "540")))
            quality = min(
                100,
                max(1, int(os.environ.get(QUALITY_ENVIRONMENT, "85"))),
            )
            max_fps = float(os.environ.get(MAX_FPS_ENVIRONMENT, "5"))
        except ValueError:
            return None
        return cls(
            Path(raw_path),
            phase=os.environ.get("MASK_PIPELINE_PROGRESS_PHASE", "inference"),
            interval_frames=interval_frames,
            width=width,
            height=height,
            jpeg_quality=quality,
            max_fps=max_fps,
            control_path=(
                Path(control)
                if (control := os.environ.get(CONTROL_ENVIRONMENT, "").strip())
                else None
            ),
        )

    def submit(self, frame: Frame, result: InferenceFrame) -> None:
        if frame.index % self.interval_frames != 0:
            return
        if self.control_path is not None and not self.control_path.is_file():
            return
        now = time.monotonic()
        if now - self._last_submit < 1.0 / self.max_fps:
            return
        self._last_submit = now
        pending = _PendingPreview(
            frame_index=frame.index,
            timestamp_sec=frame.timestamp_sec,
            image=frame.image,
            result=result,
        )
        with self._condition:
            if self._closed:
                return
            if len(self._pending) == self._pending.maxlen:
                self._pending.popleft()
                self._dropped += 1
            self._pending.append(pending)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if not self._pending and self._closed:
                    return
                pending = self._pending.popleft()
            try:
                self._write(pending)
            except Exception as exc:  # preview failure must not abort inference
                print(f"[live-preview-warning] {exc}", flush=True)

    def _write(self, pending: _PendingPreview) -> None:
        canvas, transform = _fit_canvas(
            pending.image,
            width=self.width,
            height=self.height,
        )
        if isinstance(pending.result, SegmentationFrame):
            _draw_segmentation(canvas, transform, pending.result)
        elif isinstance(pending.result, DetectionFrame):
            _draw_detections(canvas, transform, pending.result)
        else:
            raise TypeError(
                f"unsupported preview result: {type(pending.result)!r}"
            )
        success, encoded = cv2.imencode(
            ".jpg",
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not success:
            raise RuntimeError("OpenCV could not encode the live preview")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        slot = (pending.frame_index // self.interval_frames) % 3
        target = self.path.with_name(
            f"{self.path.stem}-{slot}{self.path.suffix or '.jpg'}"
        )
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_bytes(encoded.tobytes())
        os.replace(temporary, target)
        payload = {
            "path": str(target),
            "phase": self.phase,
            "frame_index": pending.frame_index,
            "timestamp_sec": pending.timestamp_sec,
            "model": pending.result.model.model_id,
            "width": self.width,
            "height": self.height,
            "generated_at_ms": int(time.time() * 1000),
            "dropped": self._dropped,
        }
        print(
            f"{PREVIEW_MARKER} "
            f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}",
            flush=True,
        )


__all__ = ["LivePreviewSink", "PREVIEW_MARKER", "render_preview"]
