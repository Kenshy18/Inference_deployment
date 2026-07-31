"""CPU video renderer for sparse frame overlays."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .face_privacy import FacePrivacyMask, derive_privacy_mask
from .models import FrameOverlay, OverlayItem, RenderSummary, SourceInfo


H264_CODECS = frozenset({"h264", "avc1", "x264"})
NVENC_CODECS = frozenset({"nvenc", "h264_nvenc"})
H264_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
)
NVENC_PRESETS = tuple(f"p{value}" for value in range(1, 8))
LOCAL_NVENC_FFMPEG = (
    Path(__file__).resolve().parents[2]
    / ".runtime"
    / "ffmpeg-nvenc-btbn-8.1"
    / "bin"
    / "ffmpeg"
)


def _progress_interval_seconds() -> float:
    try:
        value = float(
            os.environ.get("MASK_PIPELINE_PROGRESS_INTERVAL_SEC", "0.3")
        )
    except ValueError:
        value = 0.3
    return max(0.05, value)


def _emit_overlay_progress(
    completed: int,
    total: int,
    *,
    state: str,
    fps: float | None,
) -> None:
    item_index = max(
        0,
        int(os.environ.get("MASK_PIPELINE_PROGRESS_ITEM_INDEX", "0")),
    )
    item_count = max(
        1,
        int(os.environ.get("MASK_PIPELINE_PROGRESS_ITEM_COUNT", "1")),
    )
    item_name = os.environ.get("MASK_PIPELINE_PROGRESS_ITEM_NAME", "").strip()
    local_total = max(0, int(total))
    overall_completed = item_index * local_total + max(
        0,
        min(int(completed), local_total),
    )
    overall_total = item_count * local_total
    overall_state = (
        "complete"
        if state == "complete" and item_index + 1 >= item_count
        else "running"
    )
    detail = "complete" if state == "complete" else "rendering"
    print(
        "[phase-progress] "
        + json.dumps(
            {
                "phase": "overlay",
                "state": overall_state,
                "completed": overall_completed,
                "total": overall_total,
                "detail": f"{item_name}:{detail}" if item_name else detail,
                "fps": None if fps is None else max(0.0, float(fps)),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


@dataclass(frozen=True)
class RenderOptions:
    mode: str
    mask_alpha: float = 0.32
    outline_thickness: int = 2
    box_thickness: int = 2
    show_labels: bool = True
    codec: str = "mp4v"
    h264_crf: int = 18
    h264_preset: str = "veryfast"
    nvenc_cq: int = 18
    nvenc_preset: str = "p5"
    nvenc_gpu: int = 0
    target_bitrate_mbps: float | None = None
    ffmpeg_bin: Path | None = None
    start_frame: int = 0
    end_frame: int | None = None
    progress_every: int = 300
    display_style: str = "legacy"
    face_mask_threshold: float = 0.25
    face_privacy_target: str = "none"
    eye_mask_shape: str = "ellipse"
    minimum_eye_confidence: float = 0.35
    draw_face_ellipses: bool = True
    draw_face_keypoints: bool = True

    def validate(self) -> None:
        if not 0.0 <= self.mask_alpha <= 1.0:
            raise ValueError("mask_alpha must be between 0 and 1")
        if self.outline_thickness < 1 or self.box_thickness < 1:
            raise ValueError("line thickness must be at least 1")
        if len(self.codec) != 4 and self.codec.lower() not in NVENC_CODECS:
            raise ValueError("codec must be a four-character FourCC or h264_nvenc")
        if not 0 <= self.h264_crf <= 51:
            raise ValueError("h264_crf must be between 0 and 51")
        if self.h264_preset not in H264_PRESETS:
            raise ValueError(f"h264_preset must be one of {', '.join(H264_PRESETS)}")
        if not 0 <= self.nvenc_cq <= 51:
            raise ValueError("nvenc_cq must be between 0 and 51")
        if self.nvenc_preset not in NVENC_PRESETS:
            raise ValueError(f"nvenc_preset must be one of {', '.join(NVENC_PRESETS)}")
        if self.nvenc_gpu < 0:
            raise ValueError("nvenc_gpu must be non-negative")
        if self.target_bitrate_mbps is not None and self.target_bitrate_mbps <= 0:
            raise ValueError("target_bitrate_mbps must be positive")
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.end_frame is not None and self.end_frame < self.start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")
        if self.progress_every < 0:
            raise ValueError("progress_every must be non-negative")
        if self.display_style not in {"legacy", "detailed", "simple"}:
            raise ValueError("display_style must be legacy, detailed, or simple")
        if not 0.0 <= self.face_mask_threshold <= 1.0:
            raise ValueError("face_mask_threshold must be between 0 and 1")
        if self.face_privacy_target not in {"none", "face", "eyes"}:
            raise ValueError("face_privacy_target must be none, face, or eyes")
        if self.eye_mask_shape not in {"ellipse", "rectangle"}:
            raise ValueError("eye_mask_shape must be ellipse or rectangle")
        if not 0.0 <= self.minimum_eye_confidence <= 1.0:
            raise ValueError("minimum_eye_confidence must be between 0 and 1")

    @property
    def uses_libx264(self) -> bool:
        return self.codec.lower() in H264_CODECS

    @property
    def uses_nvenc(self) -> bool:
        return self.codec.lower() in NVENC_CODECS

    @property
    def uses_ffmpeg_h264(self) -> bool:
        return self.uses_libx264 or self.uses_nvenc

    @property
    def normalized_codec(self) -> str:
        if self.uses_nvenc:
            return "h264_nvenc"
        return "h264" if self.uses_libx264 else self.codec


@lru_cache(maxsize=None)
def _ffmpeg_has_encoder(executable: Path, encoder: str) -> bool:
    result = subprocess.run(
        [str(executable), "-hide_banner", "-encoders"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.returncode == 0 and encoder in result.stdout


def _resolve_ffmpeg_executable(
    configured: Path | None,
    *,
    encoder: str,
) -> Path:
    if configured is not None:
        candidate = Path(configured).expanduser().resolve()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise RuntimeError(f"FFmpeg executable is not usable: {candidate}")
        if not _ffmpeg_has_encoder(candidate, encoder):
            raise RuntimeError(
                f"FFmpeg executable does not provide {encoder}: {candidate}"
            )
        return candidate

    candidates: list[Path] = []
    configured_environment = os.environ.get("OVERLAY_FFMPEG_BIN")
    if configured_environment:
        candidates.append(Path(configured_environment).expanduser())
    if encoder == "h264_nvenc":
        candidates.append(LOCAL_NVENC_FFMPEG)
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        candidates.append(Path(system_ffmpeg))

    try:
        import imageio_ffmpeg
    except ImportError:
        pass
    else:
        candidates.append(Path(imageio_ffmpeg.get_ffmpeg_exe()))

    for value in candidates:
        candidate = value.expanduser().resolve()
        if (
            candidate.is_file()
            and os.access(candidate, os.X_OK)
            and _ffmpeg_has_encoder(candidate, encoder)
        ):
            return candidate
    raise RuntimeError(
        f"H.264 output requires FFmpeg with {encoder}; install a compatible "
        "FFmpeg build or pass --ffmpeg-bin"
    )


class _H264Writer:
    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        crf: int,
        preset: str,
        nvenc: bool,
        nvenc_cq: int,
        nvenc_gpu: int,
        target_bitrate_mbps: float | None,
        ffmpeg_bin: Path | None,
    ) -> None:
        if width % 2 or height % 2:
            raise ValueError(
                "H.264 yuv420p output requires even video width and height"
            )
        encoder = "h264_nvenc" if nvenc else "libx264"
        executable = _resolve_ffmpeg_executable(
            ffmpeg_bin,
            encoder=encoder,
        )
        command = [
            str(executable),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.12g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            encoder,
        ]
        if target_bitrate_mbps is None:
            bitrate = None
            buffer_size = None
        else:
            target_bps = round(target_bitrate_mbps * 1_000_000)
            bitrate = str(target_bps)
            buffer_size = str(target_bps * 2)
        if nvenc:
            command.extend(["-preset", preset, "-tune", "hq"])
            if bitrate is None:
                command.extend(
                    [
                        "-rc",
                        "vbr",
                        "-cq",
                        str(nvenc_cq),
                        "-b:v",
                        "0",
                    ]
                )
            else:
                command.extend(
                    [
                        "-rc",
                        "cbr",
                        "-b:v",
                        bitrate,
                        "-minrate",
                        bitrate,
                        "-maxrate",
                        bitrate,
                        "-bufsize",
                        str(buffer_size),
                        "-cbr_padding",
                        "1",
                        "-multipass",
                        "disabled",
                        "-spatial-aq",
                        "1",
                        "-temporal-aq",
                        "1",
                        "-aq-strength",
                        "8",
                    ]
                )
            command.extend(["-gpu", str(nvenc_gpu)])
        else:
            command.extend(["-preset", preset])
            if bitrate is None:
                command.extend(["-crf", str(crf)])
            else:
                command.extend(
                    [
                        "-b:v",
                        bitrate,
                        "-minrate",
                        bitrate,
                        "-maxrate",
                        bitrate,
                        "-bufsize",
                        str(buffer_size),
                    ]
                )
        # x264 lossless mode selects a profile that supports lossless coding.
        # Forcing High there makes libx264 reject CRF 0.
        if nvenc or crf != 0:
            command.extend(["-profile:v", "high"])
        command.extend(
            [
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                str(path),
            ]
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._path = path

    def write(self, frame: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("H.264 encoder input is closed")
        contiguous = np.ascontiguousarray(frame, dtype=np.uint8)
        try:
            self._process.stdin.write(memoryview(contiguous).cast("B"))
        except BrokenPipeError as exc:
            error = self._read_error()
            raise RuntimeError(f"H.264 encoder stopped unexpectedly: {error}") from exc

    def _read_error(self) -> str:
        if self._process.stderr is None:
            return "no FFmpeg error output"
        return self._process.stderr.read().decode("utf-8", errors="replace").strip()

    def close(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        error = self._read_error()
        return_code = self._process.wait()
        if self._process.stderr is not None:
            self._process.stderr.close()
        if return_code != 0:
            raise RuntimeError(
                f"H.264 encoder failed with exit code {return_code}: {error}"
            )

    def abort(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        if self._process.poll() is None:
            self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        if self._process.stderr is not None:
            self._process.stderr.close()


class _SparseFrames:
    def __init__(self, frames: Iterable[FrameOverlay] | None) -> None:
        self._iterator: Iterator[FrameOverlay] = iter(frames or ())
        self._next = next(self._iterator, None)

    def take(self, frame_index: int) -> tuple[OverlayItem, ...]:
        while self._next is not None and self._next.frame_index < frame_index:
            self._next = next(self._iterator, None)
        if self._next is None or self._next.frame_index != frame_index:
            return ()
        items = self._next.items
        self._next = next(self._iterator, None)
        return items

    def close(self) -> None:
        close = getattr(self._iterator, "close", None)
        if close is not None:
            close()


def _seek_capture(capture: cv2.VideoCapture, start_frame: int) -> int:
    """Position a capture exactly, falling back to sequential grabs."""

    if start_frame == 0:
        return 0
    positioned = capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    reported = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    if positioned and reported == start_frame:
        return start_frame

    # Some OpenCV backends either reject random access or land on a nearby
    # keyframe. Reset and advance without BGR frame retrieval in that case.
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    reported = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    if reported != 0:
        raise RuntimeError(
            f"failed to seek video to frame 0 (reported frame {reported})"
        )
    for frame_index in range(start_frame):
        if not capture.grab():
            raise RuntimeError(
                f"video ended while seeking to frame {start_frame}; "
                f"last frame={frame_index - 1}"
            )
    return start_frame


def _color(key: str) -> tuple[int, int, int]:
    if key == "genital:simple":
        # OpenCV uses BGR; this is RGB(255, 105, 180), a fixed hot pink.
        return 180, 105, 255
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # Keep colors bright enough to remain visible over dark and mid-tone video.
    values = [96 + int(value) * 159 // 255 for value in digest[:3]]
    return int(values[0]), int(values[1]), int(values[2])


def _keyframe_highlight(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Lighten an existing track color without adding another draw pass."""

    return tuple(
        min(255, int(round(channel * 0.72 + 255.0 * 0.28)))
        for channel in color
    )


def _contours(item: OverlayItem, width: int, height: int) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for polygon in item.polygons:
        if len(polygon) < 3:
            continue
        points = np.asarray(polygon, dtype=np.float32)
        points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
        points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))
        output.append(np.round(points).astype(np.int32).reshape(-1, 1, 2))
    return output


def _ascii_label(item: OverlayItem) -> str:
    components: list[str] = []
    if item.track_id is not None:
        components.append(f"T{item.track_id}")
    if item.label and item.label.isascii():
        components.append(item.label)
    if item.score is not None:
        components.append(f"{item.score:.2f}")
    return " ".join(components)


def _detailed_mask_label(item: OverlayItem) -> str:
    components: list[str] = [_track_label(item.track_id)]
    if item.label and item.label.isascii():
        components.append(item.label)
    if item.score is not None:
        components.append(f"score={item.score:.2f}")
    else:
        components.append("score=--")
    if item.provenance:
        components.append(item.provenance)
    return "  ".join(components)


def _track_label(track_id: str | None) -> str:
    if track_id is None:
        return "TRACK --"
    parts = track_id.split(":")
    if len(parts) == 3 and parts[0] == "face":
        return f"TRACK {parts[2]} / SCENE {parts[1]}"
    if len(parts) == 4 and parts[0] == "face":
        return f"TRACK {parts[3]} / SCENE {parts[2]}"
    return f"TRACK {track_id}"


def _draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 0.52,
    thickness: int = 1,
) -> None:
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(origin[0], frame.shape[1] - text_width - 4))
    y = max(text_height + 4, min(origin[1], frame.shape[0] - baseline - 2))
    cv2.rectangle(
        frame,
        (x, y - text_height - 4),
        (x + text_width + 4, y + baseline + 2),
        (18, 18, 18),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 2, y - 2),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_cut_indicator(frame: np.ndarray) -> None:
    text = "CUT"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.72, frame.shape[1] / 2200.0)
    thickness = max(2, round(frame.shape[1] / 960))
    (text_width, _text_height), _baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    _draw_label(
        frame,
        text,
        (frame.shape[1] - text_width - 16, 28),
        (40, 40, 255),
        scale=scale,
        thickness=thickness,
    )


def _draw_dotted_rectangle(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    left, top = top_left
    right, bottom = bottom_right
    length = 10
    gap = 7
    for start in range(left, right + 1, length + gap):
        end = min(right, start + length)
        cv2.line(frame, (start, top), (end, top), color, thickness, cv2.LINE_AA)
        cv2.line(
            frame,
            (start, bottom),
            (end, bottom),
            color,
            thickness,
            cv2.LINE_AA,
        )
    for start in range(top, bottom + 1, length + gap):
        end = min(bottom, start + length)
        cv2.line(frame, (left, start), (left, end), color, thickness, cv2.LINE_AA)
        cv2.line(
            frame,
            (right, start),
            (right, end),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _draw_dotted_face_mask(
    frame: np.ndarray,
    mask,
    *,
    threshold: float,
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = mask.box
    left = max(0, min(width, int(math.floor(x1))))
    top = max(0, min(height, int(math.floor(y1))))
    right = max(0, min(width, int(math.ceil(x2))))
    bottom = max(0, min(height, int(math.ceil(y2))))
    if right <= left or bottom <= top:
        return
    probability = np.frombuffer(mask.probabilities, dtype=np.uint8).reshape(
        mask.height,
        mask.width,
    )
    resized = cv2.resize(
        probability,
        (right - left, bottom - top),
        interpolation=cv2.INTER_LINEAR,
    )
    binary = (
        np.asarray(
            resized >= round(threshold * 255.0),
            dtype=np.uint8,
        )
        * 255
    )
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    spacing = max(5.0, width / 300.0)
    radius = max(1, round(width / 1200))
    minimum_area = max(4.0, binary.size * 0.0002)
    offset = np.asarray((left, top), dtype=np.float32)
    for contour in contours:
        if cv2.contourArea(contour) < minimum_area:
            continue
        points = contour.reshape(-1, 2).astype(np.float32) + offset
        if len(points) < 2:
            continue
        closed = np.concatenate((points, points[:1]), axis=0)
        segments = closed[1:] - closed[:-1]
        lengths = np.linalg.norm(segments, axis=1)
        cumulative = np.concatenate((np.zeros(1, dtype=np.float32), np.cumsum(lengths)))
        total = float(cumulative[-1])
        if total <= 0.0:
            continue
        for distance in np.arange(0.0, total, spacing):
            segment = min(
                int(np.searchsorted(cumulative, distance, side="right") - 1),
                len(segments) - 1,
            )
            fraction = (distance - cumulative[segment]) / max(
                float(lengths[segment]),
                1e-6,
            )
            point = closed[segment] + fraction * segments[segment]
            center = (round(float(point[0])), round(float(point[1])))
            cv2.circle(
                frame,
                center,
                radius + 1,
                (225, 225, 225),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                center,
                radius,
                (0, 0, 0),
                -1,
                cv2.LINE_AA,
            )


def _draw_items(
    frame: np.ndarray,
    items: tuple[OverlayItem, ...],
    options: RenderOptions,
) -> tuple[int, int]:
    height, width = frame.shape[:2]
    mask_items = [item for item in items if item.kind == "mask"]
    face_items = [item for item in items if item.kind == "face"]
    privacy_masks: list[tuple[OverlayItem, FacePrivacyMask, np.ndarray]] = []
    if options.face_privacy_target != "none":
        for item in face_items:
            privacy = derive_privacy_mask(
                options.face_privacy_target,
                item.ellipse,
                item.keypoints,
                eye_shape=options.eye_mask_shape,
                minimum_eye_confidence=options.minimum_eye_confidence,
            )
            if privacy is None:
                continue
            contour = np.asarray(
                [
                    (
                        max(0, min(width - 1, round(x))),
                        max(0, min(height - 1, round(y))),
                    )
                    for x, y in privacy.polygon
                ],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            privacy_masks.append((item, privacy, contour))

    if mask_items and options.mask_alpha > 0.0:
        tint = frame.copy()
        for item in mask_items:
            contours = _contours(item, width, height)
            color = _color(item.color_key)
            # A single detection may be represented by multiple overlapping
            # ellipses/polygons. Passing all contours to one fillPoly call uses
            # an even-odd fill rule and cancels their overlap. Fill each
            # component into the same tint canvas so the result is their union.
            for contour in contours:
                cv2.fillPoly(tint, [contour], color, cv2.LINE_AA)
        cv2.addWeighted(
            tint,
            options.mask_alpha,
            frame,
            1.0 - options.mask_alpha,
            0.0,
            dst=frame,
        )

    if privacy_masks and options.mask_alpha > 0.0:
        tint = frame.copy()
        privacy_color = (255, 70, 255)
        for _item, _privacy, contour in privacy_masks:
            cv2.fillPoly(tint, [contour], privacy_color, cv2.LINE_AA)
        cv2.addWeighted(
            tint,
            options.mask_alpha,
            frame,
            1.0 - options.mask_alpha,
            0.0,
            dst=frame,
        )
        for _item, privacy, contour in privacy_masks:
            cv2.polylines(
                frame,
                [contour],
                True,
                privacy_color,
                options.outline_thickness,
                cv2.LINE_AA,
            )
            if options.display_style == "detailed":
                minimum = contour.reshape(-1, 2).min(axis=0)
                label = (
                    f"{privacy.target.upper()} MASK {privacy.shape} "
                    f"{privacy.derivation}"
                )
                if privacy.derivation == "eye-keypoints":
                    label += f" score={privacy.confidence:.2f}"
                _draw_label(
                    frame,
                    label,
                    (int(minimum[0]), int(minimum[1]) - 3),
                    privacy_color,
                )

    if face_items and options.mask_alpha > 0.0 and options.display_style == "legacy":
        for item in face_items:
            mask = item.face_mask
            if mask is None:
                continue
            x1, y1, x2, y2 = mask.box
            left = max(0, min(width, round(x1)))
            top = max(0, min(height, round(y1)))
            right = max(0, min(width, round(x2)))
            bottom = max(0, min(height, round(y2)))
            if right <= left or bottom <= top:
                continue
            probability = np.frombuffer(
                mask.probabilities,
                dtype=np.uint8,
            ).reshape(mask.height, mask.width)
            probability = cv2.resize(
                probability,
                (right - left, bottom - top),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            alpha = probability[:, :, None] * (options.mask_alpha / 255.0)
            roi = frame[top:bottom, left:right].astype(np.float32)
            color = np.asarray(
                _color(item.color_key),
                dtype=np.float32,
            ).reshape(1, 1, 3)
            frame[top:bottom, left:right] = np.clip(
                roi * (1.0 - alpha) + color * alpha,
                0,
                255,
            ).astype(np.uint8)

    if options.display_style != "simple":
        for item in mask_items:
            contours = _contours(item, width, height)
            if not contours:
                continue
            color = _color(item.color_key)
            if options.display_style == "detailed" and item.is_keyframe:
                color = _keyframe_highlight(color)
            cv2.polylines(
                frame,
                contours,
                True,
                color,
                options.outline_thickness,
                cv2.LINE_AA,
            )
            if options.display_style == "detailed":
                all_points = np.concatenate(contours, axis=0).reshape(-1, 2)
                minimum = all_points.min(axis=0)
                maximum = all_points.max(axis=0)
                cv2.rectangle(
                    frame,
                    (int(minimum[0]), int(minimum[1])),
                    (int(maximum[0]), int(maximum[1])),
                    color,
                    options.box_thickness,
                    cv2.LINE_AA,
                )
                _draw_label(
                    frame,
                    _detailed_mask_label(item),
                    (int(minimum[0]), int(minimum[1]) - 3),
                    color,
                )
            elif options.show_labels:
                all_points = np.concatenate(contours, axis=0).reshape(-1, 2)
                minimum = all_points.min(axis=0)
                _draw_label(
                    frame,
                    _ascii_label(item),
                    (int(minimum[0]), int(minimum[1]) - 3),
                    color,
                )

    if options.display_style == "legacy":
        face_box_color = None
        ellipse_color = None
    else:
        face_box_color = (255, 170, 30)
        ellipse_color = (255, 70, 255)

    for item in face_items:
        interpolated = item.provenance == "INTERPOLATED"
        removed = item.provenance == "REMOVED_SHORT_TRACK"
        if options.display_style == "detailed" and item.face_mask is not None:
            _draw_dotted_face_mask(
                frame,
                item.face_mask,
                threshold=options.face_mask_threshold,
            )
        if removed:
            color = (70, 70, 255)
        elif interpolated:
            color = (0, 210, 255)
        else:
            color = (
                _color(item.color_key)
                if face_box_color is None
                else face_box_color
                if item.face_present is not False
                else (80, 80, 255)
            )
        label_origin: tuple[int, int] | None = None
        if item.box is not None and options.display_style != "simple":
            x1, y1, x2, y2 = item.box
            left = max(0, min(width - 1, round(x1)))
            top = max(0, min(height - 1, round(y1)))
            right = max(0, min(width - 1, round(x2)))
            bottom = max(0, min(height - 1, round(y2)))
            if interpolated:
                _draw_dotted_rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    color,
                    options.box_thickness,
                )
            else:
                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    color,
                    options.box_thickness,
                    cv2.LINE_AA,
                )
            if removed:
                cv2.line(
                    frame,
                    (left, bottom),
                    (right, top),
                    color,
                    max(3, options.box_thickness + 1),
                    cv2.LINE_AA,
                )
            label_origin = (
                (left, min(height - 2, bottom + 24))
                if (removed or interpolated) and top < 25
                else (left, max(top, 22))
            )
        if item.ellipse is not None and options.draw_face_ellipses:
            cx, cy, major, minor, theta = item.ellipse
            center = (
                max(0, min(width - 1, round(cx))),
                max(0, min(height - 1, round(cy))),
            )
            axes = (max(1, round(major)), max(1, round(minor)))
            cv2.ellipse(
                frame,
                center,
                axes,
                math.degrees(theta),
                0,
                360,
                color if ellipse_color is None or removed else ellipse_color,
                options.box_thickness,
                cv2.LINE_AA,
            )
            if options.display_style == "legacy":
                label_origin = (
                    max(0, round(cx - major)),
                    max(0, round(cy - minor)) - 3,
                )
        point_radius = max(4, round(width / 480))
        font_scale = max(0.48, width / 2400)
        for point in item.keypoints if options.draw_face_keypoints else ():
            if not point.valid or point.state == 0:
                continue
            px = max(0, min(width - 1, round(point.x)))
            py = max(0, min(height - 1, round(point.y)))
            point_color = (0, 165, 255) if point.state == 1 else (0, 255, 0)
            if point.state == 1:
                cv2.circle(
                    frame,
                    (px, py),
                    point_radius + 2,
                    point_color,
                    options.box_thickness,
                    cv2.LINE_AA,
                )
                marker = "O"
            else:
                cv2.circle(
                    frame,
                    (px, py),
                    point_radius,
                    point_color,
                    -1,
                    cv2.LINE_AA,
                )
                marker = "V"
            if options.display_style == "detailed":
                state_confidence = (
                    point.confidence
                    if point.state_confidence is None
                    else point.state_confidence
                )
                text = (
                    f"{point.class_name.upper()}:{marker} "
                    f"p{point.confidence:.2f}/s{state_confidence:.2f}"
                )
                cv2.putText(
                    frame,
                    text,
                    (px + point_radius + 3, py - point_radius - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.72,
                    point_color,
                    max(1, options.box_thickness - 1),
                    cv2.LINE_AA,
                )
        if options.display_style == "detailed" and label_origin is not None:
            head_score = "--" if item.score is None else f"{item.score:.2f}"
            face_score = "--" if item.face_score is None else f"{item.face_score:.2f}"
            compact_track = _track_label(item.track_id).replace(" / SCENE ", " / S")
            if interpolated:
                label = f"{compact_track} | INTERPOLATED"
            elif removed:
                label = f"{compact_track} | REMOVED | HEAD {head_score}"
            elif item.face_present is not False:
                label = (
                    f"{_track_label(item.track_id)} | OBSERVED | "
                    f"HEAD {head_score} | FACE {face_score}"
                )
            else:
                label = (
                    f"{_track_label(item.track_id)} | OBSERVED | "
                    f"HEAD {head_score} | NO FACE {face_score}"
                )
            _draw_label(
                frame,
                label,
                label_origin,
                color,
                scale=0.52,
                thickness=1,
            )
        elif (
            options.display_style == "legacy"
            and options.show_labels
            and label_origin is not None
        ):
            _draw_label(frame, _ascii_label(item), label_origin, color)
    return len(mask_items), len(face_items)


def _validate_source_video(
    source: SourceInfo,
    *,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
) -> None:
    if source.width is not None and source.width != width:
        raise ValueError(
            f"{source.path}: source/video width mismatch: {source.width} != {width}"
        )
    if source.height is not None and source.height != height:
        raise ValueError(
            f"{source.path}: source/video height mismatch: {source.height} != {height}"
        )
    if source.fps is not None and abs(source.fps - fps) > 0.02:
        raise ValueError(
            f"{source.path}: source/video fps mismatch: {source.fps} != {fps}"
        )
    if (
        frame_count > 0
        and source.last_frame is not None
        and source.last_frame >= frame_count
    ):
        raise ValueError(
            f"{source.path}: frame {source.last_frame} exceeds video frame count "
            f"{frame_count}"
        )


def render_video(
    *,
    video_path: Path,
    output_path: Path,
    mask_frames: Iterable[FrameOverlay] | None,
    face_frames: Iterable[FrameOverlay] | None,
    sources: tuple[SourceInfo, ...],
    options: RenderOptions,
    cut_frames: frozenset[int] | set[int] | None = None,
    overwrite: bool = False,
) -> RenderSummary:
    """Render an overlay video atomically using OpenCV and CPU decoding."""

    options.validate()
    video_path = Path(video_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if output_path == video_path:
        raise ValueError("output video must differ from input video")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_end = (
        frame_count - 1
        if options.end_frame is None
        else min(frame_count - 1, options.end_frame)
    )
    progress_total = max(0, selected_end - options.start_frame + 1)
    _emit_overlay_progress(0, progress_total, state="running", fps=None)
    for source in sources:
        _validate_source_video(
            source,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )
    if options.uses_ffmpeg_h264:
        try:
            writer: cv2.VideoWriter | _H264Writer = _H264Writer(
                temporary,
                width=width,
                height=height,
                fps=fps,
                crf=options.h264_crf,
                preset=(
                    options.nvenc_preset if options.uses_nvenc else options.h264_preset
                ),
                nvenc=options.uses_nvenc,
                nvenc_cq=options.nvenc_cq,
                nvenc_gpu=options.nvenc_gpu,
                target_bitrate_mbps=options.target_bitrate_mbps,
                ffmpeg_bin=options.ffmpeg_bin,
            )
        except BaseException:
            capture.release()
            temporary.unlink(missing_ok=True)
            raise
    else:
        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*options.codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(
                f"failed to create video with codec {options.codec!r}: {output_path}"
            )

    masks = _SparseFrames(mask_frames)
    faces = _SparseFrames(face_frames)
    cuts = cut_frames or frozenset()
    started = time.perf_counter()
    last_progress_emit = started
    progress_interval = _progress_interval_seconds()
    frames_written = 0
    masks_drawn = 0
    faces_drawn = 0
    first_written: int | None = None
    last_written: int | None = None
    frame_index = 0
    decode_seconds = 0.0
    source_seconds = 0.0
    draw_seconds = 0.0
    write_seconds = 0.0
    try:
        frame_index = _seek_capture(capture, options.start_frame)
        while True:
            phase_started = time.perf_counter()
            ok, frame = capture.read()
            decode_seconds += time.perf_counter() - phase_started
            if not ok or frame is None:
                break
            if options.end_frame is not None and frame_index > options.end_frame:
                break
            phase_started = time.perf_counter()
            frame_items = masks.take(frame_index) + faces.take(frame_index)
            source_seconds += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            mask_count, face_count = _draw_items(frame, frame_items, options)
            if options.display_style == "detailed" and frame_index in cuts:
                _draw_cut_indicator(frame)
            draw_seconds += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            writer.write(frame)
            write_seconds += time.perf_counter() - phase_started
            frames_written += 1
            masks_drawn += mask_count
            faces_drawn += face_count
            first_written = frame_index if first_written is None else first_written
            last_written = frame_index
            progress_now = time.perf_counter()
            if (
                frames_written >= progress_total
                or progress_now - last_progress_emit >= progress_interval
            ):
                _emit_overlay_progress(
                    frames_written,
                    progress_total,
                    state="running",
                    fps=frames_written / max(progress_now - started, 1e-9),
                )
                last_progress_emit = progress_now
            if options.progress_every and frames_written % options.progress_every == 0:
                print(
                    f"[overlay] frames={frames_written} "
                    f"source_frame={frame_index} masks={masks_drawn} "
                    f"faces={faces_drawn}",
                    flush=True,
                )
            frame_index += 1
    except BaseException:
        masks.close()
        faces.close()
        if isinstance(writer, _H264Writer):
            writer.abort()
        else:
            writer.release()
        capture.release()
        temporary.unlink(missing_ok=True)
        raise
    else:
        masks.close()
        faces.close()
        if isinstance(writer, _H264Writer):
            try:
                writer.close()
            except BaseException:
                capture.release()
                temporary.unlink(missing_ok=True)
                raise
        else:
            writer.release()
        capture.release()
        if frames_written == 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("no video frames were written")
        os.replace(temporary, output_path)

    elapsed_seconds = time.perf_counter() - started
    _emit_overlay_progress(
        frames_written,
        progress_total,
        state="complete",
        fps=frames_written / max(elapsed_seconds, 1e-9),
    )
    accounted_seconds = decode_seconds + source_seconds + draw_seconds + write_seconds
    return RenderSummary(
        mode=options.mode,
        output=output_path,
        frames_written=frames_written,
        masks_drawn=masks_drawn,
        faces_drawn=faces_drawn,
        first_frame=first_written,
        last_frame=last_written,
        width=width,
        height=height,
        fps=fps,
        codec=options.normalized_codec,
        elapsed_seconds=elapsed_seconds,
        decode_seconds=decode_seconds,
        source_seconds=source_seconds,
        draw_seconds=draw_seconds,
        write_seconds=write_seconds,
        other_seconds=max(0.0, elapsed_seconds - accounted_seconds),
    )


__all__ = ["RenderOptions", "render_video"]
