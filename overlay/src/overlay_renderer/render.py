"""CPU video renderer for sparse frame overlays."""

from __future__ import annotations

import hashlib
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
    / "ffmpeg-nvenc"
    / "bin"
    / "ffmpeg"
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

    def validate(self) -> None:
        if not 0.0 <= self.mask_alpha <= 1.0:
            raise ValueError("mask_alpha must be between 0 and 1")
        if self.outline_thickness < 1 or self.box_thickness < 1:
            raise ValueError("line thickness must be at least 1")
        if (
            len(self.codec) != 4
            and self.codec.lower() not in NVENC_CODECS
        ):
            raise ValueError(
                "codec must be a four-character FourCC or h264_nvenc"
            )
        if not 0 <= self.h264_crf <= 51:
            raise ValueError("h264_crf must be between 0 and 51")
        if self.h264_preset not in H264_PRESETS:
            raise ValueError(
                f"h264_preset must be one of {', '.join(H264_PRESETS)}"
            )
        if not 0 <= self.nvenc_cq <= 51:
            raise ValueError("nvenc_cq must be between 0 and 51")
        if self.nvenc_preset not in NVENC_PRESETS:
            raise ValueError(
                f"nvenc_preset must be one of {', '.join(NVENC_PRESETS)}"
            )
        if self.nvenc_gpu < 0:
            raise ValueError("nvenc_gpu must be non-negative")
        if (
            self.target_bitrate_mbps is not None
            and self.target_bitrate_mbps <= 0
        ):
            raise ValueError("target_bitrate_mbps must be positive")
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.end_frame is not None and self.end_frame < self.start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")
        if self.progress_every < 0:
            raise ValueError("progress_every must be non-negative")

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
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # Keep colors bright enough to remain visible over dark and mid-tone video.
    values = [96 + int(value) * 159 // 255 for value in digest[:3]]
    return int(values[0]), int(values[1]), int(values[2])


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


def _draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
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


def _draw_items(
    frame: np.ndarray,
    items: tuple[OverlayItem, ...],
    options: RenderOptions,
) -> tuple[int, int]:
    height, width = frame.shape[:2]
    mask_items = [item for item in items if item.kind == "mask"]
    face_items = [item for item in items if item.kind == "face"]

    if mask_items and options.mask_alpha > 0.0:
        tint = frame.copy()
        for item in mask_items:
            contours = _contours(item, width, height)
            if contours:
                cv2.fillPoly(tint, contours, _color(item.color_key), cv2.LINE_AA)
        cv2.addWeighted(
            tint,
            options.mask_alpha,
            frame,
            1.0 - options.mask_alpha,
            0.0,
            dst=frame,
        )

    for item in mask_items:
        contours = _contours(item, width, height)
        if not contours:
            continue
        color = _color(item.color_key)
        cv2.polylines(
            frame,
            contours,
            True,
            color,
            options.outline_thickness,
            cv2.LINE_AA,
        )
        if options.show_labels:
            all_points = np.concatenate(contours, axis=0).reshape(-1, 2)
            minimum = all_points.min(axis=0)
            _draw_label(
                frame,
                _ascii_label(item),
                (int(minimum[0]), int(minimum[1]) - 3),
                color,
            )

    for item in face_items:
        if item.box is None:
            continue
        color = _color(item.color_key)
        x1, y1, x2, y2 = item.box
        left = max(0, min(width - 1, round(x1)))
        top = max(0, min(height - 1, round(y1)))
        right = max(0, min(width - 1, round(x2)))
        bottom = max(0, min(height - 1, round(y2)))
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            color,
            options.box_thickness,
            cv2.LINE_AA,
        )
        if options.show_labels:
            _draw_label(
                frame,
                _ascii_label(item),
                (left, top - 3),
                color,
            )
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
                    options.nvenc_preset
                    if options.uses_nvenc
                    else options.h264_preset
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
    started = time.perf_counter()
    frames_written = 0
    masks_drawn = 0
    faces_drawn = 0
    first_written: int | None = None
    last_written: int | None = None
    frame_index = 0
    try:
        frame_index = _seek_capture(capture, options.start_frame)
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if options.end_frame is not None and frame_index > options.end_frame:
                break
            frame_items = masks.take(frame_index) + faces.take(frame_index)
            mask_count, face_count = _draw_items(frame, frame_items, options)
            writer.write(frame)
            frames_written += 1
            masks_drawn += mask_count
            faces_drawn += face_count
            first_written = frame_index if first_written is None else first_written
            last_written = frame_index
            if (
                options.progress_every
                and frames_written % options.progress_every == 0
            ):
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
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = ["RenderOptions", "render_video"]
