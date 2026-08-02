"""Replaceable video cut detectors."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol, cast

import cv2
import numpy as np

from contracts.detections import iter_detection_records


@dataclass(frozen=True)
class CutDetectionResult:
    frames: list[int]
    elapsed_seconds: float
    method: str


class CutDetector(Protocol):
    """Contract implemented by every cut-detection algorithm."""

    name: str

    def detect(self, jsonl_path: Path, video_path: Path) -> CutDetectionResult:
        """Return the first frame of every detected scene."""


def _load_frame_indices(path: Path) -> list[int]:
    return [int(record["frame_index"]) for record in iter_detection_records(path)]


def _read_video_frames(
    video_path: Path, frame_indices: list[int]
) -> Iterator[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    current_position: int | None = None
    try:
        for frame_index in frame_indices:
            if current_position != frame_index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to read video frame {frame_index}")
            current_position = frame_index + 1
            yield frame_index, frame
    finally:
        capture.release()


def _read_all_video_frames(
    video_path: Path,
    max_frames: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    frame_index = 0
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()


def _resolve_ffmpeg() -> Path | None:
    configured = os.environ.get("VIDEO_MASK_FFMPEG")
    if configured:
        path = Path(configured).expanduser().resolve()
        return path if path.is_file() else None
    executable = shutil.which("ffmpeg")
    if executable:
        return Path(executable).resolve()
    runtime_root = Path(sys.executable).resolve()
    if len(runtime_root.parents) >= 4:
        bundled = runtime_root.parents[3] / "tools" / "ffmpeg" / "bin" / "ffmpeg"
        if bundled.is_file():
            return bundled
    # The deployment image intentionally keeps native media tools with the
    # overlay repository instead of duplicating them in the Python runtime.
    # Prefer its static FFmpeg before falling back to OpenCV full-frame decode.
    repository_root = Path(__file__).resolve().parents[2]
    repository_ffmpeg = (
        repository_root
        / "overlay"
        / ".runtime"
        / "ffmpeg-nvenc-btbn-8.1"
        / "bin"
        / "ffmpeg"
    )
    if repository_ffmpeg.is_file():
        return repository_ffmpeg
    return None


def _read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_small_video_frames_ffmpeg(
    video_path: Path,
    frame_indices: list[int],
    *,
    width: int,
    height: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Decode a contiguous zero-based range directly at analysis resolution."""

    if not all(
        frame_index == expected
        for expected, frame_index in enumerate(frame_indices)
    ):
        raise ValueError("FFmpeg cut reader requires contiguous zero-based frames")
    yield from _read_all_small_video_frames_ffmpeg(
        video_path,
        width=width,
        height=height,
        max_frames=len(frame_indices),
        require_max_frames=True,
    )


def _read_all_small_video_frames_ffmpeg(
    video_path: Path,
    *,
    width: int,
    height: int,
    max_frames: int | None = None,
    require_max_frames: bool = False,
) -> Iterator[tuple[int, np.ndarray]]:
    """Decode a whole video, or its prefix, directly at analysis resolution."""

    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg was not found")
    frame_bytes = width * height * 3
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={width}:{height}:flags=area,format=bgr24",
        "-an",
        "-sn",
        "-dn",
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    if max_frames is not None:
        frames_position = command.index("-fps_mode")
        command[frames_position:frames_position] = [
            "-frames:v",
            str(max_frames),
        ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_bytes * 16,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        frame_index = 0
        while max_frames is None or frame_index < max_frames:
            payload = _read_exact(process.stdout, frame_bytes)
            if not payload and not require_max_frames:
                break
            if len(payload) != frame_bytes:
                error = process.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"FFmpeg stopped at cut-detection frame {frame_index}: {error}"
                )
            yield (
                frame_index,
                np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3),
            )
            frame_index += 1
        return_code = process.wait()
        if return_code != 0:
            error = process.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"FFmpeg cut reader failed with exit code {return_code}: {error}"
            )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _small_gray(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def _small_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _histogram_correlation(previous: np.ndarray, current: np.ndarray) -> float:
    previous_histogram = cv2.calcHist(
        [previous], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    current_histogram = cv2.calcHist(
        [current], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    previous_histogram = cv2.normalize(previous_histogram, previous_histogram).flatten()
    current_histogram = cv2.normalize(current_histogram, current_histogram).flatten()
    return float(
        cv2.compareHist(
            previous_histogram.astype("float32"),
            current_histogram.astype("float32"),
            cv2.HISTCMP_CORREL,
        )
    )


def _ssim(previous: np.ndarray, current: np.ndarray) -> float:
    first = previous.astype(np.float32)
    second = current.astype(np.float32)
    first_mean = cv2.GaussianBlur(first, (7, 7), 1.5)
    second_mean = cv2.GaussianBlur(second, (7, 7), 1.5)
    first_variance = cv2.GaussianBlur(first * first, (7, 7), 1.5) - first_mean**2
    second_variance = cv2.GaussianBlur(second * second, (7, 7), 1.5) - second_mean**2
    covariance = (
        cv2.GaussianBlur(first * second, (7, 7), 1.5) - first_mean * second_mean
    )
    score = (
        (2.0 * first_mean * second_mean + 6.5025) * (2.0 * covariance + 58.5225)
    ) / (
        (first_mean**2 + second_mean**2 + 6.5025)
        * (first_variance + second_variance + 58.5225)
    )
    return float(np.mean(score))


@dataclass(frozen=True)
class FrameDifferenceCutDetector:
    """Fast detector based on downscaled grayscale mean difference."""

    name: str = "frame_diff"
    threshold: float = 18.0
    min_gap_frames: int = 15
    width: int = 96
    height: int = 54

    def detect(self, jsonl_path: Path, video_path: Path) -> CutDetectionResult:
        started = time.perf_counter()
        from common.live_preview import PreviewGeometry, active_postprocess_preview

        preview = active_postprocess_preview()
        frames: list[int] = []
        previous: np.ndarray | None = None
        last_cut = -(10**9)
        for frame_index, frame in _read_video_frames(
            video_path, _load_frame_indices(jsonl_path)
        ):
            current = _small_gray(frame, self.width, self.height)
            if previous is not None:
                difference = float(np.mean(cv2.absdiff(current, previous)))
                if (
                    difference >= self.threshold
                    and frame_index - last_cut > self.min_gap_frames
                ):
                    frames.append(frame_index)
                    last_cut = frame_index
            previous = current
            if preview is not None and preview.should_sample("cut_detection"):
                preview.submit(
                    PreviewGeometry(
                        frame_index,
                        "cut_detection",
                        "cut detection",
                        detail=f"frame-diff · cuts {len(frames)}",
                        preview_image=cv2.cvtColor(current, cv2.COLOR_GRAY2BGR),
                    )
                )
        return CutDetectionResult(frames, time.perf_counter() - started, self.name)


@dataclass(frozen=True)
class HighPrecisionCutDetector:
    """Difference + color histogram + SSIM detector."""

    name: str = "high_precision"
    min_difference: float = 18.0
    normal_difference: float = 30.0
    strong_difference: float = 45.0
    hard_difference: float = 70.0
    color_correlation_max: float = 0.96
    strong_color_correlation_max: float = 0.98
    ssim_max: float = 0.45
    strong_ssim_max: float = 0.55
    min_gap_frames: int = 45
    width: int = 96
    height: int = 54
    accelerated_decode: bool = True

    def _is_cut(self, difference: float, color_correlation: float, ssim: float) -> bool:
        if difference < self.min_difference:
            return False
        if difference >= self.hard_difference:
            return True
        if difference >= self.strong_difference:
            return (
                color_correlation <= self.strong_color_correlation_max
                and ssim <= self.strong_ssim_max
            )
        return (
            difference >= self.normal_difference
            and color_correlation <= self.color_correlation_max
            and ssim <= self.ssim_max
        )

    def _detect_reader(
        self,
        reader: Iterator[tuple[int, np.ndarray]],
        *,
        frames_are_small: bool,
        started: float,
    ) -> CutDetectionResult:
        cuts: list[int] = []
        previous_gray: np.ndarray | None = None
        previous_bgr: np.ndarray | None = None
        last_cut = -(10**9)
        from common.live_preview import PreviewGeometry, active_postprocess_preview

        preview = active_postprocess_preview()
        for frame_index, frame in reader:
            current_bgr = (
                frame
                if frames_are_small
                else _small_bgr(frame, self.width, self.height)
            )
            current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
            is_cut = False
            difference = 0.0
            if previous_gray is not None and previous_bgr is not None:
                difference = float(np.mean(cv2.absdiff(current_gray, previous_gray)))
                if difference >= self.min_difference:
                    color_correlation = _histogram_correlation(
                        previous_bgr, current_bgr
                    )
                    structural_similarity = _ssim(previous_gray, current_gray)
                    if (
                        self._is_cut(
                            difference, color_correlation, structural_similarity
                        )
                        and frame_index - last_cut > self.min_gap_frames
                    ):
                        cuts.append(frame_index)
                        last_cut = frame_index
                        is_cut = True
            if preview is not None and preview.should_sample("cut_detection"):
                preview.submit(
                    PreviewGeometry(
                        frame_index,
                        "cut_detection",
                        "cut detection",
                        detail=(
                            f"CUT · diff {difference:.1f} · total {len(cuts)}"
                            if is_cut
                            else f"scan · diff {difference:.1f} · cuts {len(cuts)}"
                        ),
                        is_keyframe=is_cut,
                        preview_image=current_bgr.copy(),
                    )
                )
            previous_gray = current_gray
            previous_bgr = current_bgr
        return CutDetectionResult(cuts, time.perf_counter() - started, self.name)

    def detect(self, jsonl_path: Path, video_path: Path) -> CutDetectionResult:
        started = time.perf_counter()
        frame_indices = _load_frame_indices(jsonl_path)
        reader: Iterator[tuple[int, np.ndarray]]
        if (
            self.accelerated_decode
            and all(
                frame_index == expected
                for expected, frame_index in enumerate(frame_indices)
            )
            and _resolve_ffmpeg() is not None
        ):
            reader = _read_small_video_frames_ffmpeg(
                video_path,
                frame_indices,
                width=self.width,
                height=self.height,
            )
            frames_are_small = True
        else:
            reader = _read_video_frames(video_path, frame_indices)
            frames_are_small = False
        return self._detect_reader(
            reader,
            frames_are_small=frames_are_small,
            started=started,
        )

    def detect_video(
        self,
        video_path: Path,
        *,
        max_frames: int | None = None,
    ) -> CutDetectionResult:
        """Detect cuts without first materializing a detector JSONL."""

        started = time.perf_counter()
        if self.accelerated_decode and _resolve_ffmpeg() is not None:
            reader = _read_all_small_video_frames_ffmpeg(
                video_path,
                width=self.width,
                height=self.height,
                max_frames=max_frames,
                require_max_frames=False,
            )
            frames_are_small = True
        else:
            reader = _read_all_video_frames(video_path, max_frames)
            frames_are_small = False
        return self._detect_reader(
            reader,
            frames_are_small=frames_are_small,
            started=started,
        )


@dataclass(frozen=True)
class DisabledCutDetector:
    name: str = "disabled"

    def detect(self, jsonl_path: Path, video_path: Path) -> CutDetectionResult:
        return CutDetectionResult([], 0.0, self.name)


CUT_DETECTORS: dict[str, Callable[[], CutDetector]] = {
    "frame_diff": FrameDifferenceCutDetector,
    "high_precision": HighPrecisionCutDetector,
}


def register_cut_detector(
    name: str,
    factory: Callable[[], CutDetector],
    *,
    replace: bool = False,
) -> None:
    """Register a named detector factory for embedding applications."""

    if not name:
        raise ValueError("cut detector name must not be empty")
    if name in CUT_DETECTORS and not replace:
        raise ValueError(f"cut detector already registered: {name}")
    CUT_DETECTORS[name] = factory


def _load_detector(spec: str) -> CutDetector:
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            "custom cut detector must use 'python.module:attribute' syntax"
        )
    value = getattr(importlib.import_module(module_name), attribute_name)
    instance = value() if callable(value) else value
    if not hasattr(instance, "name") or not callable(getattr(instance, "detect", None)):
        raise TypeError(
            f"{spec!r} does not provide a CutDetector (name + detect method)"
        )
    return cast(CutDetector, instance)


def create_cut_detector(name: str) -> CutDetector:
    """Create a built-in or ``python.module:attribute`` detector."""

    if ":" in name:
        return _load_detector(name)
    try:
        factory = CUT_DETECTORS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown cut detector {name!r}; choose from "
            f"{sorted(CUT_DETECTORS)} or use python.module:attribute"
        ) from exc
    return factory()


def detect_cut_frames(
    jsonl_path: Path,
    video_path: Path,
    *,
    method: str = "high_precision",
) -> tuple[list[int], float, str]:
    result = create_cut_detector(method).detect(jsonl_path, video_path)
    return result.frames, result.elapsed_seconds, result.method
