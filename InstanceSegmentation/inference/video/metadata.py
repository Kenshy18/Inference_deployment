"""Video metadata independent of model families."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    frames: int
    fps: float
    width: int
    height: int


def read_video_metadata(path: Path) -> VideoMetadata:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return VideoMetadata(
            frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            fps=fps if fps > 0 else 30.0,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


__all__ = ["VideoMetadata", "read_video_metadata"]
