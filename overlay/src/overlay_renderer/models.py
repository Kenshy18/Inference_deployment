"""Data structures shared by SQLite readers and the renderer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


Point = tuple[float, float]
Polygon = tuple[Point, ...]
Box = tuple[float, float, float, float]
Ellipse = tuple[float, float, float, float, float]
OverlayKind = Literal["mask", "face"]


@dataclass(frozen=True)
class FaceKeypointOverlay:
    x: float
    y: float
    class_name: str
    state: int
    state_name: str
    confidence: float
    valid: bool


@dataclass(frozen=True)
class FaceMaskOverlay:
    width: int
    height: int
    box: Box
    probabilities: bytes


@dataclass(frozen=True)
class OverlayItem:
    """One mask or detection box to draw on a video frame."""

    identity: str
    color_key: str
    kind: OverlayKind
    label: str = ""
    score: float | None = None
    track_id: str | None = None
    polygons: tuple[Polygon, ...] = ()
    box: Box | None = None
    ellipse: Ellipse | None = None
    keypoints: tuple[FaceKeypointOverlay, ...] = ()
    face_mask: FaceMaskOverlay | None = None


@dataclass(frozen=True)
class FrameOverlay:
    frame_index: int
    items: tuple[OverlayItem, ...]


@dataclass(frozen=True)
class SourceInfo:
    """Validated facts about an input SQLite artifact."""

    path: Path
    schema: str
    role: str
    item_count: int
    first_frame: int | None
    last_frame: int | None
    width: int | None = None
    height: int | None = None
    fps: float | None = None

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["path"] = str(self.path)
        return values


@dataclass(frozen=True)
class RenderSummary:
    mode: str
    output: Path
    frames_written: int
    masks_drawn: int
    faces_drawn: int
    first_frame: int | None
    last_frame: int | None
    width: int
    height: int
    fps: float
    codec: str
    elapsed_seconds: float

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["output"] = str(self.output)
        return values
