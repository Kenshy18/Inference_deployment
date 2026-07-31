"""Standalone renderer for inference and postprocess SQLite overlays."""

from .models import OverlayItem, RenderSummary, SourceInfo
from .render import RenderOptions, render_video
from .sources import (
    OverlayContractError,
    inspect_inference_source,
    inspect_mask_source,
    iter_face_frames,
    iter_mask_frames,
    iter_raw_segmentation_frames,
    load_cut_frames,
)

__all__ = [
    "OverlayContractError",
    "OverlayItem",
    "RenderOptions",
    "RenderSummary",
    "SourceInfo",
    "inspect_inference_source",
    "inspect_mask_source",
    "iter_face_frames",
    "iter_mask_frames",
    "iter_raw_segmentation_frames",
    "load_cut_frames",
    "render_video",
]
