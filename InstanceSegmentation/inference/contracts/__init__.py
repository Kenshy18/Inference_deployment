"""Stable task contracts shared by inference model families."""

from .classification import Classification
from .common import (
    CONTRACT_VERSION,
    ColorSpace,
    Frame,
    FrameBatch,
    FrameReference,
    ModelDescriptor,
    TaskType,
    VisionAdapter,
)
from .instance_segmentation import (
    InstanceSegmentationAdapter,
    Segmentation,
    SegmentationFrame,
    SegmentationInstance,
    segmentation_frame_from_rows,
)
from .object_detection import (
    BoundingBox,
    Detection,
    DetectionFrame,
    ObjectDetectionAdapter,
)

InferenceFrame = DetectionFrame | SegmentationFrame

__all__ = [
    "CONTRACT_VERSION",
    "BoundingBox",
    "Classification",
    "ColorSpace",
    "Detection",
    "DetectionFrame",
    "Frame",
    "FrameBatch",
    "FrameReference",
    "InferenceFrame",
    "InstanceSegmentationAdapter",
    "ModelDescriptor",
    "ObjectDetectionAdapter",
    "Segmentation",
    "SegmentationFrame",
    "SegmentationInstance",
    "TaskType",
    "VisionAdapter",
    "segmentation_frame_from_rows",
]
