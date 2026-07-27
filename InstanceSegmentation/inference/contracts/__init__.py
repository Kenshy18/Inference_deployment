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
from .face_attributes import (
    KEYPOINT_CLASS_NAMES,
    KEYPOINT_STATE_NAMES,
    FaceEllipse,
    FaceKeypoint,
    FaceMask,
    FaceObservation,
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
    "FaceEllipse",
    "FaceKeypoint",
    "FaceMask",
    "FaceObservation",
    "Frame",
    "FrameBatch",
    "FrameReference",
    "InferenceFrame",
    "InstanceSegmentationAdapter",
    "KEYPOINT_CLASS_NAMES",
    "KEYPOINT_STATE_NAMES",
    "ModelDescriptor",
    "ObjectDetectionAdapter",
    "Segmentation",
    "SegmentationFrame",
    "SegmentationInstance",
    "TaskType",
    "VisionAdapter",
    "segmentation_frame_from_rows",
]
