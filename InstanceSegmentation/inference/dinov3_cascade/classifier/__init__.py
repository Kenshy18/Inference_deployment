"""DINOv3 Cascade family classifier.

Import the dependency-free checkpoint contract here.  PyTorch implementation
modules remain explicit imports so catalog and CLI discovery stay lightweight.
"""

from .contracts import (
    ClassifierCheckpointContract,
    RoiFeatureRequirements,
    contract_from_checkpoint,
)

__all__ = [
    "ClassifierCheckpointContract",
    "RoiFeatureRequirements",
    "contract_from_checkpoint",
]
