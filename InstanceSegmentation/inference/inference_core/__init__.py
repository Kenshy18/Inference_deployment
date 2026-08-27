"""Framework-neutral contracts and runtime services shared by inference models.

Model families may import this package, but ``inference_core`` never imports a
model implementation at module import time.  Keeping the shared runtime under
one namespace avoids collisions with repository-level packages such as
``orchestration`` and the postprocess package's public ``contracts`` module.
"""

from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parent.parent

from .contracts import TaskType

__all__ = ["INFERENCE_ROOT", "TaskType"]
