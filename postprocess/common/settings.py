"""Shared paths and model resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelLayout:
    root: Path
    k2_dir: Path


def project_root() -> Path:
    return PROJECT_ROOT


def default_model_root() -> Path:
    env_value = os.environ.get("POSTPROCESS_MODEL_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return PROJECT_ROOT / "models"


def resolve_models(model_root: Path | None = None) -> ModelLayout:
    root = model_root.expanduser().resolve() if model_root else default_model_root()
    return ModelLayout(
        root=root,
        k2_dir=root / "k2_v5",
    )
