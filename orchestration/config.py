"""Typed JSON configuration for the repository-level workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .config_sections import (
    ExecutionConfig,
    InferenceConfig,
    OverlayConfig,
    PostprocessConfig,
)
from .config_support import OrchestrationConfigError
from .config_loader import load_config
from .config_validation import validate_config


@dataclass(frozen=True)
class OrchestrationConfig:
    schema_version: int
    config_path: Path
    input_video: Path
    output_root: Path
    execution: ExecutionConfig
    inference: InferenceConfig
    postprocess: PostprocessConfig
    overlay: OverlayConfig

    @classmethod
    def load(cls, path: Path) -> "OrchestrationConfig":
        return load_config(cls, path)

    def validate(self) -> None:
        validate_config(self)

    def resolved_dict(self) -> dict[str, object]:
        def convert(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            return value

        values = asdict(self)
        values.pop("config_path", None)
        return convert(values)  # type: ignore[return-value]


__all__ = [
    "ExecutionConfig",
    "InferenceConfig",
    "OrchestrationConfig",
    "OrchestrationConfigError",
    "OverlayConfig",
    "PostprocessConfig",
]
