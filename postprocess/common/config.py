"""JSON configuration for feature-stage ordering, replacement, and insertion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    id: str
    implementation: str
    options: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def from_dict(cls, value: object) -> "StageSpec":
        if not isinstance(value, dict):
            raise ValueError("each pipeline stage must be an object")
        stage_id = str(value.get("id", "")).strip()
        implementation = str(value.get("implementation", "")).strip()
        if not stage_id or not implementation:
            raise ValueError("pipeline stage requires id and implementation")
        options = value.get("options", {})
        if not isinstance(options, dict):
            raise ValueError(f"stage {stage_id}: options must be an object")
        return cls(
            id=stage_id,
            implementation=implementation,
            options=dict(options),
            enabled=bool(value.get("enabled", True)),
        )


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    stages: tuple[StageSpec, ...]

    @classmethod
    def from_dict(cls, value: object) -> "PipelineConfig":
        if not isinstance(value, dict):
            raise ValueError("pipeline config must be an object")
        name = str(value.get("name", "postprocess")).strip() or "postprocess"
        raw_stages = value.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("pipeline config requires a non-empty stages list")
        stages = tuple(StageSpec.from_dict(stage) for stage in raw_stages)
        ids = [stage.id for stage in stages if stage.enabled]
        if len(ids) != len(set(ids)):
            raise ValueError("enabled pipeline stage ids must be unique")
        return cls(name=name, stages=stages)


def load_pipeline_config(path: Path) -> PipelineConfig:
    return PipelineConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def default_polygon_pipeline(*, include_preprocess: bool) -> PipelineConfig:
    stages: list[StageSpec] = []
    if include_preprocess:
        stages.extend(
            [
                StageSpec("normalization", "preprocessing.normalize"),
                StageSpec(
                    "score_policy",
                    "preprocessing.score_policy",
                    {"score_min": 0.35},
                ),
                StageSpec("nms", "nms.production_v3"),
                StageSpec(
                    "cut_detection",
                    "cut_detection.video",
                    {"enabled": True, "method": "high_precision"},
                ),
                StageSpec(
                    "tracking",
                    "tracking.greedy",
                    {"remove_short_tracks_max_frames": 10},
                ),
            ]
        )
    stages.extend(
        [
            StageSpec(
                "polygon_optimization",
                "production.polygon_v3_cpu",
                {
                    "target_interval": 6,
                    "interval_evaluation": "native_exact",
                },
            ),
            StageSpec("exact_evaluation", "evaluation.mask_iou"),
            StageSpec("output_validation", "artifacts.validate"),
        ]
    )
    return PipelineConfig("polygon_modular", tuple(stages))


def default_ellipse_pipeline(*, include_preprocess: bool) -> PipelineConfig:
    stages: list[StageSpec] = []
    if include_preprocess:
        stages.extend(
            [
                StageSpec("normalization", "preprocessing.normalize"),
                StageSpec(
                    "score_policy",
                    "preprocessing.score_policy",
                    {"score_min": 0.35},
                ),
                StageSpec("nms", "nms.production_v3"),
                StageSpec(
                    "cut_detection",
                    "cut_detection.video",
                    {"enabled": True, "method": "high_precision"},
                ),
                StageSpec(
                    "tracking",
                    "tracking.greedy",
                    {"remove_short_tracks_max_frames": 10},
                ),
            ]
        )
    stages.extend(
        [
            StageSpec(
                "ellipse_approximation",
                "approximation.ellipse.production",
            ),
            StageSpec(
                "keyframe_selection",
                "keyframes.ellipse.dense",
                {"target_ratio": 1.0 / 3.0, "dense_recall_target": 0.96},
            ),
            StageSpec(
                "mask_gap_fill",
                "gap_fill.ellipse.linear",
                {"max_gap": 30},
            ),
            StageSpec("exact_evaluation", "evaluation.ellipse.exact"),
            StageSpec("sqlite_export", "artifacts.union_sqlite"),
            StageSpec("output_validation", "artifacts.validate"),
        ]
    )
    return PipelineConfig("ellipse_modular", tuple(stages))
