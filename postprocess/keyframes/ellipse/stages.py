"""Pipeline stages owned by ellipse keyframe selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.stages import StageContext, StageResult

from .trackk_dense_recall import kftrackk_main


@dataclass(frozen=True)
class DenseEllipseKeyframesStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "ellipse_keyframes"
    requires: frozenset[str] = frozenset({"approximation_metrics_csv"})
    provides: frozenset[str] = frozenset({"keyframes_json", "interpolated_union_json"})

    def run(self, context: StageContext) -> StageResult:
        extra_args = self.options.get("extra_args", [])
        if not isinstance(extra_args, list):
            raise ValueError("extra_args must be a list")
        kftrackk_main(
            [
                "--input-metrics-csv",
                str(context.artifacts["approximation_metrics_csv"]),
                "--output-dir",
                str(context.stage_dir),
                "--target-ratio",
                str(self.options.get("target_ratio", 1.0 / 3.0)),
                "--dense-recall-target",
                str(self.options.get("dense_recall_target", 0.96)),
                *map(str, extra_args),
            ]
        )
        return StageResult(
            {
                "keyframes_json": context.stage_dir / "final_keyframes.json",
                "interpolated_union_json": (
                    context.stage_dir / "interpolated_union.json"
                ),
            }
        )
