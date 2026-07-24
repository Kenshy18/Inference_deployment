"""Pipeline stages owned by ellipse approximation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contracts.stages import StageContext, StageResult

from .inference import infer_main


def _extra_args(options: dict[str, Any]) -> list[str]:
    value = options.get("extra_args", [])
    if not isinstance(value, list):
        raise ValueError("extra_args must be a list")
    return [str(item) for item in value]


@dataclass(frozen=True)
class EllipseApproximationStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "ellipse_approximation"
    requires: frozenset[str] = frozenset({"tracked_sqlite"})
    provides: frozenset[str] = frozenset(
        {"approximated_sqlite", "approximation_metrics_csv"}
    )

    def run(self, context: StageContext) -> StageResult:
        model_root = (
            Path(self.options["model_root"])
            if self.options.get("model_root")
            else Path(__file__).resolve().parents[2] / "models"
        )
        k2_directory = model_root / "k2_v5"
        infer_main(
            [
                "--input-sqlite",
                str(context.artifacts["tracked_sqlite"]),
                "--output-dir",
                str(context.stage_dir),
                "--k2-run-dir",
                str(self.options.get("k2_run_dir", k2_directory)),
                "--k2-device",
                str(self.options.get("device", "cpu")),
                *_extra_args(self.options),
            ]
        )
        return StageResult(
            {
                "approximated_sqlite": (
                    context.stage_dir / "k1_exact_k2_v5_predictions.sqlite"
                ),
                "approximation_metrics_csv": (
                    context.stage_dir / "k1_exact_k2_v5_metrics.csv"
                ),
            }
        )
