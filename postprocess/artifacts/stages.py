"""Pipeline stages owned by final artifact export and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from contracts.stages import StageContext, StageResult

from .contract import validate_mask_sqlite
from .legacy_sqlite import export_legacy_sqlite
from .sqlite import union2sqlite_main
from .unified_sqlite import build_integrated_result


@dataclass(frozen=True)
class UnionSqliteExportStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "artifact_export"
    requires: frozenset[str] = frozenset({"filled_union_json", "tracked_sqlite"})
    provides: frozenset[str] = frozenset({"predictions_sqlite"})

    def run(self, context: StageContext) -> StageResult:
        output = context.stage_dir / "predictions.sqlite"
        union2sqlite_main(
            [
                "--input-union-json",
                str(context.artifacts["filled_union_json"]),
                "--output-sqlite",
                str(output),
                "--reference-sqlite",
                str(context.artifacts["tracked_sqlite"]),
            ]
        )
        return StageResult({"predictions_sqlite": output})


@dataclass(frozen=True)
class OutputValidationStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "output_validation"

    @property
    def requires(self) -> frozenset[str]:
        return frozenset(
            {str(self.options.get("source_artifact", "predictions_sqlite"))}
        )

    @property
    def provides(self) -> frozenset[str]:
        return frozenset(
            {str(self.options.get("output_artifact", "validation_report"))}
        )

    def run(self, context: StageContext) -> StageResult:
        source_artifact = next(iter(self.requires))
        output_artifact = next(iter(self.provides))
        stats = validate_mask_sqlite(context.artifacts[source_artifact])
        output = context.stage_dir / str(
            self.options.get("filename", "validation.json")
        )
        output.write_text(
            json.dumps(stats.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return StageResult({output_artifact: output}, stats.as_dict())


@dataclass(frozen=True)
class LegacySqliteExportStage:
    """Tentative projection for former Dinov3_postprocess consumers."""

    options: dict[str, Any] = field(default_factory=dict)
    name: str = "legacy_sqlite_export"

    @property
    def requires(self) -> frozenset[str]:
        return frozenset(
            {str(self.options.get("source_artifact", "predictions_sqlite"))}
        )

    @property
    def provides(self) -> frozenset[str]:
        return frozenset(
            {
                str(
                    self.options.get(
                        "output_artifact",
                        "legacy_predictions_sqlite",
                    )
                )
            }
        )

    def run(self, context: StageContext) -> StageResult:
        source_artifact = next(iter(self.requires))
        output_artifact = next(iter(self.provides))
        output = context.stage_dir / str(
            self.options.get("filename", "predictions.legacy.sqlite")
        )
        summary = export_legacy_sqlite(
            context.artifacts[source_artifact],
            output,
        )
        return StageResult({output_artifact: output}, summary)


@dataclass(frozen=True)
class IntegratedResultSqliteStage:
    """Publish raw inference, tracking references, and final keyframes."""

    options: dict[str, Any] = field(default_factory=dict)
    name: str = "integrated_result_sqlite"

    @property
    def requires(self) -> frozenset[str]:
        return frozenset(
            {
                "input_raw_sqlite",
                "tracked_sqlite",
                str(self.options.get("source_artifact", "predictions_sqlite")),
            }
        )

    provides: frozenset[str] = frozenset({"result_sqlite"})

    def run(self, context: StageContext) -> StageResult:
        source_artifact = str(
            self.options.get("source_artifact", "predictions_sqlite")
        )
        output = context.stage_dir / "result.sqlite"
        summary = build_integrated_result(
            context.artifacts["input_raw_sqlite"],
            context.artifacts["tracked_sqlite"],
            context.artifacts[source_artifact],
            output,
            polygon_keyframes_sqlite=context.artifacts.get(
                "keyframes_sqlite"
            ),
            ellipse_keyframes_json=context.artifacts.get("keyframes_json"),
            classwise_manifest=context.artifacts.get("classwise_manifest"),
        )
        return StageResult({"result_sqlite": output}, summary)
