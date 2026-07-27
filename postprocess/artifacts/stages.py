"""Pipeline stages owned by final artifact export and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from contracts.stages import StageContext, StageResult

from .contract import validate_mask_sqlite
from .legacy_sqlite import export_legacy_sqlite
from .sqlite import union2sqlite_main


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
    requires: frozenset[str] = frozenset({"predictions_sqlite"})
    provides: frozenset[str] = frozenset({"validation_report"})

    def run(self, context: StageContext) -> StageResult:
        stats = validate_mask_sqlite(context.artifacts["predictions_sqlite"])
        output = context.stage_dir / "validation.json"
        output.write_text(
            json.dumps(stats.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return StageResult({"validation_report": output}, stats.as_dict())


@dataclass(frozen=True)
class LegacySqliteExportStage:
    """Tentative projection for former Dinov3_postprocess consumers."""

    options: dict[str, Any] = field(default_factory=dict)
    name: str = "legacy_sqlite_export"
    requires: frozenset[str] = frozenset({"predictions_sqlite"})
    provides: frozenset[str] = frozenset({"legacy_predictions_sqlite"})

    def run(self, context: StageContext) -> StageResult:
        output = context.stage_dir / "predictions.legacy.sqlite"
        summary = export_legacy_sqlite(
            context.artifacts["predictions_sqlite"],
            output,
        )
        return StageResult({"legacy_predictions_sqlite": output}, summary)
