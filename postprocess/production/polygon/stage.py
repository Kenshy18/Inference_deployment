"""Pipeline stage for the promoted adaptive-vertex CPU-exact implementation."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from contracts.stages import StageContext, StageResult

from ..config import PRODUCTION, ProductionConfig
from .materialize import materialize_outputs
from .runtime_bridge import build_runtime_config, optimize, prepare_inputs


def _labels(path: Path) -> tuple[str, ...]:
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as db:
        tables = {
            str(row[0])
            for row in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")
        }
        if "tracks" not in tables:
            return ()
        columns = {str(row[1]) for row in db.execute("PRAGMA table_info(tracks)")}
        if "label" not in columns:
            return ()
        return tuple(
            str(row[0])
            for row in db.execute(
                "SELECT DISTINCT COALESCE(label, '') FROM tracks ORDER BY 1"
            )
        )


def _dimensions(
    path: Path,
    *,
    fallback_width: int,
    fallback_height: int,
) -> tuple[int, int]:
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as db:
        tables = {
            str(row[0])
            for row in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")
        }
        if "frames" not in tables:
            if fallback_width <= 0 or fallback_height <= 0:
                raise RuntimeError(f"source dimensions are unavailable: {path}")
            return fallback_width, fallback_height
        columns = {str(row[1]) for row in db.execute("PRAGMA table_info(frames)")}
        if not {"width", "height"}.issubset(columns):
            if fallback_width <= 0 or fallback_height <= 0:
                raise RuntimeError(f"source dimensions are unavailable: {path}")
            return fallback_width, fallback_height
        frame_column = "frame_index" if "frame_index" in columns else "frame"
        row = db.execute(
            f"SELECT width,height FROM frames ORDER BY {frame_column} LIMIT 1"
        ).fetchone()
    if row is None or int(row[0] or 0) <= 0 or int(row[1] or 0) <= 0:
        raise RuntimeError(f"source dimensions are unavailable: {path}")
    return int(row[0]), int(row[1])


@dataclass(frozen=True)
class ProductionPolygonStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "production_polygon_adaptive_recall_cpu_exact_v3"
    requires: frozenset[str] = frozenset({"tracked_sqlite"})
    provides: frozenset[str] = frozenset(
        {
            "predictions_sqlite",
            "keyframes_sqlite",
            "production_polygon_manifest",
        }
    )

    def _config(self) -> ProductionConfig:
        interval = int(
            self.options.get(
                "target_interval",
                self.options.get("interval_frames", PRODUCTION.target_interval),
            )
        )
        config = replace(
            PRODUCTION,
            target_interval=interval,
        )
        config.validate()
        evaluator = str(self.options.get("interval_evaluation", "native_exact"))
        if evaluator != "native_exact":
            raise ValueError("Production supports only CPU native_exact evaluation")
        if self.options.get("max_gap") is not None:
            maximum_gap = int(self.options["max_gap"])
            if maximum_gap != config.gapfill_max_gap:
                raise ValueError(
                    "Production polygon gap filling is fixed at "
                    f"{config.gapfill_max_gap} frames; got {maximum_gap}"
                )
        return config

    def run(self, context: StageContext) -> StageResult:
        config = self._config()
        stage_dir = Path(context.stage_dir).expanduser().resolve()
        tracked = Path(context.artifacts["tracked_sqlite"]).resolve()
        input_labels = _labels(tracked)
        unsupported_labels = tuple(
            label for label in input_labels if label not in config.labels
        )
        if unsupported_labels:
            raise ValueError(
                "Production polygon postprocess supports only "
                f"{config.labels}; found unsupported labels {unsupported_labels}"
            )
        width, height = _dimensions(
            tracked,
            fallback_width=int(self.options.get("frame_width", 1920)),
            fallback_height=int(self.options.get("frame_height", 1080)),
        )
        video = context.artifacts.get("input_video")
        source_root, preparation = prepare_inputs(
            tracked,
            stage_dir / "preparation",
            width=width,
            height=height,
            input_video=None if video is None else Path(video),
            config=config,
        )
        policy = preparation.get("vertex_policy")
        if not isinstance(policy, dict) or not isinstance(policy.get("tracks"), dict):
            raise RuntimeError("Production adaptive vertex policy is missing")
        assigned = {
            int(value["vertices_per_component"]) for value in policy["tracks"].values()
        }
        unsupported_counts = assigned - set(config.allowed_vertices_per_component)
        if unsupported_counts:
            raise RuntimeError(
                "Production vertex policy selected unsupported counts: "
                f"{sorted(unsupported_counts)}"
            )
        optimizer = optimize(
            source_root,
            stage_dir / "optimizer",
            labels=tuple(preparation["active_labels"]),
            max_tracks=max(0, int(self.options.get("max_tracks", 0))),
            force=bool(self.options.get("force", False)),
            config=config,
        )
        predictions = stage_dir / "predictions.sqlite"
        keyframes = stage_dir / "keyframes.sqlite"
        runtime = build_runtime_config(config)
        materialization = materialize_outputs(
            Path(str(optimizer["phase2_root"])),
            tracked,
            predictions,
            keyframes,
            config=config,
            runtime_profile=runtime.polygon_profile_id,
        )
        violations = sum(
            int(value["recall_violations"])
            for value in optimizer["exact_recall"].values()
        )
        payload = {
            "schema_version": 1,
            "status": "production",
            "profile": config.profile_id,
            "target_interval": config.target_interval,
            "gapfill_max_gap": config.gapfill_max_gap,
            "interval_evaluation": config.interval_evaluation,
            "vertex_policy": {
                "method": "track_q99.9_pre_border_screen_occupancy_v1",
                "allowed_vertices": list(config.allowed_vertices_per_component),
                "thresholds": list(config.screen_occupancy_thresholds),
                "summary": policy.get("summary", {}),
            },
            "border_policy": {
                "maximum_expansion_px": config.border_max_expand_px,
                "influence_px": config.border_influence_px,
                "two_axis_corner_support": config.border_corner_support,
            },
            "exact_recall_policy": (
                "best_of_persistent_or_direct_rdp_then_uniform_scale_and_audit"
            ),
            "exact_recall_violations": violations,
            "preparation": preparation,
            "optimizer": optimizer,
            "materialization": materialization,
            "runtime_bridge": "production_internal_parity_frozen_adaptive_v3",
        }
        manifest = stage_dir / "production_polygon_manifest.json"
        manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return StageResult(
            {
                "predictions_sqlite": predictions,
                "keyframes_sqlite": keyframes,
                "production_polygon_manifest": manifest,
            },
            payload,
        )


__all__ = ("ProductionPolygonStage",)
