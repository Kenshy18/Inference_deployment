"""Pipeline stage for the promoted adaptive-vertex Production implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from contracts.stages import StageContext, StageResult

from ..config import PRODUCTION, ProductionConfig
from ..source import source_dimensions, source_labels
from .materialize import materialize_outputs
from .runtime_bridge import build_runtime_config, optimize, prepare_inputs


@dataclass(frozen=True)
class ProductionPolygonStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "production_polygon_adaptive_recall_cuda_lazy_exact_v4"
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
        evaluator = str(
            self.options.get("interval_evaluation", PRODUCTION.interval_evaluation)
        )
        config = replace(config, interval_evaluation=evaluator)
        config.validate()
        return config

    def run(self, context: StageContext) -> StageResult:
        config = self._config()
        context.report_progress("polygon:preparing", 0.01)
        stage_dir = Path(context.stage_dir).expanduser().resolve()
        tracked = Path(context.artifacts["tracked_sqlite"]).resolve()
        input_labels = source_labels(tracked)
        passthrough_labels = tuple(
            label for label in input_labels if label not in config.labels
        )
        width, height = source_dimensions(
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
        context.report_progress("polygon:prepared", 0.12)
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
            progress_callback=lambda detail, fraction, fps: context.report_progress(
                detail,
                None if fraction is None else 0.12 + 0.76 * fraction,
                fps,
            ),
        )
        context.report_progress("polygon:materializing", 0.90)
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
        context.report_progress("polygon:validating", 0.98)
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
            "passthrough_labels": list(passthrough_labels),
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
