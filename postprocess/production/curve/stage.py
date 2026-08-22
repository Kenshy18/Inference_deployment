"""Pipeline stage exposing Catmull--Rom as a Production geometry choice."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from contracts.stages import StageContext, StageResult

from production.config import PRODUCTION
from production.polygon.runtime_bridge import build_runtime_config
from production.source import source_dimensions

from .config import CURVE_PRODUCTION, PROFILE_ID, CurveProductionConfig
from .engine import run_curve_optimizer
from .preparation import prepare_curve_source


def _cleanup_preparation_sqlites(
    preparation: dict[str, object],
    preparation_root: Path,
) -> dict[str, int]:
    """Remove successful-run scratch SQLite files inside the stage root."""

    root = Path(preparation_root).resolve()
    removed_files = 0
    removed_bytes = 0
    classes = preparation.get("classes", {})
    if not isinstance(classes, dict):
        return {"files": 0, "bytes": 0}
    for value in classes.values():
        if not isinstance(value, dict):
            continue
        for key in ("projected_sqlite", "border_sqlite", "endpoint_sqlite"):
            raw_path = value.get(key)
            if not raw_path:
                continue
            path = Path(str(raw_path)).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:  # pragma: no cover - safety invariant
                raise RuntimeError(
                    f"curve scratch path escaped its preparation root: {path}"
                ) from error
            for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
                if not candidate.exists():
                    continue
                removed_bytes += int(candidate.stat().st_size)
                candidate.unlink()
                removed_files += 1
    return {"files": int(removed_files), "bytes": int(removed_bytes)}


@dataclass(frozen=True)
class ProductionCurveStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = PROFILE_ID
    requires: frozenset[str] = frozenset({"tracked_sqlite"})
    provides: frozenset[str] = frozenset(
        {
            "predictions_sqlite",
            "keyframes_sqlite",
            "production_curve_manifest",
        }
    )

    def _config(self) -> CurveProductionConfig:
        value = replace(
            CURVE_PRODUCTION,
            target_interval=int(
                self.options.get(
                    "target_interval",
                    self.options.get(
                        "interval_frames", CURVE_PRODUCTION.target_interval
                    ),
                )
            ),
            native_cpu_threads=int(
                self.options.get(
                    "native_cpu_threads", CURVE_PRODUCTION.native_cpu_threads
                )
            ),
            native_reference_cache_bytes=int(
                self.options.get(
                    "native_reference_cache_bytes",
                    CURVE_PRODUCTION.native_reference_cache_bytes,
                )
            ),
            point_refine_sweeps=int(
                self.options.get(
                    "point_refine_sweeps", CURVE_PRODUCTION.point_refine_sweeps
                )
            ),
            point_refine_scheduler=str(
                self.options.get(
                    "point_refine_scheduler",
                    CURVE_PRODUCTION.point_refine_scheduler,
                )
            ),
            max_run_frames=int(
                self.options.get("max_run_frames", CURVE_PRODUCTION.max_run_frames)
            ),
            run_overlap_frames=int(
                self.options.get(
                    "run_overlap_frames", CURVE_PRODUCTION.run_overlap_frames
                )
            ),
        )
        value.validate()
        return value

    def run(self, context: StageContext) -> StageResult:
        config = self._config()
        stage_dir = Path(context.stage_dir).resolve()
        tracked = Path(context.artifacts["tracked_sqlite"]).resolve()
        selected_option = self.options.get("selected_track_ids")
        selected_track_ids = (
            None
            if selected_option is None
            else tuple(str(value) for value in selected_option)
        )
        width, height = source_dimensions(
            tracked,
            fallback_width=int(self.options.get("frame_width", 1920)),
            fallback_height=int(self.options.get("frame_height", 1080)),
        )
        context.report_progress("curve:preparing", 0.01)
        preparation_config = replace(
            PRODUCTION,
            target_interval=int(config.target_interval),
            interval_evaluation="native_exact",
        )
        preparation = prepare_curve_source(
            tracked,
            stage_dir / "preparation",
            width=width,
            height=height,
            input_video=(
                None
                if context.artifacts.get("input_video") is None
                else Path(context.artifacts["input_video"])
            ),
            config=build_runtime_config(preparation_config),
            selected_track_ids=selected_track_ids,
        )
        context.report_progress("curve:prepared", 0.10)
        engine = run_curve_optimizer(
            tracked,
            preparation,
            stage_dir / "runtime",
            config=config,
            max_tracks=max(0, int(self.options.get("max_tracks", 0))),
            progress_callback=lambda detail, fraction, fps: context.report_progress(
                detail,
                None if fraction is None else 0.10 + 0.86 * fraction,
                fps,
            ),
        )
        audit = dict(engine["audit"])
        if int(audit["recall_violations"]) or int(audit["topology_invalid_frames"]):
            raise RuntimeError(
                "Production curve component audit failed: "
                f"recall={audit['recall_violations']}, "
                f"topology={audit['topology_invalid_frames']}"
            )
        retain_preparation = bool(self.options.get("retain_preparation_sqlites", False))
        cleanup = (
            {"files": 0, "bytes": 0}
            if retain_preparation
            else _cleanup_preparation_sqlites(
                preparation,
                stage_dir / "preparation",
            )
        )
        payload = {
            "schema_version": 1,
            "status": "production",
            "profile": PROFILE_ID,
            "geometry_mode": "catmull_rom",
            "curve_contract": ("closed_uniform_catmull_rom_tension_1_factor_1_over_6"),
            "editable_variables": "interpolation_points_P_only",
            "cuda_used": False,
            "target_interval": int(config.target_interval),
            "preparation": preparation,
            "engine": engine,
            "preparation_cleanup": {
                "retained": retain_preparation,
                "removed_files": int(cleanup["files"]),
                "removed_bytes": int(cleanup["bytes"]),
            },
            "selected_track_ids": list(preparation["selected_track_ids"]),
            "passthrough_track_ids": list(preparation["passthrough_track_ids"]),
        }
        manifest = stage_dir / "production_curve_manifest.json"
        manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        context.report_progress("curve:complete", 1.0, engine["emitted_fps"])
        return StageResult(
            {
                "predictions_sqlite": Path(str(engine["predictions_sqlite"])),
                "keyframes_sqlite": Path(str(engine["keyframes_sqlite"])),
                "production_curve_manifest": manifest,
            },
            payload,
        )


__all__ = ("ProductionCurveStage",)
