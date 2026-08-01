"""Adapter for the original production-patched polygon v22 optimizer.

The validated optimizer remains isolated in ``vendor``.  This module only
maps its artifacts back onto the modular postprocess contracts so the public
result SQLite schema remains unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contracts.mask_sqlite import MaskRow, read_mask_rows, write_mask_sqlite
from contracts.stages import StageContext, StageResult

from .preparation import apply_border_expansion, apply_endpoint_extension


ROOT = Path(__file__).resolve().parents[2]
VENDOR_RUNTIME = ROOT / "vendor" / "original_polygon" / "original_run_standalone.py"
DEFAULT_PREDICTOR = ROOT / "models" / "polygon_point_predictor"
DEFAULT_NUM_WORKERS = max(1, min(4, os.cpu_count() or 1))


def _resolve_cpp_compiler() -> str | None:
    """Find the C++ compiler shipped beside the active runtime, if needed.

    The production environment intentionally does not put its Conda compiler
    on ``PATH``.  The original v22 runtime therefore used to fall back to the
    byte-identical Python DP implementation even though its native DP was
    available.  Selecting the compiler only changes how that same DP is
    executed; it does not change optimizer parameters or output geometry.
    """

    explicit = os.environ.get("CXX", "").strip()
    if explicit:
        return explicit
    for name in ("g++", "c++"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    runtime_bin = Path(sys.executable).resolve().parent
    candidates = [
        runtime_bin / "x86_64-conda-linux-gnu-g++",
        *sorted(runtime_bin.glob("*-g++")),
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _label_maps(reference: Path) -> tuple[dict[tuple[int, str], str], dict[str, str]]:
    exact: dict[tuple[int, str], str] = {}
    tracks: dict[str, str] = {}
    for row in read_mask_rows(reference):
        exact[(row.frame, row.track_id)] = row.label
        if row.label:
            tracks.setdefault(row.track_id, row.label)
    return exact, tracks


def _materialize_predictions(
    source: Path,
    reference: Path,
    output: Path,
) -> Path:
    exact_labels, track_labels = _label_maps(reference)
    rows = [
        MaskRow(
            frame=row.frame,
            track_id=row.track_id,
            polygons=row.polygons,
            label=exact_labels.get(
                (row.frame, row.track_id), track_labels.get(row.track_id, "")
            ),
            shape_type="polygon",
        )
        for row in read_mask_rows(source)
    ]
    return write_mask_sqlite(output, rows, reference_sqlite=reference)


def _materialize_keyframes(
    source: Path,
    reference: Path,
    output: Path,
) -> Path:
    exact_labels, track_labels = _label_maps(reference)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows: list[MaskRow] = []
    for value in payload:
        frame = int(value["frame"])
        track_id = str(value["track_id"])
        rows.append(
            MaskRow(
                frame=frame,
                track_id=track_id,
                polygons=json.dumps(
                    value["polygons"], ensure_ascii=False, separators=(",", ":")
                ),
                label=exact_labels.get(
                    (frame, track_id), track_labels.get(track_id, "")
                ),
                shape_type="polygon",
            )
        )
    return write_mask_sqlite(output, rows, reference_sqlite=reference)


@dataclass(frozen=True)
class ProductionPolygonV22Stage:
    """Run the exact original polygon optimizer behind current artifacts."""

    options: dict[str, Any] = field(default_factory=dict)
    name: str = "polygon_v22"
    requires: frozenset[str] = frozenset({"tracked_sqlite"})
    provides: frozenset[str] = frozenset(
        {"predictions_sqlite", "keyframes_sqlite", "polygon_v22_summary"}
    )

    def run(self, context: StageContext) -> StageResult:
        interval = int(self.options.get("interval_frames", 3))
        if interval < 1:
            raise ValueError("interval_frames must be >= 1")
        gap = int(self.options.get("max_gap", 30))
        if gap < 0:
            raise ValueError("max_gap must be >= 0")
        predictor = Path(
            self.options.get("point_predictor_model_dir", DEFAULT_PREDICTOR)
        ).expanduser().resolve()
        adaptive = bool(self.options.get("adaptive_anchor_counts", True))
        if adaptive:
            for filename in ("best.pt", "feature_stats.npz", "run_config.json"):
                if not (predictor / filename).is_file():
                    raise FileNotFoundError(predictor / filename)

        original_reference = context.artifacts["tracked_sqlite"]
        optimizer_input = original_reference
        preparation: dict[str, object] = {}
        if bool(self.options.get("border_expand", True)):
            optimizer_input, border_summary = apply_border_expansion(
                optimizer_input,
                context.stage_dir / "border_expanded.sqlite",
                width=int(self.options.get("border_width", 1920)),
                height=int(self.options.get("border_height", 1080)),
                trigger_px=float(self.options.get("border_trigger_px", 10.0)),
                expand_ratio=float(self.options.get("border_expand_ratio", 0.10)),
                min_expand_px=float(self.options.get("border_min_expand_px", 6.0)),
                max_expand_px=float(self.options.get("border_max_expand_px", 40.0)),
                influence_px=float(self.options.get("border_influence_px", 24.0)),
            )
            preparation["border_expansion"] = border_summary
        if bool(self.options.get("endpoint_extend", True)):
            optimizer_input, endpoint_summary = apply_endpoint_extension(
                optimizer_input,
                context.stage_dir / "endpoint_extended.sqlite",
                video=context.artifacts.get("input_video"),
                extend_frames=int(self.options.get("endpoint_extend_frames", 5)),
                motion_frames=int(self.options.get("endpoint_motion_frames", 10)),
                max_speed_px=float(self.options.get("endpoint_max_speed_px", 1000.0)),
            )
            preparation["endpoint_extension"] = endpoint_summary

        vendor_output = context.stage_dir / "vendor_output"
        num_workers = int(self.options.get("num_workers", DEFAULT_NUM_WORKERS))
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        command = [
            sys.executable,
            str(VENDOR_RUNTIME),
            "__onefile_polygon_optimize",
            "--input-sqlite",
            str(optimizer_input),
            "--output-dir",
            str(vendor_output),
            "--target-ratio",
            str(1.0 / float(interval)),
            "--anchors-per-contour",
            str(int(self.options.get("anchors_per_contour", 48))),
            "--point-predictor-model-dir",
            str(predictor),
            "--predictor-device",
            str(self.options.get("predictor_device", "cuda")),
            "--predictor-batch-size",
            str(int(self.options.get("predictor_batch_size", 256))),
            "--adaptive-point-quantile",
            str(float(self.options.get("adaptive_point_quantile", 0.95))),
            "--adaptive-point-offset",
            str(int(self.options.get("adaptive_point_offset", 10))),
            "--min-anchors-per-contour",
            str(int(self.options.get("min_anchors_per_contour", 8))),
            "--gapfill-max-gap",
            str(gap),
            "--max-run-frames",
            str(int(self.options.get("max_run_frames", 30000))),
            "--run-overlap-frames",
            str(int(self.options.get("run_overlap_frames", 900))),
            "--recall-min",
            str(float(self.options.get("recall_min", 0.97))),
            "--max-gap",
            str(int(self.options.get("keyframe_max_gap", 30))),
            "--num-workers",
            str(num_workers),
            "--stream-sqlite-rows",
            "--evaluate-exact",
            "--write-pred-sqlite",
            "--adaptive-anchor-counts" if adaptive else "--no-adaptive-anchor-counts",
            "--gapfill-enabled" if gap > 0 else "--no-gapfill-enabled",
        ]
        environment = os.environ.copy()
        cpp_compiler = _resolve_cpp_compiler()
        if cpp_compiler is not None:
            environment["CXX"] = cpp_compiler
        subprocess.run(
            command,
            check=True,
            cwd=str(ROOT),
            env=environment,
        )

        summary_path = vendor_output / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        predictions = context.stage_dir / "predictions.sqlite"
        keyframes = context.stage_dir / "keyframes.sqlite"
        _materialize_predictions(
            vendor_output / "pred" / "predictions.sqlite",
            original_reference,
            predictions,
        )
        _materialize_keyframes(
            vendor_output / "opt" / "final_keyframes.json",
            original_reference,
            keyframes,
        )
        return StageResult(
            {
                "predictions_sqlite": predictions,
                "keyframes_sqlite": keyframes,
                "polygon_v22_summary": summary_path,
            },
            {
                "algorithm": "original_production_patched_v22",
                "interval_frames": interval,
                "max_gap": gap,
                "adaptive_anchor_counts": adaptive,
                "num_workers": num_workers,
                "native_dp_compiler": cpp_compiler,
                "preparation": preparation,
                "optimizer": summary.get("optimizer_summary", {}),
            },
        )


__all__ = [
    "DEFAULT_NUM_WORKERS",
    "ProductionPolygonV22Stage",
]
