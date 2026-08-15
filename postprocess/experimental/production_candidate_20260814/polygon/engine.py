"""Quarantined bridge to the validated CUDA-lazy-exact optimizer runtime."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..config import CANDIDATE, CandidateConfig
from .candidate_palette import role_ids
from .dp import audit_exact_recall
from .pair_vote import pair_vote_environment

from experimental.production_candidate_polygon14.config import (
    CANDIDATE as APPROVED_POLYGON_CONTRACT,
)


POSTPROCESS_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = POSTPROCESS_ROOT.parent


def assert_runtime_bridge_contract(
    config: CandidateConfig = CANDIDATE,
) -> None:
    """Fail closed if the quarantined optimizer no longer matches this profile."""
    config.validate()
    approved = APPROVED_POLYGON_CONTRACT
    expected = {
        "profile": config.polygon_profile_id,
        "vertices": config.spatial.vertices_per_component,
        "spatial_recall": config.spatial.recall_floor,
        "spatial_recall_repair_max_scale": config.spatial.recall_repair_max_scale,
        "spatial_iou": config.spatial.iou_floor,
        "temporal_recall": config.temporal.recall_floor,
        "pair_vote_sweeps": config.temporal.pair_vote_sweeps,
    }
    actual = {
        "profile": approved.profile_id,
        "vertices": approved.vertices_per_component,
        "spatial_recall": approved.spatial_recall_floor,
        "spatial_recall_repair_max_scale": approved.spatial_recall_repair_max_scale,
        "spatial_iou": approved.spatial_iou_floor,
        "temporal_recall": approved.temporal_recall_floor,
        "pair_vote_sweeps": approved.pair_vote_sweeps,
    }
    if actual != expected:
        raise RuntimeError(
            f"approved polygon runtime contract drift: expected={expected}, actual={actual}"
        )
    runtime_root = POSTPROCESS_ROOT / "experimental/0809"
    added = str(runtime_root) not in sys.path
    if added:
        sys.path.insert(0, str(runtime_root))
    try:
        runtime = importlib.import_module("phase2_runtime")
    finally:
        if added:
            sys.path.remove(str(runtime_root))
    selector = getattr(runtime, "_class_role_state_profile")
    for label in config.labels:
        selected = tuple(
            selector(
                config.polygon_profile_id,
                label,
                float(config.temporal.target_interval),
            )
        )
        declared = role_ids(label, config.temporal.target_interval, config)
        if selected != declared:
            raise RuntimeError(
                f"candidate role palette drift for {label}: "
                f"declared={declared}, runtime={selected}"
            )
    fixed_runtime = {
        "candidate_frame_workers": 1,
        "pair_vote_threads": 8,
        "gapfill_max_gap": 15,
        "keyframe_max_gap": 30,
        "max_run_frames": 30000,
        "run_overlap_frames": 900,
        "native_batch_threads": 8,
        "cuda_prefilter_deficit_budget": 0.10,
        "cuda_prefilter_small_area": 0.0,
        "cuda_prefilter_small_deficit_budget": 0.10,
        "lazy_fallback_min_seconds": 0.5,
        "lazy_fallback_min_exact_edges": 1024,
        "lazy_fallback_infeasible_ratio": 0.875,
        "gc_interval": 8,
        "predictor_device": "cpu",
    }
    declared_runtime = {key: getattr(config.runtime, key) for key in fixed_runtime}
    if declared_runtime != fixed_runtime:
        raise RuntimeError(
            "optimizer bridge runtime drift: "
            f"declared={declared_runtime}, bridge={fixed_runtime}"
        )
    if config.runtime.interval_evaluation not in {
        "cuda_lazy_exact",
        "native_exact",
    }:
        raise RuntimeError(
            "unsupported interval evaluator: "
            f"{config.runtime.interval_evaluation!r}"
        )


def run_polygon_optimizer(
    source_root: Path,
    output_root: Path,
    *,
    config: CandidateConfig = CANDIDATE,
    labels: tuple[str, ...] | None = None,
    max_tracks: int = 0,
    force: bool = False,
) -> dict[str, object]:
    """Run the frozen optimizer without copying its experimental engine."""
    assert_runtime_bridge_contract(config)
    selected_labels = tuple(config.labels if labels is None else labels)
    if len(set(selected_labels)) != len(selected_labels):
        raise ValueError("optimizer labels must not contain duplicates")
    invalid = tuple(label for label in selected_labels if label not in config.labels)
    if invalid:
        raise ValueError(f"unsupported optimizer labels: {invalid}")
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not selected_labels:
        interval_root = output / f"interval_{config.temporal.target_interval}"
        interval_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "status": "experimental_production_candidate",
            "candidate": APPROVED_POLYGON_CONTRACT.to_dict(),
            "privacy": "SQLite mask geometry only; no video frames were opened.",
            "sqlite_output_schema_changed": False,
            "source_root": str(Path(source_root).resolve()),
            "runs": [],
            "reason": "no_active_genital_tracks",
        }
        manifest_path = output / "production_candidate_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "command": [],
            "wall_seconds": 0.0,
            "log": None,
            "manifest": str(manifest_path),
            "manifest_payload": manifest,
            "phase2_root": str(interval_root),
            "exact_recall": {},
            "active_labels": [],
        }
    command = [
        sys.executable,
        "-m",
        "experimental.production_candidate_polygon14.run",
        "--source-root",
        str(Path(source_root).resolve()),
        "--output-root",
        str(Path(output_root).resolve()),
        "--intervals",
        str(config.temporal.target_interval),
        "--labels",
        ",".join(selected_labels),
        "--label-workers",
        str(config.runtime.label_workers),
        "--num-workers",
        str(config.runtime.optimizer_workers),
        "--pair-vote-threads",
        str(config.runtime.pair_vote_threads),
        "--native-batch-threads",
        str(config.runtime.native_batch_threads),
        "--interval-evaluation",
        str(config.runtime.interval_evaluation),
        "--max-tracks",
        str(max(0, int(max_tracks))),
    ]
    if force:
        command.append("--force")
    environment = os.environ.copy()
    environment.update(pair_vote_environment(config))
    environment.update(
        {
            "MASK_PIPELINE_PHASE2_CUDA_LAZY_FALLBACK_MIN_SECONDS": str(
                config.runtime.lazy_fallback_min_seconds
            ),
            "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_BUDGET": str(
                config.runtime.cuda_prefilter_deficit_budget
            ),
            "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_SMALL_AREA": str(
                config.runtime.cuda_prefilter_small_area
            ),
            "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_SMALL_BUDGET": str(
                config.runtime.cuda_prefilter_small_deficit_budget
            ),
            "MASK_PIPELINE_PHASE2_CUDA_LAZY_FALLBACK_MIN_EDGES": str(
                config.runtime.lazy_fallback_min_exact_edges
            ),
            "MASK_PIPELINE_PHASE2_CUDA_LAZY_FALLBACK_INFEASIBLE_RATIO": str(
                config.runtime.lazy_fallback_infeasible_ratio
            ),
            "MASK_PIPELINE_PHASE2_CANDIDATE_FRAME_WORKERS": str(
                config.runtime.candidate_frame_workers
            ),
        }
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(POSTPROCESS_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    log_path = output / "optimizer.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall = time.perf_counter() - started
    if process.returncode:
        raise RuntimeError(
            f"polygon optimizer failed with exit {process.returncode}; see {log_path}"
        )
    interval_root = output / f"interval_{config.temporal.target_interval}"
    manifest_path = output / "production_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exact: dict[str, dict[str, int | float]] = {}
    for label in selected_labels:
        metrics = (
            interval_root
            / config.polygon_profile_id
            / label
            / "runtime/exact/keyframe_exact_metrics.csv"
        )
        exact[label] = audit_exact_recall(metrics, config)
    return {
        "command": command,
        "wall_seconds": wall,
        "log": str(log_path),
        "manifest": str(manifest_path),
        "manifest_payload": manifest,
        "phase2_root": str(interval_root),
        "exact_recall": exact,
        "active_labels": list(selected_labels),
        "exact_recall_gate": "repair_then_audit_and_publish",
    }


__all__ = ("assert_runtime_bridge_contract", "run_polygon_optimizer")
