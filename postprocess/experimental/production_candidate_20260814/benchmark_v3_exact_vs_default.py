#!/usr/bin/env python3
"""Benchmark the full V3 corpus with default CUDA and all-edge CPU exact DP.

The expensive upstream geometry is built once per raw inference SQLite and is
then held byte-identical across both evaluators and both target intervals.
Video decoding is used only by the local high-precision cut detector; this
script never materializes or displays video frames for an AI agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

from contracts.detections import CutList, write_cut_list
from cut_detection.detector import HighPrecisionCutDetector
from preprocessing.raw_sqlite import normalize_raw_detection_sqlite
from preprocessing.score_policy import ScorePolicy, apply_score_policy_jsonl
from tracking.builder import build_tracked_sqlite

from .config import with_interval_evaluation, with_target_interval
from .export import export_software_sqlite
from .nms.stage import run_nms_jsonl
from .polygon.engine import run_polygon_optimizer
from .polygon.preparation import prepare_classwise_source
from .validation import audit_sqlite, schema_fingerprint


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS_ROOT = (
    REPOSITORY_ROOT / "output/instance_mask_topology_20260806/inference/v3"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "output/production_candidate_20260814_v3_exact_vs_default_i2_i5_20260814"
)
MODE_TO_EVALUATOR = {
    "default_cuda": "cuda_lazy_exact",
    "cpu_exact": "native_exact",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _raw_metadata(path: Path) -> dict[str, object]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        quick_check = str(db.execute("PRAGMA quick_check").fetchone()[0])
        frame_count = int(db.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
        detection_count = int(
            db.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        )
        width, height = db.execute(
            "SELECT width,height FROM frames ORDER BY frame_index LIMIT 1"
        ).fetchone()
        video_row = db.execute(
            "SELECT path,reported_frame_count,fps,width,height "
            "FROM videos ORDER BY id LIMIT 1"
        ).fetchone()
    if quick_check != "ok":
        raise RuntimeError(f"raw SQLite quick_check failed: {path}: {quick_check}")
    if video_row is None:
        raise RuntimeError(f"raw SQLite has no source video metadata: {path}")
    video = Path(str(video_row[0])).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"source video is missing: {video}")
    return {
        "run_id": path.parent.name,
        "raw_sqlite": str(path.resolve()),
        "raw_sha256": _sha256(path),
        "raw_frames": frame_count,
        "raw_detections": detection_count,
        "width": int(width),
        "height": int(height),
        "video": str(video),
        "video_metadata_frames": int(video_row[1]),
        "video_fps": float(video_row[2]),
        "video_width": int(video_row[3]),
        "video_height": int(video_row[4]),
        "quick_check": quick_check,
    }


def discover_corpus(root: Path) -> list[dict[str, object]]:
    sqlite_paths = sorted(Path(root).expanduser().resolve().glob("*/*.sqlite"))
    if not sqlite_paths:
        raise FileNotFoundError(f"no V3 inference SQLite files below {root}")
    rows = [_raw_metadata(path) for path in sqlite_paths]
    run_ids = [str(row["run_id"]) for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError(f"duplicate run IDs: {run_ids}")
    # Short/low-detection sources fail quickly if a contract regresses.
    return sorted(rows, key=lambda row: (int(row["raw_detections"]), str(row["run_id"])))


def _build_shared_upstream(
    row: dict[str, object],
    run_root: Path,
    *,
    score_min: float,
) -> tuple[Path, Path, dict[str, object]]:
    shared_root = run_root / "shared"
    manifest_path = shared_root / "shared_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        tracked = Path(str(payload["artifacts"]["tracked_sqlite"]))
        source_root = Path(str(payload["artifacts"]["prepared_source_root"]))
        if not tracked.is_file() or not source_root.is_dir():
            raise RuntimeError(f"incomplete resumed shared artifacts: {manifest_path}")
        return tracked, source_root, payload

    shared_root.mkdir(parents=True, exist_ok=True)
    raw = Path(str(row["raw_sqlite"]))
    video = Path(str(row["video"]))
    stage_seconds: dict[str, float] = {}
    started_all = time.perf_counter()

    normalized = shared_root / "01_preprocessing/normalized.jsonl"
    scored = shared_root / "02_score_policy/scored.jsonl"
    started = time.perf_counter()
    normalization = normalize_raw_detection_sqlite(raw, normalized)
    stage_seconds["normalization"] = time.perf_counter() - started
    started = time.perf_counter()
    score = apply_score_policy_jsonl(
        normalized,
        scored,
        policy=ScorePolicy(default_min=float(score_min)),
    )
    stage_seconds["score_policy"] = time.perf_counter() - started

    nms_jsonl = shared_root / "03_nms/nms.jsonl"
    nms_trace = shared_root / "03_nms/nms_trace.jsonl.gz"
    started = time.perf_counter()
    nms = run_nms_jsonl(scored, nms_jsonl, trace_output=nms_trace)
    stage_seconds["nms"] = time.perf_counter() - started

    cuts_path = shared_root / "04_cut_detection/cuts.json"
    started = time.perf_counter()
    cuts_result = HighPrecisionCutDetector().detect(scored, video)
    cuts_path.parent.mkdir(parents=True, exist_ok=True)
    cuts_temporary = cuts_path.with_suffix(".json.tmp")
    write_cut_list(
        cuts_temporary,
        CutList(
            tuple(cuts_result.frames),
            cuts_result.method,
            cuts_result.elapsed_seconds,
        ),
    )
    os.replace(cuts_temporary, cuts_path)
    stage_seconds["cut_detection"] = time.perf_counter() - started

    tracked = shared_root / "05_tracking/tracked.sqlite"
    started = time.perf_counter()
    tracking = build_tracked_sqlite(
        nms_jsonl,
        tracked,
        cuts_path,
        remove_short_tracks_max_frames=10,
    )
    stage_seconds["tracking"] = time.perf_counter() - started

    started = time.perf_counter()
    source_root, preparation = prepare_classwise_source(
        tracked,
        shared_root / "06_polygon_preparation",
        width=int(row["width"]),
        height=int(row["height"]),
        input_video=video,
    )
    stage_seconds["polygon_preparation"] = time.perf_counter() - started
    audit = audit_sqlite(tracked)
    if not audit.ok:
        raise RuntimeError(f"invalid tracked SQLite: {tracked}: {audit.to_dict()}")
    payload = {
        "schema_version": 1,
        "privacy": (
            "Cut detection decoded 96x54 frames locally. No frame image was "
            "displayed, uploaded, or exposed to an AI agent. All other stages "
            "used SQLite/JSON geometry only."
        ),
        "run": row,
        "score_min": float(score_min),
        "stage_seconds": stage_seconds,
        "wall_seconds": time.perf_counter() - started_all,
        "normalization": normalization,
        "score": score,
        "nms": nms,
        "cuts": {
            "count": len(cuts_result.frames),
            "method": cuts_result.method,
            "detector_seconds": cuts_result.elapsed_seconds,
            "sha256": _sha256(cuts_path),
        },
        "tracking": tracking,
        "preparation": preparation,
        "tracked_audit": audit.to_dict(),
        "artifacts": {
            "normalized_jsonl": str(normalized.resolve()),
            "scored_jsonl": str(scored.resolve()),
            "nms_jsonl": str(nms_jsonl.resolve()),
            "cuts_json": str(cuts_path.resolve()),
            "tracked_sqlite": str(tracked.resolve()),
            "prepared_source_root": str(Path(source_root).resolve()),
        },
        "hashes": {
            "normalized_jsonl": _sha256(normalized),
            "scored_jsonl": _sha256(scored),
            "nms_jsonl": _sha256(nms_jsonl),
            "tracked_sqlite": _sha256(tracked),
        },
    }
    _atomic_json(manifest_path, payload)
    return tracked, Path(source_root), payload


def _phase2_matrix(phase2_root: Path) -> dict[str, Any]:
    return json.loads(
        (phase2_root / "phase2_matrix.json").read_text(encoding="utf-8")
    )


def _aggregate_metrics(matrix: dict[str, Any]) -> dict[str, Any]:
    aggregate = dict(matrix["completed_profiles"][-1])
    aggregate.pop("video_fps", None)  # upstream runner assumes the KPI length
    return aggregate


def _run_one(
    row: dict[str, object],
    run_root: Path,
    tracked: Path,
    source_root: Path,
    shared: dict[str, object],
    *,
    mode: str,
    interval: int,
) -> dict[str, object]:
    result_root = run_root / mode / f"interval_{interval}"
    result_path = result_root / "run_result.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        final_sqlite = Path(str(payload["artifacts"]["result_sqlite"]))
        if not final_sqlite.is_file() or not audit_sqlite(final_sqlite).ok:
            raise RuntimeError(f"invalid resumed final SQLite: {result_path}")
        return payload

    evaluator = MODE_TO_EVALUATOR[mode]
    config = with_interval_evaluation(
        evaluator,
        with_target_interval(interval),
    )
    started = time.perf_counter()
    optimizer = run_polygon_optimizer(
        source_root,
        result_root / "polygon",
        config=config,
        labels=tuple(shared["preparation"]["active_labels"]),
        max_tracks=0,
        force=False,
        # A corpus benchmark records the native-exact audit instead of hiding
        # any residual violation left after fixed-14 local Recall repair.
    )
    optimizer_wall = time.perf_counter() - started
    phase2_root = Path(str(optimizer["phase2_root"]))
    matrix = _phase2_matrix(phase2_root)
    aggregate = _aggregate_metrics(matrix)
    # The three labels run concurrently.  Their maximum recorded wall time is
    # the stable optimizer latency, including when a completed phase is resumed
    # after a fail-closed audit.  Wrapper wall on resume would be near zero and
    # would produce a meaningless FPS.
    optimizer_compute_wall = max(
        (float(item["wall_seconds"]) for item in matrix.get("rows", [])),
        default=float(optimizer_wall),
    )

    final_sqlite = (
        run_root.parents[1]
        / "final_sqlite"
        / mode
        / f"interval_{interval}"
        / f"{Path(str(row['raw_sqlite'])).stem}.sqlite"
    )
    started = time.perf_counter()
    software = export_software_sqlite(
        Path(str(row["raw_sqlite"])),
        tracked,
        phase2_root,
        final_sqlite,
        config=config,
    )
    export_wall = time.perf_counter() - started
    final_audit = audit_sqlite(final_sqlite)
    if not final_audit.ok:
        raise RuntimeError(
            f"invalid final SQLite: {final_sqlite}: {final_audit.to_dict()}"
        )
    raw_frames = int(row["raw_frames"])
    shared_wall = float(shared["wall_seconds"])
    payload = {
        "schema_version": 1,
        "run_id": row["run_id"],
        "mode": mode,
        "interval_evaluation": evaluator,
        "target_interval": int(interval),
        "raw_frames": raw_frames,
        "raw_detections": int(row["raw_detections"]),
        "timing": {
            "shared_upstream_seconds": shared_wall,
            "optimizer_seconds": float(optimizer_compute_wall),
            "optimizer_wrapper_seconds": float(optimizer_wall),
            "export_seconds": float(export_wall),
            "marginal_seconds": float(optimizer_compute_wall + export_wall),
            "equivalent_end_to_end_seconds": float(
                shared_wall + optimizer_compute_wall + export_wall
            ),
            "optimizer_video_fps": raw_frames
            / max(optimizer_compute_wall, 1e-9),
            "equivalent_end_to_end_fps": raw_frames
            / max(shared_wall + optimizer_compute_wall + export_wall, 1e-9),
        },
        "quality": {
            key: aggregate.get(key)
            for key in (
                "observation_rows",
                "keyframes",
                "actual_mean_interval",
                "recall_min",
                "recall_violations",
                "infeasible_streams",
                "iou_mean",
                "iou_q01_by_class_min",
                "area_ratio_max",
            )
        },
        "optimizer": optimizer,
        "software": software,
        "final_audit": final_audit.to_dict(),
        "final_schema_fingerprint": schema_fingerprint(final_sqlite),
        "artifacts": {
            "phase2_root": str(phase2_root.resolve()),
            "result_sqlite": str(final_sqlite.resolve()),
            "result_sqlite_sha256": _sha256(final_sqlite),
        },
    }
    _atomic_json(result_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--intervals", default="2,5")
    parser.add_argument("--modes", default="default_cuda,cpu_exact")
    parser.add_argument("--score-min", type=float, default=0.30)
    parser.add_argument(
        "--run-ids",
        help="optional comma-separated corpus run IDs; default is all discovered V3 runs",
    )
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    intervals = tuple(int(value) for value in args.intervals.split(",") if value)
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    if not intervals or any(value < 1 for value in intervals):
        raise ValueError("intervals must be positive integers")
    if not modes or any(value not in MODE_TO_EVALUATOR for value in modes):
        raise ValueError(f"modes must be selected from {tuple(MODE_TO_EVALUATOR)}")
    if not 0.0 <= float(args.score_min) <= 1.0:
        raise ValueError("score-min must be in [0, 1]")
    corpus = discover_corpus(args.corpus_root)
    if args.run_ids:
        selected = {value.strip() for value in args.run_ids.split(",") if value.strip()}
        corpus = [row for row in corpus if str(row["run_id"]) in selected]
        missing = selected - {str(row["run_id"]) for row in corpus}
        if missing:
            raise ValueError(f"unknown run IDs: {sorted(missing)}")
    if args.list:
        print(json.dumps(corpus, ensure_ascii=False, indent=2))
        return 0

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    corpus_manifest = {
        "schema_version": 1,
        "corpus_root": str(args.corpus_root.expanduser().resolve()),
        "output_root": str(output_root),
        "intervals": list(intervals),
        "modes": {mode: MODE_TO_EVALUATOR[mode] for mode in modes},
        "score_min": float(args.score_min),
        "runs": corpus,
        "expected_final_sqlites": len(corpus) * len(intervals) * len(modes),
    }
    _atomic_json(output_root / "corpus_manifest.json", corpus_manifest)

    results: list[dict[str, object]] = []
    batch_started = time.perf_counter()
    for run_index, row in enumerate(corpus, 1):
        run_id = str(row["run_id"])
        print(
            f"[corpus {run_index}/{len(corpus)}] {run_id} "
            f"frames={row['raw_frames']} detections={row['raw_detections']}",
            flush=True,
        )
        run_root = output_root / "runs" / run_id
        tracked, source_root, shared = _build_shared_upstream(
            row,
            run_root,
            score_min=float(args.score_min),
        )
        for interval in intervals:
            for mode in modes:
                print(
                    f"[run] {run_id} interval={interval} mode={mode}",
                    flush=True,
                )
                result = _run_one(
                    row,
                    run_root,
                    tracked,
                    source_root,
                    shared,
                    mode=mode,
                    interval=interval,
                )
                results.append(result)
                timing = result["timing"]
                quality = result["quality"]
                print(
                    f"[done] {run_id} interval={interval} mode={mode} "
                    f"fps={timing['optimizer_video_fps']:.3f} "
                    f"iou={quality['iou_mean']:.6f} "
                    f"recall_min={quality['recall_min']:.9f}",
                    flush=True,
                )
                _atomic_json(
                    output_root / "batch_state.json",
                    {
                        **corpus_manifest,
                        "completed": len(results),
                        "results": results,
                        "elapsed_seconds": time.perf_counter() - batch_started,
                    },
                )
    final_sqlites = sorted((output_root / "final_sqlite").glob("*/*/*.sqlite"))
    expected = int(corpus_manifest["expected_final_sqlites"])
    if len(final_sqlites) != expected:
        raise RuntimeError(
            f"final SQLite count mismatch: expected={expected} actual={len(final_sqlites)}"
        )
    summary = {
        **corpus_manifest,
        "status": "complete",
        "completed": len(results),
        "batch_wall_seconds": time.perf_counter() - batch_started,
        "final_sqlites": [str(path.resolve()) for path in final_sqlites],
        "results": results,
    }
    _atomic_json(output_root / "batch_summary.json", summary)
    print(
        f"[complete] results={len(results)} final_sqlites={len(final_sqlites)} "
        f"wall={summary['batch_wall_seconds']:.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
