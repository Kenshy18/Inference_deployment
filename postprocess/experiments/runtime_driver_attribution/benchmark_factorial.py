#!/usr/bin/env python3
"""Controlled runtime-factor attribution for Production geometry engines.

The benchmark keeps one real 480-observation shape sequence and changes only:

* the number of independent track streams;
* a uniform area multiplier around each component centroid; and
* the fixed editable point budget selected from Production's 14/16/18/20 set.

It is an experiment-only harness.  Production packages never import it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
from dataclasses import replace
import gc
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
from typing import Iterable

from production.config import PRODUCTION
from production.polygon.materialize import materialize_outputs
from production.polygon.runtime_bridge import (
    build_runtime_config,
    optimize,
    prepare_inputs,
)
from production.curve.config import CURVE_PRODUCTION
from production.curve.engine import run_curve_optimizer
from production.curve.preparation import prepare_curve_source


LABEL = "女性器"
SOURCE_TRACK = "26"
WIDTH = 1920
HEIGHT = 1080
OBSERVATIONS = 480
TARGET_INTERVAL = 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--geometries",
        default="polygon,catmull_rom",
        help="comma-separated subset of polygon,catmull_rom",
    )
    parser.add_argument("--factorial-repeats", type=int, default=2)
    parser.add_argument("--skip-response-curves", action="store_true")
    return parser


def _scaled_centered_polygons(raw: str, area_factor: float) -> str:
    polygons = json.loads(raw)
    linear = math.sqrt(float(area_factor))
    transformed: list[list[list[float]]] = []
    all_points = [point for polygon in polygons for point in polygon]
    if not all_points:
        return "[]"
    source_cx = sum(float(point[0]) for point in all_points) / len(all_points)
    source_cy = sum(float(point[1]) for point in all_points) / len(all_points)
    for polygon in polygons:
        transformed.append(
            [
                [
                    float(WIDTH / 2.0 + linear * (float(point[0]) - source_cx)),
                    float(HEIGHT / 2.0 + linear * (float(point[1]) - source_cy)),
                ]
                for point in polygon
            ]
        )
    return json.dumps(transformed, ensure_ascii=False, separators=(",", ":"))


def _create_source(
    source: Path,
    output: Path,
    *,
    area_factor: float,
    track_count: int,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as db:
        rows = list(
            db.execute(
                "SELECT frame,polygons FROM masks WHERE track_id=? "
                "ORDER BY frame LIMIT ?",
                (SOURCE_TRACK, OBSERVATIONS),
            )
        )
    if len(rows) != OBSERVATIONS:
        raise RuntimeError(f"expected {OBSERVATIONS} rows, got {len(rows)}")
    with sqlite3.connect(output) as db:
        db.executescript(
            """
            CREATE TABLE masks(
                frame INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                polygons TEXT NOT NULL,
                shape_type TEXT,
                dilate_px INTEGER NOT NULL DEFAULT 0,
                feather_px INTEGER NOT NULL DEFAULT 0,
                mosaic_block INTEGER NOT NULL DEFAULT 0,
                mosaic_alias REAL NOT NULL DEFAULT 0,
                label TEXT,
                PRIMARY KEY(frame,track_id)
            );
            CREATE TABLE tracks(track_id TEXT PRIMARY KEY,label TEXT);
            """
        )
        generated = []
        for index, (_source_frame, polygons) in enumerate(rows):
            partition = min(
                int(track_count) - 1,
                (index * int(track_count)) // OBSERVATIONS,
            )
            track_id = f"26{partition + 1:02d}"
            generated.append(
                (
                    int(index),
                    track_id,
                    _scaled_centered_polygons(str(polygons), float(area_factor)),
                    LABEL,
                )
            )
        db.executemany(
            """
            INSERT INTO masks(
                frame,track_id,polygons,shape_type,dilate_px,feather_px,
                mosaic_block,mosaic_alias,label
            ) VALUES (?,?,?,'polygon',0,0,0,0,?)
            """,
            generated,
        )
        tracks = sorted({row[1] for row in generated})
        db.executemany(
            "INSERT INTO tracks(track_id,label) VALUES (?,?)",
            ((track_id, LABEL) for track_id in tracks),
        )
        db.commit()
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"generated source failed integrity: {integrity}")
    return {
        "path": str(output.resolve()),
        "observations": len(generated),
        "track_ids": tracks,
        "area_factor": float(area_factor),
        "track_count": int(track_count),
    }


def _force_vertex_policy(
    policy: dict[str, object], vertices: int
) -> dict[str, object]:
    value = copy.deepcopy(policy)
    tracks = value.get("tracks")
    if not isinstance(tracks, dict):
        raise RuntimeError("vertex policy has no tracks")
    for entry in tracks.values():
        entry["vertices_per_component"] = int(vertices)
    allowed = [14, 16, 18, 20]
    value["allowed_vertices"] = allowed
    summary = value.setdefault("summary", {})
    rows = sum(int(entry.get("rows", 0)) for entry in tracks.values())
    summary["tracks_by_vertices"] = {
        str(point_count): int(len(tracks) if point_count == vertices else 0)
        for point_count in allowed
    }
    summary["track_rows_by_vertices"] = {
        str(point_count): int(rows if point_count == vertices else 0)
        for point_count in allowed
    }
    return value


def _runtime_summary(output: Path) -> dict[str, object]:
    path = (
        output
        / f"interval_{TARGET_INTERVAL}"
        / "polygon_adaptive_keyframe_v2"
        / LABEL
        / "runtime"
        / "opt"
        / "summary.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _run_polygon_case(
    source: Path,
    root: Path,
    *,
    vertices: int,
) -> dict[str, object]:
    config = replace(PRODUCTION, target_interval=TARGET_INTERVAL)
    config.validate()
    started = time.perf_counter()
    source_root, preparation = prepare_inputs(
        source,
        root / "preparation",
        width=WIDTH,
        height=HEIGHT,
        input_video=None,
        config=config,
    )
    policy = _force_vertex_policy(dict(preparation["vertex_policy"]), vertices)
    policy_path = Path(source_root) / "vertex_policy.json"
    policy_path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    prepared_seconds = time.perf_counter() - started
    optimizer_started = time.perf_counter()
    optimizer = optimize(
        source_root,
        root / "optimizer",
        labels=(LABEL,),
        max_tracks=0,
        force=True,
        config=config,
        optimizer_workers=3,
    )
    optimizer_wall = time.perf_counter() - optimizer_started
    materialize_started = time.perf_counter()
    runtime = build_runtime_config(config, optimizer_workers=3)
    materialize_outputs(
        Path(str(optimizer["phase2_root"])),
        source,
        root / "predictions.sqlite",
        root / "keyframes.sqlite",
        config=config,
        runtime_profile=runtime.polygon_profile_id,
    )
    materialize_seconds = time.perf_counter() - materialize_started
    wall_seconds = time.perf_counter() - started
    summary = _runtime_summary(root / "optimizer")
    stage_seconds = summary.get("stage_seconds_total", {})
    with sqlite3.connect(root / "predictions.sqlite") as db:
        output_rows = int(db.execute("SELECT COUNT(*) FROM masks").fetchone()[0])
    return {
        "wall_seconds": float(wall_seconds),
        "prepared_seconds": float(prepared_seconds),
        "optimizer_process_wall_seconds": float(optimizer_wall),
        "materialize_seconds": float(materialize_seconds),
        "output_rows": output_rows,
        "effective_stream_count": int(
            summary["segmentation_stats"]["effective_stream_count"]
        ),
        "mean_state_count": float(summary["mean_state_count"]),
        "interval_eval_count": int(summary["interval_eval_count"]),
        "interval_eval_frames": int(summary["interval_eval_frames"]),
        "optimizer_seconds": float(summary["optimizer_seconds"]),
        "build_candidates_seconds": float(
            stage_seconds.get("build_candidates_seconds", 0.0)
        ),
        "solve_dp_seconds": float(stage_seconds.get("solve_dp_seconds", 0.0)),
        "pair_vote_seconds": float(
            stage_seconds.get("pair_vote_refine_seconds", 0.0)
        ),
        "final_eval_seconds": float(stage_seconds.get("final_eval_seconds", 0.0)),
    }


def _curve_worker(
    source_value: str,
    track_ids: tuple[str, ...],
    root_value: str,
    vertices: int,
) -> dict[str, object]:
    source = Path(source_value)
    root = Path(root_value)
    config = replace(
        CURVE_PRODUCTION,
        target_interval=TARGET_INTERVAL,
        # Match the six-shard long-corpus Production benchmark.  Keeping this
        # fixed prevents the point-count factor from being confounded by a
        # different native thread budget.
        native_cpu_threads=4,
    )
    preparation_config = replace(
        PRODUCTION,
        target_interval=TARGET_INTERVAL,
        interval_evaluation="native_exact",
    )
    started = time.perf_counter()
    preparation = prepare_curve_source(
        source,
        root / "preparation",
        width=WIDTH,
        height=HEIGHT,
        input_video=None,
        config=build_runtime_config(preparation_config),
        selected_track_ids=track_ids,
    )
    preparation["vertex_policy"] = _force_vertex_policy(
        dict(preparation["vertex_policy"]), vertices
    )
    prepared_seconds = time.perf_counter() - started
    engine = run_curve_optimizer(
        source,
        preparation,
        root / "runtime",
        config=config,
    )
    wall_seconds = time.perf_counter() - started
    audit = engine["audit"]
    if int(audit["recall_violations"]) or int(audit["topology_invalid_frames"]):
        raise RuntimeError(f"curve audit failure: {audit}")
    initial_fit_seconds = 0.0
    dp_seconds = 0.0
    pair_vote_seconds = 0.0
    point_refine_seconds = 0.0
    final_audit_seconds = 0.0
    graph_edges = 0
    for stream in engine.get("stream_summaries", []):
        for component in stream.get("fit", []):
            initial_fit_seconds += float(component["fit"].get("elapsed_seconds", 0.0))
        for optimization in stream.get("optimization", []):
            dp_seconds += float(optimization.get("dp_seconds", 0.0))
            pair_vote_seconds += float(optimization.get("pair_vote_seconds", 0.0))
            point_refine_seconds += float(
                optimization.get("point_refine_seconds", 0.0)
            )
            final_audit_seconds += float(
                optimization.get("final_audit_seconds", 0.0)
            )
            graph_edges += int(optimization.get("graph_edges", 0))
    return {
        "wall_seconds": float(wall_seconds),
        "prepared_seconds": float(prepared_seconds),
        "output_rows": int(engine["prediction_rows"]),
        "stream_count": int(engine["streams"]),
        "dp_seconds": float(dp_seconds),
        "pair_vote_seconds": float(pair_vote_seconds),
        "point_refine_seconds": float(point_refine_seconds),
        "initial_fit_seconds": float(initial_fit_seconds),
        "final_audit_seconds": float(final_audit_seconds),
        "graph_edges": int(graph_edges),
    }


def _partition_track_ids(source: Path, workers: int) -> list[tuple[str, ...]]:
    with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as db:
        counts = [
            (str(track_id), int(count))
            for track_id, count in db.execute(
                "SELECT track_id,COUNT(*) FROM masks GROUP BY track_id "
                "ORDER BY COUNT(*) DESC,track_id"
            )
        ]
    buckets: list[list[str]] = [[] for _index in range(workers)]
    loads = [0 for _index in range(workers)]
    for track_id, count in counts:
        index = min(range(workers), key=lambda value: (loads[value], value))
        buckets[index].append(track_id)
        loads[index] += count
    return [tuple(bucket) for bucket in buckets if bucket]


def _run_curve_case(
    source: Path,
    root: Path,
    *,
    vertices: int,
) -> dict[str, object]:
    with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as db:
        track_count = int(db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0])
    workers = min(2, track_count)
    partitions = _partition_track_ids(source, workers)
    started = time.perf_counter()
    if len(partitions) == 1:
        results = [
            _curve_worker(
                str(source), partitions[0], str(root / "worker_00"), vertices
            )
        ]
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(partitions), mp_context=context
        ) as executor:
            futures = [
                executor.submit(
                    _curve_worker,
                    str(source),
                    track_ids,
                    str(root / f"worker_{index:02d}"),
                    vertices,
                )
                for index, track_ids in enumerate(partitions)
            ]
            results = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - started
    return {
        "wall_seconds": float(wall_seconds),
        "prepared_seconds_sum": float(
            sum(float(value["prepared_seconds"]) for value in results)
        ),
        "output_rows": int(sum(int(value["output_rows"]) for value in results)),
        "effective_stream_count": int(
            sum(int(value["stream_count"]) for value in results)
        ),
        "process_workers": int(len(partitions)),
        "dp_seconds_sum": float(sum(float(value["dp_seconds"]) for value in results)),
        "pair_vote_seconds_sum": float(
            sum(float(value["pair_vote_seconds"]) for value in results)
        ),
        "point_refine_seconds_sum": float(
            sum(float(value["point_refine_seconds"]) for value in results)
        ),
        "initial_fit_seconds_sum": float(
            sum(float(value["initial_fit_seconds"]) for value in results)
        ),
        "final_audit_seconds_sum": float(
            sum(float(value["final_audit_seconds"]) for value in results)
        ),
        "graph_edges": int(sum(int(value["graph_edges"]) for value in results)),
    }


def _case_key(track_count: int, area_factor: float, vertices: int) -> str:
    area = str(area_factor).replace(".", "p")
    return f"tracks_{track_count}_area_{area}_vertices_{vertices}"


def _factorial_cases(repeats: int) -> list[dict[str, object]]:
    base = [
        {"track_count": tracks, "area_factor": area, "vertices": vertices}
        for tracks in (1, 6)
        for area in (0.5, 1.5)
        for vertices in (14, 20)
    ]
    cases: list[dict[str, object]] = []
    for replicate in range(repeats):
        ordered = base if replicate % 2 == 0 else list(reversed(base))
        for value in ordered:
            cases.append({**value, "replicate": replicate + 1, "series": "factorial"})
    return cases


def _response_cases() -> list[dict[str, object]]:
    values = {
        (tracks, 1.0, 16, "track_response") for tracks in (1, 2, 3, 6)
    }
    values.update(
        (1, area, 16, "area_response") for area in (0.5, 1.0, 1.5)
    )
    values.update(
        (1, 1.0, vertices, "vertex_response")
        for vertices in (14, 16, 18, 20)
    )
    return [
        {
            "track_count": tracks,
            "area_factor": area,
            "vertices": vertices,
            "replicate": 1,
            "series": series,
        }
        for tracks, area, vertices, series in sorted(values)
    ]


def _run_geometry(
    geometry: str,
    cases: Iterable[dict[str, object]],
    source_paths: dict[tuple[int, float], Path],
    work_root: Path,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        track_count = int(case["track_count"])
        area_factor = float(case["area_factor"])
        vertices = int(case["vertices"])
        source = source_paths[(track_count, area_factor)]
        case_id = _case_key(track_count, area_factor, vertices)
        run_root = (
            work_root
            / geometry
            / str(case["series"])
            / f"rep_{int(case['replicate']):02d}"
            / case_id
        )
        run_root.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "event": "start",
                    "geometry": geometry,
                    "index": index + 1,
                    "track_count": track_count,
                    "area_factor": area_factor,
                    "vertices": vertices,
                    "series": case["series"],
                    "replicate": case["replicate"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if geometry == "polygon":
            measured = _run_polygon_case(source, run_root, vertices=vertices)
        elif geometry == "catmull_rom":
            measured = _run_curve_case(source, run_root, vertices=vertices)
        else:  # pragma: no cover - CLI validation
            raise ValueError(geometry)
        row = {
            **case,
            "geometry": geometry,
            "case_id": case_id,
            "observations": OBSERVATIONS,
            "target_interval": TARGET_INTERVAL,
            "throughput_fps": float(OBSERVATIONS / measured["wall_seconds"]),
            **measured,
        }
        results.append(row)
        print(json.dumps({"event": "complete", **row}, ensure_ascii=False), flush=True)
        shutil.rmtree(run_root)
        gc.collect()
    return results


def main() -> int:
    args = _parser().parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    geometries = tuple(
        value.strip() for value in args.geometries.split(",") if value.strip()
    )
    unsupported = set(geometries) - {"polygon", "catmull_rom"}
    if unsupported:
        raise ValueError(f"unsupported geometries: {sorted(unsupported)}")
    cases = _factorial_cases(max(1, int(args.factorial_repeats)))
    if not args.skip_response_curves:
        cases.extend(_response_cases())
    source_factors = sorted(
        {
            (int(case["track_count"]), float(case["area_factor"]))
            for case in cases
        }
    )
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="runtime_driver_attribution_") as tmp:
        work_root = Path(tmp)
        source_paths: dict[tuple[int, float], Path] = {}
        source_manifests = []
        for track_count, area_factor in source_factors:
            path = work_root / "sources" / (
                f"tracks_{track_count}_area_{str(area_factor).replace('.', 'p')}.sqlite"
            )
            source_manifests.append(
                _create_source(
                    source,
                    path,
                    area_factor=area_factor,
                    track_count=track_count,
                )
            )
            source_paths[(track_count, area_factor)] = path
        for geometry in geometries:
            results.extend(
                _run_geometry(geometry, cases, source_paths, work_root / "runs")
            )
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": str(source),
        "source_track": SOURCE_TRACK,
        "observations": OBSERVATIONS,
        "frame_dimensions": [WIDTH, HEIGHT],
        "shape_transform": (
            "each frame recentered to canvas center then uniformly scaled by "
            "sqrt(area_factor) around the aggregate component centroid"
        ),
        "target_interval": TARGET_INTERVAL,
        "factorial": {
            "track_count_levels": [1, 6],
            "area_factor_levels": [0.5, 1.5],
            "vertices_levels": [14, 20],
            "repeats": max(1, int(args.factorial_repeats)),
        },
        "curve_parallelism": "up to two balanced process shards for one class",
        "polygon_parallelism": "up to three optimizer processes for one class",
        "source_manifests": source_manifests,
        "results": results,
    }
    destination = output / "benchmark_results.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(destination, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
