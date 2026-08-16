#!/usr/bin/env python3
"""Package Phase-2 class results into the stable software-facing SQLite.

The Phase-2 optimizer emits one dense ``predictions.sqlite`` and one native
``final_keyframes.json`` per class.  This exporter merges the dense staging
rows, imports only the independently selected native keyframes, and rebuilds
the regular keyframe-primary V3 result.  The input inference data and schema
are never modified.

Video pixels are not opened.  Optional exact-recall repair inserts the source
polygon at the handful of frames where the final CPU raster audit disagrees
with the CUDA feasibility classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import tempfile
from pathlib import Path

from artifacts.unified_sqlite import build_integrated_result
from experimental.polygon_recall_optimizer.sqlite_export import schema_fingerprint
from metrics import evaluate_sqlite, write_metrics


LABELS = ("女性器", "男性器", "結合部分")


def _backup(source: Path, destination: Path) -> None:
    with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as src:
        with sqlite3.connect(destination) as dst:
            src.backup(dst)


def _phase2_label_root(root: Path, profile: str, label: str) -> Path:
    path = root / profile / label
    required = (
        path / "runtime/pred/predictions.sqlite",
        path / "runtime/opt/final_keyframes.json",
        path / "runtime/exact/keyframe_exact_metrics.csv",
        path / "metrics.json",
    )
    missing = [str(value) for value in required if not value.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete Phase-2 result for {label}: {missing}")
    return path


def _violations(path: Path, recall_floor: float) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["recall"]) + 1e-12 < float(recall_floor):
                rows.append((int(row["frame"]), str(row["track_id"])))
    return sorted(set(rows))


def _source_polygon(source: Path, frame: int, track_id: str) -> object:
    with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as db:
        row = db.execute(
            "SELECT polygons FROM masks WHERE frame=? AND track_id=?",
            (int(frame), str(track_id)),
        ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError(
            f"Recall repair source polygon is absent: frame={frame}, track={track_id}"
        )
    return json.loads(str(row[0]))


def _key_rows(
    label_root: Path,
    *,
    label: str,
    recall_floor: float,
    repair_exact_violations: bool,
) -> tuple[list[tuple[int, str, str, str]], list[dict[str, object]]]:
    metrics = json.loads((label_root / "metrics.json").read_text(encoding="utf-8"))
    source = Path(str(metrics["source_sqlite"])).resolve()
    payload = json.loads(
        (label_root / "runtime/opt/final_keyframes.json").read_text(encoding="utf-8")
    )
    keyed: dict[tuple[int, str], object] = {
        (int(row["frame"]), str(row["track_id"])): row["polygons"]
        for row in payload
    }
    repairs: list[dict[str, object]] = []
    if repair_exact_violations:
        for frame, track_id in _violations(
            label_root / "runtime/exact/keyframe_exact_metrics.csv", recall_floor
        ):
            was_key = (frame, track_id) in keyed
            keyed[(frame, track_id)] = _source_polygon(source, frame, track_id)
            repairs.append(
                {
                    "label": label,
                    "frame": frame,
                    "track_id": track_id,
                    "action": "replace_raw_key" if was_key else "insert_raw_key",
                }
            )
    rows = [
        (
            int(frame),
            str(track_id),
            json.dumps(polygons, ensure_ascii=False, separators=(",", ":")),
            label,
        )
        for (frame, track_id), polygons in sorted(
            keyed.items(), key=lambda item: (int(item[0][1]), item[0][0])
        )
    ]
    return rows, repairs


def _replace_dense_class(
    merged: sqlite3.Connection,
    prediction_path: Path,
    label: str,
) -> dict[str, int]:
    expected_tracks = {
        str(row[0])
        for row in merged.execute("SELECT track_id FROM tracks WHERE label=?", (label,))
    }
    with sqlite3.connect(f"file:{prediction_path.resolve()}?mode=ro", uri=True) as src:
        rows = [
            (int(frame), str(track_id), str(polygons))
            for frame, track_id, polygons in src.execute(
                "SELECT frame, track_id, polygons FROM masks ORDER BY track_id, frame"
            )
        ]
    actual_tracks = {track_id for _frame, track_id, _polygons in rows}
    if actual_tracks != expected_tracks:
        raise RuntimeError(
            f"{label}: Phase-2 track set differs from materialized final: "
            f"missing={sorted(expected_tracks-actual_tracks)}, "
            f"unexpected={sorted(actual_tracks-expected_tracks)}"
        )
    merged.execute(
        "DELETE FROM masks WHERE track_id IN (SELECT track_id FROM tracks WHERE label=?)",
        (label,),
    )
    merged.executemany(
        """
        INSERT INTO masks(
            frame, track_id, polygons, shape_type, dilate_px, feather_px,
            mosaic_block, mosaic_alias, label
        ) VALUES (?, ?, ?, 'polygon', 0, 0, 0, 0.0, ?)
        """,
        ((frame, track_id, polygons, label) for frame, track_id, polygons in rows),
    )
    return {"tracks": len(actual_tracks), "dense_rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--tracked-sqlite", type=Path, required=True)
    parser.add_argument("--base-final-sqlite", type=Path, required=True)
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--profile", default="orthogonal_c02_125_endpoints")
    parser.add_argument("--target-interval", type=int, default=10)
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument("--output-sqlite", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path)
    parser.add_argument(
        "--no-exact-recall-repair",
        action="store_true",
        help="preserve the raw CUDA result even when the final CPU audit disagrees",
    )
    args = parser.parse_args()

    inputs = [args.input_sqlite, args.tracked_sqlite, args.base_final_sqlite]
    for value in inputs:
        if not value.expanduser().resolve().is_file():
            raise FileNotFoundError(value)
    output = args.output_sqlite.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing SQLite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    validation_path = (
        args.validation_json.expanduser().resolve()
        if args.validation_json is not None
        else output.with_suffix(".validation.json")
    )

    phase2_root = args.phase2_root.expanduser().resolve()
    profile = str(args.profile)
    key_rows: list[tuple[int, str, str, str]] = []
    repairs: list[dict[str, object]] = []
    label_roots = {
        label: _phase2_label_root(phase2_root, profile, label) for label in LABELS
    }
    for label, label_root in label_roots.items():
        rows, label_repairs = _key_rows(
            label_root,
            label=label,
            recall_floor=float(args.recall_floor),
            repair_exact_violations=not bool(args.no_exact_recall_repair),
        )
        key_rows.extend(rows)
        repairs.extend(label_repairs)

    with tempfile.TemporaryDirectory(prefix="phase2-review-", dir=output.parent) as tmp:
        temporary = Path(tmp)
        merged_final = temporary / "merged_materialized.sqlite"
        keys_sqlite = temporary / "native_polygon_keys.sqlite"
        _backup(args.base_final_sqlite.expanduser().resolve(), merged_final)
        class_dense: dict[str, dict[str, int]] = {}
        with sqlite3.connect(merged_final) as merged:
            with merged:
                for label, label_root in label_roots.items():
                    class_dense[label] = _replace_dense_class(
                        merged,
                        label_root / "runtime/pred/predictions.sqlite",
                        label,
                    )
                for table in (
                    "class_postprocess_policies",
                    "mask_postprocess_provenance",
                ):
                    columns = {
                        str(row[1]) for row in merged.execute(f"PRAGMA table_info({table})")
                    }
                    if "keyframe_interval" in columns:
                        merged.execute(
                            f"UPDATE {table} SET keyframe_interval=? "
                            "WHERE label IN ('女性器','男性器','結合部分')",
                            (int(args.target_interval),),
                        )
            integrity = str(merged.execute("PRAGMA integrity_check").fetchone()[0])
            foreign_keys = list(merged.execute("PRAGMA foreign_key_check"))
            if integrity != "ok" or foreign_keys:
                raise RuntimeError(
                    f"invalid merged staging SQLite: {integrity}, "
                    f"foreign_keys={len(foreign_keys)}"
                )

        with sqlite3.connect(keys_sqlite) as keys:
            keys.execute(
                """
                CREATE TABLE masks(
                    frame INTEGER NOT NULL,
                    track_id TEXT NOT NULL,
                    polygons TEXT NOT NULL,
                    label TEXT NOT NULL,
                    PRIMARY KEY(frame, track_id)
                )
                """
            )
            keys.executemany(
                "INSERT INTO masks(frame, track_id, polygons, label) VALUES (?, ?, ?, ?)",
                key_rows,
            )
            keys.commit()

        build = build_integrated_result(
            args.input_sqlite.expanduser().resolve(),
            args.tracked_sqlite.expanduser().resolve(),
            merged_final,
            output,
            polygon_keyframes_sqlite=keys_sqlite,
        )

    evaluation = evaluate_sqlite(
        args.input_sqlite.expanduser().resolve(),
        output,
        recall_floor=float(args.recall_floor),
    )
    with sqlite3.connect(
        f"file:{args.input_sqlite.expanduser().resolve()}?mode=ro", uri=True
    ) as source:
        baseline_fingerprint = schema_fingerprint(source)
    with sqlite3.connect(f"file:{output}?mode=ro", uri=True) as result:
        output_fingerprint = schema_fingerprint(result)
        result_integrity = str(result.execute("PRAGMA integrity_check").fetchone()[0])
        result_foreign_keys = len(result.execute("PRAGMA foreign_key_check").fetchall())
    validation = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "experimental": True,
        "profile": profile,
        "target_interval": int(args.target_interval),
        "recall_floor": float(args.recall_floor),
        "exact_recall_repairs": repairs,
        "key_rows": len(key_rows),
        "class_dense": class_dense,
        "build": build,
        "evaluation": evaluation,
        "schema": {
            "baseline_fingerprint": baseline_fingerprint,
            "output_fingerprint": output_fingerprint,
            "unchanged": baseline_fingerprint == output_fingerprint,
            "integrity_check": result_integrity,
            "foreign_key_errors": result_foreign_keys,
        },
    }
    write_metrics(validation_path, validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
