"""Stable-schema software SQLite export boundary."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from .config import CANDIDATE, CandidateConfig


POSTPROCESS_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = POSTPROCESS_ROOT.parent
EXPORTER = POSTPROCESS_ROOT / "experimental/0809/export_phase2_review_sqlite.py"
EXACT_METRICS_HEADER = (
    "frame,track_id,run_id,has_keyframe,gt_area,pred_area,intersection,"
    "union,recall,precision,iou,weighted_error\n"
)


def _tracked_label_counts(path: Path, labels: tuple[str, ...]) -> dict[str, int]:
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as db:
        values = {
            str(label): int(count)
            for label, count in db.execute(
                "SELECT COALESCE(label, ''), COUNT(*) FROM tracks GROUP BY label"
            )
        }
    return {label: values.get(label, 0) for label in labels}


def _write_empty_label_artifacts(
    phase2_root: Path,
    tracked_sqlite: Path,
    label: str,
    config: CandidateConfig,
) -> dict[str, object]:
    """Create the export contract for a class with zero tracked instances."""
    label_root = phase2_root / config.polygon_profile_id / label
    prediction = label_root / "runtime/pred/predictions.sqlite"
    keyframes = label_root / "runtime/opt/final_keyframes.json"
    exact = label_root / "runtime/exact/keyframe_exact_metrics.csv"
    metrics = label_root / "metrics.json"
    for path in (prediction, keyframes, exact, metrics):
        path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(prediction) as db:
        db.execute("CREATE TABLE masks (frame INTEGER, track_id TEXT, polygons TEXT)")
    keyframes.write_text("[]\n", encoding="utf-8")
    exact.write_text(EXACT_METRICS_HEADER, encoding="utf-8")
    metrics.write_text(
        json.dumps(
            {
                "label": label,
                "source_sqlite": str(Path(tracked_sqlite).resolve()),
                "observation_rows": 0,
                "keyframes": 0,
                "recall_violations": 0,
                "candidate_profile": config.profile_id,
                "empty_class": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "label": label,
        "prediction_sqlite": str(prediction),
        "reason": "no_tracked_instances",
    }


def ensure_empty_label_artifacts(
    phase2_root: Path,
    tracked_sqlite: Path,
    config: CandidateConfig = CANDIDATE,
) -> list[dict[str, object]]:
    """Materialize zero-row artifacts only for truly absent classes."""
    counts = _tracked_label_counts(tracked_sqlite, config.labels)
    created: list[dict[str, object]] = []
    for label, count in counts.items():
        if count:
            continue
        label_root = Path(phase2_root) / config.polygon_profile_id / label
        if label_root.exists():
            raise RuntimeError(
                f"unexpected optimizer artifacts for absent class {label}: {label_root}"
            )
        created.append(
            _write_empty_label_artifacts(
                Path(phase2_root), Path(tracked_sqlite), label, config
            )
        )
    return created


def export_software_sqlite(
    raw_input_sqlite: Path,
    tracked_sqlite: Path,
    phase2_root: Path,
    output_sqlite: Path,
    *,
    config: CandidateConfig = CANDIDATE,
) -> dict[str, object]:
    config.validate()
    output = Path(output_sqlite).resolve()
    empty_labels = ensure_empty_label_artifacts(
        Path(phase2_root).resolve(),
        Path(tracked_sqlite).resolve(),
        config,
    )
    validation = output.with_suffix(".validation.json")
    command = [
        sys.executable,
        str(EXPORTER),
        "--input-sqlite",
        str(Path(raw_input_sqlite).resolve()),
        "--tracked-sqlite",
        str(Path(tracked_sqlite).resolve()),
        "--base-final-sqlite",
        str(Path(tracked_sqlite).resolve()),
        "--phase2-root",
        str(Path(phase2_root).resolve()),
        "--profile",
        config.polygon_profile_id,
        "--target-interval",
        str(config.temporal.target_interval),
        "--recall-floor",
        str(config.temporal.recall_floor),
        "--output-sqlite",
        str(output),
        "--validation-json",
        str(validation),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(POSTPROCESS_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    log = output.with_suffix(".export.log")
    output.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(
            f"software SQLite export failed with exit {process.returncode}; see {log}"
        )
    payload = json.loads(validation.read_text(encoding="utf-8"))
    schema = payload.get("schema", {})
    if not schema.get("unchanged"):
        raise RuntimeError("software SQLite schema changed")
    if schema.get("integrity_check") != "ok" or schema.get("foreign_key_errors"):
        raise RuntimeError(f"invalid software SQLite: {schema}")
    return {
        "output_sqlite": str(output),
        "validation": str(validation),
        "log": str(log),
        "validation_payload": payload,
        "empty_label_artifacts": empty_labels,
    }
