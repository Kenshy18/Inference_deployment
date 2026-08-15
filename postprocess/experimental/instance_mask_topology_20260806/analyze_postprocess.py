#!/usr/bin/env python3
"""Trace raw mask topology through a completed production postprocess run.

The script is intentionally read-only with respect to production artifacts.  It
joins raw detector IDs to ``tracking_assignments`` in the stable result SQLite
and writes a compact experimental audit database.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS detection_outcomes(
  run_key TEXT NOT NULL,
  detection_id INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  score REAL,
  foreground_component_count INTEGER NOT NULL,
  hole_count INTEGER NOT NULL,
  second_to_largest_ratio REAL NOT NULL,
  disposition TEXT NOT NULL,
  raw_track_id TEXT,
  final_track_id TEXT,
  scene_id INTEGER,
  removed_by_short_track INTEGER NOT NULL,
  exact_keyframe INTEGER NOT NULL,
  keyframe_component_count INTEGER,
  segment_component_count INTEGER,
  PRIMARY KEY(run_key, detection_id)
);
CREATE TABLE IF NOT EXISTS multi_component_runs(
  run_key TEXT NOT NULL,
  final_track_id TEXT NOT NULL,
  scene_id INTEGER NOT NULL,
  start_frame INTEGER NOT NULL,
  end_frame INTEGER NOT NULL,
  frame_count INTEGER NOT NULL,
  maximum_second_ratio REAL NOT NULL,
  mean_second_ratio REAL NOT NULL,
  maximum_component_count INTEGER NOT NULL,
  PRIMARY KEY(run_key, final_track_id, scene_id, start_frame)
);
CREATE INDEX IF NOT EXISTS idx_outcome_disposition
  ON detection_outcomes(run_key, disposition);
CREATE INDEX IF NOT EXISTS idx_outcome_track_frame
  ON detection_outcomes(run_key, final_track_id, frame);
"""


def _manifest_artifacts(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete"):
        raise ValueError(f"postprocess manifest is not complete: {path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"postprocess manifest has no artifacts: {path}")
    return {str(key): str(value) for key, value in artifacts.items()}


def _quote(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def analyze(
    *,
    topology: Path,
    run_key: str,
    manifest: Path,
    output: Path,
    score_min: float,
) -> dict[str, object]:
    artifacts = _manifest_artifacts(manifest)
    result = Path(artifacts["result_sqlite"])
    if not result.is_file():
        raise FileNotFoundError(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output)
    connection.executescript(SCHEMA)
    connection.execute("DELETE FROM detection_outcomes WHERE run_key=?", (run_key,))
    connection.execute("DELETE FROM multi_component_runs WHERE run_key=?", (run_key,))
    connection.execute(f"ATTACH DATABASE '{_quote(topology)}' AS topo")
    connection.execute(f"ATTACH DATABASE '{_quote(result)}' AS result")

    # Keyframe component counts are computed once.  A raw detection does not
    # necessarily become a selected keyframe, so frame+final track is the
    # authoritative relation for this audit.
    connection.execute("DROP TABLE IF EXISTS temp.key_counts")
    connection.execute(
        """
        CREATE TEMP TABLE key_counts AS
        SELECT s.track_id, k.frame, k.segment_id, COUNT(c.id) AS component_count
        FROM result.mask_track_segments s
        JOIN result.mask_keyframes k ON k.segment_id=s.id
        LEFT JOIN result.keyframe_components c ON c.keyframe_id=k.id
        GROUP BY s.track_id, k.frame, k.segment_id
        """
    )
    connection.execute(
        "CREATE INDEX temp.idx_key_counts ON key_counts(track_id, frame)"
    )
    connection.execute(
        """
        INSERT INTO detection_outcomes(
          run_key, detection_id, frame, score,
          foreground_component_count, hole_count, second_to_largest_ratio,
          disposition, raw_track_id, final_track_id, scene_id,
          removed_by_short_track, exact_keyframe, keyframe_component_count,
          segment_component_count
        )
        SELECT
          m.run_key, m.detection_id, m.frame, m.score,
          m.foreground_component_count, m.hole_count,
          m.second_to_largest_ratio,
          CASE
            WHEN a.source_detection_id IS NULL AND COALESCE(m.score, 0) < ?
              THEN 'score_filtered'
            WHEN a.source_detection_id IS NULL THEN 'nms_or_unassigned'
            WHEN a.removed_by_short_track=1 THEN 'short_track_removed'
            ELSE 'retained'
          END,
          a.raw_track_id, a.final_track_id, a.scene_id,
          COALESCE(a.removed_by_short_track, 0),
          CASE WHEN kc.segment_id IS NULL THEN 0 ELSE 1 END,
          kc.component_count,
          (
            SELECT MAX(s.component_count)
            FROM result.mask_track_segments s
            WHERE s.track_id=a.final_track_id
              AND m.frame BETWEEN s.start_frame AND s.end_frame
          )
        FROM topo.mask_topology m
        LEFT JOIN result.tracking_assignments a
          ON a.source_detection_id=m.detection_id
        LEFT JOIN key_counts kc
          ON kc.track_id=a.final_track_id AND kc.frame=m.frame
        WHERE m.run_key=?
        """,
        (float(score_min), run_key),
    )

    rows = connection.execute(
        """
        SELECT final_track_id, scene_id, frame, second_to_largest_ratio,
               foreground_component_count
        FROM detection_outcomes
        WHERE run_key=? AND disposition='retained'
          AND foreground_component_count>1 AND final_track_id IS NOT NULL
        ORDER BY final_track_id, scene_id, frame
        """,
        (run_key,),
    )
    current: list[tuple[int, float, int]] = []
    current_key: tuple[str, int] | None = None

    def flush() -> None:
        if not current or current_key is None:
            return
        ratios = [row[1] for row in current]
        connection.execute(
            """INSERT INTO multi_component_runs VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                run_key,
                current_key[0],
                current_key[1],
                current[0][0],
                current[-1][0],
                len(current),
                max(ratios),
                sum(ratios) / len(ratios),
                max(row[2] for row in current),
            ),
        )

    for track_id, scene_id, frame, ratio, component_count in rows:
        key = (str(track_id), int(scene_id))
        frame = int(frame)
        if current_key != key or (current and frame != current[-1][0] + 1):
            flush()
            current = []
            current_key = key
        current.append((frame, float(ratio), int(component_count)))
    flush()
    connection.commit()

    all_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM detection_outcomes WHERE run_key=?", (run_key,)
        ).fetchone()[0]
    )
    multi_count = int(
        connection.execute(
            """SELECT COUNT(*) FROM detection_outcomes
               WHERE run_key=? AND foreground_component_count>1""",
            (run_key,),
        ).fetchone()[0]
    )
    disposition = dict(
        connection.execute(
            """SELECT disposition, COUNT(*) FROM detection_outcomes
               WHERE run_key=? AND foreground_component_count>1
               GROUP BY disposition""",
            (run_key,),
        ).fetchall()
    )
    run_lengths = [
        int(row[0])
        for row in connection.execute(
            "SELECT frame_count FROM multi_component_runs WHERE run_key=?",
            (run_key,),
        )
    ]
    exact_keyframes = int(
        connection.execute(
            """SELECT COUNT(*) FROM detection_outcomes
               WHERE run_key=? AND foreground_component_count>1
                 AND disposition='retained' AND exact_keyframe=1""",
            (run_key,),
        ).fetchone()[0]
    )
    holes_as_multi_keyframes = int(
        connection.execute(
            """SELECT COUNT(*) FROM detection_outcomes
               WHERE run_key=? AND hole_count>0 AND exact_keyframe=1
                 AND COALESCE(keyframe_component_count,0)>1""",
            (run_key,),
        ).fetchone()[0]
    )
    total_retained = int(
        connection.execute(
            """SELECT COUNT(*) FROM detection_outcomes
               WHERE run_key=? AND disposition='retained'""",
            (run_key,),
        ).fetchone()[0]
    )
    total_retained_exact_keys = int(
        connection.execute(
            """SELECT COUNT(*) FROM detection_outcomes
               WHERE run_key=? AND disposition='retained' AND exact_keyframe=1""",
            (run_key,),
        ).fetchone()[0]
    )
    retained_multi = int(disposition.get("retained", 0))
    overall_key_rate = total_retained_exact_keys / total_retained if total_retained else 0.0
    multi_key_rate = exact_keyframes / retained_multi if retained_multi else 0.0
    varying_slot_tracks = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT s.track_id, MIN(kc.component_count) AS minimum_slots,
                     MAX(kc.component_count) AS maximum_slots
              FROM result.mask_track_segments s
              JOIN key_counts kc ON kc.segment_id=s.id
              GROUP BY s.track_id
              HAVING minimum_slots<>maximum_slots
            )
            """
        ).fetchone()[0]
    )
    multi_component_segments = int(
        connection.execute(
            """SELECT COUNT(*) FROM result.mask_track_segments
               WHERE component_count>1"""
        ).fetchone()[0]
    )
    ring_roles = dict(
        connection.execute(
            """SELECT ring_role,COUNT(*)
               FROM result.keyframe_polygon_rings GROUP BY ring_role"""
        ).fetchall()
    )
    topology_transitions = connection.execute(
        """
        WITH sequence AS (
          SELECT final_track_id, scene_id, frame,
                 foreground_component_count, hole_count,
                 LAG(frame) OVER(
                   PARTITION BY final_track_id,scene_id ORDER BY frame
                 ) AS previous_frame,
                 LAG(foreground_component_count) OVER(
                   PARTITION BY final_track_id,scene_id ORDER BY frame
                 ) AS previous_foreground_components,
                 LAG(hole_count) OVER(
                   PARTITION BY final_track_id,scene_id ORDER BY frame
                 ) AS previous_holes
          FROM detection_outcomes
          WHERE run_key=? AND disposition='retained'
        )
        SELECT
          SUM(previous_frame=frame-1 AND
              previous_foreground_components<>foreground_component_count),
          COUNT(DISTINCT CASE WHEN previous_frame=frame-1 AND
              previous_foreground_components<>foreground_component_count
              THEN final_track_id END),
          SUM(previous_frame=frame-1 AND previous_holes<>hole_count),
          COUNT(DISTINCT CASE WHEN previous_frame=frame-1 AND
              previous_holes<>hole_count THEN final_track_id END)
        FROM sequence
        """,
        (run_key,),
    ).fetchone()
    severity_outcomes: dict[str, dict[str, int]] = {}
    for severity, outcome, count, exact_count in connection.execute(
        """
        SELECT
          CASE
            WHEN second_to_largest_ratio < .001 THEN '<0.1%'
            WHEN second_to_largest_ratio < .01 THEN '0.1-1%'
            WHEN second_to_largest_ratio < .05 THEN '1-5%'
            WHEN second_to_largest_ratio < .20 THEN '5-20%'
            ELSE '>=20%'
          END AS severity,
          disposition,
          COUNT(*) AS count,
          SUM(exact_keyframe) AS exact_count
        FROM detection_outcomes
        WHERE run_key=? AND foreground_component_count>1
        GROUP BY severity, disposition
        """,
        (run_key,),
    ):
        severity_outcomes.setdefault(str(severity), {})[str(outcome)] = int(count)
        severity_outcomes[str(severity)][f"{outcome}_exact_keyframes"] = int(
            exact_count or 0
        )
    summary: dict[str, object] = {
        "run_key": run_key,
        "topology_detections": all_count,
        "multi_foreground_detections": multi_count,
        "multi_disposition": disposition,
        "retained_multi_exact_keyframes": exact_keyframes,
        "retained_observations": total_retained,
        "retained_observations_at_exact_keyframes": total_retained_exact_keys,
        "overall_exact_keyframe_rate": overall_key_rate,
        "retained_multi_exact_keyframe_rate": multi_key_rate,
        "multi_keyframe_enrichment": (
            multi_key_rate / overall_key_rate if overall_key_rate else 0.0
        ),
        "hole_detections_exported_as_multi_component_keyframes": holes_as_multi_keyframes,
        "final_multi_component_segments": multi_component_segments,
        "final_tracks_with_varying_keyframe_slot_count": varying_slot_tracks,
        "final_polygon_ring_roles": ring_roles,
        "consecutive_foreground_component_count_transitions": int(
            topology_transitions[0] or 0
        ),
        "tracks_with_foreground_component_count_transitions": int(
            topology_transitions[1] or 0
        ),
        "consecutive_hole_count_transitions": int(topology_transitions[2] or 0),
        "tracks_with_hole_count_transitions": int(topology_transitions[3] or 0),
        "multi_component_temporal_runs": len(run_lengths),
        "run_length_distribution": dict(Counter(run_lengths)),
        "maximum_run_length": max(run_lengths, default=0),
        "mean_run_length": (
            sum(run_lengths) / len(run_lengths) if run_lengths else 0.0
        ),
        "severity_outcomes": severity_outcomes,
        "result_sqlite": str(result.resolve()),
        "postprocess_manifest": str(manifest.resolve()),
    }
    connection.close()
    output.with_name(f"{output.stem}.{run_key}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--postprocess-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-min", type=float, default=0.3)
    args = parser.parse_args()
    summary = analyze(
        topology=args.topology,
        run_key=args.run_key,
        manifest=args.postprocess_manifest,
        output=args.output,
        score_min=args.score_min,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
