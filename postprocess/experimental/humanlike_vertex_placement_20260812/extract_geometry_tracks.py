#!/usr/bin/env python3
"""Extract raw tracked polygon geometry without opening the source video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-ids", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.input_sqlite) as source:
        cuts = [int(row[0]) for row in source.execute("SELECT frame FROM cuts")]
        for track_id in args.track_ids.split(","):
            rows = source.execute(
                """
                SELECT ta.frame, ta.final_label, sp.id
                FROM tracking_assignments AS ta
                JOIN segmentation_polygons AS sp
                  ON sp.detection_id = ta.source_detection_id
                WHERE ta.final_track_id = ?
                  AND ta.removed_by_short_track = 0
                ORDER BY ta.frame, sp.polygon_index
                """,
                (track_id,),
            ).fetchall()
            by_frame: dict[int, tuple[str, list[list[float]]]] = {}
            for frame, label, polygon_id in rows:
                points = [
                    [float(x), float(y)]
                    for _index, x, y in source.execute(
                        """
                        SELECT point_index, x, y
                        FROM segmentation_points
                        WHERE polygon_id = ?
                        ORDER BY point_index
                        """,
                        (polygon_id,),
                    )
                ]
                if int(frame) in by_frame:
                    # The current experiment deliberately benchmarks one closed
                    # component per track/frame so topology does not confound
                    # vertex-placement quality.
                    continue
                by_frame[int(frame)] = (str(label), points)
            destination_path = args.output_dir / f"input_track{track_id}.sqlite"
            with sqlite3.connect(destination_path) as destination:
                destination.executescript(
                    """
                    DROP TABLE IF EXISTS masks;
                    DROP TABLE IF EXISTS cuts;
                    CREATE TABLE masks (
                        frame INTEGER NOT NULL,
                        track_id TEXT NOT NULL,
                        polygons TEXT NOT NULL,
                        shape_type TEXT NOT NULL,
                        label TEXT NOT NULL
                    );
                    CREATE TABLE cuts (frame INTEGER NOT NULL);
                    """
                )
                destination.executemany(
                    "INSERT INTO cuts(frame) VALUES (?)",
                    [(value,) for value in cuts],
                )
                destination.executemany(
                    "INSERT INTO masks VALUES (?, ?, ?, 'polygon', ?)",
                    [
                        (frame, track_id, json.dumps([points]), label)
                        for frame, (label, points) in sorted(by_frame.items())
                    ],
                )
            print(track_id, len(by_frame), destination_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
