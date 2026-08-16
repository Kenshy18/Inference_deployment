#!/usr/bin/env python3
"""Compare the full HEYZO run with its independently decoded 30–45 min clip."""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=43_200)
    args = parser.parse_args()
    connection = sqlite3.connect(f"file:{args.topology.resolve()}?mode=ro", uri=True)

    def load(run_key: str, offset: int = 0) -> dict[int, list[tuple[float, int, int, float]]]:
        rows: dict[int, list[tuple[float, int, int, float]]] = collections.defaultdict(list)
        for _did, frame, score, components, holes, ratio in connection.execute(
            """SELECT detection_id,frame,score,foreground_component_count,
                      hole_count,second_to_largest_ratio
               FROM mask_topology WHERE run_key=? ORDER BY frame,detection_id""",
            (run_key,),
        ):
            rows[int(frame) - offset].append(
                (float(score), int(components), int(holes), float(ratio))
            )
        return rows

    full = load("v3lite__heyzo_3545_full", args.offset)
    clip = load("v3lite__heyzo_3545_30_45_duplicate")
    clip_frames = max(clip, default=-1) + 1
    frame_count_mismatches = 0
    row_metric_mismatches: list[dict[str, object]] = []
    compared = 0
    for frame in range(clip_frames):
        if len(full[frame]) != len(clip[frame]):
            frame_count_mismatches += 1
        for index, (left, right) in enumerate(zip(full[frame], clip[frame])):
            compared += 1
            if any(abs(a - b) > 1e-12 for a, b in zip(left, right)):
                row_metric_mismatches.append(
                    {"frame": frame, "detection_order": index, "full": left, "clip": right}
                )
    connection.close()
    payload = {
        "offset_frames": args.offset,
        "clip_frames": clip_frames,
        "detections_compared": compared,
        "frame_detection_count_mismatches": frame_count_mismatches,
        "row_metric_mismatch_count": len(row_metric_mismatches),
        "exact_row_rate": (compared - len(row_metric_mismatches)) / compared if compared else 1.0,
        "mismatches": row_metric_mismatches[:100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
