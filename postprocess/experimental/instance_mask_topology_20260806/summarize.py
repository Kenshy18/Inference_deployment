#!/usr/bin/env python3
"""Write aggregate JSON/CSV/Markdown for the topology inference matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _inference_elapsed(output_root: Path) -> float | None:
    manifest = output_root / "logs" / "run_manifest.json"
    if not manifest.is_file():
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for stage in payload.get("stages", []):
        if stage.get("name") == "inference" and stage.get("status") == "complete":
            return float(stage["elapsed_seconds"])
    return None


def _totals(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    totals: dict[str, dict[str, object]] = {}
    for model in sorted({str(row["model"]) for row in rows}):
        group = [row for row in rows if row["model"] == model]
        detections = sum(int(row["detections"]) for row in group)
        multi = sum(int(row["multi_foreground"]) for row in group)
        holes = sum(int(row["with_holes"]) for row in group)
        low, high = _wilson(multi, detections)
        totals[model] = {
            "runs": len(group),
            "frames": sum(int(row["frames"]) for row in group),
            "detections": detections,
            "multi_foreground": multi,
            "multi_rate": multi / detections if detections else 0.0,
            "multi_rate_ci95": [low, high],
            "with_holes": holes,
            "hole_rate": holes / detections if detections else 0.0,
        }
    return totals


def build(topology: Path, matrix_path: Path) -> dict[str, object]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    by_key = {str(row["run_key"]): row for row in matrix}
    connection = sqlite3.connect(f"file:{topology.resolve()}?mode=ro", uri=True)
    rows: list[dict[str, object]] = []
    for audit in connection.execute(
        "SELECT run_key,model_key,video_slug,input_video,frame_count,detection_count "
        "FROM audit_runs ORDER BY model_key,video_slug"
    ):
        run_key, model, slug, video, frames, detections = audit
        aggregate = connection.execute(
            """
            SELECT
              SUM(foreground_component_count>1), SUM(hole_count>0),
              MAX(foreground_component_count), MAX(hole_count),
              MAX(second_to_largest_ratio),
              SUM(foreground_component_count>1 AND second_to_largest_ratio<.001),
              SUM(second_to_largest_ratio>=.001 AND second_to_largest_ratio<.01),
              SUM(second_to_largest_ratio>=.01 AND second_to_largest_ratio<.05),
              SUM(second_to_largest_ratio>=.05 AND second_to_largest_ratio<.20),
              SUM(second_to_largest_ratio>=.20)
            FROM mask_topology WHERE run_key=?
            """,
            (run_key,),
        ).fetchone()
        multi = int(aggregate[0] or 0)
        holes = int(aggregate[1] or 0)
        low, high = _wilson(multi, int(detections))
        item = by_key[str(run_key)]
        elapsed = _inference_elapsed(Path(str(item["output_root"])))
        rows.append(
            {
                "run_key": run_key,
                "model": model,
                "video_slug": slug,
                "input_video": video,
                "max_frames_requested": item.get("max_frames"),
                "frames": int(frames),
                "detections": int(detections),
                "multi_foreground": multi,
                "multi_rate": multi / int(detections) if detections else 0.0,
                "multi_rate_ci95_low": low,
                "multi_rate_ci95_high": high,
                "with_holes": holes,
                "hole_rate": holes / int(detections) if detections else 0.0,
                "max_foreground_components": int(aggregate[2] or 0),
                "max_holes": int(aggregate[3] or 0),
                "max_second_to_largest_ratio": float(aggregate[4] or 0.0),
                "second_ratio_lt_0_001": int(aggregate[5] or 0),
                "second_ratio_0_001_to_0_01": int(aggregate[6] or 0),
                "second_ratio_0_01_to_0_05": int(aggregate[7] or 0),
                "second_ratio_0_05_to_0_20": int(aggregate[8] or 0),
                "second_ratio_ge_0_20": int(aggregate[9] or 0),
                "inference_elapsed_seconds": elapsed,
                "inference_fps": (int(frames) / elapsed if elapsed else None),
            }
        )
    totals = _totals(rows)
    # The 30–45 minute HEYZO clip is byte-for-byte video coverage already
    # contained in the full HEYZO run.  Keep it as a repeatability run but do
    # not double-weight it in the unique-coverage estimate.
    unique_rows = [
        row for row in rows
        if str(row["video_slug"]) != "heyzo_3545_30_45_duplicate"
    ]
    unique_totals = _totals(unique_rows)
    paired: list[dict[str, object]] = []
    by_slug: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_slug.setdefault(str(row["video_slug"]), {})[str(row["model"])] = row
    for slug, models in sorted(by_slug.items()):
        if "v3" not in models or "v3lite" not in models:
            continue
        v3, lite = models["v3"], models["v3lite"]
        comparable_frames = min(int(v3["frames"]), int(lite["frames"]))
        # Calculate both rates on the same leading-frame window.  Reusing the
        # full V3-lite rate when V3 is sampled would bias the model contrast.
        pair_counts: dict[str, tuple[int, int]] = {}
        for model_name, row in (("v3", v3), ("v3lite", lite)):
            detections, multi = connection.execute(
                """SELECT COUNT(*),SUM(foreground_component_count>1)
                   FROM mask_topology
                   WHERE run_key=? AND frame<?""",
                (row["run_key"], comparable_frames),
            ).fetchone()
            pair_counts[model_name] = (int(detections), int(multi or 0))
        v3_rate = pair_counts["v3"][1] / pair_counts["v3"][0] if pair_counts["v3"][0] else 0.0
        lite_rate = pair_counts["v3lite"][1] / pair_counts["v3lite"][0] if pair_counts["v3lite"][0] else 0.0
        paired.append(
            {
                "video_slug": slug,
                "comparable_frames": comparable_frames,
                "v3_detections": pair_counts["v3"][0],
                "v3_multi": pair_counts["v3"][1],
                "v3_multi_rate": v3_rate,
                "v3lite_detections": pair_counts["v3lite"][0],
                "v3lite_multi": pair_counts["v3lite"][1],
                "v3lite_multi_rate": lite_rate,
                "rate_difference_v3lite_minus_v3": lite_rate - v3_rate,
                "coverage_note": (
                    "equal" if int(v3["frames"]) == int(lite["frames"])
                    else "V3 is a leading-frame sample"
                ),
            }
        )
    score_specs = (
        ("0.3–0.5", 0.3, 0.5),
        ("0.5–0.7", 0.5, 0.7),
        ("0.7–0.9", 0.7, 0.9),
        (">=0.9", 0.9, None),
    )
    score_accumulator = {
        (model, label): [0, 0]
        for model in ("v3", "v3lite")
        for label, _lower, _upper in score_specs
    }
    # As with the paired rate chart, both models must see the same frame
    # windows.  V3-lite full-video tails are therefore excluded here whenever
    # the matching V3 run is deliberately sampled.
    for pair in paired:
        slug = str(pair["video_slug"])
        comparable_frames = int(pair["comparable_frames"])
        for model in ("v3", "v3lite"):
            run_key = str(by_slug[slug][model]["run_key"])
            for label, lower, upper in score_specs:
                where_upper = "" if upper is None else "AND score<?"
                parameters: list[object] = [run_key, comparable_frames, lower]
                if upper is not None:
                    parameters.append(upper)
                detections, multi = connection.execute(
                    f"""
                    SELECT COUNT(*),SUM(foreground_component_count>1)
                    FROM mask_topology
                    WHERE run_key=? AND frame<? AND score>=? {where_upper}
                    """,
                    parameters,
                ).fetchone()
                score_accumulator[(model, label)][0] += int(detections)
                score_accumulator[(model, label)][1] += int(multi or 0)
    score_bins: list[dict[str, object]] = []
    for model in ("v3", "v3lite"):
        for label, _lower, _upper in score_specs:
            detections, multi = score_accumulator[(model, label)]
            score_bins.append(
                {
                    "model": model,
                    "score_bin": label,
                    "detections": detections,
                    "multi_foreground": multi,
                    "multi_rate": multi / detections if detections else 0.0,
                }
            )
    topology_profiles: list[dict[str, object]] = []
    for model in ("v3", "v3lite"):
        for is_multi in (0, 1):
            count, average_score, average_bbox_area = connection.execute(
                """
                SELECT COUNT(*),AVG(m.score),
                       AVG((m.bbox_x2-m.bbox_x1)*(m.bbox_y2-m.bbox_y1))
                FROM mask_topology m
                JOIN audit_runs a USING(run_key)
                WHERE a.model_key=?
                  AND a.video_slug<>'heyzo_3545_30_45_duplicate'
                  AND (m.foreground_component_count>1)=?
                """,
                (model, is_multi),
            ).fetchone()
            topology_profiles.append(
                {
                    "model": model,
                    "topology": "multi" if is_multi else "single",
                    "detections": int(count),
                    "average_score": float(average_score or 0.0),
                    "average_bbox_area": float(average_bbox_area or 0.0),
                }
            )
    connection.close()
    return {
        "runs": rows,
        "model_totals_including_repeatability_run": totals,
        "model_totals_unique_video_coverage": unique_totals,
        "paired": paired,
        "score_bins_unique_video_coverage": score_bins,
        "topology_profiles_unique_video_coverage": topology_profiles,
    }


def write_report(payload: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = payload["runs"]
    assert isinstance(rows, list)
    if rows:
        with (output_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Instance-mask topology audit",
        "",
        "A foreground component is an even-depth contour (an actual disconnected island).",
        "An odd-depth contour is a hole and is counted separately.",
        "",
        "## Model totals",
        "",
        "| model | runs | frames | detections | multi foreground | rate (95% CI) | holes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    totals = payload["model_totals_unique_video_coverage"]
    assert isinstance(totals, dict)
    for model, data in totals.items():
        lo, hi = data["multi_rate_ci95"]
        lines.append(
            f"| {model} | {data['runs']} | {data['frames']:,} | {data['detections']:,} | "
            f"{data['multi_foreground']:,} | {100*data['multi_rate']:.3f}% "
            f"({100*lo:.3f}–{100*hi:.3f}%) | {data['with_holes']:,} |"
        )
    lines.extend(
        [
            "",
            "## Per run",
            "",
            "| model | video | frames | detections | multi | rate | max components | >=5% secondary | inference FPS |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        fps = row["inference_fps"]
        fps_text = f"{fps:.2f}" if fps is not None else "n/a"
        lines.append(
            f"| {row['model']} | {row['video_slug']} | {row['frames']:,} | "
            f"{row['detections']:,} | {row['multi_foreground']:,} | "
            f"{100*row['multi_rate']:.3f}% | {row['max_foreground_components']} | "
            f"{row['second_ratio_0_05_to_0_20'] + row['second_ratio_ge_0_20']:,} | "
            f"{fps_text} |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.topology, args.matrix)
    write_report(payload, args.output_dir)
    print(json.dumps(payload["model_totals_unique_video_coverage"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
