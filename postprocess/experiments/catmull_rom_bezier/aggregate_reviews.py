"""Aggregate multiple real-track polygon/curve review runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty aggregate")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(roots: list[Path]) -> dict[str, Any]:
    manifests = []
    groups: dict[tuple[int, int, str], dict[str, Any]] = {}
    detail_rows: list[dict[str, Any]] = []
    for root in roots:
        resolved = root.resolve()
        manifest_path = resolved / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append(
            {
                "root": str(resolved),
                "track_id": str(manifest["track_id"]),
                "label": str(manifest["label"]),
                "frames": int(manifest["frames"]),
                "frame_range": list(manifest["actual_frame_range"]),
                "wall_seconds": float(manifest["wall_seconds"]),
            }
        )
        for summary_path in sorted(resolved.glob("points_*/interval_*/summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics_rows = _read_csv(summary_path.parent / "frame_metrics.csv")
            points = int(summary["point_count"])
            interval = int(summary["target_interval"])
            for representation, prefix in (("polygon", "polygon"), ("curve", "curve")):
                key = (points, interval, representation)
                bucket = groups.setdefault(
                    key,
                    {
                        "ious": [],
                        "recalls": [],
                        "area_ratios": [],
                        "keys": 0,
                        "elapsed": 0.0,
                        "dp_seconds": 0.0,
                        "pair_vote_seconds": 0.0,
                        "topology_invalid": 0,
                        "pair_vote_gain": 0.0,
                        "temporal_control_residual_sum": 0.0,
                        "tracks": 0,
                        "recall_floor": float(manifest["recall_floor"]),
                    },
                )
                if (
                    abs(float(bucket["recall_floor"]) - float(manifest["recall_floor"]))
                    > 1e-12
                ):
                    raise ValueError("all review roots must use one Recall floor")
                info = summary[representation]
                ious = np.asarray(
                    [float(row[f"{prefix}_iou"]) for row in metrics_rows],
                    dtype=np.float64,
                )
                recalls = np.asarray(
                    [float(row[f"{prefix}_recall"]) for row in metrics_rows],
                    dtype=np.float64,
                )
                ratios = np.asarray(
                    [float(row[f"{prefix}_area_ratio"]) for row in metrics_rows],
                    dtype=np.float64,
                )
                bucket["ious"].append(ious)
                bucket["recalls"].append(recalls)
                bucket["area_ratios"].append(ratios)
                bucket["keys"] += int(info["chosen_keyframes"])
                bucket["elapsed"] += float(info["elapsed_seconds"])
                bucket["dp_seconds"] += float(info.get("dp_seconds", 0.0))
                bucket["pair_vote_seconds"] += float(info.get("pair_vote_seconds", 0.0))
                bucket["topology_invalid"] += int(info["topology_invalid_frames"])
                bucket["pair_vote_gain"] += float(info["pair_vote_iou_gain"])
                bucket["temporal_control_residual_sum"] += float(
                    info["temporal_control_residual"]
                ) * len(ious)
                bucket["tracks"] += 1
                detail_rows.append(
                    {
                        "track_id": str(manifest["track_id"]),
                        "label": str(manifest["label"]),
                        "point_count": points,
                        "target_interval": interval,
                        "representation": representation,
                        "frames": len(ious),
                        "keys": int(info["chosen_keyframes"]),
                        "effective_interval": float(info["effective_interval"]),
                        "mean_iou": float(np.mean(ious)),
                        "minimum_iou": float(np.min(ious)),
                        "minimum_recall": float(np.min(recalls)),
                        "area_ratio_q95": float(np.quantile(ratios, 0.95)),
                        "elapsed_seconds": float(info["elapsed_seconds"]),
                    }
                )

    rows: list[dict[str, Any]] = []
    for (points, interval, representation), bucket in sorted(groups.items()):
        ious = np.concatenate(bucket["ious"])
        recalls = np.concatenate(bucket["recalls"])
        ratios = np.concatenate(bucket["area_ratios"])
        elapsed = float(bucket["elapsed"])
        rows.append(
            {
                "point_count": points,
                "target_interval": interval,
                "representation": representation,
                "tracks": int(bucket["tracks"]),
                "frames": int(len(ious)),
                "chosen_keyframes": int(bucket["keys"]),
                "effective_interval": float(len(ious) / max(bucket["keys"], 1)),
                "mean_iou": float(np.mean(ious)),
                "minimum_iou": float(np.min(ious)),
                "q01_iou": float(np.quantile(ious, 0.01)),
                "q05_iou": float(np.quantile(ious, 0.05)),
                "minimum_recall": float(np.min(recalls)),
                "recall_floor": float(bucket["recall_floor"]),
                "recall_violations": int(
                    np.count_nonzero(recalls + 1e-12 < float(bucket["recall_floor"]))
                ),
                "area_ratio_mean": float(np.mean(ratios)),
                "area_ratio_q95": float(np.quantile(ratios, 0.95)),
                "area_ratio_maximum": float(np.max(ratios)),
                "topology_invalid_frames": int(bucket["topology_invalid"]),
                "temporal_control_residual": float(
                    bucket["temporal_control_residual_sum"] / max(len(ious), 1)
                ),
                "pair_vote_iou_gain_sum": float(bucket["pair_vote_gain"]),
                "dp_seconds": float(bucket["dp_seconds"]),
                "pair_vote_seconds": float(bucket["pair_vote_seconds"]),
                "temporal_seconds": elapsed,
                "temporal_fps": float(len(ious) / max(elapsed, 1e-12)),
            }
        )
    overall_rows: list[dict[str, Any]] = []
    overall_keys = sorted(
        {(interval, representation) for _points, interval, representation in groups}
    )
    for interval, representation in overall_keys:
        selected = [
            bucket
            for (
                points,
                candidate_interval,
                candidate_representation,
            ), bucket in groups.items()
            if candidate_interval == interval
            and candidate_representation == representation
        ]
        ious = np.concatenate(
            [values for bucket in selected for values in bucket["ious"]]
        )
        recalls = np.concatenate(
            [values for bucket in selected for values in bucket["recalls"]]
        )
        ratios = np.concatenate(
            [values for bucket in selected for values in bucket["area_ratios"]]
        )
        recall_floors = {float(bucket["recall_floor"]) for bucket in selected}
        if len(recall_floors) != 1:  # pragma: no cover - guarded above as well
            raise ValueError("all review roots must use one Recall floor")
        recall_floor = recall_floors.pop()
        keys = sum(int(bucket["keys"]) for bucket in selected)
        elapsed = sum(float(bucket["elapsed"]) for bucket in selected)
        overall_rows.append(
            {
                "target_interval": interval,
                "representation": representation,
                "point_counts": sorted(
                    {
                        points
                        for points, candidate_interval, candidate_representation in groups
                        if candidate_interval == interval
                        and candidate_representation == representation
                    }
                ),
                "tracks": sum(int(bucket["tracks"]) for bucket in selected),
                "frames": int(len(ious)),
                "chosen_keyframes": int(keys),
                "effective_interval": float(len(ious) / max(keys, 1)),
                "mean_iou": float(np.mean(ious)),
                "minimum_iou": float(np.min(ious)),
                "q01_iou": float(np.quantile(ious, 0.01)),
                "q05_iou": float(np.quantile(ious, 0.05)),
                "minimum_recall": float(np.min(recalls)),
                "recall_floor": float(recall_floor),
                "recall_violations": int(
                    np.count_nonzero(recalls + 1e-12 < recall_floor)
                ),
                "area_ratio_mean": float(np.mean(ratios)),
                "area_ratio_q95": float(np.quantile(ratios, 0.95)),
                "area_ratio_maximum": float(np.max(ratios)),
                "topology_invalid_frames": sum(
                    int(bucket["topology_invalid"]) for bucket in selected
                ),
                "pair_vote_iou_gain_sum": sum(
                    float(bucket["pair_vote_gain"]) for bucket in selected
                ),
                "dp_seconds": sum(float(bucket["dp_seconds"]) for bucket in selected),
                "pair_vote_seconds": sum(
                    float(bucket["pair_vote_seconds"]) for bucket in selected
                ),
                "temporal_seconds": float(elapsed),
                "temporal_fps": float(len(ious) / max(elapsed, 1e-12)),
            }
        )
    return {
        "schema_version": 1,
        "status": "experimental_not_production",
        "comparison_contract": (
            "same spatial point budget and hard minimum Recall; polygon uses the "
            "current persistent line-fit plus its single-state DP, while curve uses "
            "fixed-handle Catmull-Rom whole-boundary fitting, a bounded five-state "
            "curve DP and exact-gated low-tail rescue"
        ),
        "sources": manifests,
        "aggregate": rows,
        "overall": overall_rows,
        "per_track": detail_rows,
    }


def main() -> int:
    args = _parser().parse_args()
    result = aggregate([Path(root) for root in args.roots])
    output = Path(args.output_dir).resolve()
    generated = (
        output / "aggregate.json",
        output / "aggregate.csv",
        output / "overall.csv",
        output / "per_track.csv",
    )
    if output.exists() and any(path.exists() for path in generated) and not args.force:
        raise SystemExit(f"aggregate output exists: {output} (use --force)")
    output.mkdir(parents=True, exist_ok=True)
    (output / "aggregate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output / "aggregate.csv", result["aggregate"])
    _write_csv(output / "overall.csv", result["overall"])
    _write_csv(output / "per_track.csv", result["per_track"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("aggregate", "main")
