#!/usr/bin/env python3
"""Build a comprehensive, locally-rendered NMS/topology review pack.

The pack deliberately separates direct NMS/topology decisions from tracking,
polygon approximation, keyframe DP, and pair-vote.  No network API is used.
Frames are decoded with local OpenCV and are never opened by an AI viewer.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS = ROOT / "postprocess"
import sys

if str(POSTPROCESS) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS))

from contracts.detections import iter_detection_records  # noqa: E402
from nms.adaptive import DEFAULT_NMS  # noqa: E402
from nms.component_aware import DEFAULT_COMPONENT_AWARE_NMS  # noqa: E402
from experimental.nms_ablation_20260813.render_component_candidate_v2_review_gallery import (  # noqa: E402,E501
    CATEGORY_ORDER,
    _decision_frames,
    _detection_id,
    _open_ro,
    _render_one,
    _scan_candidates,
    _select_diverse,
    _write_csv,
)


DEFAULT_ABLATION = ROOT / "output/nms_component_candidate_v2_ablation_finalcode_20260813"
DEFAULT_THRESHOLDS = ROOT / "output/nms_component_candidate_v2_thresholds_20260813"
DEFAULT_TOPOLOGY = ROOT / "output/instance_mask_topology_20260806/topology.sqlite"
DEFAULT_OUTPUT = ROOT / "output/nms_topology_comprehensive_review_20260813"


REVIEW_CATEGORIES = (
    "01_holes_all",
    "02_tiny_islands_all",
    "03_redundant_islands_all",
    "04_candidate_nms_all_suppressions",
    "05_legacy_only_stratified",
    "06_candidate_kept_near_070",
    "07_cross_class_or_chain_stratified",
)


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["run_key"]), int(row["frame"])


def _unique_frames(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _mask_iou_bin(value: float) -> str:
    for lower, upper, label in (
        (0.00, 0.10, "00_10"),
        (0.10, 0.25, "10_25"),
        (0.25, 0.50, "25_50"),
        (0.50, 0.65, "50_65"),
        (0.65, 0.70 + 1e-12, "65_70"),
    ):
        if lower <= value < upper:
            return label
    return "other"


def _legacy_stratified(rows: list[dict[str, Any]], per_cell: int) -> list[dict[str, Any]]:
    """Sample every run x Mask-IoU stratum, then add rare cross-class cases."""
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    cross_class: list[dict[str, Any]] = []
    for row in rows:
        event = row.get("event") or {}
        value = float(event.get("mask_iou", 0.0))
        cells[(str(row["run_key"]), _mask_iou_bin(value))].append(row)
        if event.get("winner_class") != event.get("loser_class"):
            cross_class.append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(cells):
        ordered = sorted(
            cells[key],
            key=lambda row: (
                float((row.get("event") or {}).get("mask_iou", 0.0)),
                int(row["frame"]),
            ),
        )
        if len(ordered) <= per_cell:
            selected.extend(ordered)
            continue
        # Evenly cover each bucket, rather than selecting only its most extreme tail.
        indices = {
            round(index * (len(ordered) - 1) / (per_cell - 1))
            for index in range(per_cell)
        }
        selected.extend(ordered[index] for index in sorted(indices))
    selected.extend(
        sorted(
            cross_class,
            key=lambda row: (
                -float((row.get("event") or {}).get("mask_iou", 0.0)),
                str(row["run_key"]),
                int(row["frame"]),
            ),
        )[:40]
    )
    return _unique_frames(selected)


def _load_threshold_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if abs(float(raw["nms_threshold"]) - 0.70) > 1e-12:
                continue
            mask_iou = float(raw["mask_iou"])
            if not 0.68 <= mask_iou < 0.70:
                continue
            rows.append(
                {
                    "category": "06_candidate_kept_near_070",
                    "run_key": raw["run_key"],
                    "frame": int(raw["frame_index"]),
                    "reason": (
                        f"candidate keeps pair D{raw['first_id']}/D{raw['second_id']} "
                        f"just below threshold; Mask-IoU={mask_iou:.6f}"
                    ),
                    "input_count": 0,
                    "input_ids": [],
                    "legacy_ids": [],
                    "candidate_ids": [],
                    "legacy_only_suppressed_ids": [],
                    "candidate_only_suppressed_ids": [],
                    "both_suppressed_ids": [],
                    "event": {
                        "winner_id": raw["first_id"],
                        "loser_id": raw["second_id"],
                        "winner_class": raw["first_class"],
                        "loser_class": raw["second_class"],
                        "reason": "candidate_retains_below_0.70",
                        "mask_iou": mask_iou,
                    },
                    "component": None,
                    "event_count": 0,
                }
            )
    return _unique_frames(
        sorted(rows, key=lambda row: (str(row["run_key"]), int(row["frame"])))
    )


def _load_input_records(path: Path, wanted: set[int]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in iter_detection_records(path):
        frame = int(record["frame_index"])
        if frame in wanted:
            result[frame] = record
        if len(result) == len(wanted):
            break
    missing = sorted(wanted - set(result))
    if missing:
        raise RuntimeError(f"missing selected frames in {path}: {missing[:20]}")
    return result


def _arm_records(
    input_record: dict[str, Any], row: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frame = int(input_record["frame_index"])
    inputs = copy.deepcopy(list(input_record["detections"]))
    legacy_detections = DEFAULT_NMS.apply(copy.deepcopy(inputs))
    candidate_detections = DEFAULT_COMPONENT_AWARE_NMS.apply(copy.deepcopy(inputs))
    legacy_record = {**input_record, "detections": legacy_detections}
    candidate_record = {**input_record, "detections": candidate_detections}

    def ids(detections: list[dict[str, Any]]) -> list[int | str]:
        return [
            _detection_id(detection, frame, index)
            for index, detection in enumerate(detections)
        ]

    input_ids = ids(inputs)
    legacy_ids = ids(legacy_detections)
    candidate_ids = ids(candidate_detections)
    row.update(
        {
            "input_count": len(input_ids),
            "input_ids": input_ids,
            "legacy_ids": legacy_ids,
            "candidate_ids": candidate_ids,
            "legacy_only_suppressed_ids": sorted(
                set(candidate_ids) - set(legacy_ids), key=str
            ),
            "candidate_only_suppressed_ids": sorted(
                set(legacy_ids) - set(candidate_ids), key=str
            ),
            "both_suppressed_ids": sorted(
                set(input_ids) - set(legacy_ids) - set(candidate_ids), key=str
            ),
        }
    )
    return input_record, legacy_record, candidate_record


def _write_all_decisions(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "source_category",
        "run_key",
        "frame",
        "reason",
        "input_count",
        "input_ids",
        "legacy_ids",
        "candidate_ids",
        "legacy_only_suppressed_ids",
        "candidate_only_suppressed_ids",
        "both_suppressed_ids",
        "event",
        "component",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["source_category"] = encoded.pop("category", "")
            for key in (
                "input_ids",
                "legacy_ids",
                "candidate_ids",
                "legacy_only_suppressed_ids",
                "candidate_only_suppressed_ids",
                "both_suppressed_ids",
                "event",
                "component",
            ):
                encoded[key] = json.dumps(
                    encoded.get(key), ensure_ascii=False, separators=(",", ":")
                )
            writer.writerow(encoded)


def _verify_jpegs(root: Path, expected: int) -> dict[str, Any]:
    paths = sorted(root.rglob("*.jpg"))
    failures: list[str] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or min(image.shape[:2]) <= 0:
            failures.append(str(path))
    return {
        "expected_unique_images": expected,
        "decoded_unique_images": len(paths),
        "decode_failures": failures,
        "passed": len(paths) == expected and not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--threshold-root", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--topology-sqlite", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--legacy-per-run-iou-bin", type=int, default=4)
    parser.add_argument("--chain-samples", type=int, default=80)
    args = parser.parse_args()

    ablation = args.ablation_root.expanduser().resolve()
    thresholds = args.threshold_root.expanduser().resolve()
    topology_path = args.topology_sqlite.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    topology = _open_ro(topology_path)
    metadata = {
        str(row["run_key"]): {
            "video": Path(str(row["input_video"])),
            "video_slug": str(row["video_slug"]),
        }
        for row in topology.execute(
            "SELECT run_key,input_video,video_slug FROM audit_runs WHERE model_key='v3'"
        )
    }
    topology.close()

    all_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_sources: dict[str, Path] = {}
    run_decisions: dict[str, dict[int, dict[str, Any]]] = {}
    for run_dir in sorted((ablation / "runs").glob("v3__*")):
        if run_dir.name not in metadata:
            continue
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        scored = Path(str(summary["input"]["jsonl"])).resolve()
        decisions, components = _decision_frames(run_dir)
        for category, rows in _scan_candidates(
            ablation, run_dir.name, scored, decisions, components
        ).items():
            all_candidates[category].extend(rows)
        run_sources[run_dir.name] = scored
        run_decisions[run_dir.name] = decisions

    candidate_suppression = copy.deepcopy(_unique_frames(
        sorted(
            all_candidates["02_both_suppress_high_iou"]
            + all_candidates["03_candidate_only_suppressed"],
            key=lambda row: (str(row["run_key"]), int(row["frame"])),
        )
    ))
    for row in candidate_suppression:
        row["category"] = "04_candidate_nms_all_suppressions"

    review: dict[str, list[dict[str, Any]]] = {
        "01_holes_all": copy.deepcopy(
            _unique_frames(all_candidates["04_hole_fill"])
        ),
        "02_tiny_islands_all": copy.deepcopy(
            _unique_frames(all_candidates["05_tiny_island_at_most_1pct"])
        ),
        "03_redundant_islands_all": copy.deepcopy(
            _unique_frames(all_candidates["06_redundant_island_80_50"])
        ),
        "04_candidate_nms_all_suppressions": candidate_suppression,
        "05_legacy_only_stratified": copy.deepcopy(
            _legacy_stratified(
                all_candidates["01_legacy_only_low_mask_iou"],
                int(args.legacy_per_run_iou_bin),
            )
        ),
        "06_candidate_kept_near_070": _load_threshold_rows(
            thresholds / "residual_pair_events.csv"
        ),
        "07_cross_class_or_chain_stratified": copy.deepcopy(
            _select_diverse(
                all_candidates["07_cross_class_or_three_detection_chain"],
                "07_cross_class_or_three_detection_chain",
                int(args.chain_samples),
            )
        ),
    }
    for category, rows in review.items():
        for row in rows:
            row["category"] = category

    wanted_by_run: dict[str, set[int]] = defaultdict(set)
    for rows in review.values():
        for row in rows:
            wanted_by_run[str(row["run_key"])].add(int(row["frame"]))
    records = {
        run: _load_input_records(run_sources[run], wanted)
        for run, wanted in wanted_by_run.items()
    }

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    (staging / "images").mkdir(parents=True)
    for category in REVIEW_CATEGORIES:
        (staging / category).mkdir(parents=True)

    rows_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for rows in review.values():
        for row in rows:
            rows_by_frame[_row_key(row)].append(row)

    rendered_by_frame: dict[tuple[str, int], dict[str, Any]] = {}
    image_manifest: list[dict[str, Any]] = []
    try:
        for index, ((run_key, frame), rows) in enumerate(sorted(rows_by_frame.items()), 1):
            primary = rows[0]
            input_record = records[run_key][frame]
            arms = _arm_records(input_record, primary)
            name = f"{metadata[run_key]['video_slug']}_f{frame}.jpg"
            master = staging / "images" / name
            rendered = _render_one(
                primary,
                arms,
                metadata[run_key]["video"],
                master,
                int(args.panel_width),
            )
            rendered_by_frame[(run_key, frame)] = rendered
            image_manifest.append(rendered)
            for row in rows:
                # Synchronize exact IDs produced by the final policies.
                for key in (
                    "input_count",
                    "input_ids",
                    "legacy_ids",
                    "candidate_ids",
                    "legacy_only_suppressed_ids",
                    "candidate_only_suppressed_ids",
                    "both_suppressed_ids",
                ):
                    row[key] = primary[key]
                link = staging / str(row["category"]) / name
                if not link.exists():
                    os.link(master, link)
                row["image"] = str(output / str(row["category"]) / name)
                row["topology_changed_ids"] = rendered["topology_changed_ids"]
                row["sample_index"] = index

        all_rows = [row for category in REVIEW_CATEGORIES for row in review[category]]
        _write_csv(staging / "review_manifest.csv", all_rows)
        (staging / "review_manifest.json").write_text(
            json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_decisions = [row for category in CATEGORY_ORDER for row in all_candidates[category]]
        _write_all_decisions(staging / "all_decisions.csv", all_decisions)

        aggregate = json.loads((ablation / "summary.json").read_text(encoding="utf-8"))
        arm_csv = list(csv.DictReader((ablation / "aggregate_arm_summary.csv").open()))
        arm = {row["arm"]: row for row in arm_csv}
        counts = {
            "frames": 477691,
            "input_detections": 431815,
            "legacy_suppressed": int(arm["legacy"]["suppressed_detections"]),
            "candidate_suppressed": int(
                arm["component_candidate_v2"]["suppressed_detections"]
            ),
            "candidate_retains_vs_legacy": (
                int(arm["component_candidate_v2"]["retained_detections"])
                - int(arm["legacy"]["retained_detections"])
            ),
            "holes_filled": int(arm["component_candidate_v2"]["holes_filled"]),
            "tiny_islands_removed": int(
                arm["component_candidate_v2"]["tiny_islands_removed"]
            ),
            "redundant_islands_removed": int(
                arm["component_candidate_v2"]["redundant_islands_removed"]
            ),
        }
        summary = {
            "schema_version": 1,
            "scope": "direct NMS/topology only; tracking/DP/pair-vote excluded",
            "privacy": "All video decoding was local OpenCV; no image was uploaded or opened by an AI viewer.",
            "source_ablation": str(ablation),
            "source_threshold_analysis": str(thresholds),
            "counts": counts,
            "available_direct_events": {
                category: len(all_candidates[category]) for category in CATEGORY_ORDER
            },
            "review_rows": {category: len(review[category]) for category in REVIEW_CATEGORIES},
            "unique_rendered_frames": len(rendered_by_frame),
            "source_config": aggregate.get("config", {}),
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        readme = f"""# NMS・穴・島の包括監査オーバーレイ

この成果物は、追跡・ポリゴン近似・キーフレームDP・pair-voteを混ぜず、NMSと穴・島処理そのものだけを比較します。

## 見方

- 左: 旧Production NMS
- 右: 新component-aware candidate v2
- 同一検出IDは左右で同じ色
- 実線＋半透明塗り: その方式が保持したマスク
- 破線: その方式が削除した入力マスク、または穴・島処理前の輪郭

## 全体数

- V3 9 run: {counts['frames']:,} frames / {counts['input_detections']:,} detections
- 穴埋め: {counts['holes_filled']} components（全該当frameを画像化）
- 1%以下の島削除: {counts['tiny_islands_removed']} components（全該当frameを画像化）
- 80%被覆・50%面積比の冗長島削除: {counts['redundant_islands_removed']} components（全件画像化）
- 旧NMS削除: {counts['legacy_suppressed']:,} detections
- 新NMS削除: {counts['candidate_suppressed']:,} detections（全該当frameを画像化）
- 新方式が旧方式より追加保持: {counts['candidate_retains_vs_legacy']:,} detections

## フォルダ

- `01_holes_all`: 穴埋めの全該当frame
- `02_tiny_islands_all`: 1%以下の島削除の全該当frame
- `03_redundant_islands_all`: 80/50規則による島削除の全件
- `04_candidate_nms_all_suppressions`: 新NMSが削除した全該当frame
- `05_legacy_only_stratified`: 旧だけが削除した4,561件をrun×Mask-IoU帯×クラス跨ぎで層化抽出
- `06_candidate_kept_near_070`: 新NMSが0.70直下で保持した全pair（0.68以上0.70未満）
- `07_cross_class_or_chain_stratified`: クラス跨ぎ・3検出以上の連鎖を層化抽出

`review_manifest.csv` は画像化対象、`all_decisions.csv` は直接判断の全件台帳です。

注意: 目視GTがないため、このオーバーレイで判定できるのは「規則が意図どおり適用されたか」「明らかな削除しすぎ・残しすぎがあるか」です。意味的正解率を保証するものではありません。
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")

        # Validate only unique physical JPEGs; category entries are hard links.
        qa = _verify_jpegs(staging / "images", len(rendered_by_frame))
        (staging / "qa.json").write_text(
            json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if not qa["passed"]:
            raise RuntimeError(f"JPEG QA failed: {qa}")
        os.replace(staging, output)
    except BaseException:
        raise

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
