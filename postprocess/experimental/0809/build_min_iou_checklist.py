#!/usr/bin/env python3
"""Build an editor-facing minimum-IoU checklist from geometry-only diagnostics."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _ndf(frame: int) -> str:
    frame %= 30 * 60 * 60 * 24
    hours, frame = divmod(frame, 30 * 60 * 60)
    minutes, frame = divmod(frame, 30 * 60)
    seconds, frames = divmod(frame, 30)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def _df(frame: int) -> str:
    frame %= 2_589_408
    ten_minutes, remainder = divmod(frame, 17_982)
    frame += 18 * ten_minutes
    if remainder >= 2:
        frame += 2 * ((remainder - 2) // 1_798)
    hours, frame = divmod(frame, 30 * 60 * 60)
    minutes, frame = divmod(frame, 30 * 60)
    seconds, frames = divmod(frame, 30)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d};{frames:02d}"


def _as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else default


def _as_bool(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() in {"1", "true", "yes"}


def _priority(iou: float) -> str:
    if iou < 0.30:
        return "P0"
    if iou < 0.40:
        return "P1"
    return "P2"


def _interpretation(row: dict[str, str]) -> tuple[str, str]:
    recall = _as_float(row, "recall")
    area_ratio = _as_float(row, "area_ratio")
    local_drop = _as_float(row, "local_iou_drop")
    if recall >= 0.97 and area_ratio >= 1.5 and not _as_bool(row, "has_keyframe"):
        return (
            "欠損補完・置きっぱなし候補",
            "rawが縮小・欠落していないか、最終マスクが対象を正しく覆い続けているか、過剰に残っていないか確認",
        )
    if local_drop >= 0.10:
        return (
            "局所IoU急落",
            "中心の前後3フレームをコマ送りし、一瞬の膨張・縮小・位置飛び・回転がないか確認",
        )
    if _as_bool(row, "has_keyframe"):
        return (
            "キーフレーム形状要確認",
            "キー位置だけ輪郭や頂点対応が跳ねず、前後補間が連続しているか確認",
        )
    return (
        "低IoU要確認",
        "rawと最終マスクを比較し、漏れ防止の有益な補完か、過剰マスク・位置ずれか判定",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--context", type=int, default=15)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["frame"] = str(int(row["frame"]))

    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if _as_float(row, "iou") < args.threshold:
            grouped[(row["label"], row["track_id"], int(row["run_id"]))].append(row)

    sections: list[list[dict[str, str]]] = []
    for values in grouped.values():
        values.sort(key=lambda row: int(row["frame"]))
        current: list[dict[str, str]] = []
        for row in values:
            if current and int(row["frame"]) - int(current[-1]["frame"]) > args.max_gap:
                sections.append(current)
                current = []
            current.append(row)
        if current:
            sections.append(current)

    checklist: list[dict[str, object]] = []
    for section in sections:
        minimum = min(section, key=lambda row: _as_float(row, "iou"))
        section_start = min(int(row["frame"]) for row in section)
        section_end = max(int(row["frame"]) for row in section)
        center = int(minimum["frame"])
        category, check = _interpretation(minimum)
        checklist.append(
            {
                "priority": _priority(_as_float(minimum, "iou")),
                "class": minimum["label"],
                "track_id": minimum["track_id"],
                "run_id": int(minimum["run_id"]),
                "context_tc_ndf": f"{_ndf(max(0, section_start - args.context))}–{_ndf(section_end + args.context)}",
                "minimum_tc_ndf": _ndf(center),
                "minimum_tc_df": _df(center),
                "minimum_frame": center,
                "minimum_iou": _as_float(minimum, "iou"),
                "recall_at_minimum": _as_float(minimum, "recall"),
                "area_ratio_at_minimum": _as_float(minimum, "area_ratio"),
                "local_iou_drop": _as_float(minimum, "local_iou_drop"),
                "pair_vote_iou_delta": _as_float(minimum, "iou_delta_vs_best_v4"),
                "minimum_is_keyframe": _as_bool(minimum, "has_keyframe"),
                "low_iou_rows": len(section),
                "low_iou_frame_span": f"{section_start}–{section_end}",
                "interpretation": category,
                "what_to_check": check,
                "normal_playback": "[ ]",
                "frame_step": "[ ]",
                "verdict": "",
                "comment": "",
            }
        )
    checklist.sort(key=lambda row: (float(row["minimum_iou"]), int(row["minimum_frame"])))
    for index, row in enumerate(checklist, 1):
        row["check_id"] = f"LIOU-{index:03d}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "minimum_iou_checklist.csv"
    fields = [
        "check_id", "priority", "class", "track_id", "run_id",
        "context_tc_ndf", "minimum_tc_ndf", "minimum_tc_df", "minimum_frame",
        "minimum_iou", "recall_at_minimum", "area_ratio_at_minimum",
        "local_iou_drop", "pair_vote_iou_delta", "minimum_is_keyframe",
        "low_iou_rows", "low_iou_frame_span", "interpretation", "what_to_check",
        "normal_playback", "frame_step", "verdict", "comment",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(checklist)

    counts = {priority: sum(row["priority"] == priority for row in checklist) for priority in ("P0", "P1", "P2")}
    md = [
        "# 最低IoUチェックリスト（時系列安定化前ベースライン）",
        "",
        f"対象は `IoU < {args.threshold:.2f}` の連続区間です。同一track/runで低IoUフレーム間の間隔が{args.max_gap}以下なら1区間へ統合し、各区間の最低IoUフレームを代表点にしています。",
        "",
        f"- 合計: **{len(checklist)}区間**",
        f"- P0（IoU < 0.30）: **{counts['P0']}区間**",
        f"- P1（0.30 ≤ IoU < 0.40）: **{counts['P1']}区間**",
        f"- P2（0.40 ≤ IoU < 0.50）: **{counts['P2']}区間**",
        "- TC: 29.97 NDFを主表記。DF中心TCもCSVに保存。",
        "- 判定候補: `問題なし（有益な補完）` / `過剰マスク` / `漏れ` / `位置ずれ` / `時間的不連続` / `判断保留`",
        "",
        "## 確認手順",
        "",
        "1. `確認範囲`を通常再生し、点滅・遅れ・置きっぱなし・過剰膨張を確認する。",
        "2. `最低TC`の前後3フレームをコマ送りし、rawと最終マスクを比較する。",
        "3. Recallが高く面積比が大きい場合は、生マスクの一時縮小を救った有益な補完かを先に判断する。",
        "4. 結果をチェック欄と判定欄へ記録する。",
        "",
        "## 全区間",
        "",
        "| 完了 | ID | 優先 | 確認範囲 NDF | 最低TC | class / track | IoU | Recall | 面積比 | 解釈 | 判定 |",
        "|---|---|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in checklist:
        md.append(
            "| [ ] | {check_id} | {priority} | {context_tc_ndf} | {minimum_tc_ndf} | "
            "{class} / {track_id} | {minimum_iou:.4f} | {recall_at_minimum:.4f} | "
            "{area_ratio_at_minimum:.2f} | {interpretation} |  |".format(**row)
        )
    md.extend(
        [
            "",
            "## 判定上の注意",
            "",
            "最低IoUはAI生マスクとの一致度であり、人手GTとの品質ではありません。Recallが高く、最終マスクが時間的に滑らかで対象を正しく覆う場合、低IoUは生マスクの一時的な縮小を救った結果である可能性があります。",
        ]
    )
    (args.output_dir / "minimum_iou_checklist.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"saved {csv_path}")
    print(f"saved {args.output_dir / 'minimum_iou_checklist.md'}")
    print(f"sections={len(checklist)} P0={counts['P0']} P1={counts['P1']} P2={counts['P2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
