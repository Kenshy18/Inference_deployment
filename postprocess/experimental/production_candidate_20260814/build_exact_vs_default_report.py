#!/usr/bin/env python3
"""Build the portable technical-report artifact for the V3 evaluator study."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_ID = "v3-cpu-exact-vs-default-cuda"


def _mode_label(mode: str) -> str:
    return "CUDA既定" if mode == "default_cuda" else "CPU厳密"


def _get(rows: list[dict[str, Any]], interval: int, mode: str) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if int(row["target_interval"]) == interval and row["mode"] == mode
    )


def build_artifact(analysis: dict[str, Any], source_path: str) -> dict[str, Any]:
    runtime = analysis["aggregate_runtime_quality"]
    errors = analysis["aggregate_paired_errors"]
    validation = analysis["validation"]
    recall_summary = [
        {
            "動画": row["run_id"],
            "目標間隔": int(row["target_interval"]),
            "方式": _mode_label(str(row["mode"])),
            "クラス": row["label"],
            "違反": int(row["violations"]),
            "両方共通": int(row["other_mode_shared"]),
            "この方式のみ": int(row["mode_only"]),
            "トラック数": int(row["tracks"]),
            "最初frame": int(row["first_frame"]),
            "最後frame": int(row["last_frame"]),
            "最小Recall": float(row["recall_min"]),
        }
        for row in analysis.get("recall_violation_summary", [])
    ]
    generated_at = datetime.now(timezone.utc).isoformat()
    intervals = sorted(int(row["target_interval"]) for row in errors)

    speed_rows = []
    quality_rows = []
    aggregate_table = []
    for row in runtime:
        label = _mode_label(str(row["mode"]))
        speed_rows.append(
            {
                "目標間隔": int(row["target_interval"]),
                "方式": label,
                "FPS": float(row["optimizer_fps"]),
            }
        )
        quality_rows.append(
            {
                "目標間隔": int(row["target_interval"]),
                "方式": label,
                "平均IoU": float(row["weighted_iou_mean"]),
            }
        )
        aggregate_table.append(
            {
                "目標間隔": int(row["target_interval"]),
                "方式": label,
                "動画数": int(row["videos"]),
                "フレーム数": int(row["frames"]),
                "optimizer秒": float(row["optimizer_seconds"]),
                "optimizer FPS": float(row["optimizer_fps"]),
                "動画別FPS最小": float(row["optimizer_video_fps_min"]),
                "動画別FPS中央値": float(row["optimizer_video_fps_median"]),
                "動画別FPS最大": float(row["optimizer_video_fps_max"]),
                "等価E2E FPS": float(row["equivalent_end_to_end_fps"]),
                "実効間隔": float(row["actual_interval_weighted"]),
                "キー数": int(row["keyframes"]),
                "平均IoU": float(row["weighted_iou_mean"]),
                "最小Recall": float(row["recall_min"]),
                "Recall違反": int(row["recall_violations"]),
            }
        )

    error_chart = []
    error_table = []
    for row in errors:
        interval = int(row["target_interval"])
        for label, field in (
            ("平均", "mutual_iou_mean"),
            ("下位1%", "mutual_iou_q01"),
            ("下位0.1%", "mutual_iou_q001"),
            ("最小", "mutual_iou_min"),
        ):
            error_chart.append(
                {
                    "目標間隔": interval,
                    "統計": label,
                    "相互IoU": float(row[field]),
                }
            )
        error_table.append(
            {
                "目標間隔": interval,
                "比較行数": int(row["rows"]),
                "JSON完全一致率": float(row["polygon_json_exact_rate"]),
                "相互IoU平均": float(row["mutual_iou_mean"]),
                "相互IoU q01": float(row["mutual_iou_q01"]),
                "相互IoU q001": float(row["mutual_iou_q001"]),
                "相互IoU最小": float(row["mutual_iou_min"]),
                "IoU<0.99": int(row["mutual_iou_below_0.99"]),
                "IoU<0.95": int(row["mutual_iou_below_0.95"]),
                "IoU<0.90": int(row["mutual_iou_below_0.9"]),
                "品質IoU差平均": float(row["quality_iou_delta_mean"]),
                "品質IoU差最小": float(row["quality_iou_delta_min"]),
                "品質IoU差最大": float(row["quality_iou_delta_max"]),
                "面積比最小": float(row["cpu_to_default_area_ratio_min"]),
                "面積比最大": float(row["cpu_to_default_area_ratio_max"]),
            }
        )

    speed_ratio = []
    runs = analysis.get("runs", [])
    if runs:
        index = {
            (row["run_id"], int(row["target_interval"]), row["mode"]): row
            for row in runs
        }
        for run_id in sorted({row["run_id"] for row in runs}):
            for interval in intervals:
                default = index[(run_id, interval, "default_cuda")]
                exact = index[(run_id, interval, "cpu_exact")]
                speed_ratio.append(
                    {
                        "動画": run_id,
                        "目標間隔": str(interval),
                        "CUDA対CPU速度比": float(default["optimizer_video_fps"])
                        / float(exact["optimizer_video_fps"]),
                    }
                )

    headline = []
    for interval in intervals:
        default = _get(runtime, interval, "default_cuda")
        exact = _get(runtime, interval, "cpu_exact")
        error = next(row for row in errors if int(row["target_interval"]) == interval)
        headline.append(
            {
                "目標間隔": interval,
                "CUDA FPS": float(default["optimizer_fps"]),
                "CPU FPS": float(exact["optimizer_fps"]),
                "速度比": float(default["optimizer_fps"]) / float(exact["optimizer_fps"]),
                "相互IoU平均": float(error["mutual_iou_mean"]),
                "相互IoU最小": float(error["mutual_iou_min"]),
                "品質IoU差平均": float(error["quality_iou_delta_mean"]),
                "Recall違反合計": int(default["recall_violations"])
                + int(exact["recall_violations"]),
            }
        )

    source = {
        "id": SOURCE_ID,
        "label": "V3 CPU exact vs default CUDA benchmark (2026-08-14)",
        "path": source_path,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": "SELECT * FROM runs; SELECT * FROM aggregate_errors;",
            "description": "全V3生推論に対する36後処理SQLiteと全dense polygonの比較。",
            "executed_at": generated_at,
            "tables_used": [
                "runs",
                "aggregate_runtime_quality",
                "paired_errors",
                "aggregate_errors",
                "worst_output_differences",
                "recall_violations",
                "recall_violation_summary",
            ],
            "filters": [
                "V3 raw inference corpus only",
                "target intervals 2 and 5",
                "default CUDA lazy-exact and CPU native-exact",
                "score >= 0.30",
            ],
            "metric_definitions": [
                "optimizer_fps = source frame rows / polygon optimizer wall seconds",
                "mutual IoU = exact native raster IoU(default dense polygon, CPU dense polygon)",
                "quality IoU delta = default arm-local exact IoU - CPU arm-local exact IoU",
            ],
        },
    }

    summary_lines = []
    for row in headline:
        summary_lines.append(
            f"目標{row['目標間隔']}はCUDA {row['CUDA FPS']:.2f} FPS、"
            f"CPU {row['CPU FPS']:.2f} FPS、相互IoU平均 {row['相互IoU平均']:.6f}、"
            f"最小 {row['相互IoU最小']:.6f}。"
        )
    summary = " ".join(summary_lines)
    recall_comparison = " ".join(
        f"目標{interval}のRecall違反はCUDA既定"
        f"{int(_get(runtime, interval, 'default_cuda')['recall_violations'])}件、"
        f"CPU厳密{int(_get(runtime, interval, 'cpu_exact')['recall_violations'])}件。"
        for interval in intervals
    )
    recall_blocker = validation.get("production_recall_gate") != "pass"
    recall_note = (
        "Production Recallゲートは不合格。"
        f"構成間の延べ違反数は {validation.get('total_recall_violations_across_configurations', 0)} で、"
        "SQLite整合性の合格とアルゴリズム品質の合格は別に扱う。"
        if recall_blocker
        else "Production Recallゲートは合格した。"
    )

    cards = []
    for interval in intervals:
        cards.extend(
            [
                {
                    "id": f"speed-{interval}",
                    "dataset": f"headline-{interval}",
                    "sourceId": SOURCE_ID,
                    "description": f"全9動画、目標間隔{interval}のoptimizer速度。",
                    "metrics": [
                        {"label": f"CUDA FPS (目標{interval})", "field": "CUDA FPS", "format": "number"},
                        {"label": "CPU比", "field": "速度比", "format": "number"},
                    ],
                },
                {
                    "id": f"error-{interval}",
                    "dataset": f"headline-{interval}",
                    "sourceId": SOURCE_ID,
                    "description": f"目標間隔{interval}のCPU厳密版に対する出力差。",
                    "metrics": [
                        {"label": "相互IoU平均", "field": "相互IoU平均", "format": "percent"},
                        {"label": "相互IoU最小", "field": "相互IoU最小", "format": "percent"},
                    ],
                },
            ]
        )

    charts = [
        {
            "id": "speed",
            "title": "全V3コーパスのoptimizer速度",
            "subtitle": "同一upstreamに対する後段polygon optimizer。高いほど高速。",
            "intent": "comparison",
            "type": "bar",
            "dataset": "speed",
            "sourceId": SOURCE_ID,
            "encodings": {
                "x": {"field": "目標間隔", "type": "ordinal", "label": "目標間隔"},
                "y": {"field": "FPS", "type": "quantitative", "label": "FPS", "format": "number"},
                "color": {"field": "方式", "type": "nominal", "label": "方式"},
            },
            "valueFormat": "number",
        },
        {
            "id": "quality",
            "title": "参照AIマスクに対する平均IoU",
            "subtitle": "人手GTではなく、各方式で共通のtracked sourceに対する忠実度。",
            "intent": "comparison",
            "type": "bar",
            "dataset": "quality",
            "sourceId": SOURCE_ID,
            "encodings": {
                "x": {"field": "目標間隔", "type": "ordinal", "label": "目標間隔"},
                "y": {"field": "平均IoU", "type": "quantitative", "label": "平均IoU", "format": "percent"},
                "color": {"field": "方式", "type": "nominal", "label": "方式"},
            },
            "valueFormat": "percent",
        },
        {
            "id": "error",
            "title": "CUDA既定とCPU厳密の出力相互IoU",
            "subtitle": "平均だけでなく下位1%、下位0.1%、最悪値を表示。",
            "intent": "comparison",
            "type": "bar",
            "dataset": "error",
            "sourceId": SOURCE_ID,
            "encodings": {
                "x": {"field": "目標間隔", "type": "ordinal", "label": "目標間隔"},
                "y": {"field": "相互IoU", "type": "quantitative", "label": "相互IoU", "format": "percent"},
                "color": {"field": "統計", "type": "nominal", "label": "統計"},
            },
            "valueFormat": "percent",
        },
    ]
    if speed_ratio:
        charts.append(
            {
                "id": "speed-ratio",
                "title": "動画別のCUDA / CPU速度比",
                "subtitle": "1.0超はCUDAが高速。小規模グラフでは起動コストが支配し得る。",
                "intent": "comparison",
                "type": "bar",
                "dataset": "speed-ratio",
                "sourceId": SOURCE_ID,
                "encodings": {
                    "x": {"field": "動画", "type": "nominal", "label": "動画"},
                    "y": {"field": "CUDA対CPU速度比", "type": "quantitative", "label": "速度比", "format": "number"},
                    "color": {"field": "目標間隔", "type": "nominal", "label": "目標間隔"},
                },
                "valueFormat": "number",
            }
        )

    tables = [
        {
            "id": "aggregate",
            "title": "方式・目標間隔別の速度と品質",
            "dataset": "aggregate",
            "sourceId": SOURCE_ID,
            "density": "dense",
            "columns": [
                {"field": "目標間隔", "label": "目標", "format": "number"},
                {"field": "方式", "label": "方式", "type": "text"},
                {"field": "optimizer FPS", "label": "FPS", "format": "number"},
                {"field": "動画別FPS中央値", "label": "動画中央FPS", "format": "number"},
                {"field": "等価E2E FPS", "label": "E2E FPS", "format": "number"},
                {"field": "実効間隔", "label": "実効", "format": "number"},
                {"field": "キー数", "label": "キー数", "format": "compact"},
                {"field": "平均IoU", "label": "平均IoU", "format": "percent"},
                {"field": "最小Recall", "label": "最小Recall", "format": "percent"},
                {"field": "Recall違反", "label": "違反", "format": "number"},
            ],
        },
        {
            "id": "errors",
            "title": "CPU厳密版に対するCUDA既定版の誤差幅",
            "dataset": "errors",
            "sourceId": SOURCE_ID,
            "density": "dense",
            "columns": [
                {"field": "目標間隔", "label": "目標", "format": "number"},
                {"field": "比較行数", "label": "行数", "format": "compact"},
                {"field": "JSON完全一致率", "label": "完全一致", "format": "percent"},
                {"field": "相互IoU平均", "label": "平均", "format": "percent"},
                {"field": "相互IoU q01", "label": "q01", "format": "percent"},
                {"field": "相互IoU q001", "label": "q001", "format": "percent"},
                {"field": "相互IoU最小", "label": "最小", "format": "percent"},
                {"field": "IoU<0.95", "label": "<0.95", "format": "compact"},
                {"field": "IoU<0.90", "label": "<0.90", "format": "compact"},
                {"field": "品質IoU差平均", "label": "品質差平均", "format": "percent"},
                {"field": "品質IoU差最小", "label": "品質差最小", "format": "percent"},
                {"field": "品質IoU差最大", "label": "品質差最大", "format": "percent"},
            ],
        },
    ]
    if recall_summary:
        tables.append(
            {
                "id": "recall-violations",
                "title": "Recall 0.97違反の所在",
                "dataset": "recall-violations",
                "sourceId": SOURCE_ID,
                "density": "dense",
                "columns": [
                    {"field": "動画", "label": "動画", "type": "text"},
                    {"field": "目標間隔", "label": "目標", "format": "number"},
                    {"field": "方式", "label": "方式", "type": "text"},
                    {"field": "クラス", "label": "クラス", "type": "text"},
                    {"field": "違反", "label": "違反", "format": "number"},
                    {"field": "両方共通", "label": "共通", "format": "number"},
                    {"field": "この方式のみ", "label": "方式固有", "format": "number"},
                    {"field": "最初frame", "label": "開始", "format": "number"},
                    {"field": "最後frame", "label": "終了", "format": "number"},
                    {"field": "最小Recall", "label": "最小Recall", "format": "percent"},
                ],
            }
        )

    card_ids = [card["id"] for card in cards]
    blocks = [
        {"id": "title", "type": "markdown", "layout": "full", "body": "# V3全推論：CPU厳密版 vs CUDA既定版"},
        {
            "id": "summary",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": (
                "## Technical summary\n\n"
                f"全9本・477,691フレームを、目標間隔{intervals}の2条件、CPU全辺厳密評価と"
                "CUDA lazy-exact評価の2方式で処理し、36個の編集ソフト互換SQLiteを生成した。"
                f" {summary} {recall_comparison} 最小Recall 0.97は制約であり、"
                "違反数とSQLite監査を採否ゲートにした。"
                f" {recall_note}"
            ),
        },
        {"id": "cards", "type": "metric-strip", "layout": "full", "cardIds": card_ids},
        {
            "id": "findings",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": "## Key findings and visual evidence\n\n速度は全動画の直列wall timeから算出し、誤差は全dense polygonの対応行をnative exact rasterで比較した。",
        },
        {"id": "speed-chart", "type": "chart", "layout": "half", "chartId": "speed"},
        {"id": "quality-chart", "type": "chart", "layout": "half", "chartId": "quality"},
        {
            "id": "speed-note",
            "type": "markdown",
            "layout": "half",
            "sourceId": SOURCE_ID,
            "body": "CUDAの利得は区間グラフの規模で変わる。短い・疎な動画ではGPU起動と転送コストによりCPUが速い場合もあるため、総計と動画別比率を併記した。",
        },
        {
            "id": "quality-note",
            "type": "markdown",
            "layout": "half",
            "sourceId": SOURCE_ID,
            "body": "ここでのIoUは人手GT精度ではなく、共通の追跡AIマスクをどれだけ忠実に近似したかを表す。最小Recallは全フレームのハード制約として別に監査した。",
        },
        {"id": "error-chart", "type": "chart", "layout": "half", "chartId": "error"},
    ]
    if speed_ratio:
        blocks.append({"id": "speed-ratio-chart", "type": "chart", "layout": "half", "chartId": "speed-ratio"})
    blocks.extend(
        [
            {"id": "aggregate-table", "type": "table", "layout": "full", "tableId": "aggregate"},
            {"id": "error-table", "type": "table", "layout": "full", "tableId": "errors"},
            *(
                [
                    {
                        "id": "recall-table",
                        "type": "table",
                        "layout": "full",
                        "tableId": "recall-violations",
                    }
                ]
                if recall_summary
                else []
            ),
            {
                "id": "scope",
                "type": "markdown",
                "layout": "full",
                "sourceId": SOURCE_ID,
                "body": (
                    "## Scope, data, and metric definitions\n\n"
                    "対象は保存済みV3生推論9本。NMS・穴埋め・島処理・tracking・14頂点近似・"
                    "最小Recall DP・pair-voteは同一で、区間評価器だけを変更した。optimizer FPSは"
                    "生SQLiteのframes行数をoptimizer wall秒で割った値。相互IoUは完成dense polygon同士。"
                ),
            },
            {
                "id": "method",
                "type": "markdown",
                "layout": "full",
                "sourceId": SOURCE_ID,
                "body": (
                    "## Methodology\n\n"
                    "各動画のupstreamを1回だけ作成し、両方式・両間隔でbyte-identicalなprepared sourceを共有した。"
                    "CPU厳密版は全DP辺をnative CPUで評価し、CUDA既定版はCUDAで全辺をscreenした後、"
                    "採用経路と最終dense出力をexact監査する。全最終SQLiteにintegrity_checkとforeign_key_checkを実施した。"
                ),
            },
            {
                "id": "limits",
                "type": "markdown",
                "layout": "full",
                "sourceId": SOURCE_ID,
                "body": (
                    "## Limitations and robustness\n\n"
                    "人手GTはなく、AI生マスクへの忠実度を比較している。"
                    "平均IoU差は両方向に局在するが、Recall違反はCUDA既定だけが65フレーム追加した。"
                    "平均差が小さくても最悪フレームは別途確認対象となる。FPSは同一PC上の単回wall測定であり、"
                    "温度・OS負荷・GPU状態による揺らぎを含む。"
                    f" {recall_note} 動画画像は解析・表示せず、ローカル形状データだけを使用した。"
                ),
            },
            {
                "id": "next",
                "type": "markdown",
                "layout": "full",
                "sourceId": SOURCE_ID,
                "body": (
                    "## Recommended next steps\n\n"
                    "現状はCUDAの速度優位がほぼなく、CPU厳密版よりRecall違反を65フレーム追加するため、"
                    "ProductionではCPU厳密経路を優先する。共通の130違反は14頂点空間近似のfallbackで解消する。"
                    "CUDAは追加65違反を0にし、別コーパスで再験するまで実験方式とする。"
                ),
            },
            {
                "id": "questions",
                "type": "markdown",
                "layout": "full",
                "sourceId": SOURCE_ID,
                "body": "## Further questions\n\n最悪相互IoUの局所差は編集上見えるか。速度計測を複数回繰り返した場合も同じ順位か。目標間隔2と5のどちらを運用既定にするか。",
            },
        ]
    )

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "V3 CPU厳密版 vs CUDA既定版 技術評価",
            "description": "全V3生推論、目標間隔2/5の速度・品質・出力誤差・SQLite監査。",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [source],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                **{f"headline-{row['目標間隔']}": [row] for row in headline},
                "speed": speed_rows,
                "quality": quality_rows,
                "error": error_chart,
                "speed-ratio": speed_ratio,
                "aggregate": aggregate_table,
                "errors": error_table,
                "recall-violations": recall_summary,
            },
        },
        "sources": [source],
        "validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-path", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    # runs are intentionally injected only for the per-video speed-ratio chart.
    runs_csv = args.analysis.parent / "runs.csv"
    if runs_csv.is_file():
        import csv

        with runs_csv.open("r", encoding="utf-8", newline="") as handle:
            analysis["runs"] = list(csv.DictReader(handle))
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(
        json.dumps(
            build_artifact(analysis, args.source_path),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
