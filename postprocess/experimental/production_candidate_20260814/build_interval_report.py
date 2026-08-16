#!/usr/bin/env python3
"""Build reproducible notebook and portable-report payload for interval 1..6."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import nbformat
except ModuleNotFoundError:  # Artifact-only environments do not need Jupyter.
    nbformat = None


SOURCE_ID = "kpi-interval-1-6-comparison"


def _indexed(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["arm"]), int(row["target_interval"])): row
        for row in payload["rows"]
    }


def _arm_name(value: str) -> str:
    return "旧Production" if value == "legacy_production" else "新候補"


def build_artifact(payload: dict[str, Any], source_path: str) -> dict[str, Any]:
    indexed = _indexed(payload)
    generated_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    actual_chart: list[dict[str, Any]] = []
    iou_chart: list[dict[str, Any]] = []
    recall_chart: list[dict[str, Any]] = []
    speed_chart: list[dict[str, Any]] = []
    legacy_violations = 0
    candidate_violations = 0

    for interval in range(1, 7):
        for arm in ("legacy_production", "production_candidate_20260814"):
            source = indexed[(arm, interval)]
            name = _arm_name(arm)
            item = {
                "方式": name,
                "目標間隔": interval,
                "実効間隔": source["actual_interval"],
                "キー数": source["keyframes"],
                "Recall平均": source["recall_mean"],
                "Recall最小": source["recall_min"],
                "Recall違反": source["recall_violations"],
                "IoU平均": source["iou_mean"],
                "IoU下位1%": source["iou_q01"],
                "IoU最小": source["iou_min"],
                "面積比q99": source["area_ratio_q99"],
                "共通raw IoU": source["common_raw_pixel_weighted_iou"],
                "処理秒": source["optimizer_wall_seconds"],
                "処理FPS": source["video_fps"],
                "不正ポリゴン": source["invalid_polygon_rings"],
            }
            rows.append(item)
            actual_chart.append(
                {"目標間隔": interval, "方式": name, "実効間隔": source["actual_interval"]}
            )
            iou_chart.extend(
                [
                    {"目標間隔": interval, "方式": f"{name} 平均", "IoU": source["iou_mean"]},
                    {"目標間隔": interval, "方式": f"{name} 下位1%", "IoU": source["iou_q01"]},
                ]
            )
            recall_chart.append(
                {"目標間隔": interval, "方式": name, "最小Recall": source["recall_min"]}
            )
            speed_chart.append(
                {"目標間隔": interval, "方式": name, "処理秒": source["optimizer_wall_seconds"]}
            )
            if arm == "legacy_production":
                legacy_violations += int(source["recall_violations"])
            else:
                candidate_violations += int(source["recall_violations"])

    candidate_rows = [row for row in rows if row["方式"] == "新候補"]
    legacy_rows = [row for row in rows if row["方式"] == "旧Production"]
    deltas = payload["by_interval"]
    headline = [
        {
            "新候補Recall合格設定": sum(row["Recall最小"] >= 0.97 - 1e-9 for row in candidate_rows),
            "旧Recall違反合計": legacy_violations,
            "新Recall違反合計": candidate_violations,
            "IoU平均差平均": statistics.mean(float(row["candidate_minus_legacy_iou_mean"]) for row in deltas),
            "速度倍率中央値": statistics.median(float(row["candidate_vs_legacy_runtime_ratio"]) for row in deltas),
            "新目標達成率平均": statistics.mean(min(row["実効間隔"], row["目標間隔"]) / max(row["実効間隔"], row["目標間隔"]) for row in candidate_rows),
            "旧目標達成率平均": statistics.mean(min(row["実効間隔"], row["目標間隔"]) / max(row["実効間隔"], row["目標間隔"]) for row in legacy_rows),
        }
    ]
    source = {
        "id": SOURCE_ID,
        "label": "KPI V3 mask-geometry interval 1..6 benchmark (2026-08-14)",
        "path": source_path,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": (
                "SELECT * FROM interval_metrics "
                "ORDER BY target_interval, arm"
            ),
            "description": (
                "旧Productionと新Production候補の目標間隔1〜6における"
                "品質・実効間隔・速度・SQLite監査結果を読み出す。"
            ),
            "executed_at": generated_at,
            "tables_used": ["interval_metrics"],
            "filters": [
                "V3 KPI scored detections; score >= 0.30",
                "target intervals 1 through 6",
                "SQLite/JSON mask geometry only; no video frame decoded",
            ],
            "metric_definitions": [
                "actual_interval = prediction_rows / keyframes",
                "recall_violations = arm-local exact Recall below 0.97",
                "common_raw_pixel_weighted_iou uses the same scored AI masks",
            ],
        },
    }
    title = "旧Production vs 新Production候補：目標キーフレーム間隔1〜6"

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}", "layout": "full"},
        {
            "id": "summary",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": (
                "## Technical Summary\n\n"
                "同一KPI V3入力で、旧Production全系統と、新しい仮想成分Mask NMS・穴/島処理・"
                "14頂点ポリゴン・最小Recall制約DP・制約付きpair-voteを比較した。"
                "新候補はRecall 0.97をハード制約として扱う一方、旧Productionは最終pair-vote後の"
                "同制約を保証しない。品質値は人手GTではなく各armの追跡AIマスク基準であり、"
                "共通raw AIマスク基準も併記した。"
            ),
        },
        {"id": "cards", "type": "metric-strip", "cardIds": ["recall", "iou", "target", "speed"], "layout": "full"},
        {"id": "findings", "type": "markdown", "layout": "full", "body": "## Findings and visual evidence\n\n目標追尾、平均/裾IoU、最小Recall、処理時間を同じ6設定で比較する。", "sourceId": SOURCE_ID},
        {"id": "actual", "type": "chart", "chartId": "actual", "layout": "half"},
        {"id": "iou", "type": "chart", "chartId": "iou", "layout": "half"},
        {"id": "recall-chart", "type": "chart", "chartId": "recall-chart", "layout": "half"},
        {"id": "speed-chart", "type": "chart", "chartId": "speed-chart", "layout": "half"},
        {"id": "metrics", "type": "table", "tableId": "metrics", "layout": "full"},
        {
            "id": "scope",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": (
                "## Scope, data, and definitions\n\n"
                "対象は12月KPI動画のV3推論SQLite。目標間隔は1〜6、score閾値0.30、短命トラック削除10、"
                "同一カットを使用。実効間隔はdense prediction行数 / keyframe数。Recall違反は"
                "各armの追跡参照に対する0.97未満。処理FPSは23,510フレーム / optimizer wall秒。"
            ),
        },
        {
            "id": "method",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": (
                "## Methodology\n\n"
                "旧armはlegacy adaptive NMS、旧ポリゴン近似、production_v22 DP/pair-vote。"
                "新armはvirtual-component Mask NMS v4、全穴埋め、所有本体比1%以下の島除去、"
                "14頂点近似、各補間フレームの最小Recall制約DP、2 sweep pair-vote。"
                "SQLite/JSONのマスク形状のみを解析し、動画フレームは復号していない。"
            ),
        },
        {
            "id": "limits",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": (
                "## Limitations and robustness\n\n"
                "人手GTがなく、armごとにNMS・tracking結果が異なるためarm-local IoU/Recallは完全な"
                "意味精度比較ではない。共通raw AIマスクIoUを補助指標にしたが、AI自体の誤検出を"
                "正解として扱う限界がある。単一動画の結果なので、Production昇格前には別V3動画と"
                "編集ソフトでの限定目視が必要。全SQLiteはintegrity/FK/schemaを監査する。"
            ),
        },
        {
            "id": "next",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": (
                "## Recommended next steps\n\n"
                "1. 実用候補の間隔3〜6を編集ソフトで局所確認する。\n"
                "2. 別V3動画で同じ12点比較を再実行する。\n"
                "3. 速度差が許容外なら候補形状生成とpair-voteをプロファイルする。"
            ),
        },
        {"id": "questions", "type": "markdown", "layout": "full", "body": "## Further questions\n\nNMS差で新しく残るトラックは真の別インスタンスか。最小Recall維持のための面積増加は編集上許容されるか。実運用の既定目標を3〜6のどこに置くか。", "sourceId": SOURCE_ID},
    ]

    cards = [
        {"id": "recall", "dataset": "headline", "description": "6設定中のRecall制約合格数と違反総数。", "sourceId": SOURCE_ID, "metrics": [{"label": "新候補合格", "field": "新候補Recall合格設定", "format": "number"}, {"label": "旧違反", "field": "旧Recall違反合計", "format": "number"}]},
        {"id": "iou", "dataset": "headline", "description": "6設定における新候補−旧Productionの平均IoU差の平均。", "sourceId": SOURCE_ID, "metrics": [{"label": "平均IoU差", "field": "IoU平均差平均", "format": "percent"}]},
        {"id": "target", "dataset": "headline", "description": "目標間隔への平均到達率。", "sourceId": SOURCE_ID, "metrics": [{"label": "新候補", "field": "新目標達成率平均", "format": "percent"}, {"label": "旧", "field": "旧目標達成率平均", "format": "percent"}]},
        {"id": "speed", "dataset": "headline", "description": "新候補optimizer wall / 旧Production wall の中央値。", "sourceId": SOURCE_ID, "metrics": [{"label": "速度コスト倍率", "field": "速度倍率中央値", "format": "number"}]},
    ]

    def chart(chart_id: str, title_text: str, dataset: str, y_field: str, y_label: str, fmt: str = "number") -> dict[str, Any]:
        return {
            "id": chart_id,
            "title": title_text,
            "intent": "comparison",
            "type": "bar",
            "dataset": dataset,
            "sourceId": SOURCE_ID,
            "encodings": {
                "x": {"field": "目標間隔", "type": "ordinal", "label": "目標間隔"},
                "y": {"field": y_field, "type": "quantitative", "label": y_label, "format": fmt},
                "color": {"field": "方式", "type": "nominal", "label": "方式"},
            },
            "valueFormat": fmt,
        }

    charts = [
        chart("actual", "目標と実効キーフレーム間隔", "actual_chart", "実効間隔", "実効間隔"),
        chart("iou", "平均IoUと下位1% IoU", "iou_chart", "IoU", "IoU", "percent"),
        chart("recall-chart", "最小Recall", "recall_chart", "最小Recall", "最小Recall", "percent"),
        chart("speed-chart", "optimizer処理時間", "speed_chart", "処理秒", "秒"),
    ]
    tables = [
        {
            "id": "metrics",
            "title": "目標間隔1〜6の全比較",
            "dataset": "metrics",
            "sourceId": SOURCE_ID,
            "density": "dense",
            "defaultSort": {"field": "目標間隔", "direction": "asc"},
            "columns": [
                {"field": "方式", "label": "方式", "type": "text"},
                {"field": "目標間隔", "label": "目標", "format": "number"},
                {"field": "実効間隔", "label": "実効", "format": "number"},
                {"field": "キー数", "label": "キー数", "format": "number"},
                {"field": "Recall最小", "label": "Recall最小", "format": "percent"},
                {"field": "Recall違反", "label": "違反", "format": "number"},
                {"field": "IoU平均", "label": "IoU平均", "format": "percent"},
                {"field": "IoU下位1%", "label": "IoU q01", "format": "percent"},
                {"field": "IoU最小", "label": "IoU最小", "format": "percent"},
                {"field": "面積比q99", "label": "面積比q99", "format": "number"},
                {"field": "共通raw IoU", "label": "共通raw IoU", "format": "percent"},
                {"field": "処理秒", "label": "秒", "format": "number"},
                {"field": "処理FPS", "label": "FPS", "format": "number"},
                {"field": "不正ポリゴン", "label": "不正", "format": "number"},
            ],
        }
    ]
    return {
        "surface": "report",
        "manifest": {"version": 1, "surface": "report", "title": title, "description": "KPI V3の目標キーフレーム間隔1〜6比較", "generatedAt": generated_at, "cards": cards, "charts": charts, "tables": tables, "sources": [source], "blocks": blocks},
        "snapshot": {"version": 1, "generatedAt": generated_at, "status": "ready", "datasets": {"headline": headline, "metrics": rows, "actual_chart": actual_chart, "iou_chart": iou_chart, "recall_chart": recall_chart, "speed_chart": speed_chart}},
        "sources": [source],
    }


def build_notebook(comparison: Path) -> Any:
    relative = str(comparison)
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell("# tl;dr\n\n旧Productionと新Production候補を目標間隔1〜6で比較し、Recall、IoU、実効間隔、速度、SQLite健全性を監査する。"),
        nbformat.v4.new_markdown_cell("## Context & Methods\n\n同一KPI V3入力を用い、動画フレームを復号せずSQLite/JSONマスク形状だけを集計する。arm-local品質と共通raw AIマスク基準を分ける。"),
        nbformat.v4.new_code_cell(f"from pathlib import Path\nimport json, pandas as pd\np = Path({relative!r})\ndata = json.loads(p.read_text(encoding='utf-8'))\ndf = pd.DataFrame(data['rows'])\nassert len(df) == 12\nassert set(df.target_interval) == set(range(1, 7))\nassert set(df.arm) == {{'legacy_production', 'production_candidate_20260814'}}\ndf[['arm','target_interval','actual_interval','recall_min','recall_violations','iou_mean','iou_q01','optimizer_wall_seconds']]"),
        nbformat.v4.new_markdown_cell("## Data\n\n12セル（2方式×6目標）。各セルは完成したソフトウェア互換SQLiteとexact評価CSVへ対応する。"),
        nbformat.v4.new_code_cell("assert data['validation']['all_sqlite_integrity_ok']\nassert data['validation']['all_sqlite_foreign_keys_ok']\nassert (df.query(\"arm == 'production_candidate_20260814'\").recall_violations == 0).all()\ndf.groupby('arm').agg({'result_sqlite':'count','invalid_polygon_rings':'sum','recall_violations':'sum'})"),
        nbformat.v4.new_markdown_cell("## Results"),
        nbformat.v4.new_code_cell("cols=['arm','target_interval','actual_interval','keyframes','recall_min','recall_violations','iou_mean','iou_q01','common_raw_pixel_weighted_iou','optimizer_wall_seconds','video_fps']\ndf[cols].sort_values(['target_interval','arm']).reset_index(drop=True)"),
        nbformat.v4.new_code_cell("delta = pd.DataFrame(data['by_interval'])\ndelta"),
        nbformat.v4.new_markdown_cell("## Takeaways\n\n新候補は最小Recallを制約として保証する。IoU・実効間隔・速度の優劣は上表から判断し、人手GT不在とNMS差による参照差を必ず併記する。"),
    ]
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--source-path", required=True)
    args = parser.parse_args()
    payload = json.loads(args.comparison.read_text(encoding="utf-8"))
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(build_artifact(payload, args.source_path), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    nbformat.write(build_notebook(args.comparison.resolve()), args.notebook)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
