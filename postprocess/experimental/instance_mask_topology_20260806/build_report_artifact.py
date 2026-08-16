#!/usr/bin/env python3
"""Build the canonical portable-report artifact for the topology audit."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path


TITLE = "1インスタンス複数連結成分：V3 / V3-lite監査レポート"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:.{digits}f}%"


def _source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    generated_at: str,
    sql: str,
    tables_used: list[str],
) -> dict[str, object]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": sql,
            "description": description,
            "executed_at": generated_at,
            "tables_used": tables_used,
            "filters": [
                "元動画フレームをAIへ送信・表示せず、ローカル推論SQLiteだけを集計",
                "重複する30–45分クリップは固有動画カバレッジの総計から除外",
            ],
        },
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    summary = _read(args.summary)
    postprocess = [_read(path) for path in args.postprocess_summary]
    repeatability = _read(args.repeatability)
    raster = _read(args.raster_validation)
    audit = _read(args.artifact_audit)
    postprocess_audit = _read(args.postprocess_audit)
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()

    runs = list(summary["runs"])
    unique_totals = summary["model_totals_unique_video_coverage"]
    paired = list(summary["paired"])
    score_bins = [
        {
            "model": row["model"],
            "score_bin": row["score_bin"],
            "detections": row["detections"],
            "multi_count": row["multi_foreground"],
            "multi_rate_pct": 100.0 * row["multi_rate"],
        }
        for row in summary["score_bins_unique_video_coverage"]
    ]
    lite_total = unique_totals["v3lite"]
    v3_total = unique_totals["v3"]

    per_run = [
        {
            "model": row["model"],
            "video": row["video_slug"],
            "frames": row["frames"],
            "detections": row["detections"],
            "multi_count": row["multi_foreground"],
            "multi_rate_pct": 100.0 * row["multi_rate"],
            "holes": row["with_holes"],
            "max_components": row["max_foreground_components"],
            "severe_secondary_ge_5pct": (
                row["second_ratio_0_05_to_0_20"] + row["second_ratio_ge_0_20"]
            ),
            "inference_fps": row["inference_fps"],
        }
        for row in runs
    ]
    paired_rows = [
        {
            "video": row["video_slug"],
            "model": model,
            "comparable_frames": row["comparable_frames"],
            "detections": row[f"{model}_detections"],
            "multi_count": row[f"{model}_multi"],
            "multi_rate_pct": 100.0 * row[f"{model}_multi_rate"],
            "coverage": row["coverage_note"],
        }
        for row in paired
        for model in ("v3", "v3lite")
    ]
    severity = []
    for row in runs:
        if row["model"] != "v3lite" or row["video_slug"] == "heyzo_3545_30_45_duplicate":
            continue
        severity.extend(
            {
                "video": row["video_slug"],
                "severity": label,
                "count": row[field],
            }
            for label, field in (
                ("<0.1%", "second_ratio_lt_0_001"),
                ("0.1–1%", "second_ratio_0_001_to_0_01"),
                ("1–5%", "second_ratio_0_01_to_0_05"),
                ("5–20%", "second_ratio_0_05_to_0_20"),
                (">=20%", "second_ratio_ge_0_20"),
            )
        )
    post_rows = []
    for item in postprocess:
        disposition = item["multi_disposition"]
        post_rows.extend(
            {
                "run": item["run_key"],
                "stage": stage,
                "count": value,
            }
            for stage, value in (
                ("raw multi-component", item["multi_foreground_detections"]),
                ("retained after NMS/tracking", disposition["retained"]),
                ("retained as exact keyframe", item["retained_multi_exact_keyframes"]),
            )
        )
    post_table = [
        {
            "run": item["run_key"],
            "raw_multi": item["multi_foreground_detections"],
            "retained": item["multi_disposition"]["retained"],
            "nms_or_unassigned": item["multi_disposition"]["nms_or_unassigned"],
            "short_track_removed": item["multi_disposition"]["short_track_removed"],
            "exact_keyframes": item["retained_multi_exact_keyframes"],
            "keyframe_enrichment": item["multi_keyframe_enrichment"],
            "final_multi_segments": item["final_multi_component_segments"],
            "tracks_with_varying_slots": item["final_tracks_with_varying_keyframe_slot_count"],
        }
        for item in postprocess
    ]

    summary_source = _source(
        "topology_summary",
        "トポロジー集計",
        "summary.json",
        "推論SQLiteの各検出について輪郭包含深度を再構成し、偶数深度を前景成分、奇数深度を穴として集計。",
        generated_at,
        "SELECT run_key, COUNT(*) AS detections, SUM(foreground_component_count > 1) AS multi_foreground, SUM(hole_count > 0) AS with_holes FROM mask_topology GROUP BY run_key",
        ["topology.sqlite.mask_topology", "topology.sqlite.audit_runs"],
    )
    post_source = _source(
        "postprocess_trace",
        "Production後処理追跡",
        "postprocess_trace.sqlite",
        "source_detection_idを使い、複数前景成分の検出がNMS、追跡、短命削除、キーフレーム、最終geometryへどう到達したかを追跡。",
        generated_at,
        "SELECT run_key, disposition, COUNT(*) AS detections, SUM(exact_keyframe) AS exact_keyframes FROM detection_outcomes WHERE foreground_component_count > 1 GROUP BY run_key, disposition",
        ["postprocess_trace.sqlite.detection_outcomes"],
    )
    validation_source = _source(
        "validation_evidence",
        "成果物整合性検証",
        "validation.json",
        "SQLite quick_check、行数・外部キー相当の整合、スキーマ署名、ラスタ再検証、重複区間の再現性を監査。",
        generated_at,
        "SELECT run_key, frame_count, detection_count FROM audit_runs ORDER BY run_key",
        ["topology.sqlite.audit_runs"],
    )

    headline = (
        f"V3-liteでは固有動画カバレッジ{int(lite_total['detections']):,}検出中"
        f"{int(lite_total['multi_foreground']):,}件（{_pct(float(lite_total['multi_rate']))}）に"
        "複数の独立前景成分がありました。同一区間比較ではV3が一貫して低率です。"
    )
    post_statement = (
        "Production後処理はインスタンス内部の成分を個別に除去しません。"
        f"代表2 runではraw {sum(int(x['multi_foreground_detections']) for x in postprocess):,}件中、"
        f"{sum(int(x['multi_disposition']['retained']) for x in postprocess):,}件が追跡結果へ残り、"
        f"{sum(int(x['retained_multi_exact_keyframes']) for x in postprocess):,}件が厳密なキーフレームになりました。"
    )
    raster_checked = sum(int(row["checked_multi_contour_detections"]) for row in raster["runs"])
    raster_foreground_mismatches = sum(
        int(row.get("foreground_count_mismatches", row["mismatches"]))
        for row in raster["runs"]
    )
    raster_foreground_agreement = (
        1.0 - raster_foreground_mismatches / raster_checked if raster_checked else 1.0
    )
    validation_statement = (
        f"成果物監査は{'合格' if audit.get('success') else '不合格'}。"
        f"代表後処理SQLite監査も{'合格' if postprocess_audit.get('success') else '不合格'}。"
        f"重複区間の行単位一致率は{_pct(float(repeatability['exact_row_rate']), 4)}、"
        f"ラスタ再構成の前景成分一致は{_pct(raster_foreground_agreement, 4)}です。"
    )

    cards = [
        {
            "id": "lite_rate",
            "description": "重複クリップを除くV3-lite固有動画カバレッジ。",
            "dataset": "headline_metrics",
            "sourceId": "topology_summary",
            "filter": {"metric": "v3lite_multi_rate"},
            "metrics": [{"label": "V3-lite 複数前景成分率", "field": "value", "format": "percent"}],
        },
        {
            "id": "lite_coverage",
            "description": "V3-liteで監査した検出インスタンス数。",
            "dataset": "headline_metrics",
            "sourceId": "topology_summary",
            "filter": {"metric": "v3lite_detections"},
            "metrics": [{"label": "V3-lite監査検出数", "field": "value", "format": "number"}],
        },
        {
            "id": "v3_rate",
            "description": "時間予算内で選択したV3カバレッジ。モデル間比較は同一区間チャートを参照。",
            "dataset": "headline_metrics",
            "sourceId": "topology_summary",
            "filter": {"metric": "v3_multi_rate"},
            "metrics": [{"label": "V3 複数前景成分率", "field": "value", "format": "percent"}],
        },
        {
            "id": "retention_rate",
            "description": "代表2 runにおけるraw複数成分検出の追跡結果残存率。",
            "dataset": "headline_metrics",
            "sourceId": "postprocess_trace",
            "filter": {"metric": "postprocess_retention"},
            "metrics": [{"label": "後処理後の残存率", "field": "value", "format": "percent"}],
        },
    ]
    charts = [
        {
            "id": "paired_rates",
            "title": "同一先頭フレーム区間での複数前景成分率",
            "subtitle": "時間窓を揃え、V3の部分推論runも公平に比較します。",
            "type": "bar",
            "intent": "comparison",
            "question": "同じ動画区間でV3とV3-liteの複数前景成分率はどれだけ違うか。",
            "rationale": "動画ごとにモデルを並べることで、動画構成差を混ぜずにモデル差を比較できる。",
            "combinationRationale": "色をモデルへ割り当て、各動画内で2モデルを並列比較する。",
            "dataset": "paired_rates",
            "sourceId": "topology_summary",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "video", "type": "nominal", "label": "動画"},
                "y": {"field": "multi_rate_pct", "type": "quantitative", "label": "複数前景成分率 (%)"},
                "color": {"field": "model", "type": "nominal", "label": "モデル"},
                "tooltip": [
                    {"field": "detections", "type": "quantitative", "label": "検出数"},
                    {"field": "multi_count", "type": "quantitative", "label": "複数成分数"},
                    {"field": "comparable_frames", "type": "quantitative", "label": "比較フレーム"},
                ],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "title": "モデル"},
        },
        {
            "id": "lite_per_video",
            "title": "V3-liteの動画別発生率",
            "subtitle": "0.77%前後から2.33%まで、入力動画による差があります。",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "V3-liteの複数前景成分率は動画によってどれだけ変わるか。",
            "rationale": "長い動画識別子を横軸値のbarへ並べ、発生率の順位と幅を読み取りやすくする。",
            "dataset": "v3lite_per_video",
            "sourceId": "topology_summary",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "video", "type": "nominal", "label": "動画"},
                "y": {"field": "multi_rate_pct", "type": "quantitative", "label": "複数前景成分率 (%)"},
                "tooltip": [
                    {"field": "detections", "type": "quantitative", "label": "検出数"},
                    {"field": "multi_count", "type": "quantitative", "label": "複数成分数"},
                ],
            },
        },
        {
            "id": "severity",
            "title": "V3-lite副成分の相対面積分布",
            "subtitle": "小さい島が多数ですが、主成分に近い大きさの重大例も存在します。",
            "type": "stackedBar",
            "intent": "composition",
            "question": "各動画の複数前景成分は副成分面積比のどの帯域に分布するか。",
            "rationale": "面積比の帯域を積み上げ、総件数と深刻度構成を同時に比較する。",
            "dataset": "severity",
            "sourceId": "topology_summary",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "video", "type": "nominal", "label": "動画"},
                "y": {"field": "count", "type": "quantitative", "label": "検出数"},
                "color": {"field": "severity", "type": "nominal", "label": "副成分 / 主成分面積比"},
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "title": "副成分 / 主成分面積比"},
        },
        {
            "id": "score_relationship",
            "title": "検出確信度帯別の複数前景成分率",
            "subtitle": "V3-liteでは低確信度帯ほど発生率が高い傾向です。",
            "type": "bar",
            "intent": "comparison",
            "question": "検出確信度と複数前景成分の発生率にはどのような関係があるか。",
            "rationale": "共通の確信度帯でモデルを並べ、発生率の単調傾向とモデル差を同時に確認する。",
            "combinationRationale": "色をモデルへ割り当て、同一確信度帯で発生率を比較する。",
            "dataset": "score_bins",
            "sourceId": "topology_summary",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "score_bin", "type": "ordinal", "label": "検出確信度"},
                "y": {"field": "multi_rate_pct", "type": "quantitative", "label": "複数前景成分率 (%)"},
                "color": {"field": "model", "type": "nominal", "label": "モデル"},
                "tooltip": [
                    {"field": "detections", "type": "quantitative", "label": "検出数"},
                    {"field": "multi_count", "type": "quantitative", "label": "複数成分数"},
                ],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "title": "モデル"},
        },
        {
            "id": "postprocess_flow",
            "title": "複数前景成分検出のProduction後処理経路",
            "subtitle": "NMS・短命削除は検出/トラック単位で、内部成分専用の除去ではありません。",
            "type": "bar",
            "intent": "funnel",
            "question": "raw複数前景成分検出のうち、何件が追跡・キーフレームへ残るか。",
            "rationale": "同じ母集団を段階別件数で並べ、後処理による減少と残存を直接比較する。",
            "combinationRationale": "色を代表runへ割り当て、入力ごとの後処理挙動を同一段階で比較する。",
            "dataset": "postprocess_flow",
            "sourceId": "postprocess_trace",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "stage", "type": "nominal", "label": "段階"},
                "y": {"field": "count", "type": "quantitative", "label": "件数"},
                "color": {"field": "run", "type": "nominal", "label": "run"},
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "title": "run"},
        },
    ]
    tables = [
        {
            "id": "run_table",
            "title": "全推論runの監査結果",
            "subtitle": "実測フレーム、検出、発生率、深刻例、実効推論速度。",
            "dataset": "per_run",
            "sourceId": "topology_summary",
            "defaultSort": {"field": "multi_rate_pct", "direction": "desc"},
            "columns": [
                {"field": "model", "label": "モデル", "type": "text"},
                {"field": "video", "label": "動画", "type": "text"},
                {"field": "frames", "label": "フレーム", "format": "number"},
                {"field": "detections", "label": "検出", "format": "number"},
                {"field": "multi_count", "label": "複数成分", "format": "number"},
                {"field": "multi_rate_pct", "label": "発生率 (%)", "format": "number"},
                {"field": "severe_secondary_ge_5pct", "label": "副成分>=5%", "format": "number"},
                {"field": "inference_fps", "label": "推論 FPS", "format": "number"},
            ],
        },
        {
            "id": "post_table",
            "title": "Production後処理の追跡結果",
            "subtitle": "同じsource_detection_idを最終geometryまで追跡。",
            "dataset": "postprocess_table",
            "sourceId": "postprocess_trace",
            "defaultSort": {"field": "raw_multi", "direction": "desc"},
            "columns": [
                {"field": "run", "label": "run", "type": "text"},
                {"field": "raw_multi", "label": "raw", "format": "number"},
                {"field": "retained", "label": "残存", "format": "number"},
                {"field": "nms_or_unassigned", "label": "NMS/未割当", "format": "number"},
                {"field": "short_track_removed", "label": "短命削除", "format": "number"},
                {"field": "exact_keyframes", "label": "厳密キー", "format": "number"},
                {"field": "keyframe_enrichment", "label": "キー濃縮倍率", "format": "number"},
                {"field": "tracks_with_varying_slots", "label": "slot数変動track", "format": "number"},
            ],
        },
    ]

    manifest_sources = [
        {"id": source["id"], "label": source["label"], "path": source["path"]}
        for source in (summary_source, post_source, validation_source)
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "V3/V3-liteの出力マスク輪郭トポロジーとProduction後処理での扱いを監査。",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": manifest_sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
                {"id": "summary_heading", "type": "markdown", "body": "## 技術サマリー"},
                {"id": "summary_finding", "type": "markdown", "body": headline, "sourceId": "topology_summary"},
                {"id": "metrics", "type": "metric-strip", "cardIds": ["lite_rate", "lite_coverage", "v3_rate", "retention_rate"]},
                {"id": "paired_heading", "type": "markdown", "body": "## モデル間比較"},
                {"id": "paired", "type": "chart", "chartId": "paired_rates"},
                {"id": "per_video", "type": "chart", "chartId": "lite_per_video"},
                {"id": "severity_heading", "type": "markdown", "body": "## 発生の深刻度"},
                {"id": "severity_chart", "type": "chart", "chartId": "severity"},
                {"id": "score_chart", "type": "chart", "chartId": "score_relationship"},
                {"id": "post_heading", "type": "markdown", "body": "## Production後処理での扱い"},
                {"id": "post_finding", "type": "markdown", "body": post_statement, "sourceId": "postprocess_trace"},
                {"id": "post_chart", "type": "chart", "chartId": "postprocess_flow"},
                {"id": "post_table_block", "type": "table", "tableId": "post_table"},
                {
                    "id": "mechanism",
                    "type": "markdown",
                    "body": (
                        "### 実装上の原因\n\n"
                        "- NMSはbbox/検出単位であり、1検出内の島を分離しません。\n"
                        "- 追跡も検出を1 track観測として扱うため、島は同じtrackへ入ります。\n"
                        "- polygon最適化は成分数が変わるとrunを分割しますが、最終segmentは必ずしも分割せず、可変slot数を保持します。\n"
                        "- 穴は中間表現でring roleを失い、最終geometryでは外部成分として書き出されます。"
                    ),
                    "sourceId": "postprocess_trace",
                },
                {"id": "validation_heading", "type": "markdown", "body": "## 妥当性・再現性"},
                {"id": "validation_finding", "type": "markdown", "body": validation_statement, "sourceId": "validation_evidence"},
                {"id": "run_table_block", "type": "table", "tableId": "run_table"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## スコープと定義\n\n"
                        "- **複数前景成分**: 輪郭包含木で偶数深度となる、互いに非接続な前景島が2個以上ある1検出。\n"
                        "- **穴**: 奇数深度の輪郭。複数前景成分とは別集計。\n"
                        "- V3-liteは全選択動画を全長推論。V3は10時間予算内で短尺を全長、長尺を先頭サンプル。\n"
                        "- 比較チャートは両モデルに存在する同じ先頭フレーム数へ限定。"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 制約\n\n"
                        "- 監査対象はSQLiteへ保存されたpolygonです。polygon化前の確率logitそのものではありません。\n"
                        "- polygon化には点数上限があり、極小島は保存前に落ちる可能性があるため、発生率は下限推定です。\n"
                        "- V3の長尺動画は時間予算による先頭区間サンプルで、全長母集団の推定ではありません。\n"
                        "- ローカル生成レビュー画像は人間確認用です。本監査では機密動画フレームをAIへ提示していません。"
                    ),
                },
                {
                    "id": "recommendations",
                    "type": "markdown",
                    "body": (
                        "## 推奨対応\n\n"
                        "1. polygon正規化時に包含関係を保持し、穴を意図的に塗り潰すかring roleとして明示的に伝播する。\n"
                        "2. optimizer前に、時間的に短命かつ小面積の副成分だけを対象にしたcomponent-levelフィルタを追加する。\n"
                        "3. 成分数が変わる区間ではslot identityを明示し、最終segment内でslotが入れ替わらない契約をテストする。\n"
                        "4. 大きい副成分は誤検出と決めつけず、局所レビューでNMS/分類/モデル側の原因を分離する。"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline_metrics": [
                    {"metric": "v3lite_multi_rate", "value": lite_total["multi_rate"]},
                    {"metric": "v3lite_detections", "value": lite_total["detections"]},
                    {"metric": "v3_multi_rate", "value": v3_total["multi_rate"]},
                    {
                        "metric": "postprocess_retention",
                        "value": sum(int(x["multi_disposition"]["retained"]) for x in postprocess)
                        / sum(int(x["multi_foreground_detections"]) for x in postprocess),
                    },
                ],
                "paired_rates": paired_rows,
                "v3lite_per_video": [row for row in per_run if row["model"] == "v3lite" and row["video"] != "heyzo_3545_30_45_duplicate"],
                "severity": severity,
                "score_bins": score_bins,
                "postprocess_flow": post_rows,
                "postprocess_table": post_table,
                "per_run": per_run,
            },
        },
        "sources": [summary_source, post_source, validation_source],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--postprocess-summary", type=Path, action="append", required=True)
    parser.add_argument("--postprocess-trace", type=Path, required=True)
    parser.add_argument("--repeatability", type=Path, required=True)
    parser.add_argument("--raster-validation", type=Path, required=True)
    parser.add_argument("--artifact-audit", type=Path, required=True)
    parser.add_argument("--postprocess-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_copy = args.output.parent / "summary.json"
    trace_copy = args.output.parent / "postprocess_trace.sqlite"
    if args.summary.resolve() != summary_copy.resolve():
        shutil.copy2(args.summary, summary_copy)
    if args.postprocess_trace.resolve() != trace_copy.resolve():
        shutil.copy2(args.postprocess_trace, trace_copy)
    (args.output.parent / "validation.json").write_text(
        json.dumps(
            {
                "artifact_audit": _read(args.artifact_audit),
                "postprocess_audit": _read(args.postprocess_audit),
                "raster_validation": _read(args.raster_validation),
                "repeatability": _read(args.repeatability),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
