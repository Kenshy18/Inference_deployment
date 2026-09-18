#!/usr/bin/env python3
"""Build a portable report for the diverse Production runtime validation."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = ROOT / "output/runtime_speed_diverse_validation_20260831"
RESULTS_PATH = EXPERIMENT_ROOT / "benchmark_results.json"
REPEAT_PATH = EXPERIMENT_ROOT / "repeat_comparison.json"
REPORT_ROOT = EXPERIMENT_ROOT / "report"

DISPLAY_NAMES = {
    "kpi_full_multiclass": "KPI全編・多トラック",
    "simple_3min_long_tracks": "3分・少数長トラック",
    "kpi_excerpt_many_tracks": "KPI抜粋・多数短トラック",
    "release_qa_sparse_with_cuts": "短尺・疎・カットあり",
    "single_large_long_track": "単一・長大・20頂点",
}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(source_id: str, label: str, table: str, sql: str) -> dict[str, object]:
    return {
        "id": source_id,
        "label": label,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": sql,
            "description": (
                "Query against the normalized diverse-runtime validation ledger, "
                "generated deterministically from saved Production benchmark JSON."
            ),
            "tables_used": [f"runtime_diverse.{table}"],
            "filters": [
                "Production polygon runtime",
                "target keyframe interval 3",
                "minimum Recall floor 0.97",
                "serial benchmark execution without a competing GPU workload",
            ],
            "metric_definitions": [
                "timeline FPS = source video frames / polygon optimizer wall seconds",
                "observation FPS = mask observations / polygon optimizer wall seconds",
                "stream supply = min(independent runs, worker slots) / worker slots",
                "repeat range = (maximum FPS - minimum FPS) / median FPS",
            ],
        },
    }


def _write_ledger(
    workloads: list[dict[str, object]],
    repeats: list[dict[str, object]],
    validation: list[dict[str, object]],
) -> None:
    ledger = REPORT_ROOT / "runtime_diverse_validation.sqlite"
    if ledger.exists():
        ledger.unlink()
    with sqlite3.connect(ledger) as database:
        database.execute(
            """
            CREATE TABLE workload_results (
                sort_order INTEGER,
                dataset TEXT,
                workload TEXT,
                stratum TEXT,
                video_frames INTEGER,
                observations INTEGER,
                tracks INTEGER,
                active_labels INTEGER,
                longest_track INTEGER,
                independent_runs INTEGER,
                worker_slots INTEGER,
                stream_supply REAL,
                vertex20_share REAL,
                optimizer_seconds REAL,
                timeline_fps REAL,
                observation_fps REAL,
                end_to_end_polygon_fps REAL,
                evaluation_mfps REAL,
                evaluation_work_per_video_frame REAL,
                peak_rss_mib REAL,
                actual_interval REAL,
                mean_iou REAL,
                p01_iou REAL,
                minimum_iou REAL,
                minimum_recall REAL,
                recall_violations INTEGER,
                area_ratio_p95 REAL,
                area_ratio_max REAL
            )
            """
        )
        database.executemany(
            "INSERT INTO workload_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                (
                    index,
                    row["dataset"],
                    row["workload"],
                    row["stratum"],
                    row["video_frames"],
                    row["observations"],
                    row["tracks"],
                    row["active_labels"],
                    row["longest_track"],
                    row["independent_runs"],
                    row["worker_slots"],
                    row["stream_supply"],
                    row["vertex20_share"],
                    row["optimizer_seconds"],
                    row["timeline_fps"],
                    row["observation_fps"],
                    row["end_to_end_polygon_fps"],
                    row["evaluation_mfps"],
                    row["evaluation_work_per_video_frame"],
                    row["peak_rss_mib"],
                    row["actual_interval"],
                    row["mean_iou"],
                    row["p01_iou"],
                    row["minimum_iou"],
                    row["minimum_recall"],
                    row["recall_violations"],
                    row["area_ratio_p95"],
                    row["area_ratio_max"],
                )
                for index, row in enumerate(workloads)
            ),
        )
        database.execute(
            """
            CREATE TABLE repeatability (
                dataset TEXT,
                workload TEXT,
                fps_repeat1 REAL,
                fps_repeat2 REAL,
                fps_median REAL,
                fps_range_over_median REAL,
                compared_rows INTEGER,
                changed_rows INTEGER,
                keyframes_exact_equal INTEGER,
                minimum_recall_repeat1 REAL,
                minimum_recall_repeat2 REAL
            )
            """
        )
        database.executemany(
            "INSERT INTO repeatability VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                (
                    row["dataset"],
                    row["workload"],
                    row["fps_repeat1"],
                    row["fps_repeat2"],
                    row["fps_median"],
                    row["fps_range_over_median"],
                    row["rows"],
                    row["changed_rows"],
                    int(bool(row["keyframes_exact_equal"])),
                    row["minimum_recall_repeat1"],
                    row["minimum_recall_repeat2"],
                )
                for row in repeats
            ),
        )
        database.execute(
            """
            CREATE TABLE robustness_checks (
                input_kind TEXT,
                status TEXT,
                stage TEXT,
                evidence TEXT,
                interpretation TEXT
            )
            """
        )
        database.executemany(
            "INSERT INTO robustness_checks VALUES (?,?,?,?,?)",
            (
                (
                    row["input_kind"],
                    row["status"],
                    row["stage"],
                    row["evidence"],
                    row["interpretation"],
                )
                for row in validation
            ),
        )


def build() -> dict[str, object]:
    raw_results = _load(RESULTS_PATH)["results"]
    raw_repeats = _load(REPEAT_PATH)["comparisons"]
    workloads: list[dict[str, object]] = []
    for result in raw_results:
        quality = result["quality"]
        characteristics = result["source_characteristics"]
        execution = result["execution"]
        vertex_rows = result["vertex_policy_summary"]["track_rows_by_vertices"]
        runs = int(quality["run_count"])
        slots = int(execution["maximum_concurrent_dp_workers"])
        evaluation_frames = int(quality["interval_eval_frames"])
        observations = int(characteristics["observation_rows"])
        workloads.append(
            {
                "dataset": result["dataset"],
                "workload": DISPLAY_NAMES[result["dataset"]],
                "stratum": result["stratum"],
                "video_frames": int(result["video_frames"]),
                "observations": observations,
                "tracks": int(characteristics["tracks"]),
                "active_labels": int(characteristics["active_labels"]),
                "longest_track": int(characteristics["track_length_max"]),
                "independent_runs": runs,
                "worker_slots": slots,
                "stream_supply": min(runs, slots) / slots,
                "vertex20_share": int(vertex_rows.get("20", 0)) / observations,
                "optimizer_seconds": float(result["optimizer_wall_seconds"]),
                "timeline_fps": float(result["timeline_fps"]),
                "observation_fps": float(result["observation_fps"]),
                "end_to_end_polygon_fps": float(result["end_to_end_timeline_fps"]),
                "evaluation_mfps": evaluation_frames
                / float(result["optimizer_wall_seconds"])
                / 1_000_000,
                "evaluation_work_per_video_frame": evaluation_frames
                / int(result["video_frames"]),
                "peak_rss_mib": float(result["peak_child_rss_mib"]),
                "actual_interval": float(result["actual_mean_interval"]),
                "mean_iou": float(quality["mean_iou"]),
                "p01_iou": float(quality["p01_iou"]),
                "minimum_iou": float(quality["minimum_iou"]),
                "minimum_recall": float(quality["minimum_recall"]),
                "recall_violations": int(quality["recall_violations"]),
                "area_ratio_p95": float(quality["area_ratio_p95"]),
                "area_ratio_max": float(quality["area_ratio_max"]),
            }
        )
    repeats = [
        {**row, "workload": DISPLAY_NAMES[row["dataset"]]} for row in raw_repeats
    ]
    validation = [
        {
            "input_kind": "現行Production形式の1080p tracked入力（5種類）",
            "status": "PASS",
            "stage": "polygon optimizer",
            "evidence": "5/5完走、Recall違反0件",
            "interpretation": "評価対象の現行形式では完走性とRecall制約を確認",
        },
        {
            "input_kind": "2026-08-04旧720p tracked中間物を直接流用",
            "status": "FAIL",
            "stage": "topology guard",
            "evidence": "stream 3:run7:instance, frame 0でinvalid selected endpoint",
            "interpretation": (
                "旧中間SQLiteを現行追跡/NMSを通さず直接入力する互換経路は未保証。"
                "現行raw→canonical 1080p経路の720p対応不良を意味しない"
            ),
        },
    ]
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_ledger(workloads, repeats, validation)

    total_frames = sum(row["video_frames"] for row in workloads)
    total_observations = sum(row["observations"] for row in workloads)
    total_wall = sum(row["optimizer_seconds"] for row in workloads)
    aggregate_fps = total_frames / total_wall
    fps_values = [row["timeline_fps"] for row in workloads]
    repeat_ranges = [row["fps_range_over_median"] for row in repeats]
    changed_rows = sum(row["changed_rows"] for row in repeats)
    compared_rows = sum(row["rows"] for row in repeats)

    sources = [
        _source(
            "workloads-source",
            "Diverse Production polygon workloads",
            "workload_results",
            "SELECT * FROM workload_results ORDER BY sort_order",
        ),
        _source(
            "repeatability-source",
            "Repeated-run timing and exact output comparison",
            "repeatability",
            "SELECT * FROM repeatability ORDER BY fps_median DESC",
        ),
        _source(
            "robustness-source",
            "Input compatibility and completion checks",
            "robustness_checks",
            "SELECT * FROM robustness_checks ORDER BY status DESC",
        ),
    ]
    title = "多様なデータにおける後処理速度安定性の検証"
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    report = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Production Polygon後処理の入力構造別FPS・品質・再現性検証。",
            "generatedAt": generated,
            "sources": sources,
            "charts": [
                {
                    "id": "throughput-by-workload",
                    "title": "入力構造別の後処理FPS",
                    "subtitle": "目標間隔3。240 FPSは多トラック時に到達するが、全入力の固定値ではない。",
                    "intent": "comparison",
                    "question": "入力構造が変わるとProduction後処理の速度はどこまで変わるか。",
                    "rationale": "5種類の離散ワークロードを同一指標で比較するため横棒を使う。",
                    "type": "bar",
                    "dataset": "workload_results",
                    "sourceId": "workloads-source",
                    "encodings": {
                        "x": {"field": "workload", "type": "nominal"},
                        "y": {
                            "field": "timeline_fps",
                            "type": "quantitative",
                            "label": "動画FPS",
                        },
                    },
                    "valueFormat": "number",
                    "unit": "FPS",
                    "layout": "full",
                    "surface": {"orientation": "horizontal"},
                },
                {
                    "id": "parallel-supply",
                    "title": "独立run供給率と評価スループット",
                    "subtitle": "供給率=使用可能な独立run数÷ワーカー枠（上限100%）。少数長トラックは枠を埋められない。",
                    "intent": "relationship",
                    "question": "速度差は評価量より、並列枠を埋める独立run供給に対応するか。",
                    "rationale": "並列供給率と実評価速度の対応を見るため散布図を使う。",
                    "type": "scatter",
                    "dataset": "workload_results",
                    "sourceId": "workloads-source",
                    "encodings": {
                        "x": {
                            "field": "stream_supply",
                            "type": "quantitative",
                            "label": "独立run供給率",
                            "format": "percent",
                        },
                        "y": {
                            "field": "evaluation_mfps",
                            "type": "quantitative",
                            "label": "区間評価速度",
                        },
                        "color": {"field": "workload", "type": "nominal"},
                    },
                    "valueFormat": "number",
                    "unit": "百万評価フレーム/秒",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "workload-table",
                    "title": "ワークロード別の速度・品質・負荷",
                    "subtitle": "FPSだけでなく、独立run数、頂点、Recall、IoU、面積膨張を併記。",
                    "dataset": "workload_results",
                    "sourceId": "workloads-source",
                    "defaultSort": {"field": "timeline_fps", "direction": "desc"},
                    "density": "compact",
                    "layout": "full",
                    "columns": [
                        {"field": "workload", "label": "入力", "type": "text"},
                        {"field": "timeline_fps", "label": "FPS", "format": "number"},
                        {"field": "tracks", "label": "トラック", "format": "number"},
                        {"field": "longest_track", "label": "最長", "format": "number"},
                        {"field": "stream_supply", "label": "run供給", "format": "percent"},
                        {"field": "vertex20_share", "label": "20頂点行", "format": "percent"},
                        {"field": "actual_interval", "label": "実効間隔", "format": "number"},
                        {"field": "mean_iou", "label": "平均IoU", "format": "number"},
                        {"field": "p01_iou", "label": "p01 IoU", "format": "number"},
                        {"field": "minimum_iou", "label": "最小IoU", "format": "number"},
                        {"field": "minimum_recall", "label": "最小Recall", "format": "number"},
                        {"field": "area_ratio_p95", "label": "面積p95", "format": "number"},
                        {"field": "peak_rss_mib", "label": "最大RSS MiB", "format": "number"},
                    ],
                },
                {
                    "id": "repeat-table",
                    "title": "同一入力の再実行再現性",
                    "subtitle": "速度差と、全出力行・キーフレームの完全一致を確認。",
                    "dataset": "repeatability",
                    "sourceId": "repeatability-source",
                    "defaultSort": {"field": "fps_range_over_median", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "workload", "label": "入力", "type": "text"},
                        {"field": "fps_repeat1", "label": "FPS #1", "format": "number"},
                        {"field": "fps_repeat2", "label": "FPS #2", "format": "number"},
                        {"field": "fps_range_over_median", "label": "レンジ/中央値", "format": "percent"},
                        {"field": "changed_rows", "label": "変更行", "format": "number"},
                        {"field": "keyframes_exact_equal", "label": "キー一致", "format": "number"},
                    ],
                },
                {
                    "id": "robustness-table",
                    "title": "完走性と入力互換性",
                    "subtitle": "旧中間生成物を直接投入する経路は、現行のraw入力経路と分けて評価。",
                    "dataset": "robustness_checks",
                    "sourceId": "robustness-source",
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "input_kind", "label": "入力経路", "type": "text"},
                        {"field": "status", "label": "結果", "type": "text"},
                        {"field": "stage", "label": "段階", "type": "text"},
                        {"field": "evidence", "label": "証拠", "type": "text"},
                        {"field": "interpretation", "label": "解釈", "type": "text"},
                    ],
                },
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical-summary",
                    "type": "markdown",
                    "sourceId": "workloads-source",
                    "body": (
                        "## 技術サマリー\n\n"
                        f"**240 FPSは固定値ではない。** 現行Productionを5種類・計{total_frames:,}動画フレーム、"
                        f"{total_observations:,}マスク観測で測ると、後処理は{min(fps_values):.1f}～{max(fps_values):.1f} FPS、"
                        f"直列合算では{aggregate_fps:.1f} FPSだった。多トラックで並列枠を満たす2件は"
                        f"243.1/288.5 FPSだが、単一長大トラックは32.6 FPSである。\n\n"
                        "品質面では5件すべて最小Recall 0.97以上、違反0件、目標間隔3に対する実効間隔は"
                        "2.99～3.04だった。ただしKPI全編に最小IoU 0.289・最大面積比3.435の局所外れ値があり、"
                        "速度検証の成功を局所品質の完全保証とは扱わない。"
                    ),
                },
                {
                    "id": "key-findings",
                    "type": "markdown",
                    "sourceId": "workloads-source",
                    "body": (
                        "## 主要所見\n\n"
                        "速度差の主因は、独立して処理できるトラック/runがワーカー枠を埋められるかである。"
                        "多トラック2件はrun供給率100%で区間評価速度6.10～6.44百万評価フレーム/秒、"
                        "単一トラックは供給率11.1%で0.96百万/秒だった。一方、1動画フレーム当たり評価量は"
                        "約1.75万～2.94万（1.68倍差）に対し、評価速度は6.72倍差であり、評価量だけでは"
                        "FPSレンジを説明できない。20頂点は負荷要因だが、多トラックなら20頂点行19%を含んでも"
                        "243 FPSなので、頂点数単独が原因ではない。"
                    ),
                },
                {"id": "throughput-chart-block", "type": "chart", "chartId": "throughput-by-workload", "layout": "full"},
                {"id": "supply-chart-block", "type": "chart", "chartId": "parallel-supply", "layout": "full"},
                {"id": "workload-table-block", "type": "table", "tableId": "workload-table", "layout": "full"},
                {
                    "id": "repeatability",
                    "type": "markdown",
                    "sourceId": "repeatability-source",
                    "body": (
                        "## 同一入力では安定し、出力も完全一致\n\n"
                        f"4入力の再実行レンジは中央値比{min(repeat_ranges):.1%}～{max(repeat_ranges):.1%}。"
                        f"比較した{compared_rows:,}出力行は変更{changed_rows}行で、全ケースのキーフレームも完全一致した。"
                        "したがって現在の問題はランダムな出力や大きな実行時揺れではなく、入力ごとに利用可能な"
                        "並列性が違うことによる予測不能さである。"
                    ),
                },
                {"id": "repeat-table-block", "type": "table", "tableId": "repeat-table", "layout": "full"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "sourceId": "workloads-source",
                    "body": (
                        "## 範囲・データ・指標\n\n"
                        "目標キーフレーム間隔3、最小Recall 0.97、Production Polygonの同一設定で、"
                        "全編多クラス、多数短トラック、少数長トラック、カットを含む疎な短尺、単一長大20頂点"
                        "という5層を直列実行した。timeline FPSは動画総フレーム÷optimizer wall timeで、"
                        "NMS・追跡・最終Windows同期は含まない。end-to-end polygon FPSだけが候補準備を含む。"
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "workloads-source",
                    "body": (
                        "## 方法\n\n"
                        "コミット78f2d0cのProduction Python 3.10環境を使用し、GPU競合がない状態で各データを"
                        "別プロセスとして実行した。入力特性、実行スケジュール、区間評価数、stage work time、"
                        "RSS、最終SQLite品質を保存した。4データは同条件で再実行し、SQLiteの全polygon行と"
                        "キーフレームJSONを完全比較した。"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "sourceId": "robustness-source",
                    "body": (
                        "## 限界とロバスト性\n\n"
                        "有効な実データは5種類で、統計モデルによる一般化保証には不足する。今回のFPSはPolygon"
                        "optimizer段階であり、GUI完了時間や推論FPSではない。旧2026-08-04の720p tracked中間物を"
                        "直接現行optimizerへ入れる互換試験はtopology guardで停止した。この入力は現行NMS/追跡を"
                        "通っていないため、通常の720p raw→canonical 1080p経路の障害とは断定できないが、"
                        "旧中間SQLiteの直接再利用は未保証である。"
                    ),
                },
                {"id": "robustness-table-block", "type": "table", "tableId": "robustness-table", "layout": "full"},
                {
                    "id": "next-steps",
                    "type": "markdown",
                    "body": (
                        "## 推奨する次の対応\n\n"
                        "- GUIや見積りでは240 FPSを固定値にせず、独立run数・最長トラック・頂点構成から速度帯を表示する。\n"
                        "- exact出力を保ったまま、クラス別プールを跨ぐグローバルwork queueで少数クラス時の空き枠を減らす。\n"
                        "- 単一長大トラックは現状の最悪速度層として別SLOにし、厳密同値を確認できる内側並列だけを検討する。\n"
                        "- KPI全編の最小IoU/最大膨張外れ値を目視レビューし、速度改善とは別の品質課題として扱う。\n"
                        "- 現行raw入力から作った720p/1080p/4K短尺で、前処理からの解像度互換を改めて確認する。"
                    ),
                },
                {
                    "id": "further-questions",
                    "type": "markdown",
                    "body": (
                        "## 次に答えるべき問い\n\n"
                        "グローバルwork queueで少数長トラック層をどこまで引き上げられるか。"
                        "単一トラックの完全同値な区間評価並列は可能か。"
                        "raw入力から得た追加動画で観測された3速度帯が再現するか。"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "workload_results": workloads,
                "repeatability": repeats,
                "robustness_checks": validation,
            },
        },
        "sources": sources,
    }
    notes = {
        "schema_version": 1,
        "generated_at": generated,
        "audience": "technical",
        "delivery_mode": "portable_html",
        "inputs": [str(RESULTS_PATH), str(REPEAT_PATH)],
        "coverage": {
            "valid_workloads": len(workloads),
            "video_frames": total_frames,
            "observations": total_observations,
            "target_interval": 3,
        },
        "omissions": [
            "No causal regression is reported because five valid workloads are insufficient for stable coefficient estimates.",
            "No universal FPS guarantee is inferred from workload-stratified observations.",
            "Curve mode and end-to-end inference/GUI completion are outside this benchmark.",
        ],
    }
    (REPORT_ROOT / "artifact.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (REPORT_ROOT / "source_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    build()
    print(REPORT_ROOT / "artifact.json")
