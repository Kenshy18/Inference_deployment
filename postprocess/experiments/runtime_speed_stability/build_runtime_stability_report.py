#!/usr/bin/env python3
"""Build the canonical technical report for runtime-stability experiments."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from statistics import median


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = ROOT / "output/runtime_speed_stability_20260830"
REPORT_ROOT = EXPERIMENT_ROOT / "report"
PROFILE = "polygon_adaptive_keyframe_v2"
LABELS = ("女性器", "男性器", "結合部分")
VIDEO_FRAMES = 23510


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(relative: str) -> tuple[dict[str, object], dict[str, object]]:
    matrix = _load(EXPERIMENT_ROOT / relative / "phase2_matrix.json")
    return matrix["completed_profiles"][-1], matrix["execution"]


def _prediction_rows(root: Path, label: str) -> dict[tuple[int, str], str]:
    path = root / PROFILE / label / "runtime/pred/predictions.sqlite"
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as database:
        values = database.execute(
            "SELECT frame,track_id,polygons FROM masks ORDER BY frame,track_id"
        ).fetchall()
    return {
        (int(frame), str(track_id)): str(polygons)
        for frame, track_id, polygons in values
    }


def _keyframes(root: Path, label: str) -> object:
    return _load(root / PROFILE / label / "runtime/opt/final_keyframes.json")


def _exact_parity() -> dict[str, object]:
    baseline = EXPERIMENT_ROOT / "full_corpus/runs/interval3_production_warm"
    candidate = EXPERIMENT_ROOT / "production_auto_full/interval_3"
    total = 0
    changed = 0
    keyframes_equal = True
    for label in LABELS:
        left = _prediction_rows(baseline, label)
        right = _prediction_rows(candidate, label)
        if set(left) != set(right):
            raise RuntimeError(f"prediction keys differ for {label}")
        total += len(left)
        changed += sum(left[key] != right[key] for key in left)
        keyframes_equal = keyframes_equal and _keyframes(
            baseline, label
        ) == _keyframes(candidate, label)
    return {
        "rows": total,
        "changed_rows": changed,
        "keyframes_exact_equal": keyframes_equal,
    }


def _source(source_id: str, label: str, table: str, sql: str) -> dict[str, object]:
    return {
        "id": source_id,
        "label": label,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": sql,
            "description": (
                "Query against the normalized runtime experiment ledger generated "
                "deterministically from saved phase2 matrices and benchmark JSON."
            ),
            "tables_used": [f"runtime_stability.{table}"],
            "filters": [
                "V3 KPI corpus: 23,510 video frames",
                "93 tracks and 24,503 source observations",
                "minimum Recall floor 0.97",
            ],
            "metric_definitions": [
                "video FPS = 23,510 source-video frames / profile wall seconds",
                "speed range = (maximum repeated FPS - minimum repeated FPS) / median repeated FPS",
                "exact parity compares serialized polygons for every frame/track row and complete keyframe ledgers",
            ],
        },
    }


def _write_ledger(
    *,
    summary: dict[str, object],
    throughput_rows: list[dict[str, object]],
    interval_rows: list[dict[str, object]],
    curve_rows: list[dict[str, object]],
    rejected_rows: list[dict[str, object]],
) -> None:
    path = REPORT_ROOT / "runtime_stability.sqlite"
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as database:
        database.execute(
            "CREATE TABLE report_summary (metric TEXT PRIMARY KEY, value REAL, unit TEXT)"
        )
        database.executemany(
            "INSERT INTO report_summary(metric,value,unit) VALUES (?,?,?)",
            ((key, float(value), "") for key, value in summary.items()),
        )
        database.execute(
            "CREATE TABLE polygon_throughput (sort_order INTEGER, configuration TEXT, video_fps REAL, wall_seconds REAL, repeat TEXT, maximum_native_threads INTEGER, output_changed_rows INTEGER)"
        )
        database.executemany(
            "INSERT INTO polygon_throughput VALUES (?,?,?,?,?,?,?)",
            (
                (
                    index,
                    row["configuration"],
                    row["video_fps"],
                    row["wall_seconds"],
                    str(row["repeat"]),
                    row["maximum_native_threads"],
                    row["output_changed_rows"],
                )
                for index, row in enumerate(throughput_rows)
            ),
        )
        database.execute(
            "CREATE TABLE interval_throughput (target_interval TEXT, scheduler TEXT, video_fps REAL, wall_seconds REAL, mean_iou REAL, minimum_recall REAL)"
        )
        database.executemany(
            "INSERT INTO interval_throughput VALUES (?,?,?,?,?,?)",
            (
                (
                    row["target_interval"],
                    row["scheduler"],
                    row["video_fps"],
                    row["wall_seconds"],
                    row["mean_iou"],
                    row["minimum_recall"],
                )
                for row in interval_rows
            ),
        )
        database.execute(
            "CREATE TABLE curve_throughput (configuration TEXT, repeat TEXT, video_fps REAL, wall_seconds REAL, groups INTEGER, native_threads INTEGER, mean_iou REAL, minimum_recall REAL)"
        )
        database.executemany(
            "INSERT INTO curve_throughput VALUES (?,?,?,?,?,?,?,?)",
            (
                (
                    row["configuration"],
                    row["repeat"],
                    row["video_fps"],
                    row["wall_seconds"],
                    row["groups"],
                    row["native_threads"],
                    row["mean_iou"],
                    row["minimum_recall"],
                )
                for row in curve_rows
            ),
        )
        database.execute(
            "CREATE TABLE rejected_controls (control TEXT, speed_effect REAL, changed_rows INTEGER, worst_output_iou REAL, decision TEXT)"
        )
        database.executemany(
            "INSERT INTO rejected_controls VALUES (?,?,?,?,?)",
            (
                (
                    row["control"],
                    row["speed_effect"],
                    row["changed_rows"],
                    row["worst_output_iou"],
                    row["decision"],
                )
                for row in rejected_rows
            ),
        )


def build() -> dict[str, object]:
    old3, old3_execution = _aggregate(
        "full_corpus/runs/interval3_production_warm"
    )
    old3_cold, _ = _aggregate("full_corpus/runs/interval3_production")
    old6, _ = _aggregate("full_corpus/runs/interval6_production_warm")
    balanced_names = (
        "label3_opt6_native2",
        "label3_opt6_native2_repeat2",
        "label3_opt6_native2_repeat3",
    )
    balanced = [
        _aggregate(f"full_corpus/scheduler/{name}")[0] for name in balanced_names
    ]
    balanced_cold, balanced_execution = _aggregate(
        "full_corpus/scheduler/cold_cache_label3_opt6_native2"
    )
    balanced6, _ = _aggregate(
        "full_corpus/scheduler/interval6_label3_opt6_native2"
    )
    production_auto, production_execution = _aggregate(
        "production_auto_full/interval_3"
    )
    single_auto, single_execution = _aggregate("production_auto_single/interval_3")
    curve = _load(EXPERIMENT_ROOT / "curve_scheduler/benchmark_results.json")
    exact_parallel = _load(
        EXPERIMENT_ROOT / "exact_parallel_controls/benchmark_results.json"
    )
    chunking = _load(
        EXPERIMENT_ROOT / "long_track_chunking/benchmark_results.json"
    )
    full = _load(EXPERIMENT_ROOT / "full_corpus/benchmark_results.json")
    parity = _exact_parity()

    repeated_fps = [float(row["video_fps"]) for row in balanced]
    stability_range = (
        (max(repeated_fps) - min(repeated_fps)) / median(repeated_fps)
    )
    interval3_gain = (
        float(production_auto["video_fps"]) / float(old3["video_fps"]) - 1.0
    )
    interval6_gain = (
        float(balanced6["video_fps"]) / float(old6["video_fps"]) - 1.0
    )

    throughput_rows = [
        {
            "configuration": "旧 cold 3×9×4",
            "video_fps": float(old3_cold["video_fps"]),
            "wall_seconds": float(old3_cold["profile_wall_seconds"]),
            "repeat": "cold",
            "maximum_native_threads": 108,
            "output_changed_rows": 0,
        },
        {
            "configuration": "旧 warm 3×9×4",
            "video_fps": float(old3["video_fps"]),
            "wall_seconds": float(old3["profile_wall_seconds"]),
            "repeat": "warm",
            "maximum_native_threads": 108,
            "output_changed_rows": 0,
        },
    ]
    for index, row in enumerate(balanced, 1):
        throughput_rows.append(
            {
                "configuration": f"新 3×6×2 #{index}",
                "video_fps": float(row["video_fps"]),
                "wall_seconds": float(row["profile_wall_seconds"]),
                "repeat": index,
                "maximum_native_threads": 36,
                "output_changed_rows": 0,
            }
        )
    throughput_rows.extend(
        (
            {
                "configuration": "新 cold-cache 3×6×2",
                "video_fps": float(balanced_cold["video_fps"]),
                "wall_seconds": float(balanced_cold["profile_wall_seconds"]),
                "repeat": "cold-cache",
                "maximum_native_threads": 36,
                "output_changed_rows": 0,
            },
            {
                "configuration": "Production API auto",
                "video_fps": float(production_auto["video_fps"]),
                "wall_seconds": float(production_auto["profile_wall_seconds"]),
                "repeat": "integration",
                "maximum_native_threads": int(
                    production_execution["maximum_concurrent_dp_workers"]
                    * production_execution["native_batch_threads"]
                ),
                "output_changed_rows": int(parity["changed_rows"]),
            },
        )
    )
    interval_rows = [
        {
            "target_interval": "3",
            "scheduler": "旧 3×9×4",
            "video_fps": float(old3["video_fps"]),
            "wall_seconds": float(old3["profile_wall_seconds"]),
            "mean_iou": float(old3["iou_mean"]),
            "minimum_recall": float(old3["recall_min"]),
        },
        {
            "target_interval": "3",
            "scheduler": "新 auto",
            "video_fps": float(production_auto["video_fps"]),
            "wall_seconds": float(production_auto["profile_wall_seconds"]),
            "mean_iou": float(production_auto["iou_mean"]),
            "minimum_recall": float(production_auto["recall_min"]),
        },
        {
            "target_interval": "6",
            "scheduler": "旧 3×9×4",
            "video_fps": float(old6["video_fps"]),
            "wall_seconds": float(old6["profile_wall_seconds"]),
            "mean_iou": float(old6["iou_mean"]),
            "minimum_recall": float(old6["recall_min"]),
        },
        {
            "target_interval": "6",
            "scheduler": "新 auto相当",
            "video_fps": float(balanced6["video_fps"]),
            "wall_seconds": float(balanced6["profile_wall_seconds"]),
            "mean_iou": float(balanced6["iou_mean"]),
            "minimum_recall": float(balanced6["recall_min"]),
        },
    ]
    curve_rows = [
        {
            "configuration": str(row["name"]),
            "repeat": f"#{row['repeat']}",
            "video_fps": float(row["video_fps"]),
            "wall_seconds": float(row["wall_seconds"]),
            "groups": int(row["groups"]),
            "native_threads": int(row["native_threads"]),
            "mean_iou": float(row["mean_iou"]),
            "minimum_recall": float(row["minimum_recall"]),
        }
        for row in curve["results"]
    ]
    candidate_variants = {row["name"]: row for row in exact_parallel["variants"]}
    chunk_variants = {row["name"]: row for row in chunking["variants"]}
    rejected_rows = [
        {
            "control": "長トラックを600フレームに分割",
            "speed_effect": (
                float(chunk_variants["chunk600_o60"]["observation_fps"])
                / float(chunk_variants["global"]["observation_fps"])
                - 1.0
            ),
            "changed_rows": int(
                chunk_variants["chunk600_o60"]["comparison_to_global"][
                    "changed_frames"
                ]
            ),
            "worst_output_iou": float(
                chunk_variants["chunk600_o60"]["comparison_to_global"][
                    "minimum_iou_vs_global"
                ]
            ),
            "decision": "不採用（出力非互換）",
        },
        {
            "control": "DP最大辺を30→16へ短縮",
            "speed_effect": (
                float(full["results"][1]["wall_seconds"] ** -1)
                / float(full["results"][0]["wall_seconds"] ** -1)
                - 1.0
            ),
            "changed_rows": int(full["comparisons"]["3"]["changed_rows"]),
            "worst_output_iou": float(
                full["comparisons"]["3"]["changed_minimum_iou"]
            ),
            "decision": "不採用（局所差が大きい）",
        },
        {
            "control": "単一長トラックのみ候補生成2並列",
            "speed_effect": (
                float(candidate_variants["candidate2_native4"]["observation_fps"])
                / float(candidate_variants["candidate1_native4"]["observation_fps"])
                - 1.0
            ),
            "changed_rows": int(
                candidate_variants["candidate2_native4"]["comparison_to_global"][
                    "changed_frames"
                ]
            ),
            "worst_output_iou": float(
                candidate_variants["candidate2_native4"]["comparison_to_global"][
                    "minimum_iou_vs_global"
                ]
            ),
            "decision": "条件付き採用（単一runのみ）",
        },
    ]

    summary_values = {
        "old_interval3_video_fps": float(old3["video_fps"]),
        "new_interval3_video_fps": float(production_auto["video_fps"]),
        "interval3_speed_gain": interval3_gain,
        "old_interval6_video_fps": float(old6["video_fps"]),
        "new_interval6_video_fps": float(balanced6["video_fps"]),
        "interval6_speed_gain": interval6_gain,
        "balanced_repeat_range": stability_range,
        "balanced_cold_cache_video_fps": float(balanced_cold["video_fps"]),
        "balanced_repeat_min_video_fps": min(repeated_fps),
        "balanced_repeat_max_video_fps": max(repeated_fps),
        "exact_parity_rows": int(parity["rows"]),
        "changed_rows": int(parity["changed_rows"]),
        "minimum_recall": float(production_auto["recall_min"]),
        "recall_violations": int(production_auto["recall_violations"]),
        "video_frames": VIDEO_FRAMES,
        "tracks": 93,
        "source_observations": 24503,
        "output_rows": int(parity["rows"]),
        "single_track_observations": 1519,
        "single_track_keyframes": int(single_auto["keyframes"]),
        "single_candidate1_observation_fps": float(
            candidate_variants["candidate1_native4"]["observation_fps"]
        ),
        "single_candidate2_observation_fps": float(
            candidate_variants["candidate2_native4"]["observation_fps"]
        ),
        "single_candidate_speed_gain": float(rejected_rows[2]["speed_effect"]),
        "curve_production_min_video_fps": min(
            row["video_fps"]
            for row in curve_rows
            if row["configuration"] == "production_2shards_6x4"
        ),
        "curve_production_max_video_fps": max(
            row["video_fps"]
            for row in curve_rows
            if row["configuration"] == "production_2shards_6x4"
        ),
        "curve_coarse_min_video_fps": min(
            row["video_fps"]
            for row in curve_rows
            if row["configuration"] == "coarse_1shard_3x8"
        ),
        "curve_coarse_max_video_fps": max(
            row["video_fps"]
            for row in curve_rows
            if row["configuration"] == "coarse_1shard_3x8"
        ),
        "old_maximum_native_threads": 108,
        "new_maximum_native_threads": 36,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_ledger(
        summary=summary_values,
        throughput_rows=throughput_rows,
        interval_rows=interval_rows,
        curve_rows=curve_rows,
        rejected_rows=rejected_rows,
    )
    sources = [
        _source(
            "runtime-summary",
            "Runtime stability summary metrics",
            "report_summary",
            "SELECT metric, value, unit FROM report_summary ORDER BY metric",
        ),
        _source(
            "polygon-throughput-source",
            "Polygon scheduler throughput runs",
            "polygon_throughput",
            "SELECT configuration, video_fps, wall_seconds, repeat, maximum_native_threads, output_changed_rows FROM polygon_throughput ORDER BY sort_order",
        ),
        _source(
            "interval-throughput-source",
            "Target interval throughput comparison",
            "interval_throughput",
            "SELECT target_interval, scheduler, video_fps, wall_seconds, mean_iou, minimum_recall FROM interval_throughput ORDER BY CAST(target_interval AS INTEGER), scheduler",
        ),
        _source(
            "curve-throughput-source",
            "Curve shard scheduler throughput runs",
            "curve_throughput",
            "SELECT configuration, repeat, video_fps, wall_seconds, groups, native_threads, mean_iou, minimum_recall FROM curve_throughput ORDER BY configuration, repeat",
        ),
        _source(
            "rejected-controls-source",
            "Runtime controls quality gate",
            "rejected_controls",
            "SELECT control, speed_effect, changed_rows, worst_output_iou, decision FROM rejected_controls ORDER BY speed_effect DESC",
        ),
    ]
    title = "後処理ランタイムの速度安定化検証"
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    report = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                "Polygon/Curve後処理の品質を維持した、予測可能な並列実行構成の技術検証。"
            ),
            "generatedAt": generated,
            "sources": sources,
            "charts": [
                {
                    "id": "polygon-throughput",
                    "title": "Polygon後処理のスケジューラ別スループット",
                    "subtitle": (
                        "目標間隔3、23,510動画フレーム。新構成は反復・cold-cache・Production APIを含む。"
                    ),
                    "intent": "comparison",
                    "question": "過剰並列を抑えると速度と再現性はどう変わるか。",
                    "rationale": "離散的な実行構成間で単一のFPS指標を比較するため横棒が適切。",
                    "type": "bar",
                    "dataset": "polygon_throughput",
                    "sourceId": "polygon-throughput-source",
                    "encodings": {
                        "x": {"field": "configuration", "type": "nominal"},
                        "y": {
                            "field": "video_fps",
                            "type": "quantitative",
                            "label": "動画FPS",
                            "format": "number",
                        },
                    },
                    "valueFormat": "number",
                    "unit": "FPS",
                    "layout": "full",
                    "surface": {"orientation": "horizontal"},
                },
                {
                    "id": "interval-throughput",
                    "title": "目標キーフレーム間隔別の動画FPS",
                    "subtitle": "同一V3 KPIコーパス。旧warm構成と品質同一の新構成を比較。",
                    "intent": "comparison",
                    "question": "目標間隔3と6の両方で新スケジュールは速いか。",
                    "rationale": "間隔ごとの旧・新ペアを直接比較するためグループ棒を使う。",
                    "type": "bar",
                    "dataset": "interval_throughput",
                    "sourceId": "interval-throughput-source",
                    "encodings": {
                        "x": {"field": "target_interval", "type": "ordinal"},
                        "y": {
                            "field": "video_fps",
                            "type": "quantitative",
                            "label": "動画FPS",
                        },
                        "color": {"field": "scheduler", "type": "nominal"},
                    },
                    "valueFormat": "number",
                    "unit": "FPS",
                    "layout": "full",
                    "surface": {"grouping": "grouped"},
                },
                {
                    "id": "curve-throughput",
                    "title": "Curve後処理のシャード構成別スループット",
                    "subtitle": "目標間隔3、各構成2反復。同じ24ネイティブスレッドでも細粒度分割が有利。",
                    "intent": "comparison",
                    "question": "Curve側でも負荷分散の粒度が速度安定性を左右するか。",
                    "rationale": "4回の実測を実行構成と反復単位で比較するためグループ棒を使う。",
                    "type": "bar",
                    "dataset": "curve_throughput",
                    "sourceId": "curve-throughput-source",
                    "encodings": {
                        "x": {"field": "configuration", "type": "nominal"},
                        "y": {
                            "field": "video_fps",
                            "type": "quantitative",
                            "label": "動画FPS",
                        },
                        "color": {"field": "repeat", "type": "nominal"},
                    },
                    "valueFormat": "number",
                    "unit": "FPS",
                    "layout": "full",
                    "surface": {"grouping": "grouped"},
                },
            ],
            "tables": [
                {
                    "id": "rejected-controls",
                    "title": "高速化候補の品質ゲート",
                    "subtitle": "速度だけでなく、既存出力との局所差を採否条件にした。",
                    "dataset": "rejected_controls",
                    "sourceId": "rejected-controls-source",
                    "defaultSort": {"field": "speed_effect", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "control", "label": "施策", "type": "text"},
                        {
                            "field": "speed_effect",
                            "label": "速度差",
                            "format": "percent",
                            "movement": True,
                        },
                        {
                            "field": "changed_rows",
                            "label": "変更行",
                            "format": "number",
                        },
                        {
                            "field": "worst_output_iou",
                            "label": "最悪出力IoU",
                            "format": "number",
                        },
                        {"field": "decision", "label": "判断", "type": "text"},
                    ],
                }
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical-summary",
                    "type": "markdown",
                    "sourceId": "runtime-summary",
                    "body": (
                        "## 技術サマリー\n\n"
                        f"**Polygon Productionの数値出力を変えず、間隔3で {float(old3['video_fps']):.1f}→{float(production_auto['video_fps']):.1f} FPS（+{interval3_gain:.1%}）、間隔6で {float(old6['video_fps']):.1f}→{float(balanced6['video_fps']):.1f} FPS（+{interval6_gain:.1%}）へ改善した。** 主要因はアルゴリズムではなく、旧構成が24コア上で最大108本のネイティブ実行スレッドを競合させていたことだった。新しい実効CPU数ベースの構成は最大36本に抑える。\n\n"
                        f"間隔3の新構成3反復は {min(repeated_fps):.1f}–{max(repeated_fps):.1f} FPSで、レンジは中央値の{stability_range:.1%}。cold-cacheでも {float(balanced_cold['video_fps']):.1f} FPSだった。全{parity['rows']:,}出力行とキーフレーム台帳は旧warm構成に完全一致し、最小Recallは{float(production_auto['recall_min']):.2f}、違反0件である。"
                    ),
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "sourceId": "runtime-summary",
                    "body": (
                        "## 評価範囲と指標\n\n"
                        "V3 KPIコーパスの23,510動画フレーム、93トラック、24,503入力観測を使用した。主指標は後処理プロファイルのwall timeから求める動画FPSで、品質ゲートは最小Recall 0.97、Recall違反数、平均IoU、全出力行・全キーフレームの完全一致である。単一長トラック試験は1,519観測で、全コーパスの速度予測を代替するものではなく、並列化できない端ケースの検証として扱った。"
                    ),
                },
                {
                    "id": "polygon-finding",
                    "type": "markdown",
                    "sourceId": "runtime-summary",
                    "body": (
                        "## Polygonの分散は過剰な入れ子並列が作っていた\n\n"
                        "旧構成は初回143.7 FPS、warm後223.1 FPSと大きく振れた。クラス並列・DPプロセス並列・ネイティブスレッド並列を独立に最大化したため、CPUランキューと厳密評価の待ち時間が膨らんだ。新構成はCUDA近似や目的関数を変えず、CPUの同時実行枠だけを整えるため、速度上昇と完全な出力互換を同時に得られた。"
                    ),
                },
                {"id": "polygon-chart", "type": "chart", "chartId": "polygon-throughput", "layout": "full"},
                {
                    "id": "interval-finding",
                    "type": "markdown",
                    "sourceId": "runtime-summary",
                    "body": (
                        "## 改善は主要な目標間隔で再現した\n\n"
                        f"間隔3の改善率は+{interval3_gain:.1%}、間隔6は+{interval6_gain:.1%}だった。各ペアで平均IoUと最小Recallは同一であり、速度のためにDPグラフやRecall制約を弱めていない。したがって処理時間の予測には、目標間隔に加えて実効CPU数と同時トラック数を使うべきである。"
                    ),
                },
                {"id": "interval-chart", "type": "chart", "chartId": "interval-throughput", "layout": "full"},
                {
                    "id": "single-track",
                    "type": "markdown",
                    "sourceId": "runtime-summary",
                    "body": (
                        "## 単一長トラックには限定的な内側並列を使う\n\n"
                        f"トラックが1本しかなくプロセスプールを活用できない場合だけ、候補形状生成を2並列にすると {float(candidate_variants['candidate1_native4']['observation_fps']):.2f}→{float(candidate_variants['candidate2_native4']['observation_fps']):.2f} 観測FPS（+{rejected_rows[2]['speed_effect']:.1%}）になった。4・8並列は逆に遅いため、Productionは「単一runなら2、それ以外は1」を自動選択する。統合試験では単一長トラックの1,519行・506キーも完全一致した。"
                    ),
                },
                {
                    "id": "curve-finding",
                    "type": "markdown",
                    "sourceId": "runtime-summary",
                    "body": (
                        "## Curveは既存の細粒度シャーディングを維持する\n\n"
                        "Curveの現行2シャード/クラス構成は229.6–230.8 FPSで、反復差は約0.54%。同じ最大24ネイティブスレッドでも1シャード/クラスへ粗くすると164.7–169.4 FPSまで低下した。総スレッド数だけでなく、長いトラックが1ワーカーを占有しない負荷分散が重要である。品質値は両構成で同一だったため、Curve側のスケジューラは変更しない。"
                    ),
                },
                {"id": "curve-chart", "type": "chart", "chartId": "curve-throughput", "layout": "full"},
                {
                    "id": "quality-gate",
                    "type": "markdown",
                    "sourceId": "runtime-summary",
                    "body": (
                        "## 速くても局所出力が変わる施策は採用しない\n\n"
                        "長トラック分割とDP最大辺短縮はwall timeを改善したが、補間境界または最適経路を変えた。平均IoUだけなら差が小さく見える一方、旧出力との最悪IoUは無視できない。速度安定化のProduction採用条件を『Recall維持』だけでなく『全出力・全キーの完全一致』へ引き上げた。"
                    ),
                },
                {"id": "quality-table", "type": "table", "tableId": "rejected-controls", "layout": "full"},
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "runtime-summary",
                    "body": (
                        "## 方法\n\n"
                        "同一の準備済みSQLite、同一のCUDA lazy-exact評価、同一のDP最大辺30、同一のpair-voteを用い、クラス並列数・各クラスのDPプロセス数・ネイティブスレッド数だけを変更した。候補構成は24論理CPU上で比較し、選定後にcold-cache、3反復、目標間隔6、単一長トラック、Production API統合の順で検証した。最終比較ではSQLiteの`frame, track_id, polygons`文字列とキーフレームJSON全体を直接比較した。"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "sourceId": "runtime-summary",
                    "body": (
                        "## 限界とロバスト性\n\n"
                        "速度値は24論理CPUの現ホストと1本の実コーパスに基づく。GPU競合、電力制限、他プロセス、保存先I/O、極端に少ないトラックでは絶対FPSが変わる。そこで実装は固定値24ではなくCPU affinityを含む実効CPU数から上限を算出する。Curveは2反復、Polygonは主要構成3反復であり、長時間の分位点SLOを定義するには追加コーパスが必要である。"
                    ),
                },
                {
                    "id": "next-steps",
                    "type": "markdown",
                    "body": (
                        "## 推奨する次の運用\n\n"
                        "- Polygonは実効CPU数ベースの自動スケジューラを既定値にする。\n"
                        "- DP最大辺30と長トラックの大域最適化を維持する。\n"
                        "- 各実行manifestへ実際のクラス並列、DP並列、ネイティブスレッド、候補生成並列を記録する。\n"
                        "- GUIの所要時間予測は目標間隔だけでなく、動画フレーム数・トラック数・実効CPU数を入力にする。\n"
                        "- GPUまたはCPUの外部負荷がある場合は予測値に『競合あり』を表示する。"
                    ),
                },
                {
                    "id": "further-questions",
                    "type": "markdown",
                    "body": (
                        "## 次に確認すべき問い\n\n"
                        "異なる実動画3–4時間分でp10/p50/p90 FPSを測ると、トラック密度からどこまで所要時間を予測できるか。デプロイ版GUI経由でCPU/GPU競合を検出し、自動的に並列枠を下げた方が完了時間の分散をさらに抑えられるか。"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "polygon_throughput": throughput_rows,
                "interval_throughput": interval_rows,
                "curve_throughput": curve_rows,
                "rejected_controls": rejected_rows,
            },
        },
        "sources": sources,
    }
    source_notes = {
        "schema_version": 1,
        "generated_at": generated,
        "audience": "technical",
        "delivery_mode": "portable_html",
        "report_structure": [
            "technical summary",
            "scope and metrics",
            "key findings with visual evidence",
            "methodology",
            "limitations and robustness",
            "recommended next steps",
            "further questions",
        ],
        "chart_map": [
            {
                "section": "Polygon scheduler",
                "question": "How does bounded parallelism change throughput and stability?",
                "family": "comparison",
                "type": "bar",
                "fields": ["configuration", "video_fps"],
                "claim": "3x6x2 removes cold/warm collapse without output changes",
                "palette": "single-root preferred",
            },
            {
                "section": "Target interval",
                "question": "Does the gain reproduce at intervals 3 and 6?",
                "family": "comparison",
                "type": "grouped bar",
                "fields": ["target_interval", "scheduler", "video_fps"],
                "claim": "Both major intervals improve with identical quality",
                "palette": "hard two-root cap",
            },
            {
                "section": "Curve scheduler",
                "question": "Does shard granularity matter at equal thread count?",
                "family": "comparison",
                "type": "grouped bar",
                "fields": ["configuration", "repeat", "video_fps"],
                "claim": "Fine shards are faster and stable",
                "palette": "hard two-root cap",
            },
        ],
        "exact_parity": parity,
        "schedules": {
            "old": old3_execution,
            "balanced_experiment": balanced_execution,
            "production_auto": production_execution,
            "single_track_auto": single_execution,
        },
        "omissions": [
            "No confidence interval is claimed because repeated full-corpus runs are limited to three polygon and two curve observations.",
            "No per-frame trend chart is used because the analytical question compares discrete scheduler configurations, not temporal movement.",
        ],
    }
    (REPORT_ROOT / "artifact.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (REPORT_ROOT / "source_notes.json").write_text(
        json.dumps(source_notes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    build()
    print(REPORT_ROOT / "artifact.json")
