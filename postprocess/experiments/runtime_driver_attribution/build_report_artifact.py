#!/usr/bin/env python3
"""Build the canonical portable-report artifact for runtime attribution."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any


LABELS = {
    "polygon": "ポリゴン",
    "catmull_rom": "Catmull–Rom曲線",
    "track_count": "トラック構造",
    "area_factor": "マスク面積",
    "vertices": "頂点数",
    "track_x_area": "トラック×面積",
    "track_x_vertices": "トラック×頂点",
    "area_x_vertices": "面積×頂点",
    "track_x_area_x_vertices": "3要因交互作用",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-notes", type=Path, required=True)
    parser.add_argument("--evidence-db", type=Path, required=True)
    return parser


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def main() -> None:
    args = _parser().parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    polygon = analysis["geometries"]["polygon"]
    curve = analysis["geometries"]["catmull_rom"]
    ps = polygon["shapley_wall_time_gap"]
    cs = curve["shapley_wall_time_gap"]
    pr = polygon["response_curves"]
    cr = curve["response_curves"]

    contributions = []
    for geometry, values in (("polygon", ps), ("catmull_rom", cs)):
        for driver, share in values["contribution_percent"].items():
            contributions.append(
                {
                    "geometry": LABELS[geometry],
                    "driver": LABELS[driver],
                    "share": share / 100.0,
                    "share_percent": share,
                    "seconds": values["contribution_seconds"][driver],
                }
            )

    response_datasets: dict[str, list[dict[str, Any]]] = {}
    for series, factor in (
        ("track_response", "track_count"),
        ("area_response", "area_factor"),
        ("vertex_response", "vertices"),
    ):
        rows = []
        for geometry, curves in (("polygon", pr), ("catmull_rom", cr)):
            for point in curves[series]:
                rows.append(
                    {
                        "geometry": LABELS[geometry],
                        "level": str(point[factor]),
                        "level_number": float(point[factor]),
                        "fps": point["throughput_fps_median"],
                        "wall_seconds": point["wall_seconds_median"],
                    }
                )
        response_datasets[series] = rows

    anova_rows = []
    for geometry, values in (("polygon", polygon), ("catmull_rom", curve)):
        anova = values["factorial_anova"]["wall_seconds"]
        for term, share in anova["term_percent_of_total"].items():
            anova_rows.append(
                {
                    "geometry": LABELS[geometry],
                    "term": LABELS[term],
                    "variance_share": share / 100.0,
                }
            )
        anova_rows.append(
            {
                "geometry": LABELS[geometry],
                "term": "実行間の残差",
                "variance_share": anova["residual_percent_of_total"] / 100.0,
            }
        )

    factorial_rows = []
    for geometry, values in (("polygon", polygon), ("catmull_rom", curve)):
        for row in values["factorial_cell_medians"]:
            factorial_rows.append(
                {
                    "geometry": LABELS[geometry],
                    "tracks": row["track_count"],
                    "area_factor": row["area_factor"],
                    "vertices": row["vertices"],
                    "seconds": row["wall_seconds_median"],
                    "fps": row["throughput_fps_median"],
                }
            )

    summary = [{
        "polygon_track_share": ps["contribution_percent"]["track_count"] / 100.0,
        "polygon_area_share": ps["contribution_percent"]["area_factor"] / 100.0,
        "polygon_vertex_share": ps["contribution_percent"]["vertices"] / 100.0,
        "curve_track_share": cs["contribution_percent"]["track_count"] / 100.0,
        "curve_area_share": cs["contribution_percent"]["area_factor"] / 100.0,
        "curve_vertex_share": cs["contribution_percent"]["vertices"] / 100.0,
    }]

    evidence_path = "output/runtime_driver_attribution_20260828/report/analysis_evidence.sqlite"
    common_query = {
        "engine": "sqlite",
        "language": "sql",
        "executed_at": analysis["generated_at"],
        "filters": {
            "observations": analysis["benchmark_definition"]["observations"],
            "target_interval": analysis["benchmark_definition"]["target_interval"],
            "track_count_levels": [1, 6],
            "area_factor_levels": [0.5, 1.5],
            "vertices_levels": [14, 20],
            "repetitions": 2,
        },
        "metric_definitions": {
            "Shapley contribution": (
                "Average marginal wall-time increase over all six orders in which "
                "the three factors move from the fast corner to the slow corner."
            ),
            "Throughput FPS": "480 mask observations divided by end-to-end geometry wall time.",
            "Track structure": (
                "The same 480 observations partitioned into independent tracks; this "
                "changes worker utilization and legal interval-graph boundaries."
            ),
        },
    }

    def make_source(source_id: str, label: str, table: str) -> dict[str, Any]:
        return {
            "id": source_id,
            "label": label,
            "path": evidence_path,
            "query": {
                **common_query,
                "description": f"Reviewed rows used by {label}.",
                "sql": f"SELECT * FROM {table}",
                "tables_used": [table],
            },
        }

    sources = [
        make_source("headline_source", "Headline contribution metrics", "headline_summary"),
        make_source("contribution_source", "Shapley driver contribution", "contributions"),
        make_source("track_source", "Track-count response curve", "track_response"),
        make_source("area_source", "Mask-area response curve", "area_response"),
        make_source("vertex_source", "Vertex-count response curve", "vertex_response"),
        make_source("anova_source", "Factorial variance decomposition", "anova"),
        make_source("factorial_source", "Factorial benchmark cells", "factorial_cells"),
    ]
    title = "後処理速度の要因分解：トラック・面積・頂点数"
    technical_summary = f"""## 技術サマリー

**ポリゴンではトラック構造が支配的です。** 同じ480マスクを使い、1→6トラック、面積0.5→1.5倍、14→20頂点を同時に変えた処理時間差を100%に配分すると、トラック構造 **{_pct(ps['contribution_percent']['track_count'])}**、面積 **{_pct(ps['contribution_percent']['area_factor'])}**、頂点数 **{_pct(ps['contribution_percent']['vertices'])}** でした。

**Catmull–Rom曲線では面積とトラック構造がほぼ同格です。** 面積 **{_pct(cs['contribution_percent']['area_factor'])}**、トラック構造 **{_pct(cs['contribution_percent']['track_count'])}**、頂点数 **{_pct(cs['contribution_percent']['vertices'])}** です。

したがって50〜200 FPSの差は「頂点数だけ」では説明できません。ポリゴンは長い少数トラックで並列度が不足し、区間グラフも大きくなることが主因です。曲線はさらに、ラスタ評価するマスク面積の影響が大きく出ます。"""

    contribution_text = f"""## ポリゴンはトラック構造、曲線は面積が最大要因

下図の割合は、最速条件（6トラック・面積0.5倍・14点）から最遅条件（1トラック・面積1.5倍・20点）までの**処理時間増分**を、要因の適用順に依存しないShapley法で100%に配分したものです。2回の独立測定でも、ポリゴンのトラック寄与は **{polygon['shapley_contribution_range_percent']['track_count']['min']:.1f}〜{polygon['shapley_contribution_range_percent']['track_count']['max']:.1f}%**、曲線の面積寄与は **{curve['shapley_contribution_range_percent']['area_factor']['min']:.1f}〜{curve['shapley_contribution_range_percent']['area_factor']['max']:.1f}%** に収まり、順位は変わりませんでした。"""

    track_text = f"""## トラック分割の効果はワーカー数で飽和する

面積1.0倍・16点に固定すると、ポリゴンは1トラック **{pr['track_response'][0]['throughput_fps_median']:.1f} FPS** から3トラック **{pr['track_response'][2]['throughput_fps_median']:.1f} FPS** へ上がり、6トラックでは **{pr['track_response'][3]['throughput_fps_median']:.1f} FPS** とほぼ横ばいです。3プロセスが埋まると追加トラックの利得がなくなるためです。曲線は2ワーカー構成なので、1→2トラックで大きく改善し、その後は測定揺らぎの範囲です。

これはトラック数自体が計算を重くする、という意味ではありません。**総マスク数が同じなら、長い1本のトラックより複数の独立トラックの方が速い**という結果です。"""

    area_text = f"""## 面積は曲線で強く、ポリゴンでは中程度に効く

1トラック・16点に固定して面積を0.5→1.5倍へ変えると、処理時間はポリゴンで **{pr['area_response_endpoints']['wall_time_change_percent']:.1f}%増**、曲線で **{cr['area_response_endpoints']['wall_time_change_percent']:.1f}%増** でした。曲線のFPSは **{cr['area_response'][0]['throughput_fps_median']:.1f}→{cr['area_response'][-1]['throughput_fps_median']:.1f}** まで下がり、面積依存のラスタ評価が明確に現れています。"""

    vertex_text = f"""## 頂点数は無視できないが、最大要因ではない

1トラック・面積1.0倍に固定して14→20点へ増やすと、処理時間はポリゴンで **{pr['vertex_response_endpoints']['wall_time_change_percent']:.1f}%増**、曲線で **{cr['vertex_response_endpoints']['wall_time_change_percent']:.1f}%増** でした。ポリゴンの14点と16点はほぼ同速ですが、18点から明確に遅くなります。したがって面積連動の可変頂点は速度へ効くものの、今回の大きなFPS振れの主原因ではありません。"""

    definition_text = """## 測定範囲と指標の定義

- 入力は実データの同一トラックから取得した480マスクです。運動・画面端の影響を除くため中央へ再配置しました。
- 目標キーフレーム間隔は3、解像度は1920×1080、状態数はProductionと同じ8です。
- トラック数は総マスク数480を保ったまま1・2・3・6本へ分割しました。
- 面積係数0.5・1.0・1.5は、各フレーム形状を重心周りに一様スケールした結果です。
- 頂点数はProductionの候補14・16・18・20点を強制しました。
- FPSは、480マスク観測 ÷ 形状準備・最適化・書き出しを含む壁時計時間です。動画フレームFPSではありません。"""

    method_text = """## 寄与率は三要因の実験操作から算出

2×2×2の全8条件をポリゴンと曲線で各2回実行しました。各セルの中央値を使い、最速角から最遅角への時間差について、3要因を変える全6順序の限界増分を平均してShapley寄与率を算出しました。

別途、各要因だけを動かした応答曲線を1回測定し、寄与率の方向と飽和点を確認しました。分散分析でも順位は整合し、ポリゴンではトラック主効果が88.9%、曲線では面積47.3%・トラック39.9%でした。Shapleyと分散分析の百分率が異なるのは、前者が「端点間の増分」、後者が「全16観測のばらつき」を配分するためです。"""

    limitation_text = f"""## 解釈上の制約と再現性

**寄与率は選んだレンジに依存します。** 1→6トラック、面積3倍、14→20点という比較範囲を変えれば割合も変わるため、全動画に普遍的な定数ではありません。

また、トラックを分割するとワーカー利用率だけでなく、トラック境界を跨ぐ区間候補が消えます。したがって「トラック構造」には並列化と区間グラフ縮小の両方が含まれます。品質を保つために実トラックを恣意的に分割する推奨ではありません。

2回のセル間相対差の中央値はポリゴン **{polygon['replicate_stability']['median_relative_range_percent']:.2f}%**、曲線 **{curve['replicate_stability']['median_relative_range_percent']:.2f}%** でした。ポリゴンの最大差はコールド側1セルで **{polygon['replicate_stability']['max_relative_range_percent']:.1f}%** でしたが、独立ラン別の寄与率順位は不変です。応答曲線は各1回なので、細かな数FPS差は順位ではなく傾向として読むべきです。"""

    next_text = """## 次に最も効く改善

1. **ポリゴン:** 長い少数トラックでも3ワーカーを埋められるよう、トラック内区間評価を安全にチャンク化・バッチ化する。
2. **曲線:** ROIまたはラスタ評価領域を形状周辺へ厳密に制限し、面積比例コストを下げる。
3. **両方式:** 18・20点トラックだけを対象に候補評価の共有・ベクトル化を行う。14→16点の最適化優先度は低い。
4. 本番ログへ、総マスク数、トラック長分布、面積分布、頂点数分布、実ワーカー利用率を保存し、動画単位のFPS予測モデルで外部妥当性を確認する。"""

    questions_text = """## 追加で確認すべき問い

- 実動画で50 FPSになるケースは、1〜2本の長大トラックに偏っているか。
- 大面積かつ20点のフレームが同時に集中した際、面積×頂点の交互作用は長尺データでも小さいままか。
- トラック内並列化でDPの最適解と決定性を完全に維持できる分割単位はどこか。"""

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Production形状処理の速度差を三要因の制御実験で分解した技術レポート。",
            "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
            "cards": [
                {
                    "id": "polygon_track",
                    "description": "ポリゴンの処理時間差に占めるトラック構造の寄与。",
                    "dataset": "summary",
                    "sourceId": "headline_source",
                    "metrics": [{"label": "ポリゴン：トラック", "field": "polygon_track_share", "format": "percent"}],
                },
                {
                    "id": "polygon_area",
                    "description": "ポリゴンの処理時間差に占める面積の寄与。",
                    "dataset": "summary",
                    "sourceId": "headline_source",
                    "metrics": [{"label": "ポリゴン：面積", "field": "polygon_area_share", "format": "percent"}],
                },
                {
                    "id": "curve_area",
                    "description": "曲線の処理時間差に占める面積の寄与。",
                    "dataset": "summary",
                    "sourceId": "headline_source",
                    "metrics": [{"label": "曲線：面積", "field": "curve_area_share", "format": "percent"}],
                },
                {
                    "id": "curve_track",
                    "description": "曲線の処理時間差に占めるトラック構造の寄与。",
                    "dataset": "summary",
                    "sourceId": "headline_source",
                    "metrics": [{"label": "曲線：トラック", "field": "curve_track_share", "format": "percent"}],
                },
            ],
            "charts": [
                {
                    "id": "driver_contribution",
                    "title": "処理時間差への寄与率",
                    "subtitle": "最速条件から最遅条件への時間増分を100%配分、各2回測定",
                    "type": "bar",
                    "dataset": "contributions",
                    "sourceId": "contribution_source",
                    "encodings": {
                        "x": {"field": "driver", "type": "ordinal", "label": "要因"},
                        "y": {"field": "share", "type": "quantitative", "label": "寄与率", "format": "percent"},
                        "color": {"field": "geometry", "type": "nominal", "label": "形状方式"},
                    },
                    "yAxisTitle": "寄与率",
                    "valueFormat": "percent",
                    "layout": "full",
                },
                {
                    "id": "track_response",
                    "title": "トラック数を変えたときの処理速度",
                    "subtitle": "総マスク数480、面積1.0倍、16点に固定",
                    "type": "bar",
                    "dataset": "track_response",
                    "sourceId": "track_source",
                    "encodings": {
                        "x": {"field": "level", "type": "ordinal", "label": "トラック数"},
                        "y": {"field": "fps", "type": "quantitative", "label": "マスク観測FPS"},
                        "color": {"field": "geometry", "type": "nominal", "label": "形状方式"},
                    },
                    "yAxisTitle": "マスク観測FPS",
                    "valueFormat": "number",
                    "layout": "full",
                },
                {
                    "id": "area_response",
                    "title": "面積を変えたときの処理速度",
                    "subtitle": "1トラック・16点に固定、面積係数0.5〜1.5",
                    "type": "bar",
                    "dataset": "area_response",
                    "sourceId": "area_source",
                    "encodings": {
                        "x": {"field": "level", "type": "ordinal", "label": "面積係数"},
                        "y": {"field": "fps", "type": "quantitative", "label": "マスク観測FPS"},
                        "color": {"field": "geometry", "type": "nominal", "label": "形状方式"},
                    },
                    "yAxisTitle": "マスク観測FPS",
                    "valueFormat": "number",
                    "layout": "full",
                },
                {
                    "id": "vertex_response",
                    "title": "頂点数を変えたときの処理速度",
                    "subtitle": "1トラック・面積1.0倍に固定、14〜20点",
                    "type": "bar",
                    "dataset": "vertex_response",
                    "sourceId": "vertex_source",
                    "encodings": {
                        "x": {"field": "level", "type": "ordinal", "label": "頂点数"},
                        "y": {"field": "fps", "type": "quantitative", "label": "マスク観測FPS"},
                        "color": {"field": "geometry", "type": "nominal", "label": "形状方式"},
                    },
                    "yAxisTitle": "マスク観測FPS",
                    "valueFormat": "number",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "anova",
                    "title": "分散分析による主効果と交互作用",
                    "subtitle": "2回×8条件の壁時計時間のばらつきを配分",
                    "dataset": "anova",
                    "sourceId": "anova_source",
                    "columns": [
                        {"field": "geometry", "label": "方式", "type": "text"},
                        {"field": "term", "label": "項", "type": "text"},
                        {"field": "variance_share", "label": "分散寄与", "format": "percent"},
                    ],
                },
                {
                    "id": "factorial_cells",
                    "title": "全要因実験の実測セル",
                    "subtitle": "各条件2回の壁時計時間中央値、480マスク観測",
                    "dataset": "factorial_cells",
                    "sourceId": "factorial_source",
                    "columns": [
                        {"field": "geometry", "label": "方式", "type": "text"},
                        {"field": "tracks", "label": "トラック数", "format": "number"},
                        {"field": "area_factor", "label": "面積係数", "format": "number"},
                        {"field": "vertices", "label": "頂点数", "format": "number"},
                        {"field": "seconds", "label": "秒", "format": "number"},
                        {"field": "fps", "label": "FPS", "format": "number"},
                    ],
                },
            ],
            "sources": [
                {"id": source["id"], "label": source["label"], "path": source["path"]}
                for source in sources
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {"id": "technical_summary", "type": "markdown", "body": technical_summary},
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["polygon_track", "polygon_area", "curve_area", "curve_track"]},
                {"id": "contribution_text", "type": "markdown", "body": contribution_text},
                {"id": "contribution_chart", "type": "chart", "chartId": "driver_contribution", "layout": "full"},
                {"id": "track_text", "type": "markdown", "body": track_text},
                {"id": "track_chart", "type": "chart", "chartId": "track_response", "layout": "full"},
                {"id": "area_text", "type": "markdown", "body": area_text},
                {"id": "area_chart", "type": "chart", "chartId": "area_response", "layout": "full"},
                {"id": "vertex_text", "type": "markdown", "body": vertex_text},
                {"id": "vertex_chart", "type": "chart", "chartId": "vertex_response", "layout": "full"},
                {"id": "scope", "type": "markdown", "body": definition_text},
                {"id": "method", "type": "markdown", "body": method_text},
                {"id": "anova_table", "type": "table", "tableId": "anova", "layout": "full"},
                {"id": "robustness", "type": "markdown", "body": limitation_text},
                {"id": "factorial_table", "type": "table", "tableId": "factorial_cells", "layout": "full"},
                {"id": "next", "type": "markdown", "body": next_text},
                {"id": "questions", "type": "markdown", "body": questions_text},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": analysis["generated_at"],
            "status": "ready",
            "datasets": {
                "summary": summary,
                "contributions": contributions,
                **response_datasets,
                "anova": anova_rows,
                "factorial_cells": factorial_rows,
            },
        },
        "sources": sources,
        "package_info": {},
    }

    args.evidence_db.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_db.unlink(missing_ok=True)
    with sqlite3.connect(args.evidence_db) as db:
        db.execute(
            "CREATE TABLE headline_summary("
            "polygon_track_share REAL,polygon_area_share REAL,polygon_vertex_share REAL,"
            "curve_track_share REAL,curve_area_share REAL,curve_vertex_share REAL)"
        )
        db.execute(
            "INSERT INTO headline_summary VALUES (?,?,?,?,?,?)",
            tuple(summary[0].values()),
        )
        db.execute(
            "CREATE TABLE contributions(geometry TEXT,driver TEXT,share REAL,share_percent REAL,seconds REAL)"
        )
        db.executemany(
            "INSERT INTO contributions VALUES (:geometry,:driver,:share,:share_percent,:seconds)",
            contributions,
        )
        for table in ("track_response", "area_response", "vertex_response"):
            db.execute(
                f"CREATE TABLE {table}(geometry TEXT,level TEXT,level_number REAL,fps REAL,wall_seconds REAL)"
            )
            db.executemany(
                f"INSERT INTO {table} VALUES (:geometry,:level,:level_number,:fps,:wall_seconds)",
                response_datasets[table],
            )
        db.execute("CREATE TABLE anova(geometry TEXT,term TEXT,variance_share REAL)")
        db.executemany(
            "INSERT INTO anova VALUES (:geometry,:term,:variance_share)", anova_rows
        )
        db.execute(
            "CREATE TABLE factorial_cells("
            "geometry TEXT,tracks INTEGER,area_factor REAL,vertices INTEGER,seconds REAL,fps REAL)"
        )
        db.executemany(
            "INSERT INTO factorial_cells VALUES ("
            ":geometry,:tracks,:area_factor,:vertices,:seconds,:fps)",
            factorial_rows,
        )
        db.commit()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source_notes = {
        "audience": "technical",
        "delivery_mode": "html",
        "required_structure_mapping": {
            "technical_summary": "technical_summary",
            "key_findings": ["contribution_text", "track_text", "area_text", "vertex_text"],
            "scope_data_definitions": "scope",
            "methodology": ["method", "anova_table"],
            "limitations_uncertainty_robustness": ["robustness", "factorial_table"],
            "recommended_next_steps": "next",
            "further_questions": "questions",
        },
        "chart_map": [
            {"segment": "driver contribution", "type": "grouped bar", "dataset": "contributions", "claim": "relative contribution differs by geometry"},
            {"segment": "track response", "type": "grouped bar", "dataset": "track_response", "claim": "parallelism saturates at worker count"},
            {"segment": "area response", "type": "grouped bar", "dataset": "area_response", "claim": "curve cost is strongly area-sensitive"},
            {"segment": "vertex response", "type": "grouped bar", "dataset": "vertex_response", "claim": "points matter but rank third"},
        ],
        "omissions": [
            "No causal claim outside the controlled 480-observation benchmark.",
            "One-factor response curves have one run per non-center level; small differences are directional evidence only.",
        ],
    }
    args.source_notes.parent.mkdir(parents=True, exist_ok=True)
    args.source_notes.write_text(
        json.dumps(source_notes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
