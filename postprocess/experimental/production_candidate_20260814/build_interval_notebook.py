#!/usr/bin/env python3
"""Build and execute the interval benchmark notebook without Jupyter deps."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import traceback
from pathlib import Path
from typing import Any


def markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def build(comparison: Path) -> dict[str, Any]:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "cells": [
            markdown(
                "# tl;dr\n\n旧Productionと新Production候補を目標間隔1〜6で比較し、"
                "Recall、IoU、実効間隔、速度、SQLite健全性を監査する。"
            ),
            markdown(
                "## Context & Methods\n\n同一KPI V3入力を用い、動画フレームを復号せず"
                "SQLite/JSONマスク形状だけを集計する。arm-local品質と共通raw AIマスク基準を分ける。"
            ),
            code(
                "from pathlib import Path\nimport json\n"
                f"comparison = Path({str(comparison.resolve())!r})\n"
                "data = json.loads(comparison.read_text(encoding='utf-8'))\n"
                "rows = data['rows']\n"
                "assert len(rows) == 12\n"
                "assert {int(r['target_interval']) for r in rows} == set(range(1, 7))\n"
                "assert {r['arm'] for r in rows} == {'legacy_production', 'production_candidate_20260814'}\n"
                "print(json.dumps({'rows': len(rows), 'arms': sorted({r['arm'] for r in rows})}, ensure_ascii=False))"
            ),
            markdown(
                "## Data\n\n12セル（2方式×6目標）。各セルは完成したソフトウェア互換SQLiteと"
                "exact評価CSVへ対応する。"
            ),
            code(
                "assert data['validation']['all_sqlite_integrity_ok']\n"
                "assert data['validation']['all_sqlite_foreign_keys_ok']\n"
                "new = [r for r in rows if r['arm'] == 'production_candidate_20260814']\n"
                "assert all(int(r['recall_violations']) == 0 for r in new)\n"
                "print(json.dumps({'schema_hash_count': data['validation']['schema_hash_count'], "
                "'candidate_recall_violations': sum(int(r['recall_violations']) for r in new)}, ensure_ascii=False))"
            ),
            markdown("## Results"),
            code(
                "cols=['arm','target_interval','actual_interval','keyframes','recall_min',"
                "'recall_violations','iou_mean','iou_q01','common_raw_pixel_weighted_iou',"
                "'optimizer_wall_seconds','video_fps']\n"
                "selected=[{k:r[k] for k in cols} for r in sorted(rows, "
                "key=lambda x:(x['target_interval'],x['arm']))]\n"
                "print(json.dumps(selected, ensure_ascii=False, indent=2))"
            ),
            code(
                "deltas = data['by_interval']\nassert len(deltas) == 6\n"
                "print(json.dumps(deltas, ensure_ascii=False, indent=2))"
            ),
            markdown(
                "## Takeaways\n\n新候補は最小Recallを制約として保証する。IoU・実効間隔・速度の"
                "優劣は上表から判断し、人手GT不在とNMS差による参照差を必ず併記する。"
            ),
        ],
    }


def execute(notebook: dict[str, Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "__notebook__"}
    errors: list[str] = []
    count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(cell["source"], f"<cell-{count}>", "exec"), namespace)
        except Exception:
            detail = traceback.format_exc()
            errors.append(detail)
            cell["outputs"] = [
                {
                    "output_type": "error",
                    "ename": detail.splitlines()[-1].split(":", 1)[0],
                    "evalue": detail.splitlines()[-1],
                    "traceback": detail.splitlines(),
                }
            ]
        else:
            text = output.getvalue()
            cell["outputs"] = (
                [{"output_type": "stream", "name": "stdout", "text": text}]
                if text
                else []
            )
        cell["execution_count"] = count
        if errors:
            break
    return {
        "code_cells": count,
        "errors": len(errors),
        "executed_top_to_bottom": not errors,
        "executor": "deterministic_stdlib_shared_namespace",
        "privacy": "No video frames decoded.",
        "error_messages": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    notebook = build(args.comparison)
    receipt = execute(notebook)
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    args.notebook.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if receipt["errors"]:
        raise RuntimeError("notebook execution failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
