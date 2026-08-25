"""Create and execute the reproducible geometry trade-off notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("output/geometry_tradeoff_kpi_15min_20260825/analysis"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    analysis = args.analysis_dir.resolve()
    notebook_path = analysis / "geometry_tradeoff_analysis.ipynb"
    notebook = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        }
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Polygon vs closed uniform Catmull–Rom\n\n"
            "Real V3 masklet benchmark on the 13:04 KPI video. Quality uses only "
            "the 24,503 source observations common to both methods; effective "
            "interval and point burden use all 25,090 materialized rows."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            f"ANALYSIS = Path({str(analysis)!r})\n"
            "summary = pd.read_csv(ANALYSIS / 'summary.csv')\n"
            "by_class = pd.read_csv(ANALYSIS / 'summary_by_class.csv')\n"
            "paired = pd.read_csv(ANALYSIS / 'paired_delta_summary.csv')\n"
            "manifest = json.loads((ANALYSIS / 'summary.json').read_text())\n"
            "summary.shape, by_class.shape, paired.shape"
        ),
        nbformat.v4.new_markdown_cell("## Validation gates"),
        nbformat.v4.new_code_cell(
            "assert manifest['validation']['actual_run_count'] == 14\n"
            "assert manifest['validation']['all_quality_rows_equal_source']\n"
            "assert manifest['validation']['all_recall_constraints_satisfied']\n"
            "assert manifest['validation']['all_topology_valid']\n"
            "assert manifest['validation']['all_outputs_exist']\n"
            "assert manifest['validation']['all_sqlite_integrity_checks_passed']\n"
            "manifest['validation']"
        ),
        nbformat.v4.new_markdown_cell("## Complete comparison"),
        nbformat.v4.new_code_cell(
            "columns = ['geometry','target_interval','effective_interval',"
            "'source_video_fps','iou_mean','iou_q05','iou_q01','iou_minimum',"
            "'recall_minimum','area_ratio_q95','area_ratio_maximum',"
            "'editable_points_mean_per_keyframe','keyframe_rows']\n"
            "summary[columns].sort_values(['target_interval','geometry']).round(4)"
        ),
        nbformat.v4.new_markdown_cell("## Speed and achieved frequency"),
        nbformat.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "for geometry, group in summary.groupby('geometry'):\n"
            "    group = group.sort_values('target_interval')\n"
            "    axes[0].plot(group.target_interval, group.source_video_fps, marker='o', label=geometry)\n"
            "    axes[1].plot(group.target_interval, group.effective_interval, marker='o', label=geometry)\n"
            "axes[0].axhline(200, color='gray', ls='--', lw=1)\n"
            "axes[0].set(title='Throughput', xlabel='Target interval', ylabel='Source-video FPS')\n"
            "axes[1].plot(range(1,8), range(1,8), color='gray', ls='--', lw=1, label='target')\n"
            "axes[1].set(title='Target attainment', xlabel='Target interval', ylabel='Actual effective interval')\n"
            "for ax in axes: ax.grid(alpha=.25); ax.legend()\n"
            "fig.tight_layout(); fig.savefig(ANALYSIS / 'speed_and_frequency.png', dpi=160); plt.show()"
        ),
        nbformat.v4.new_markdown_cell("## Quality and expansion"),
        nbformat.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n"
            "for geometry, group in summary.groupby('geometry'):\n"
            "    group = group.sort_values('target_interval')\n"
            "    axes[0].plot(group.target_interval, group.iou_mean, marker='o', label=geometry)\n"
            "    axes[1].plot(group.target_interval, group.iou_q05, marker='o', label=geometry)\n"
            "    axes[2].plot(group.target_interval, group.area_ratio_q95, marker='o', label=geometry)\n"
            "axes[0].set(title='Mean IoU', ylabel='IoU')\n"
            "axes[1].set(title='5th-percentile IoU', ylabel='IoU')\n"
            "axes[2].set(title='95th-percentile area ratio', ylabel='output / source area')\n"
            "for ax in axes: ax.set_xlabel('Target interval'); ax.grid(alpha=.25); ax.legend()\n"
            "fig.tight_layout(); fig.savefig(ANALYSIS / 'quality_and_expansion.png', dpi=160); plt.show()"
        ),
        nbformat.v4.new_markdown_cell("## Point burden and paired deltas"),
        nbformat.v4.new_code_cell(
            "display(summary[['geometry','target_interval','editable_points_mean_per_keyframe',"
            "'editable_points_total','track_point_budget_distribution']].round(4))\n"
            "display(paired.round(5))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Interpretation\n\n"
            "The same-target comparison is useful for operational settings, but it "
            "is not an equal-sparsity comparison when achieved intervals differ. "
            "For an approximately equal effective interval of 3, compare Polygon "
            "target 3 with Catmull–Rom target 4."
        ),
        nbformat.v4.new_code_cell(
            "polygon3 = summary.query(\"geometry == 'polygon' and target_interval == 3\").iloc[0]\n"
            "curve4 = summary.query(\"geometry == 'catmull_rom' and target_interval == 4\").iloc[0]\n"
            "pd.DataFrame([polygon3, curve4])[columns].round(5)"
        ),
    ]
    nbformat.write(notebook, notebook_path)
    client = NotebookClient(notebook, timeout=300, kernel_name="python3")
    executed = client.execute(cwd=str(Path.cwd()))
    nbformat.write(executed, notebook_path)
    print(notebook_path)


if __name__ == "__main__":
    main()
