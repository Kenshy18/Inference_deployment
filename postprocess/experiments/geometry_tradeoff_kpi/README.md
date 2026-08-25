# Geometry trade-off KPI analysis

This utility aggregates the 14 real-video runs under
`output/geometry_tradeoff_kpi_15min_20260825`: polygon and closed uniform
Catmull–Rom, each at target intervals 1 through 7.

Quality is evaluated on the common 24,503 source `(track_id, frame)` rows.
Gap-filled rows are deliberately excluded from IoU, Recall, and area-ratio
statistics, while effective interval and editable-point burden use the complete
25,090-row materialized output.

Run with the Production Python environment:

```bash
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python \
  postprocess/experiments/geometry_tradeoff_kpi/analyze.py
```

The analysis directory contains:

- `summary.csv` and `summary_by_class.csv`: complete numeric results.
- `paired_delta_summary.csv`: paired curve-minus-polygon deltas.
- `review_candidates.csv`: low-IoU, high-expansion, and largest-delta frames.
- `sqlite_integrity.csv`: schema population and SQLite integrity checks.
- `geometry_tradeoff_analysis.ipynb`: executed reproducible notebook.
- `report.html`: self-contained technical report.
