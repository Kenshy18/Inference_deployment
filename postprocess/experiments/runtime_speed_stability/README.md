# Runtime speed stability experiments

This experiment isolates postprocess throughput variance without weakening the
minimum-Recall constraint or changing the serialized output geometry.

## Production decision

- Polygon: budget nested label/process/native-thread parallelism from the CPUs
  available to the process. On the 24-core validation host the three-label
  schedule is `3 x 6 x 2`, instead of the oversubscribed `3 x 9 x 4` schedule.
- Polygon: use two candidate-frame workers only for a single optimizer run;
  use one when track-level process parallelism is available.
- Curve: retain the existing two cost-balanced shards per class. Coarsening to
  one shard per class was slower despite the same total native-thread ceiling.
- Preserve the Production DP maximum gap of 30 and global long-track solve.
  Faster alternatives changed local output geometry and were rejected.

## Reproducible programs

- `benchmark_long_track_chunking.py`: long-track chunking, candidate-frame
  worker count, native-thread count, and DP maximum-gap controls.
- `benchmark_full_corpus_controls.py`: full V3 KPI corpus quality and exact
  output comparisons.
- `benchmark_curve_scheduler.py`: Curve shard scheduling comparison.
- `build_runtime_stability_report.py`: exact SQLite/keyframe parity check,
  normalized audit ledger, and canonical report artifact generation.

The generated, self-contained technical report is:

`output/runtime_speed_stability_20260830/report/report.html`

The report is generated from saved benchmark results and is not a live data
connection.
