# Production + raw-Recall guard results (2026-08-06)

## Scope

- Source: `12月KPI動画.sqlite`
- Baseline: Production polygon output with requested interval 10
- Label: `男性器`
- Frames: `8681..20053`
- Raw mask instances: `6,089` across 24 track/scene segments
- Constraint: minimum per-instance raw-observation Recall `0.97`
- Temporal consensus/reference: disabled
- Video pixels: not opened

The guard retains every Production key position. A Production key shape is
changed only when it violates the floor, and a new key is inserted only when no
safe direct interpolation edge exists. Within each original Production
interval, dynamic programming selects the fewest added keys and then the
highest raw-mask IoU among equal-key paths.

## Result

| Metric | Production interval 10 | Production + Recall guard | Delta |
|---|---:|---:|---:|
| Keyframes | 636 | 1,882 | +1,246 (+195.91%) |
| Mean temporal key interval | 10.157 | 3.346 | -6.811 |
| Adjusted existing key shapes | 0 | 178 | +178 |
| Minimum Recall | 0.679331 | 0.970032 | +0.290701 |
| Recall < 0.97 | 1,542 | 0 | -1,542 |
| Recall q01 | 0.878502 | 0.970624 | +0.092122 |
| Mean IoU | 0.871633 | 0.932388 | +0.060755 |
| IoU q01 | 0.537198 | 0.657414 | +0.120216 |
| Minimum IoU | 0.297091 | 0.404249 | +0.107157 |
| Mean precision | 0.890842 | 0.947121 | +0.056279 |
| Mean output/raw area ratio | 1.110906 | 1.047488 | -0.063418 |
| Mean excess-area ratio | 0.134914 | 0.063302 | -0.071613 |
| Adjacent output IoU | 0.952828 | 0.933917 | -0.018911 |
| Mean output-area log delta | 0.012491 | 0.022555 | +0.010064 |
| Mean centroid-residual acceleration (px) | 6.723151 | 5.242102 | -1.481049 |

The hard floor removes every raw-reference violation and improves overlap and
precision, but it almost triples the key count. Adjacent-mask IoU declines and
area change increases because the guarded result follows raw-frame variation
more closely. This is the central cost of applying an unconditional per-frame
raw Recall constraint to Production.

## Performance and contract

- Optimizer wall time: `93.49 s`
- Candidate edges evaluated: `47,061`
- Recall-safe edges: `18,231`
- Baseline SQLite: `284,278,784 bytes`
- Guarded SQLite: `286,490,624 bytes`
- Size increase: `2,211,840 bytes` (`0.78%`)
- SQLite schema fingerprint: unchanged
- `PRAGMA integrity_check`: `ok`
- Foreign-key errors: `0`
- Exported schema version: `3`
- Postprocess test suite: `102 passed`, `41 subtests passed`

## Artifacts

- SQLite: `output/production_recall_guard_20260806/exact_recall097/12月KPI動画_production_exact_recall097.sqlite`
- Aggregate report: `output/production_recall_guard_20260806/exact_recall097/comparison_report.json`
- Paired per-frame metrics: `output/production_recall_guard_20260806/exact_recall097/frame_metrics.csv`

The SQLite is a copy of the Production-compatible V3 result. Only the selected
polygon keyframe rows and their provenance are replaced; tables, columns,
indexes, triggers, views, face data, raw masks, cuts, and metadata retain the
same contract.
