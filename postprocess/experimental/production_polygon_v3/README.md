# Production Polygon V3 experiment

This experiment starts from the validated Production polygon keyframes and
adds two independent hard constraints:

1. minimum full-mask Recall at every observed frame; and
2. minimum side-local Recall plus Production-compatible off-canvas extent at
   every observed frame touching a screen edge.

Every Production key position is retained. A Production key shape is retained
when it already satisfies both constraints. Each original Production interval
is then densely reconstructed with the editor's stored `point_index` semantics.
The local DP inserts the fewest additional keys required for feasibility and
uses accumulated original-mask IoU only to break equal-key ties.

V3 intentionally has no fixed global key count, optional 4--30% expansion
states, post-decode pair-vote mutation, Production mean-IoU threshold, or mean
Recall budget. Scaling is permitted only as the minimum repair needed to make
an anchor feasible.

The runner reads SQLite geometry only and never opens video frames:

```bash
PYTHONPATH=postprocess:overlay/src \
python -m experimental.production_polygon_v3.run \
  --source-sqlite /path/to/source.sqlite \
  --baseline-sqlite /path/to/production.sqlite \
  --output-dir /path/to/output \
  --output-sqlite /path/to/output.sqlite \
  --label 男性器 --start-frame 8681 --end-frame 20059 \
  --normal-recall-floor 0.97 --border-recall-floor 0.97
```

The exported SQLite schema is copied from the baseline and fingerprinted before
and after the keyframe replacement. The runner independently reloads it and
rechecks normal Recall, border Recall/extent, SQLite integrity, foreign keys,
and editor-versus-Overlay interpolation agreement.

By default (`--point-count 0`), every segment retains its original Production
vertex count. A positive override exists only for controlled experiments.

## Soft key-penalty refinement

`run_penalty.py` takes a fully dual-safe V3 SQLite and removes redundant keys
with the Production-style objective `sum(1 - frame IoU) + lambda * key count`.
Lambda is calibrated toward the requested interval, but the count is never
fixed. Every eligible edge has already passed both per-frame minimum Recall
constraints.

For diagnostics only, `--constraint-mode normal-only` and
`--constraint-mode border-only` isolate the key-count cost of each constraint.
The intended algorithm is the default `--constraint-mode both`; the ablation
outputs must not be promoted as production candidates.
