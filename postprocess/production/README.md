# Production post-processing

The default polygon pipeline uses the promoted 2026-08 profile:

- `nms.production_v3`: hole filling, <=1% island cleanup, virtual-component
  NMS, and adaptive exact-mask comparisons. Bounding boxes are broad-phase
  only.
- `production.polygon_v3_cpu`: fixed 14-point spatial approximation,
  multi-state DP with a soft keyframe interval, exact CPU interval evaluation,
  per-key pair-vote, and topology guards.

The prior `nms.adaptive` and `approximation.polygon.production_v22` stage IDs
remain registered as rollback paths. The parity-frozen optimizer is reached
only through `production/polygon/runtime_bridge.py`; no other Production module
may import dated experimental runtime modules.

Final exact Recall is always audited. The selected CPU corpus contains a small
number of spatially infeasible fixed-14-point observations, so the default
publishes them with explicit violation counts in
`production_polygon_manifest.json`. Set
`require_zero_exact_recall_violations=true` to fail closed instead.
