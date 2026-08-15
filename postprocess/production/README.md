# Production post-processing

The default polygon pipeline uses the promoted 2026-08 profile:

- `nms.production_v3`: hole filling, <=1% island cleanup, virtual-component
  NMS, and adaptive exact-mask comparisons. Bounding boxes are broad-phase
  only.
- `production.polygon_v3_cpu`: track-wise 14/16/18/20-point spatial fallback,
  multi-state DP with a soft keyframe interval, exact CPU interval evaluation,
  per-key pair-vote, and topology guards.

The prior `nms.adaptive` and `approximation.polygon.production_v22` stage IDs
remain registered as rollback paths. The parity-frozen optimizer is reached
only through `production/polygon/runtime_bridge.py`; no other Production module
may import dated experimental runtime modules.

Final exact Recall is always audited and Production is fail-closed: one
observed frame below 0.97 rejects the artifact.  Each continuous track segment
first tries 14 points per component, then 16, 18, and 20.  The smallest
native-exact quality-feasible count is retained; no frame may bypass the final
Recall gate.
