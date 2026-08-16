# Production post-processing

The default polygon pipeline uses the promoted adaptive CPU-exact profile:

- `nms.production_v3`: fills true holes, removes owner-relative islands of at
  most 1%, and runs virtual-component adaptive Mask NMS. Bounding boxes are a
  broad phase only; native-pixel masks make every suppression decision.
- `production.polygon_v3_cpu`: selects one point count per track from the
  q99.9 pre-border mask area divided by the real frame area. Counts are 14,
  16, 18, or 20 at strict 3%, 10%, and 25% crossings.
- Edge preparation caps both the influence band and outward displacement at
  16 px. A mask supported by two perpendicular screen edges keeps explicit
  two-axis corner support.
- Spatial fitting, multistate DP, and per-key pair-vote all use the selected
  track point count. Exact Recall is at least 0.97, the keyframe interval is a
  soft target, and the final topology guard rejects invalid optimization
  trials without stopping the complete video.
- Interval evaluation uses the native CPU-exact implementation by default.

The prior `nms.adaptive` and `approximation.polygon.production_v22` stage IDs
remain registered as explicit rollback paths. The parity-frozen optimizer is
reached only through `production/polygon/runtime_bridge.py`; no other
Production module imports dated experimental runtime modules.

Final exact Recall violations, rejected pair-vote trials, selected vertex
counts, border settings, and SQLite integrity are recorded in manifests. A
rejected local trial falls back to the last valid geometry; it is not a final
output violation. The public SQLite schema remains V3/revision 5.
