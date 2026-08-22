# Production post-processing

Production offers two editable genital-mask geometries.  Both share the same
NMS, tracking, screen-edge preparation, track-adaptive point-count policy,
minimum Recall contract, topology gates, classwise scheduling, and V3 SQLite
schema:

- `polygon` (default) uses line segments and the promoted adaptive CUDA-lazy
  interval screen. Every selected edge and final mask is still validated by
  the native exact evaluator.
- `catmull_rom` uses a closed uniform Catmull--Rom spline with tension 1.0.
  The editable points are interpolation points `P`; cubic Bezier handles are
  always derived from adjacent points with the fixed `1/6` conversion factor.
  The deployed default remains the exact CPU evaluator, so curve
  post-processing does not compete with GPU inference. A separately gated
  OpenCV-compatible CUDA evaluator is available for polygon, ellipse, and
  Catmull--Rom Recall/IoU validation; it is loaded only when explicitly
  selected and does not change raster semantics.

Select the mode with `--mask-geometry polygon|catmull_rom`, or with the GUI
mask-shape selector.  The default remains `polygon` for compatibility.

The shared promoted input contract is:

- `nms.production_v3`: fills true holes, removes owner-relative islands of at
  most 1%, and runs virtual-component adaptive Mask NMS. Bounding boxes are a
  broad phase only; native-pixel masks make every suppression decision.
- The selected geometry stage chooses one point count per track from the
  q99.9 pre-border mask area divided by the real frame area. Counts are 14,
  16, 18, or 20 at strict 3%, 10%, and 25% crossings.
- Edge preparation caps both the influence band and outward displacement at
  16 px. A mask supported by two perpendicular screen edges keeps explicit
  two-axis corner support.
- Spatial fitting, multistate DP, and per-key refinement all use the selected
  track point count. Exact Recall is at least 0.97, the keyframe interval is a
  soft target, and the final topology guard rejects invalid optimization
  trials without stopping the complete video.  The curve engine has a
  deterministic same-point-count completion envelope for otherwise
  infeasible spatial fits and reports every use in its manifest.
- Polygon interval screening uses CUDA by default. Every selected polygon
  edge and every final polygon mask is then audited with the native exact
  evaluator, so the Recall floor and topology gates remain exact. The curve
  engine uses exact CPU evaluation by default. Validation runs may select the
  pixel-identical CUDA or exact hybrid backend with
  `MASK_CURVE_EXACT_RASTER_BACKEND`; the backend is recorded in the manifest.
- Polygon routes run as three concurrent class threads.  Catmull--Rom routes
  use deterministic, cost-balanced process shards (two shards per class and
  at most six workers by default), because the exact CPU path also contains
  Python DP and audit work that cannot benefit from threads alone.  Each
  process reads the immutable tracked SQLite directly and writes a disjoint
  route; merge order is stable. Different target intervals do not serialize
  the class jobs.

The default runner exposes only the promoted NMS, polygon and Catmull--Rom
stage IDs. It never falls back to a retired optimizer: unsupported semantic
labels or contract options fail with an actionable error. The parity-frozen
polygon compatibility engine is reached only through
`production/polygon/runtime_bridge.py`; deployed code does not import
development-only research packages. Historical experiment identifiers are
never emitted by Production manifests.

Gap filling is fixed at 15 frames. The supported user quality control is the
soft target keyframe interval; Production preparation, Recall floors,
topology checks, point-count policy, interpolation method, and evaluator are
frozen as one tested contract.  A curve SQLite is distinguished by
`catmull_rom_uniform_tension_1_v1` in its interpolation metadata while keeping
the public V3/revision-5 schema unchanged.

For the supported target interval range 1--6, the curve DP evaluates at most
24 frames per edge. This limit is four times the largest supported target and
was parity-checked against the former 30-frame graph: selected frames, states,
geometry and exact metrics remained identical across the V3 six-track suite.
After the initial DP, a tail-quality rescue may add keys inside each bounded
600-frame optimization chunk.  Production intentionally has no artificial
rescue-key quota: a key is accepted only where the independent spatial fit
materially improves a low-IoU or inflated interpolated frame and both
neighbouring intervals still satisfy exact Recall and topology.  This keeps
the requested interval soft while preventing one difficult frame from
carrying the area cost of an otherwise sparse run.  Difficult motion therefore
pays with extra keys rather than a silently poor frame; experiments may still
set an explicit positive insertion cap when measuring that trade-off.

Spatial fitting normally keeps one persistent Catmull--Rom point phase for the
whole track.  If that shared placement leaves a rare source frame below the
local IoU floor or above the area cap, Production refits only that frame with
the same point count, aligns its cyclic phase to the track controls, and tests
exact-valid blends before DP.  The conservative spatial envelope remains a
non-stopping last resort only when the local refit cannot meet the hard Recall
floor.  Both paths are counted separately in the engine manifest.

Final exact Recall violations, rejected pair-vote trials, selected vertex
counts, border settings, and SQLite integrity are recorded in manifests. A
rejected local trial falls back to the last valid geometry; it is not a final
output violation. The public SQLite schema remains V3/revision 5.

The promoted curve contract and its nine-run V3 acceptance evidence are
recorded in [`curve/VALIDATION_20260822.md`](curve/VALIDATION_20260822.md).
