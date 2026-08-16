# Postprocess geometry and keyframe quality audit (2026-08-04)

## Scope and privacy

This audit used only local processes and numerical artifacts. No video frame
was displayed, exported for inspection, passed to the AI agent, or uploaded.
The agent inspected video metadata, SQLite rows, polygon/ellipse coordinates,
logs, and aggregate metrics only. The public result SQLite schema was not
changed.

Three locally generated mask sets were examined:

- A: 4,024 tracked rows from a 1920x1080, 3-minute sample;
- B: 3,319 tracked rows from a 1280x720 sample;
- C: 2,003 tracked rows from a 1920x1080 sample.

## Release-relevant findings

### P0: keyframe count is optimized, but worst-frame quality is not bounded

The ellipse optimizer searches for a fixed global keyframe ratio. Its DP cost
is calculated in a smoothed parameter space, and the selected keyframe values
are subsequently moved by a global least-squares fit. Dense-recall repair then
enforces an area-weighted average recall target by inflating ellipse axes.
There is no per-frame IoU, precision, or recall floor.

At a target ratio of one keyframe per three source rows:

| sample | raw ellipse global IoU | keyframed global IoU | frames losing >0.05 IoU | worst IoU loss |
|---|---:|---:|---:|---:|
| B | 0.937636 | 0.919504 | 561 | -0.4153 |
| C | 0.972908 | 0.954946 | 192 | -0.3470 |

For B, 988 of 3,319 rows fell below 0.90 IoU after keyframe interpolation.
For C, 153 of 2,003 rows did. Aggregate recall remained high enough to hide
these tail failures.

Ablation on sample B isolated the main causes:

| variant | global IoU | p1 IoU | rows below 0.90 IoU | recall below 0.90 |
|---|---:|---:|---:|---:|
| current: smoothing=1, min-gap=2, confidence blend, global LS | 0.919504 | 0.6992 | 988 | 47 |
| raw signal, min-gap=1, global LS | **0.924330** | **0.8271** | **658** | **2** |
| raw signal, min-gap=1, no value fit | 0.921313 | 0.8158 | 898 | 11 |
| raw signal, min-gap=2, no value fit | 0.917480 | 0.6495 | 996 | 63 |

Removing pre-selection smoothing and allowing adjacent keyframes materially
improves the tail. Global LS is helpful once it is fitted to the raw signal,
but it still provides no worst-frame guarantee.

The measured stabilization (`smooth_alpha=0`, `min_gap=1`, raw anchors with
global LS) has therefore been promoted as the current default. It improved all
three examined inputs without changing keyframe density or the SQLite schema.
The quality-constrained vNext work below is still required for a real tail
guarantee and for adversarial short-period motion.

An end-to-end rerun of sample B confirmed the promoted defaults: the keyframe
stage decreased from 0.871 s to 0.639 s, total measured stage time decreased
from 6.947 s to 6.708 s, global IoU increased from 0.919504 to 0.924330, p1 IoU
increased from 0.6992 to 0.8271, and recall-below-0.90 rows fell from 47 to 2.

Synthetic periodic center motion confirms aliasing:

- period 24 frames: 1.24 px RMSE;
- period 8 frames: 9.61 px RMSE;
- period 6 frames: 21.77 px RMSE, 91.5 px worst error;
- period 4 frames: only 5% keyframes selected, 56.24 px RMSE;
- alternating period 2: 78.87 px RMSE and only 27.3% amplitude retained.

The period-4 failure occurs because smoothing plus a fixed density objective
can select equal-phase anchors and interpret a fast cycle as nearly static.
`min_gap=2` makes a true period-2 signal impossible to represent.

### P0: ellipse boundary geometry is hard-coded to 1920x1080

`approximation/ellipse/runtime_fst.py` fixes the raster extent to 1920x1080.
This is valid for the 4K analysis proxy, but not for ordinary 1280x720 input,
which is not proxied by orchestration.

On sample B, numerical source-coordinate checks found 30 masks touching the
actual 1280x720 boundary (29 bottom, 3 right, with overlap), while the ellipse
stage reported zero edge rows. Thirteen of these rows were routed to K2 and 17
to K1. The K1 edge rows had median IoU 0.7830 and minimum IoU 0.6462.

Video geometry must therefore be passed into ellipse and polygon preparation;
it must not be inferred from mask maxima or silently defaulted to 1080p.

### P1: K2 edge features lose the true edge after square padding

K2 constructs its 10-channel input by rasterizing a clipped local crop,
square-padding the crop, and only then computing four edge-touch flags. When
padding is added to the touched side, a true screen-edge mask no longer touches
the padded-mask edge. In sample A all 61 true edge cases had a mismatch in the
post-padding flag (56 bottom and 5 right).

Forced K2 on sample A showed a distinct edge tail:

- non-edge p1 IoU: 0.9411; minimum: 0.9302;
- edge p10 IoU: 0.9089; minimum: 0.8395;
- 11 edge rows fell below 0.90 IoU.

Changing inference alone to compute the flags before padding is unsafe. With
the existing checkpoint it improved 27 edge rows and regressed 34, including
regressions near 0.10 IoU. The checkpoint was evidently calibrated against a
different/legacy channel meaning. The correct fix needs matching training and
inference semantics, followed by an edge-heavy validation set.

### P1: K2 slots have no explicit anti-crossing/coherence constraint

K2 predicts two unordered SPD ellipses. Temporal slot order is stabilized
afterward with a greedy keep-or-swap comparison, but the model objective does
not directly penalize visually implausible crossed ellipses. Low-IoU cases
contain close centers, angle differences of roughly 50--88 degrees, and in
some cases a very small second component. This numerically matches the reported
bow-tie appearance.

K2 was still useful: every production-selected K2 row in samples B and C had
higher exact IoU than its K1 alternative. Therefore a blanket K1 fallback is
not an acceptable correction. A constrained K2 refinement or retrained slot
objective is required.

### P1: keyframe saliency under-targets affine and local shape change

The reusable `tentative.analyze_temporal_geometry` diagnostic decomposes
adjacent contour motion into translation, similarity, full affine, and local
post-affine residual without reading video frames.

For sample A, exact-keyframe selection increased coverage of the top 10% of
translation events by 11.2 percentage points over its baseline selection
rate, but only increased affine-event coverage by 0.9 points and local-shape
coverage by 4.2 points. Direction reversals were exact keyframes only 20.8% of
the time for polygon and 25.8% for ellipse.

Across samples B and C, ellipse keyframes selected top-decile translation
events 4.0--5.0 points more often than baseline, but local-deformation event
selection changed by +0.6 points and -1.8 points respectively. The existing
system therefore reacts mainly to centroid motion, not to the requested
affine-versus-local shape distinction.

The polygon v22 optimizer removes translation, rotation, and uniform scale via
a similarity transform. Anisotropic scale and shear remain mixed with local
shape deformation. Its interval cost does evaluate exact masks, but its recall
constraint is an aggregate budget, so it also permits tail failures. In sample
A, 40 rows fell below 0.90 IoU and four rows below 0.90 recall.

### P1: evaluation disagrees at screen boundaries

The production ellipse solver clips out-of-frame predictions to the 1920x1080
canvas when calculating exact metrics. The generic ellipse and mask evaluators
instead allocate an unbounded box around both polygons, counting invisible
off-screen ellipse area as false positive. This produced a reported minimum
edge IoU of 0.2756 where a canvas-clipped recomputation gave 0.6546.

The evaluator must use the same explicit video geometry as production. Until
then, screen-edge quality reports are internally inconsistent.

### P2 (fixed): polygon subprocess paths depended on absolute paths

The polygon adapter passed stage artifact paths to a subprocess and changed the
subprocess working directory. Relative output roots could consequently resolve
to a different location and open a new empty SQLite (`no such table: masks`).
The adapter now resolves the stage directory, tracked SQLite, predictor, and
derived artifacts before spawning the vendor runtime. A regression test runs
the production polygon pipeline from a relative output root. This robustness
fix does not alter mask geometry or the public SQLite schema.

## Measured performance context

No frame pixels were inspected during these measurements.

- v3-lite local inference: 134.3 fps on B and 144.2 fps on C (2,400 frames).
- Ellipse postprocess stage totals: 6.95 s on B and 3.73 s on C.
- Ellipse approximation itself: 3.87 s on B (480 K2 rows) and 2.08 s on C
  (8 K2 rows).
- Sample A polygon processing: 34.33 s total; about 27.43 s in the polygon
  optimizer. Polygon quality improvements must preserve the exact v22 output
  contract while optimizing candidate/interval evaluation.

## Recommended implementation order and acceptance criteria

1. **Explicit geometry contract (no SQLite schema change).** Probe video width
   and height once in orchestration/postprocess, pass them into ellipse,
   polygon boundary preparation, and all exact evaluators. Add 720p, 1080p,
   and 4K-proxy boundary fixtures. Expected: every true boundary row receives
   the same edge classification in preparation, K1/K2, and evaluation.
2. **Fix evaluator clipping.** Relative subprocess paths are already fixed and
   covered by regression tests. The remaining isolated robustness change is to
   make all three evaluators use the same explicit canvas at image edges.
3. **Keyframe vNext behind an experimental switch.** Detect events on the raw
   unsmoothed signal, allow `min_gap=1`, use a shared frame set for both K2
   slots, and make the DP quality-constrained rather than count-constrained.
   The interval constraint should include sampled union IoU/precision/recall,
   not only parameter RMSE. Keyframe ratio becomes an observed outcome or soft
   storage ceiling, not the primary objective. Expected on current samples:
   no frame below the configured recall floor, p1 IoU materially above 0.827,
   and no synthetic period-2/4 amplitude collapse.
4. **Add full-affine/local-deformation saliency for polygon.** Keep translation,
   full affine, and post-affine residual as separate signals and record their
   contribution in diagnostics. Preserve v22 exact interval evaluation and
   vertex quality. Expected: top-decile event capture rises above baseline for
   all three signal families without increasing the same-quality keyframe
   count.
5. **K2 vNext.** Correct edge channels in both training and inference; add
   edge-heavy and crossed-slot data; use permutation-invariant union loss plus
   slot separation/coherence and temporal assignment constraints. Before a
   retrained model is accepted, use exact source-mask refinement only on
   suspicious K2 outputs. Expected: no edge/non-edge p1 gap of the current
   magnitude, no crossed-slot visual proxy violations, and no regression in
   global recall or throughput beyond an agreed budget.

Production promotion should be gated by the existing output-schema validator;
none of these changes requires altering the final SQLite schema.
