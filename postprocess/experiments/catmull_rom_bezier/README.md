# Closed Catmull–Rom / Bézier approximation experiment

This isolated experiment implements the editor's exact curve contract:

- closed, uniform Catmull–Rom spline;
- tension `1.0`;
- one cubic Bézier segment from `P[i]` to `P[i+1]`;
- Bézier handle factor `1/6`;
- only interpolation points `P[i]` are editable or optimized.

`B1` and `B2` are always derived by `model.bezier_segments()`.  There is no
free-handle fitting API.

## Point placement

The fitter first aligns a dense version of every source contour in time.  It
then selects one persistent set of contour locations for the entire track.
Curvature and chord-deviation saliency attract points to corners and deep
concavities.  The initial points are followed by a whole-boundary linear fit
using the exact Catmull--Rom basis matrix, so all sampled curve pixels (rather
than only the points `P`) contribute to the fit.  Corrections are temporally
regularized and clipped relative to local chord length.  Exact raster
IoU/Recall and low-tail IoU select the final blend.  A location may move only
between its two neighbours, so point identities cannot swap.  Finally, a
small temporally regularized isotropic scale repairs raster Recall while
keeping the same number and correspondence of points.

After fitting, cyclic point numbering receives a conservative temporal phase
gate.  Ordinary motion keeps the original shape-based correspondence.  A
cyclic roll is changed only when its translation-normalized XY error is at
least `0.50` and at least eight times worse than another roll.  Rolling a
closed loop does not change its curve; it only prevents a half-contour point
number jump from collapsing the linearly interpolated intermediate curve.

The output includes the editable `P` values and the derived `B0..B3` values for
every frame, plus a same-point-count RDP polygon baseline.

## Run

```bash
PYTHONPATH=postprocess \
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  -m experiments.catmull_rom_bezier \
  --input-sqlite /path/to/tracked.sqlite \
  --track-id 12 \
  --control-points 6,8,10,12 \
  --output-dir output/catmull_rom_bezier_track12
```

Use `--list-tracks` to inspect available tracks.  This first experiment fits
one explicitly selected connected component.  It fails instead of silently
switching component identity when that component is absent.

This package is not imported by Production.

## Keyframe-DP comparison

`run_keyframe_review` compares two endpoint representations under one shared
hard-minimum-Recall keyframe graph:

1. Production's current persistent line-fit polygon placement;
2. the closed Catmull–Rom interpolation points from this experiment.

Both use the same point count, target interval and exact per-frame minimum
Recall constraint.  The polygon baseline uses the current persistent line-fit
and its single-state DP.  The final curve candidate additionally uses five
small bounded scale states (`1.000` through `1.035`), a quadratic low-IoU loss,
and an exact-gated quality rescue.  The target interval remains soft: rescue
keys are admitted only when their spatial evidence materially repairs a bad
interpolation, and the allowance shrinks with the remaining key-density
slack.  A support key at an adjacent frame may be inserted unchanged to
localize a shape correction that a long edge would otherwise prevent.

For every curve edge, the optimizer first linearly interpolates only `P[i]`,
derives all `B1/B2` handles again with factor `1/6`, and only then rasterizes
the curve.  Pair-vote and the curve-specific local refinement also edit only
`P`; free Bézier handles never enter the state space.  Every accepted state,
edge, rescue and final dense frame passes exact raster Recall and strict
self-intersection gates.

```bash
PYTHONPATH=postprocess \
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  -m experiments.catmull_rom_bezier.run_keyframe_review \
  --input-sqlite /path/to/tracked.sqlite \
  --track-id 12 \
  --point-counts 8,10,12 \
  --target-intervals 3,6 \
  --source-video /path/to/source.mp4 \
  --output-dir output/catmull_rom_bezier_dp_review
```

The generated `curve_dense_review.sqlite` stores a densely sampled curve as a
polygon so existing overlay tools can display it.  It is not the semantic
curve interchange format.  `curve_keyframes.json` is authoritative and stores
the editable `P` values plus the automatically derived `B0..B3` values.

When `--source-video` is supplied, the runner also writes:

- `review_gallery/`: numerically selected low-IoU, regression and improvement
  frames;
- `review_sequence.mp4`: every evaluated frame under one fixed ROI, with the
  source mask, polygon result and curve result side by side;
- per-panel IoU, Recall, area ratio and `KEY`/`interp` status.

The MP4 is reopened and every frame is decoded before the run is accepted.
Multiple track runs can be combined without rerunning the optimizer:

```bash
PYTHONPATH=postprocess \
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  -m experiments.catmull_rom_bezier.aggregate_reviews \
  output/review_male output/review_female output/review_joint \
  --output-dir output/catmull_rom_bezier_multi_track_review
```

Use `--single-state-curve` to reproduce the initial representation-only
ablation.  The default is the validated curve-specific multistate path.  The
state palette contains no large-mask candidate and can be overridden with
`--curve-state-scales` for ablation.

The initial experiment used a twelve-key exact-gated rescue cap; those numbers
are retained only as historical ablation context.  Promoted Production has no
artificial rescue-key quota by default, because Recall and local-quality gates
take priority over the soft target interval.  A positive
`--curve-rescue-maximum-keys` remains available only for controlled Pareto
experiments.

Dense sampled curves use an exact sweep-line broad phase around Production's
existing strict segment-crossing predicate.  It is a computation-only
replacement: the experiment test suite verifies parity with the scalar
Production predicate.  On a 288-edge sampled curve this reduced 10,000 checks
from about 22.7 seconds to 3.7 seconds without changing the result.

Input frames must be contiguous.  The runner rejects a track slice with a
frame gap instead of silently treating non-adjacent observations as one-frame
neighbours.  Dense SQLite files are review artifacts; the authoritative curve
interchange is `curve_keyframes.json` containing `P` and the exactly derived
`B0..B3` segments.
