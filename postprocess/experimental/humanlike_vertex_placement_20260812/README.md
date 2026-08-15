# Human-like fixed-count polygon placement (2026-08-12)

This experiment studies replacing Production's adaptive polygon anchors with a
fast, corner-aware fixed-count approximation.  The 48-point equal-arclength
contour used in the earliest experiment was an artificial cap/reference, not
the actual Production output contract.  It operates on SQLite polygon geometry
only.  No video frame was decoded, displayed, or sent outside the local machine
during this work.

## Current best: persistent section line fit

The earlier native contour-point DP documented below is now a measured
baseline, not the current winner.  The current candidate is implemented in
`persistent_line_fit.py`, `shared_area_dp.py`, and `quality_repair.py`.

It keeps old Production's image-coordinate phase convention, allocates a fixed
non-overlapping contour section to every vertex for the whole track, fits a
supporting boundary line to each section, and uses regularized adjacent-line
intersections as vertices.  Consequently a vertex is no longer restricted to
one raw contour sample, while its semantic section cannot swap with a neighbor.
Signed-area and maximum-chord-distance terms jointly protect broad curvature,
deep narrow concavities, and corners.  Exact raster guards minimally blend only
failing frames toward a phase-aligned RDP fallback.

On five comparable Production tracks (3,401 frames), every fixed count from 14
through 18 beats the old Production fixed-count placement at the same count in
both mean IoU and interpolation IoU.  At 14 points the frame-weighted mean IoU
is 0.974783 versus 0.950572, the worst IoU is 0.950141, worst Recall is
0.970304, self-intersections are zero, and guarded throughput is 337 FPS.  Only
2/3,401 frames used the sparse repair.  Actual Production uses a frame-weighted
21.95 points on these tracks, so 14 points is a 36.2% reduction.  Full results:
`output/humanlike_vertex_placement_20260812/production_superiority_final/REPORT.md`.

## Why the first decimator failed

The first shared-index deletion experiment could only delete points from a
48-point equal grid.  It could not move a surviving point to a real corner.
One Track 55 result consequently left 9/48 of the perimeter without a vertex
and bridged a deep notch with an invalid shortcut.  Recall alone did not reject
that over-coverage, and a track-mean temporal objective hid the rare bad frame.

Several replacements were measured and rejected:

- fixed shared arclength indices: stable but missed moving concavities;
- curvature-density quantiles: fast, but minimum Recall stayed below 0.97;
- curvature-DTW registration: too inaccurate and only 64--100 FPS;
- Python band DP: better correspondence, but only 5--27 FPS;
- global-strength temporal smoothing: one strength could not serve both easy
  and critical frames;
- adaptive smoothing: safe and better, but only about 140 FPS.

## Selected algorithm

1. Convert every contour to one CCW orientation.
2. Build an exact-count closed-curve RDP hierarchy.  Vertices are actual source
   contour points, so points concentrate at corners and deep concavities.
3. Assign cyclic vertex IDs from the temporal centre outward.  Only translation
   is removed while comparing phases; rotation remains observable.  This is
   essential: full Procrustes normalization made symmetric shapes ambiguous
   and caused half-perimeter phase flips.
4. Run a C++ fixed-count dynamic program over the complete contour.  Each edge
   is scored by missed/excess signed area, maximum chord deviation, and a small
   penalty from the temporally predicted vertex position.
5. Validate the complete track/cut segment with exact raster metrics.  A vertex
   count is accepted only when minimum Recall >= 0.97, minimum IoU >= 0.95, and
   self-intersections are zero.  The same count is retained for the entire
   contiguous segment.
6. Compression mode tries 16, 17, 18, 20, 24, 32, then 48 vertices.  Fast-safe
   mode starts directly at 20 and uses 24/32/48 only as a fallback.

The default native weights selected across classes are:

```text
temporal_weight      = 0.003
distance_weight      = 2.0
missing_area_weight  = 1.0
minimum Recall       = 0.97
minimum IoU          = 0.95
```

## Full-track geometry benchmark

The validation set contains 4,077 frames from six long single-component tracks
covering male genital, female genital, and union-region classes.  All results
below have zero self-intersections.

| Track | Class | Frames | Selected vertices | Mean IoU | Min IoU | Min Recall | Guarded FPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 55 | male | 979 | 18 | 0.98318 | 0.96552 | 0.97133 | 294 |
| 47 | male | 456 | 18 | 0.97690 | 0.96654 | 0.97036 | 313 |
| 29 | female | 676 | 16 | 0.98141 | 0.97582 | 0.97586 | 947 |
| 36 | male | 661 | 18 | 0.97807 | 0.96472 | 0.97202 | 323 |
| 66 | union | 680 | 17 | 0.97790 | 0.96440 | 0.97228 | 766 |
| 97 | union | 625 | 17 | 0.98139 | 0.96232 | 0.97146 | 461 |

Compression mode averages 17.35 vertices weighted by frames.  This is 63.9%
below the artificial 48-point experimental reference, but that percentage is
not a reduction from Production.  It runs at 517 FPS mean and 294 FPS worst
case including every failed lower-count attempt and exact validation.

At those same selected counts, equal-arclength sampling obtained a
frame-weighted mean IoU of 0.96720 and violated minimum Recall 0.97 on every
track.  The selected algorithm obtained 0.98020 weighted mean IoU
(**+0.01300**) and passed the Recall guard on every track.  The gain therefore
comes from putting points at useful corners, not merely accepting less detail.

Fast-safe mode uses 20 vertices on all six tracks: **58.3% fewer vertices**,
1,022--1,950 FPS for generation, and 811--1,380 FPS including exact validation.
Its mean IoU range is 0.98076--0.98832 and minimum Recall range is
0.97057--0.98409.

The previous Python RDP seed and phase stage consumed 1.63 s of a 2.51 s Track
55 run.  Moving it to C++ changed mean IoU by only 0.0000017 while increasing
generation from 390 to 1,089 FPS on that track.

For reference, the experimental 48-point Python equal-arclength construction ran at
about 560 FPS on the same tracks.  The 20-point native candidate is therefore
both smaller and faster, although its intended quality contract is the explicit
0.97 Recall / 0.95 IoU guard rather than exact reproduction of the 48-point
mask.

## Comparison with actual Production vertex counts

The actual `data/12月KPI動画.sqlite` Production keyframe polygons are not fixed
at 48 points.  Across 4,689 stored polygon rings they contain 22.02 points on
average (median 22, range 11--33).  On five directly comparable polygon tracks,
Production uses 21--23 points while the quality-guarded candidate selects
17--18.  Weighted by the evaluated Production keyframes this is 21.93 versus
17.60 points, a **19.7% reduction**.

An exhaustive 8--23 point sweep found that 8--16 points cannot satisfy both
minimum Recall 0.97 and minimum IoU 0.95 on all tested tracks.  The selected
counts were 18 for tracks 55/47/36 and 17 for tracks 66/97.  Starting at 8 and
testing every integer runs at 108 mask-frames/s including exact validation;
starting at 16 reaches the same choices at 378 mask-frames/s.

The reduced representation is not yet a complete Production replacement.  At
Production keyframe times it raises weighted mean IoU from 0.96338 to 0.97968
and the worst keyframe Recall from 0.77794 to 0.97146, but its
similarity-corrected vertex movement is 2.14x Production when normalized by
object radius.  Dense interpolation at the unchanged Production key schedule
is essentially tied (mean IoU 0.95548 Production versus 0.95499 candidate).
This isolates semantic vertex correspondence, not vertex count, as the next
blocking problem.  Reproduce the comparison with
`compare_production_vertex_budget.py`; detailed results are in
`output/humanlike_vertex_placement_20260812/production_budget_comparison_8_23/`.

## Phase-flip incident and fix

On Track 47, the rejected Procrustes phase alignment changed the cyclic phase
by 10 positions out of 20 over three frames.  Endpoint masks had areas around
7,500--8,200 px, but the linearly interpolated polygon collapsed to roughly
800--1,000 px.  Translation-only phase scoring selected shift zero and removed
the collapse.  The same fix removed previously observed catastrophic
interpolation failures on female and male tracks 29 and 36.

## Files

- `spatial.py`: reference RDP/Visvalingam and translation-only phase logic.
- `native_temporal/`: optimized C++ RDP, phase assignment, and polygon DP.
- `native_dp.py`: cut/gap-aware Python bridge.
- `candidate.py`: exact quality guard and fixed-count fallback.
- `interpolation_audit.py`: keyframe interpolation QA.
- `benchmark_*.py`, `tune_native_dp.py`: reproducible experiments.
- `postprocess/tests/test_humanlike_vertex_placement.py`: focused regression
  tests.

Build the native module with:

```bash
postprocess/experimental/humanlike_vertex_placement_20260812/native_temporal/build.sh
```

## Remaining scope before Production promotion

- **Local vertex allocation is not yet temporally invariant.**  The total
  count, winding, order, and gross cyclic phase are fixed, but the DP may move
  one vertex from one local arc to another.  In a six-track audit (4,071
  adjacent frame pairs / 70,625 vertex transitions), a heuristic for
  corner-to-straight or straight-to-corner allocation changes found 288 events
  whose motion exceeded half of the local vertex spacing.  Some are genuine
  shape changes, but the current objective does not distinguish them from
  unwanted local split/merge events.  Persistent corner anchors and fixed
  per-arc vertex quotas, with hysteretic controlled redistribution, are needed
  before Production promotion.  See `allocation_audit.py`.
- The experiment handles one outer polygon component.  Multi-component slots,
  holes, and component birth/death need an explicit policy.
- Metrics use the AI/raw polygon as reference, not human GT.  A final editor
  review is still required before replacing Production.
- Vertex placement cannot make genuinely nonlinear ten-frame motion linear.
  Keyframe selection must still add keys where interpolation quality falls.
- The Production SQLite schema was not changed.
