# Initial-shape search handoff: `production_candidate_best_v4`

## Conclusion

`production_candidate_best_v4` is the best initial-shape palette found in the
2026-08-11 search.  It improves the frozen `production_candidate_baseline_v1`
Pareto points at requested intervals 5 and 8 while keeping pair-vote and
post-decode repair disabled.  It is a research handoff, not yet the Production
default: isolated local regressions and the CUDA/exact Recall boundary must be
handled in the next objective/repair phase.

No video was decoded or opened.  All candidates and measurements use polygon
geometry already stored in SQLite.  Production source and the final SQLite
schema were not changed.

## Selected palettes

Every tuple below also includes the `raw` state.

| Class / requested interval | Candidate states |
|---|---|
| Female, interval < 2 | `C02_125, G02, G04, A06, F3_P1, D6_P1` |
| Female, interval >= 2 | baseline tuple + `F3_Q75_P1` |
| Male, all intervals | `C02_125, G02_H3, G04_H3, A06_K3, F3_P1, D6_R5_P1` |
| Joint, interval < 4 | `C02_125, G02, G04, A06, F3_P1, D6_P1` |
| Joint, 4 <= interval < 7 | baseline tuple + forward `VF8_P1` |
| Joint, interval >= 7 | `C02_125, G02, G04, A06, VF8_P1, VB8_P1` |

The interval-dependent joint palette is intentional.  At interval 5 the
forward VF8 superset has the better lower-tail quality.  Around interval 8,
the paired forward/backward vertex trend replaces F3/D6 and gives more reach
with essentially unchanged mean IoU and better q01/q05 than the one-direction
variant.

## Full-data Pareto result

The source contains 24,501 class-observation rows over 23,510 video frames.
Recall floor is 0.97 and the exact metrics below include every emitted frame.

| Requested interval | Profile | Effective interval | Mean IoU | Exact-feasible mean IoU | Worst class q01 IoU | Max area ratio | Infeasible streams | video FPS |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | frozen baseline v1 | 4.732904 | 0.899494 | 0.900417 | 0.493196 | 4.371832 | 5 | 269.996 |
| 5 | **best v4** | **4.750903** | **0.901221** | **0.904804** | **0.501421** | 4.371832 | 6 | **252.637** |
| 8 | frozen baseline v1 | 6.351042 | 0.870233 | 0.870486 | 0.404253 | 4.404167 | 3 | 272.982 |
| 8 | **best v4** | **6.456072** | **0.871793** | **0.874686** | **0.420098** | **4.371832** | 4 | **253.840** |

Thus the selected shapes move both tested points outward in effective interval
and upward in mean IoU.  The strict-Recall-feasible subset gains +0.00439 IoU
at target 5 and +0.00420 at target 8.  Both final points remain above the
240-video-FPS requirement.

The complete final runs are:

- `output/phase2_best_v4_parallel3_cv8_full_i1_20260811/`
- `output/phase2_best_v4_parallel3_cv8_full_i3_20260811/`
- `output/phase2_best_v4_parallel3_cv8_full_i5_20260811/`
- `output/phase2_best_v4_parallel3_cv8_full_i8_20260811/`

The final profile is monotonic over the complete 1/3/5/8 validation curve:

| Requested interval | Effective interval | Mean IoU | Exact-feasible mean IoU | Worst class q01 | video FPS |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.201202 | 0.967883 | 0.969261 | 0.852372 | 256.934 |
| 3 | 2.980081 | 0.935742 | 0.942515 | 0.674561 | 256.751 |
| 5 | 4.750903 | 0.901221 | 0.904804 | 0.501421 | 252.637 |
| 8 | 6.456072 | 0.871793 | 0.874686 | 0.420098 | 253.840 |

## What the search established

- One common six-state palette is inferior to class-specific palettes.
- Increasing the C02 area cap to 1.50 or 1.75 extends difficult joint tracks,
  but lowers joint mean IoU to about 0.778 or 0.759.  It is a low-quality
  reach tradeoff, not a Production candidate.
- Short/mid/long 30%-support temporal shapes, broader directional support,
  trajectory residuals, and robust vertex fits were either redundant with the
  baseline states or worsened the Pareto point.
- Horizon-8 per-vertex line fitting is the only new family that consistently
  improves difficult joint tracks.  Forward-only is preferable near interval
  5; paired forward/backward states are preferable near interval 8.
- The male D6-base reverse ablation improves mean IoU and uses slightly fewer
  keys than the selected male palette, but loses q01/CVaR quality.  It remains
  an explicit reach-biased alternative and is not the default.
- Large masks can always buy keyframe reach.  The accepted palette therefore
  excludes the aggressive cap ladder even when it improves target proximity.

## Remaining local-quality problem

The aggregate lower tail improves for male and joint masks, but adding states
changes the globally selected path.  Some formerly keyed frames become
interpolated and can be locally much worse even though the track objective is
better.  The worst comparisons to the frozen baseline were:

| Target | Class | Frame | Baseline IoU | v4 IoU | v4 area/raw area | Baseline key -> v4 key |
|---:|---|---:|---:|---:|---:|---|
| 5 | Female | 2120 | 0.9376 | 0.6727 | 1.487 | 1 -> 0 |
| 5 | Male | 10175 | 0.9553 | 0.6614 | 1.490 | 1 -> 0 |
| 5 | Joint | 22975 | 0.9458 | 0.4659 | 2.140 | 1 -> 0 |
| 8 | Female | 3784 | 0.9201 | 0.6680 | 1.435 | 1 -> 0 |
| 8 | Male | 9890 | 0.8317 | 0.4988 | 1.925 | 1 -> 0 |
| 8 | Joint | 22195 | 0.9846 | 0.4787 | 2.037 | 1 -> 0 |

This is not solved by adding another initial shape.  It is the expected next
objective-level problem: the current mean interval loss plus lambda-per-key
can trade one bad frame for a better whole-track score.  The next phase should
measure pair-vote first, then add a low-tail/local-expansion term or an exact
post-path guard.  A blanket minimum IoU-to-raw constraint is not recommended,
because it would also prevent desirable correction of genuinely broken raw
masks.

The candidate search therefore stops here instead of hiding this problem with
larger masks or post-hoc edits.

## Recall boundary

The CUDA scanline classifier and exact OpenCV rasterizer differ at a small
number of polygon boundary pixels.  The final exact audit has zero violations
on every stream classified as feasible, but v4 exposes one more infeasible
stream than the frozen baseline at each full point.  This issue also exists in
the baseline and is not caused by SQLite or the candidate generator.  Before
promotion, accepted CUDA paths need exact edge revalidation/repair or a proven
conservative margin.

## Performance and determinism

The stable fast schedule is three class processes in parallel with one CUDA
worker per class and eight OpenCV threads per class.  Spawning multiple CuPy
workers inside each class intermittently raises `cudaErrorInitializationError`
under WSL.  The runner now exposes the isolated CuPy site to the parent so the
real exception is preserved instead of being masked by
`ModuleNotFoundError: cupy_backends`.

The OpenCV thread allocation changed scheduling only.  Before/after outputs
were byte-identical for:

- `opt/final_keyframes.json`
- `opt/interpolated_union.json`
- `exact/keyframe_exact_metrics.csv`
- `pred/predictions.sqlite`

The final target-8 wall time is 92.62 seconds (253.84 video FPS); target 5 is
93.06 seconds (252.64 video FPS).

Validation completed:

- 56 focused polygon/Recall/Pareto tests passed;
- native raster parity: 207 scalar cases and 120 batch edges;
- native exact evaluator speedup in its microbenchmark: 10.66x; and
- final target-5 artifacts are byte-identical to the screened v2 composition,
  while target-8 artifacts are byte-identical before/after thread tuning.

## Next work order

1. Freeze this v4 initial-shape palette for subsequent ablations.
2. Measure pair-vote's selection, IoU-tail, Recall, area, and key-count effects
   against the v4 no-pair-vote outputs.
3. Add a local-tail/area-acceleration guard to both DP and any pair-vote update.
4. Add exact validation of selected CUDA edges and eliminate the extra
   infeasible stream before Production promotion.
5. Re-run targets 1, 3, 5, and 8, then export the software-compatible SQLite.
