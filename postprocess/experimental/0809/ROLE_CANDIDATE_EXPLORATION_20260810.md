# Role-based initial-shape exploration — 2026-08-10

## Scope

- Experimental code only; Production and the SQLite schema were not changed.
- Video pixels were never opened.  All candidates and metrics use SQLite polygon geometry.
- Pair-vote and post-decode repair remained disabled.
- Hard per-frame raw-mask Recall was fixed at 0.97.
- Interval edges were evaluated with the exact C++ evaluator for the reported quality results.

The first-priority pool from the external proposal was implemented: A2, A4,
D6, B3, C1, C6, E2, F3, G3 and Z1.  Small-ROI TSDF aggregation is shared by
the relevant candidates.  This is the prescribed first screening stage, not a
claim that all 50 proposed rules have been implemented.

## Representative source

One whole track nearest 650 observations was selected deterministically from
each class in the existing half-track source:

| Class | Track | Observations |
|---|---:|---:|
| female genital | 26 | 676 |
| male genital | 33 | 661 |
| joint | 69 | 708 |

Total: 2,045 observations.  No track was cut.

## Candidate sets

- `scale6`: raw + fixed isotropic scales 1.02/1.04/1.06/1.08/1.12/1.16.
- `roles6`: raw + A2/D6/B3/C1/E2/F3, exactly the proposed initial set.
- `hybrid6`: raw + S102/F3/E2/B3/C1/D6.  S102 is retained as a minimal
  coverage state after `roles6` was found unable to form a feasible path.

## Exact target-interval-5 results

| Class | Set | Actual interval | Keys | Mean IoU | Min IoU | IoU q01 | IoU q05 | Area q99 | Area max | Infeasible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| female | scale6 | 5.000 | 136 | .972714 | .900443 | .927104 | .947201 | 1.061742 | 1.079929 | 0 |
| female | roles6 | 5.037 | 135 | .968768 | .895705 | .918796 | .939947 | 1.063329 | 1.094017 | 0 |
| female | hybrid6 | 5.000 | 136 | **.973262** | .897211 | .923864 | **.950412** | 1.062718 | 1.089370 | 0 |
| male | scale6 | 5.038 | 132 | .956449 | .684871 | .888137 | .921784 | 1.101046 | 1.394992 | 0 |
| male | roles6 | 3.511 | 189 | .958008 | .714520 | .868596 | .920490 | 1.134239 | 1.360737 | 0 |
| male | hybrid6 | 5.038 | 132 | **.956541** | **.716694** | .887543 | **.924806** | **1.083418** | **1.328489** | 0 |
| joint | scale6 | 4.497 | 164 | .773307 | .487448 | .530387 | .619779 | 1.832127 | 2.051500 | 0 |
| joint | roles6 | 1.000 | 734 | .966955 | .950694 | .955774 | .959259 | .999371 | 1.002506 | **1** |
| joint | hybrid6 | 1.837 | 400 | .932007 | .653514 | .747174 | .834275 | 1.302764 | 1.459046 | 0 |

All feasible paths had exact minimum Recall >= 0.97.  Every interpolated
polygon from these nine outputs was checked with Shapely: 6,135/6,135 were
valid, with zero self-intersections or non-positive-area polygons.

## Representative Pareto points

Aggregate across the three representative tracks:

| Requested interval | Set | Actual interval | Keys | Mean IoU | Area max | Infeasible |
|---:|---|---:|---:|---:|---:|---:|
| 1 | scale6 | 1.302 | 1,591 | .973295 | 1.818600 | 0 |
| 1 | roles6 | 1.176 | 1,762 | .974314 | 1.152390 | 1 |
| 1 | hybrid6 | 1.234 | 1,679 | **.974409** | 1.302905 | 0 |
| 5 | scale6 | 4.821 | 432 | .898420 | 2.051500 | 0 |
| 5 | roles6 | 1.960 | 1,058 | .964662 | 1.360737 | 1 |
| 5 | hybrid6 | 3.110 | 668 | .953574 | 1.459046 | 0 |
| 10 | scale6 | 7.010 | 298 | .874976 | 2.051500 | 0 |
| 10 | roles6 | 1.966 | 1,055 | .963858 | 1.360737 | 1 |
| 10 | hybrid6 | 3.464 | 600 | .943907 | 1.459046 | 0 |

The sets are not compared at identical lambda; each row is the solution
closest to its requested interval under the hard Recall constraint.  The table
therefore describes the reachable trade-off region, not a same-key-count
quality comparison except where actual intervals match.

## Candidate use and ablation findings

- In `roles6`, selected male key states were raw 55, B3 4, C1 6, E2 27 and
  F3 97.  A2 and D6 were never selected on that track.
- In `hybrid6`, S102 remained the dominant coverage state.  F3 and E2 supplied
  most of the non-scale corrections; B3/C1/D6 had smaller, class-dependent use.
- Batch 1 (A2/A4/D6/B3/C1/C6) was infeasible for male and joint tracks.
- Batch 2 (E2/F3/G3/Z1/A2/C1) was feasible for female and male but infeasible
  for the joint track.
- G3 was equivalent to F3 in this representative sample because none of the
  selected tracks touched a known image boundary.  Right/bottom-edge detection
  also requires frame dimensions to be supplied to the experimental runtime.
- The joint track required every isotropic scale state, including 1.12 and
  1.16, to approach interval 5.  Replacing all scale states by role states
  improved mask fidelity but removed the shape coverage needed for long edges.

## Runtime

Target interval 5, exact C++ evaluation, 2,045 observations:

| Set | Candidate build | DP/evaluation | Profile wall |
|---|---:|---:|---:|
| scale6 | 1.00 s total | 48.61 s total | 58.00 s |
| roles6 | 23.62 s total | 34.09 s total | 65.06 s |
| hybrid6 | 17.65 s total | 34.75 s total | 60.21 s |

TSDF role generation is currently exploratory Python/OpenCV code and is not
yet cached across profiles.  Its output quality, rather than its current build
speed, was the target of this pass.

## CUDA diagnostic

On male track 33, exact C++ evaluation found `roles6` feasible with actual
interval 3.511.  The current approximation-only CUDA graph marked the same
track infeasible and retained only 4,107 of 948,885 edges.  Fixed scale states
hide this raster false-negative by outward expansion.  Consequently the
existing zero-deficit CUDA prefilter is not a valid sole judge for raw-like
role candidates; exact verification or a calibrated conservative margin is
required during future candidate screening.

## Result

The proposed fixed set `{A2,D6,B3,C1,E2,F3}` is **not** a replacement for the
fixed-scale states under the current hard raw-Recall formulation.  It improves
shape fidelity but lacks sufficient outward coverage, and it is infeasible for
the representative joint track.

The first useful result is the hybrid point.  For female and male tracks it
matches interval 5 while improving several quality/tail/expansion metrics.  For
the joint track it forms a different non-dominated point: substantially better
IoU and area control, but many more keys.  The evidence therefore supports
class-specific state palettes and retaining at least one calibrated coverage
state; it does not support replacing all six scale states globally.
