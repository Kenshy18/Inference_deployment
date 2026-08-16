# Interval-8 bounded candidate search (2026-08-11)

## Scope and privacy

This search used polygon geometry already stored in SQLite. It did not open,
decode, inspect, or upload video frames. Pair-vote and post-decode shape repair
were disabled. No Production schema or Production pipeline implementation was
changed.

## Objective

The requested point was mean keyframe interval 8 under the exact per-frame
Recall floor 0.97, without buying keyframe reach through excessive endpoint
shape change or mask inflation. Candidate and final-path gates included mean
IoU, q01/q05 IoU, area-ratio tails, maximum area ratio, and infeasible streams.

## New candidate family

`IVF8`/`IVB8` invert the interval-8 linear interpolation equation. With the
opposite endpoint fixed, every intermediate raw polygon implies the endpoint
shape needed to reproduce that intermediate polygon. The candidate retains
only recurrent, local additions from those inverse projections. Translation,
rotation, and isotropic scale are removed before fitting, so the candidate
models residual non-rigid change rather than duplicating ordinary keyframe
motion interpolation.

The emitted endpoint is deterministic and is hard-capped to 1.10 or 1.15 of
the raw endpoint area. The 1.10 and 1.15 variants, forward/backward variants,
least-squares alternatives, replacements, supersets, and state-pruning
ablations were evaluated.

## What was rejected

- Endpoint least squares with 1.10/1.15 area caps did not create enough new
  Recall-feasible interval-8 edges.
- Inverse projection at 1.20 improved reach but reduced mean and lower-tail
  IoU; it was rejected.
- Adding both inverse directions to every class improved aggregate means but
  changed the joint-region path and worsened its q01/area tail; it was rejected.
- The forward male candidate caused a local long-edge expansion around track
  55/frame 13377. It was removed even though aggregate metrics improved.
- Replacing the joint backward VF state or either G state lost too much reach
  or Recall feasibility. The joint v4 palette was retained unchanged.

## Physical limit at requested interval 8

The difficult joint track 71 has 463 contiguous observations. For an
eight-frame chord, its centroid's maximum deviation from linear endpoint
interpolation is:

| statistic | pixels | fraction of equivalent object radius |
|---|---:|---:|
| median | 38.22 | 0.810 |
| q95 | 84.98 | 1.807 |
| maximum | 107.68 | 2.407 |

Track 71 therefore needs 200 keys (effective interval 2.315), and track 69
needs 198 keys (3.707), even in the strongest bounded candidate sets. A shape
candidate cannot cover a centroid excursion of roughly 0.8--1.8 object radii
while also remaining within a 10--15% area cap and satisfying per-frame Recall
0.97. The only quality-preserving action is to add keys in those nonlinear
motion sections.

Consequently, exact interval 8 is infeasible for this dataset under the stated
three conditions: per-frame Recall 0.97, linear polygon interpolation, and no
large shape inflation. Reaching an aggregate interval of exactly 8 would
require a solver-level global key budget that lets difficult tracks use short
intervals and compensates with intervals above 8 on quiet tracks. It cannot be
obtained honestly by another bounded endpoint shape alone.

## Safest interval-oriented point

`production_candidate_interval8_safe_v7` changes only the male palette:

```text
raw, C02_125, G02_H3, G04_H3, A06_K3, D6_R5_P1, IVB8_115
```

Female and joint palettes are byte-for-byte the v4 compositions. On the full
24,501 class observations:

| metric | best v4 | bounded safe v7 | delta |
|---|---:|---:|---:|
| effective interval | 6.456072 | 6.481193 | +0.025121 |
| keyframes | 3,973 | 3,958 | -15 |
| mean IoU | 0.871793 | 0.871630 | -0.000163 |
| feasible mean IoU | 0.874686 | 0.874516 | -0.000169 |
| worst-class q01 IoU | 0.420098 | 0.420098 | unchanged |
| maximum area ratio | 4.371832 | 4.371832 | unchanged |
| infeasible streams | 4 | 4 | unchanged |
| video FPS | 253.840 | 227.961 | -25.879 |

The v7 point is a genuine Pareto trade: 15 fewer keys and unchanged worst-tail
and maximum-inflation metrics for a 0.00016 mean-IoU cost. It is not a strict
replacement for v4, and its candidate-generation path still needs performance
work before Production promotion.

For the 134 selected non-gapfilled `IVB8_115` male keys:

| candidate geometry metric | median | q95 | maximum |
|---|---:|---:|---:|
| area / raw area | 1.0991 | 1.1279 | 1.1343 |
| raw-to-candidate centroid shift / radius | 0.0125 | 0.0366 | 0.0477 |
| Hausdorff distance / radius | 0.1326 | 0.2457 | 0.2878 |

Thus the accepted candidate itself does not contain the earlier whole-mask
overexpansion or large position shift.

## Software review artifact

The editor-compatible SQLite is:

`output/phase2_production_candidate_interval8_safe_v7_full_20260811/software_review/12月KPI動画_production_candidate_interval8_safe_v7_目標間隔8_Recall097.sqlite`

- size: 270,667,776 bytes
- SHA-256: `b133660aafd1103a7666bca38e36c9e45e320d21c910bf329967b443db61a649`
- schema fingerprint unchanged: yes
- `PRAGMA integrity_check`: `ok`
- foreign-key errors: 0
- frames: 23,510
- mask segments: 101
- editable mask keyframes: 3,961 (including exact-Recall boundary repairs)
- polygon points: 85,900
- cuts: 41

The adjacent validation JSON records the exact repairs and schema audit.

## Validation

- 56 focused polygon/Recall/Pareto tests passed.
- All final feasible streams have exact minimum Recall at or above 0.97.
- Pair-vote remained disabled.
- No video frame was opened.
