# Fixed-cardinality Catmull--Rom DP validation (2026-08-25)

## Final contract

- Minimum per-frame Recall: `0.97` (hard edge and final-audit constraint).
- Topology: no invalid interpolated frame.
- Key count: solve the requested cardinality directly; if it is infeasible,
  choose the smallest feasible count above it.
- States: first solve the inexpensive two-state graph, then use the validated
  four-state isotropic palette only for streams that miss the density or local
  quality gates.
- Local quality: after global selection, the existing exact quality guard may
  add keys for a local `IoU < 0.85` or `area ratio > 1.20`. The target interval
  is a soft goal; Recall and these catastrophic-failure guards take priority.
- Editable geometry: Catmull--Rom interpolation points `P` only; handles remain
  derived by the frozen `1/6` conversion.

The rejected variant forced a four-state temporal-endpoint palette for every
target from 4 through 6 and disabled the final local-quality guard. On the full
KPI corpus it reduced throughput and produced minimum IoU `0.2399` and maximum
area ratio `4.1543`. It is not part of Production.

## Controlled speed comparison

Both revisions were run consecutively on the same host, same 15-minute KPI
tracked SQLite, same six classwise workers, target interval 6, and CPU-only
exact raster path. The baseline is commit `efeb989`; the candidate uses the
fixed-cardinality decoder and the final contract above.

| Metric | Previous penalty DP | Fixed-cardinality candidate | Change |
|---|---:|---:|---:|
| Output mask rows | 25,090 | 25,090 | identical |
| Classwise wall seconds | 106.063 | 107.548 | +1.40% |
| Output-mask FPS | 236.56 | 233.29 | -1.38% |
| Maximum worker RSS KiB | 1,790,076 | 1,785,940 | -0.23% |
| Keyframes | 7,570 | 7,559 | -11 |
| Effective interval | 3.3144 | 3.3192 | +0.15% |

The 1.38% throughput difference is within the run-to-run spread observed on
this host. Two isolated checks support that interpretation:

- stable 300-frame track: baseline median `139.46 FPS`, candidate median
  `139.83 FPS`;
- difficult 1,332-frame track 48: baseline `55.19 FPS`, candidate
  `56.77 FPS`.

The native cardinality decode itself is about 3 ms for a 300-frame graph. The
dominant work remains exact interval rasterization, curve fitting and point
refinement. No expensive temporal-endpoint state family is used in Production.

## Full-corpus quality comparison

| Metric | Previous penalty DP | Fixed-cardinality candidate |
|---|---:|---:|
| Mean IoU | 0.958740 | 0.958556 |
| IoU q01 | 0.882244 | 0.884279 |
| IoU q05 | 0.907584 | 0.908823 |
| Minimum IoU | 0.851086 | 0.852016 |
| Minimum Recall | 0.970000 | 0.970000 |
| Recall violations | 0 | 0 |
| Mean area ratio | 1.015704 | 1.015495 |
| Area ratio q99 | 1.110255 | 1.109652 |
| Maximum area ratio | 1.167614 | 1.166193 |
| Local-quality violations | 0 | 0 |

The candidate preserves mean quality to within `0.000184` IoU while slightly
improving q01, q05, minimum IoU and area statistics. The difficult track 48
also remained stable: minimum IoU changed from `0.86542` to `0.86645`, maximum
area ratio from `1.10953` to `1.11213`, and effective interval from `5.026` to
`5.065`, with zero Recall violations.

## Decision

The fixed-cardinality decoder is retained, but only behind the adaptive
two-state/four-state isotropic search and the exact local-quality guard. This
provides direct Pareto-point selection where feasible without paying for a
larger graph on easy streams or forcing sparse but visibly broken masks on
difficult streams.
