# Catmull--Rom interval trade-off validation (2026-08-25)

## Production contract

- Per-frame Recall `>= 0.97` and valid interpolated topology are hard graph
  and final-audit constraints.
- IoU and keyframe count are the trade-off. A low-IoU or enlarged frame does
  not add a key after DP and does not veto the requested Pareto point.
- The requested target interval is a soft target. If no path at that key count
  satisfies the hard constraints, the smallest feasible key count above the
  target is selected.
- Targets 1 through 6 use the same fixed-cardinality optimizer. There are no
  target-specific quality thresholds.
- Editable geometry remains Catmull--Rom interpolation points `P`; every
  Bezier handle is derived by the frozen closed uniform `1/6` conversion.

## State palette

The former isotropic-only palette could not express sparse valid paths for
several male and joint streams. Production now retains the inexpensive raw
and temporal-coverage probe first, and expands only streams that miss the
requested density:

- female: the existing compact isotropic palette;
- male: `raw`, `C02_125`, `A06_K3`, `D6_R5_P1`;
- joint: `raw`, `C02_125`, `A06`, `VF8_P1`.

The role transforms operate directly on the fixed Catmull points, preserving
point count and cyclic phase. They do not introduce free Bezier handles.

## Full KPI trade-off

All rows below come from the same 25,090-mask V3 KPI SQLite, six classwise
workers and the authoritative exact CPU raster path. Pair-vote and point
refinement remained enabled.

| Target | Keys | Effective interval | Mean IoU | IoU q05 | Minimum IoU | Mean area ratio | Area q95 | Recall violations | FPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25,090 | 1.000 | 0.985337 | 0.975841 | 0.934211 | 1.000899 | 1.007407 | 0 | 195.40 |
| 2 | 12,569 | 1.996 | 0.972148 | 0.927748 | 0.373807 | 1.006487 | 1.053266 | 0 | 219.92 |
| 3 | 8,485 | 2.957 | 0.956718 | 0.869131 | 0.339171 | 1.020606 | 1.128962 | 0 | 244.07 |
| 4 | 6,516 | 3.851 | 0.941434 | 0.829633 | 0.316516 | 1.039177 | 1.184096 | 0 | 253.02 |
| 5 | 5,404 | 4.643 | 0.927597 | 0.789901 | 0.240613 | 1.058051 | 1.238491 | 0 | 254.29 |
| 6 | 4,747 | 5.285 | 0.915507 | 0.757105 | 0.240613 | 1.075845 | 1.295262 | 0 | 244.58 |

Minimum Recall was at least `0.97` and topology-invalid component frames were
zero at every target. Effective interval rises monotonically while mean IoU
falls monotonically, so targets 1--6 now expose the intended key-count/IoU
trade-off. Target 6 does not reach an exact effective interval of 6 because
the hard Recall constraint makes that key count infeasible on some streams.

The low IoU tail and area maxima at sparse settings are deliberately reported,
not hidden: maximum area ratio was `4.1428` at targets 5 and 6. Under this
contract those are costs of selecting the sparse Pareto point, not reasons to
silently insert keys and change the user's requested trade-off.

## Controlled target-6 comparison

| Metric | Previous quality-guarded palette | Role-palette trade-off |
|---|---:|---:|
| Keys | 7,559 | 4,747 |
| Effective interval | 3.319 | 5.285 |
| Mean IoU | 0.958556 | 0.915507 |
| Recall violations | 0 | 0 |
| Classwise wall seconds | 107.548 | 102.585 |
| Output-mask FPS | 233.29 | 244.58 |

The adaptive two-state probe reduced the role-palette DP time enough to make
the new sparse trade-off faster than the former quality-guarded path. The
37.2% key reduction is paid for by a 0.0430 reduction in mean IoU, which is the
explicit behavior of the requested trade-off rather than a quality regression
at a fixed key count.

## Regression evidence

- Production Catmull--Rom tests: `37 passed`.
- All six full runs: Recall violations `0`, topology-invalid frames `0`.
- The runtime imports only Production modules; no experimental module is used.
- SQLite geometry and interpolation identifiers are unchanged.
