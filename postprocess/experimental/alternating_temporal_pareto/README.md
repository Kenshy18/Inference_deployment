# Alternating temporal Pareto experiment

This experiment is intentionally isolated from the production post-processing
pipeline. It never uses Production keyframe geometry. The source SQLite is
used only for raw AI polygons, track/cut topology, metadata, and schema-safe
export.

## Shape states

Every frame starts with seven fixed-count polygon states:

1. raw observation;
2. short-window IoU median;
3. short-window Recall quantile;
4. medium-window IoU median;
5. medium-window Recall quantile;
6. long-window IoU median; and
7. long-window Recall quantile.

The default window radii are 2, 5, and 10 observations. Neighboring polygons
are rigidly aligned to the target frame before aggregation. Border constraints
are built directly from raw observations by
`independent_raw_border_v1`; no Production border transform is called.

## Alternation

1. DP evaluates the seven states and every interpolation edge densely under
   hard global Recall and side-local border Recall constraints.
2. Selected keys and low-quality intervals receive continuous vertex blends
   between feasible temporal states.
3. A second DP re-evaluates all original and refined states.

The target key interval is a Pareto-front selection preference, not a hard
key-count constraint. The experiment saves exact stage audits, per-refinement
local before/after metrics, cProfile data, schema-preserving SQLite outputs,
and an independent temporal expansion audit.

## 2026-08-09 full-range result

The full 8,681..20,059 experiment maintained minimum Recall 0.97 and reached
mean intervals 1.50, 3.00, 5.00, 8.00, and 9.01. Stage-two refinement improved
mean IoU from 0.8283 to 0.8343, but did **not** exceed the reference new Pareto
result (0.8641 at approximately interval 10). It is therefore not approved for
promotion. The result is retained as a profiled negative/partial result for the
next candidate-state and optimizer revision.

No video pixels or GPU processing are used by this experiment.
