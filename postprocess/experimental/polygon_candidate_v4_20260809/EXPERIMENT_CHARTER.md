# Polygon candidate V4 overnight experiment charter

## Scope and privacy

- Start: 2026-08-09 05:22 JST
- Target finish: 2026-08-09 13:22 JST
- Input: SQLite mask geometry only. Video pixels must not be opened.
- Production pipeline and SQLite schema must not be changed.
- High-freedom optimization is a bounded diagnostic, not the main project.

## Fixed comparison contract

- Raw reference: male-mask observations from frame 8681 through 20059.
- Hard global Recall floor: 0.97 on every evaluated frame.
- Hard border Recall floor: 0.97 and all directional extent constraints.
- Primary target mean key interval: 5.
- Secondary targets: 3, 8, and 9. Targets 1 and 10 are boundary checks only.
- All fair comparisons use the same exact total key count at a target.
- Stored polygon contract: 23 vertices with canonical correspondence.

## Metrics

1. Mean, q01, q05, and minimum IoU.
2. Mean and minimum Recall, plus violation count.
3. Mean, q95, q99, and maximum output/raw area ratio.
4. Key count and achieved mean key interval.
5. Candidate-family selection count and unique feasible-edge contribution.
6. Anchor states, evaluated edges, feasible-edge ratio, wall time, and peak RSS.
7. Vertex reversal/correspondence, editor/Overlay reconstruction agreement, and border audit.

## Time boxes and stop rules

| JST | Phase | Stop rule |
|---|---|---|
| 05:22-05:50 | Freeze harness and baselines | Do not change metrics after this gate. |
| 05:50-07:10 | Candidate superset and ablation | Stop a family after two representative segments if gain < 0.0005 and it adds > 20% edges. |
| 07:10-09:10 | C1 affine and C2 low-dimensional contour optimization | Reject if it violates Recall/border or costs > 2x legacy for gain < 0.002. |
| 09:10-10:20 | Alternating DP/shape/DP | Maximum two rounds; stop if the second round gains < 0.0005 IoU. |
| 10:20-11:20 | Multiple initial-shape hypotheses | High-freedom teacher work is capped at 45 minutes and difficult segments only. |
| 11:20-12:35 | Full-range targets 3/5/8/9 | Prefer target 5 completion, then 3, 8, 9 if time tight. |
| 12:35-13:00 | SQLite and failure audit | No production/schema edits. |
| 13:00-13:22 | Final report | Record rejected hypotheses as well as winners. |

## Candidate hypotheses

The legacy family is retained as a subset before adding any new family:

1. minimally Recall-repaired raw/legacy source;
2. independent directional border repair;
3. staged expansion levels;
4. raw-only pair-vote endpoint regression.

Additive hypotheses to test independently and in small unions:

1. short/medium/long temporal median;
2. short/medium/long temporal outward quantile;
3. interval-regression endpoint;
4. anisotropic principal-axis expansion;
5. one-sided motion/edge-aware expansion;
6. low-frequency normal-offset shapes from local geometry statistics;
7. a small, explicitly time-capped summary of high-freedom optimized residuals.

## Success gates

A candidate V4 configuration is a valid superior candidate only if:

- every Recall and border constraint passes;
- target-5 mean IoU is no worse than the complete legacy candidate set;
- q01 IoU and maximum area ratio do not show a new catastrophic tail;
- no vertex reversal or editor/Overlay disagreement is detected;
- improvement repeats outside the segment used to design the candidate;
- practical configuration is <= 2x legacy wall time, with <= 1.5x preferred.
