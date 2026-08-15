# Exact target-only and process-parallel validation (2026-08-08)

## Scope

- SQLite geometry only; no video pixels were opened.
- Male polygon masks: 6,499 observations across frames 8,681–20,059.
- Recall floor 0.97, requested mean interval 8, 23 vertices, five candidate
  anchor shapes, and maximum edge span 30.
- Stable SQLite schema and stored interpolation contract were unchanged.

## Quality-preserving process parallelism

The former 6 x 4 process layout completed the full Pareto optimizer in
272.915 seconds. A 5 x 5 layout completed it in 241.578 seconds, an 11.5%
reduction, or 47.10 video frames per second. All 5,163 frontier points,
4,551,487 evaluated edges, 1,855,347 feasible edges, selected keyframes, and
quality metrics were exactly equal.

A 6 x 4 thread variant was rejected. Although four threads improved one
248-frame segment from 52.27 to 49.57 seconds, the full run exceeded six
minutes because Python work outside GEOS remained GIL-bound. Eight and 24
threads were also rejected after clear regressions.

## Exact target-only mode

Target-only computes the best solution at the requested global key count and
does not materialize unrelated global Pareto points. It still evaluates the
same anchors and all hard-Recall edges, so mask quality and feasibility are
not approximated.

| Metric | Complete frontier | Exact target-only |
|---|---:|---:|
| Optimizer seconds | 241.578 | 231.641 |
| Video-frame throughput | 47.10 FPS | 49.12 FPS |
| Frontier points emitted | 5,163 | 1 |
| Selected keys | 851 | 851 |
| Actual mean interval | 8.003636 | 8.003636 |
| Minimum Recall | 0.9700000069 | 0.9700000069 |
| Mean IoU | 0.8904161942 | 0.8904161942 |
| Evaluated / feasible edges | 4,551,487 / 1,855,347 | identical |

The target-only point is exactly equal to the complete frontier's 851-key
point. Selected-keyframe JSON has the same SHA-256:
`2d93cd1f5864c0c144bab55b48404158c48e19d2e6f0b1436d5f5b45a42daf83`.

The exported target-only SQLite and complete-front SQLite contained the same
65 schema objects, 44 tables, and 5,241,413 rows. Only the expected
`annotation_state.updated_at_utc` timestamp differed. Both passed integrity
and foreign-key checks.

## Interpretation

Target-only removes 5,162 unused reported Pareto points (99.98%) but saves
only 4.1% optimizer time because geometry feasibility dominates runtime:
anchor construction and 4.55 million edge evaluations are deliberately
unchanged. Larger speedups require either better exact edge scheduling and
conservative pruning, or an explicitly approximate target-only search.

## Duplicate anchor-intersection removal

Anchor evaluation previously executed the same GEOS intersection twice when
the quality and hard-constraint masks were the same `RawMask` object. Reusing
that exact intersection changes no floating-point input or selected geometry.
On the full 26-segment interval-10 target-only workload, an immediate A/B run
reduced optimizer time from `244.551` to `239.352` seconds (`2.13%`) while the
selected-keyframe JSON SHA-256, minimum Recall, and mean IoU remained exactly
equal. A broader dictionary-based anchor cache was rejected: it improved the
longest single track but slowed the full workload to `246.219` seconds.
