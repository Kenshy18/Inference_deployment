# Superior Pareto output-preserving performance work (2026-08-08)

## Scope

- SQLite geometry only; no video pixels were opened.
- Source range: class `男性器`, frames 8681 through 20059, 6,499 raw masks,
  26 track segments.
- Configuration: Recall floor 0.97, target interval 8, 23 points, four core
  anchor states plus pair-vote state, maximum edge span 30.
- Hardware: Intel Core Ultra 9 285K exposed as 24 WSL CPUs.
- Acceptance required identical selected keyframes, complete Pareto frontier,
  edge counts, Recall, IoU, schema, and exported SQLite rows other than the
  expected export timestamp.

## Profile

The interrupted single-core profile of the largest 979-frame segment captured
482.3 seconds and 667 million Python calls. The dominant work was:

- edge evaluation: 394.1 seconds;
- GEOS polygon intersection: 221.5 seconds;
- interpolated geometry construction: 126.8 seconds;
- anchor construction: 83.3 seconds.

The run was intentionally stopped after 8:47 because enough samples had been
collected and a single-core completion was not needed.

## Accepted changes

1. Reuse already-computed right-endpoint anchor metrics for every edge ending
   at the same anchor.
2. Reuse the constraint intersection when the quality mask is the identical
   unexpanded `RawMask` (51.5% of observations in this fixture).
3. Replace scalar `numpy.clip` with equivalent scalar branches in the
   per-edge utility function.
4. Bypass redundant `make_valid -> unary_union -> make_valid` for one already
   valid polygon; invalid and degenerate polygons retain the original path.
5. Compute each base anchor's minimum feasible scale once and reuse it during
   state expansion.
6. Use four outer segment workers and six edge processes per segment on this
   24-core CPU. The previous default was six by four.
7. Evaluate independent anchor pairs for the same frame through Shapely 2's
   GEOS array API. Recall rejection remains frame-by-frame, metric sums remain
   in source-frame order, and invalid polygons retain the scalar repair path.
8. Retune the final parallel layout to six outer workers by four edge
   processes after GEOS batching changed the per-edge cost.

## Timing

| Variant | Workers x edge processes | Optimizer seconds | Wall seconds | SQLite export |
|---|---:|---:|---:|---:|
| Baseline | 6 x 4 | 412.278 | 466.29 | yes |
| Endpoint/intersection reuse | 6 x 4 | 374.057 | 420.36 | yes |
| Valid single-polygon fast path | 6 x 4 | 360.606 | 409.41 | yes |
| Parallel schedule trial | 4 x 6 | 346.348 | 388.00 | no |
| Parallel schedule trial (rejected) | 3 x 8 | 406.616 | 456.64 | no |
| Final: anchor reuse + 4 x 6 | 4 x 6 | 335.628 | 385.92 | yes |
| Extra quality-bbox check (rejected) | 4 x 6 | 335.722 | 380.28 | no |
| Frame-direction GEOS batch (rejected, track 40) | 1 x 1 | 78.675 | 88.34 | no |
| Anchor-pair GEOS batch, exported | 4 x 6 | 276.206 | 324.45 | yes |
| Anchor-pair GEOS batch, final schedule | 6 x 4 | 272.915 | 314.30 | no |
| Anchor-pair GEOS batch, 25-process schedule | 5 x 5 | 241.578 | 279.58 | no |
| Exact target-only, 25-process schedule | 5 x 5 | 231.641 | 273.70 | yes |

Before the low-level GEOS work, the accepted optimizer time was 18.6% lower
(1.23x faster) than the original baseline. End-to-end wall time at that stage,
including SQLite export and independent audit, was 17.2% lower (1.21x faster).
Maximum RSS changed from 986,252 KiB to 982,704 KiB.

The low-level anchor-pair batching reduces optimizer time a further 18.7%
relative to the former accepted 335.628-second implementation. Relative to
the original 412.278-second baseline, the final 272.915-second optimizer is
33.8% faster (1.51x). The fully exported and independently audited 4 x 6 run
reduced end-to-end wall time from 466.29 to 324.45 seconds, a 30.4% reduction.
Maximum RSS remained effectively flat at 983,256 KiB.

Retuning the process topology to five outer segment workers by five edge
workers reduced the complete-front optimizer by another 11.5%, from 272.915
to 241.578 seconds, without changing one frontier point. Exact target-only
global combination reduced this to 231.641 seconds. Target-only retains the
same selected 851-key point but intentionally emits only that one point rather
than all 5,163 complete-front points. It does not skip any anchor or edge
geometry evaluation, so its additional speedup is a measured 4.1%, not the
multi-fold improvement that would require approximate edge pruning.

## Output equivalence

- selected keyframe JSON SHA-256, before and after:
  `2d93cd1f5864c0c144bab55b48404158c48e19d2e6f0b1436d5f5b45a42daf83`
- frontier SHA-256, before and after:
  `a79d61b3500b1d26ca9a998b6a8a463d72659224b6501abceed9ae06648c8a8f`
- frontier points: 5,163 both;
- edge evaluations: 4,551,487 both;
- feasible edges: 1,855,347 both;
- selected keys: 851 both;
- actual mean interval: 8.003636363636364 both;
- minimum Recall: 0.9700000068999916 both;
- mean IoU: 0.8904161942436251 both.

The two exported SQLite files contained 65 identical schema objects, 44 tables,
and 5,241,413 rows. Every table was set-equal except `annotation_state`, whose
only difference was the expected `updated_at_utc` export timestamp. Both passed
`integrity_check` and had zero foreign-key violations.

Because every target interval selects from the same complete frontier, the
identical frontier hash also covers target intervals 1, 3, 5, 8, and 10; only
the selected frontier index differs.

The GEOS-batched implementation retained the same selected-key SHA-256, the
same complete frontier, all 4,551,487 edge evaluations, all 1,855,347 feasible
edges, and every reported quality metric. Its exported SQLite again contained
65 identical schema objects, 44 tables, and 5,241,413 rows; only the expected
`annotation_state.updated_at_utc` export timestamp differed.
