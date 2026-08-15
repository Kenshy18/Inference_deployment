# Phase 1 findings

## Question

How far can the unchanged Production quality-plus-lambda keyframe objective
follow requested keyframe intervals when all of the following are enforced?

- one aligned/resampled `raw` polygon state per frame;
- pair-vote disabled;
- post-decode shape repair disabled;
- every prepared frame exposed as a key-position candidate; and
- Recall >= 0.97 on every frame of every feasible interpolation edge.

This experiment reads SQLite polygon geometry only. It never opens a video
frame. Production code and the public SQLite schema are unchanged.

## Correctness guard

The fast Production raster cache is retained for the IoU objective. Recall
feasibility is independently checked using the same component-fill and vector
interpolation semantics as the final exact evaluator. This is necessary
because:

- filling multiple components in one OpenCV call can apply an even/odd rule to
  overlaps and disagree with the final component-by-component evaluator; and
- float32 polygon interpolation can round differently at half pixels from the
  final vector interpolation path.

All 18 cells pass the following audit:

- exactly one `raw` candidate state;
- dense candidate-position pool;
- pair-vote disabled;
- post-decode repair disabled;
- SQLite integrity `ok`; and
- zero final-exact Recall violations among streams declared feasible.

## Result

The full result is in:

`output/phase1_raw_hard_recall_no_pair_20260809_final_v6/PHASE1_RESULTS.md`

Aggregate metrics below contain only the 66 streams for which a hard-Recall
raw-only path exists. The other 37 of 103 streams are explicitly marked
infeasible and are not silently expanded or repaired.

| Requested interval | Actual interval | Mean IoU | Min Recall | Recall violations | Wall seconds | Prepared rows/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.211 | 0.978771 | 0.970017 | 0 | 257.8 | 97.3 |
| 3 | 2.279 | 0.962036 | 0.970010 | 0 | 260.5 | 96.3 |
| 5 | 2.588 | 0.957385 | 0.970010 | 0 | 262.0 | 95.8 |
| 8 | 2.664 | 0.955041 | 0.970010 | 0 | 279.9 | 89.6 |
| 10 | 2.669 | 0.954763 | 0.970010 | 0 | 367.5 | 68.3 |
| 15 | 2.674 | 0.954026 | 0.970010 | 0 | 669.1 | 37.5 |

At requested interval 10, the classwise feasible limits are:

| Class | Actual interval | Mean IoU | Min Recall | Infeasible streams |
|---|---:|---:|---:|---:|
| 女性器 | 3.963 | 0.961765 | 0.970017 | 21 / 43 |
| 男性器 | 2.744 | 0.956258 | 0.970022 | 5 / 26 |
| 結合部分 | 1.930 | 0.945602 | 0.970010 | 11 / 34 |

The requested interval has effectively saturated by 8. Increasing it from 8
to 15 changes the aggregate actual interval only from 2.664 to 2.674, while
optimizer time grows from 273 to 663 seconds.

## Comparison with unchanged Production

Unchanged Production follows the requested interval much more closely, but it
does so without a per-frame Recall guarantee and with pair-vote enabled.

| Requested | Production actual | Production mean IoU | Production min Recall | Production Recall violations |
|---:|---:|---:|---:|---:|
| 1 | 2.470 | 0.944835 | 0.474318 | 4,443 |
| 3 | 3.358 | 0.931143 | 0.567606 | 5,531 |
| 5 | 5.192 | 0.900072 | 0.557851 | 5,910 |
| 8 | 8.134 | 0.859418 | 0.402695 | 6,313 |
| 10 | 10.291 | 0.831348 | 0.000000 | 6,193 |

This is not an apples-to-apples quality contest: its purpose is to show what
the new hard constraint costs when the shape state remains raw-only.

## Conclusion and Phase 2 gate

Phase 1 is complete. A hard minimum Recall constraint correctly transfers its
cost into extra keyframes, but raw-only shape freedom is insufficient:

- 37 streams have no complete feasible graph under the Production gap limits;
- even the feasible subset saturates at aggregate interval about 2.67; and
- requested intervals above 8 add compute without meaningful sparsification.

Phase 2 should add shape candidates without changing the DP priority. It is an
improvement only if it simultaneously:

1. keeps final-exact minimum Recall >= 0.97 with zero violations;
2. reduces the 37 infeasible streams;
3. increases the feasible actual interval at the same or better mean IoU; and
4. avoids local area explosions and single-frame IoU collapses.
