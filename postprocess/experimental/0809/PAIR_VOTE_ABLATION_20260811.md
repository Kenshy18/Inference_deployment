# best_v4 post-DP pair-vote ablation (2026-08-11)

## Contract

- baseline: `production_candidate_best_v4`, requested interval 8;
- Recall floor: 0.97;
- candidate generation, CUDA DP, key positions, and key count: unchanged;
- only Production's post-DP least-squares pair-vote was enabled;
- post-decode Recall expansion repair: disabled in both arms;
- video frames were not opened; only SQLite polygon geometry and exact metric
  CSV were inspected.

The path-locked analyzer aborts if any `(class, track_id, run_id, frame)` key
identity differs. All 3,973 key identities matched.

## Result

| class | keys | moved keys | mean vertex move | q95 vertex move | dense mean IoU delta | Recall < 0.97 after | vote time |
|---|---:|---:|---:|---:|---:|---:|---:|
| female | 1,260 | 1,256 | 6.309 px | 19.535 px | +0.030219 | 5,146 | 2.328 s |
| male | 989 | 976 | 14.243 px | 48.144 px | +0.031892 | 3,804 | 2.508 s |
| joint | 1,724 | 1,715 | 13.203 px | 51.461 px | +0.068460 | 7,034 | 3.881 s |
| **all** | **3,973** | **3,947** | **11.335 px** | — | **+0.044510** | **15,984** | **8.718 s summed** |

The maximum individual vertex displacement was 201.822 px. No polygon became
self-intersecting, but geometric validity alone does not make the result safe.

## Shape effect

Across selected keys, pair-vote output area divided by the selected best-v4
key area was:

| statistic | ratio |
|---|---:|
| q05 | 0.710613 |
| median | 0.884220 |
| q95 | 0.987550 |
| minimum | 0.411543 |
| maximum | 1.020382 |

The median key lost 11.6% of its area; the strongest contraction retained only
41.2%. Before/after key-shape IoU had median 0.880736, q05 0.708408, and minimum
0.411543. Pair-vote is therefore not a no-op.

## Quality trade

Dense mean IoU increased from 0.871793 to 0.916303 because the voted shapes
moved closer to the raw masks and removed much of best-v4's Recall-oriented
extra area. This average improvement is not acceptable under the declared hard
Recall policy:

- Recall violations increased from 4 to 15,984 of 24,501 rows;
- minimum Recall decreased from 0.941303 to 0.476472;
- 5,382 rows lost IoU despite the positive mean delta;
- 677 rows lost more than 0.05 IoU;
- at keyframes, 2,733 of 3,958 emitted key rows violated Recall 0.97.

## Why it happens

Production pair-vote does not refine the selected best-v4 candidate around its
current position. For each adjacent key pair, it solves least-squares endpoint
vectors directly against `run.anchors`, the aligned raw polygon sequence. It
then replaces each selected key with the length-weighted average of the
left/right interval endpoint proposals. The selected candidate vector is used
only as the initial array and is overwritten whenever the key has an interval
proposal.

This behavior was reasonable for the historical raw-centered candidate path,
but it conflicts with best-v4: best-v4 deliberately chooses supported expanded
states to satisfy minimum Recall. Pair-vote regresses those states toward raw,
removing their coverage without revalidating the DP edges.

The earlier recollection that pair-vote was almost inactive most likely refers
to the separate **additive pre-DP pair-vote proposal** experiment. In that
version a pair-vote shape was merely another candidate state and could be
selected rarely. The Production post-DP mutation measured here applies to
almost every selected key.

## Decision

Do not enable the original post-DP pair-vote on best-v4. Its mean-IoU benefit
is real, but it invalidates the minimum-Recall contract. Any successor must be
an in-DP candidate or a constrained/blended refinement that is densely
revalidated and rejected when Recall, local IoU tail, area acceleration, or
vertex topology regresses.

## IoU-only constrained follow-up

A second mode was implemented with no movement penalty and no temporal
smoothness penalty. For each track it searches the line

```text
best_v4 + alpha * (Production_pair_vote - best_v4), 0 <= alpha <= 1
```

and selects the alpha with maximum exact dense mean IoU subject to exact
per-frame minimum Recall >= 0.97. The coarse grid has 33 points and the best
neighborhood is refined at 1/256. A track with no feasible candidate stays at
best-v4. This is deliberately a one-dimensional first experiment, not free
per-key or per-vertex optimization.

On the full target-8 data:

| metric | best-v4 | constrained pair-vote | delta |
|---|---:|---:|---:|
| effective interval | 6.456072 | 6.456072 | unchanged |
| mean IoU | 0.871793 | 0.875031 | +0.003238 |
| q01 IoU | 0.516062 | 0.520162 | +0.004100 |
| q05 IoU | 0.683759 | 0.687287 | +0.003529 |
| minimum Recall | 0.941303 | 0.941303 | unchanged |
| Recall violations | 4 | 4 | unchanged known infeasible streams |
| mean area/raw | 1.148152 | 1.141698 | -0.006454 |
| maximum area/raw | 4.371832 | 4.326949 | -0.044883 |

It changed 3,516 of 3,973 keys, but the movement was bounded by the Recall
constraint rather than by a movement penalty:

- mean vertex movement: 0.497 px;
- q95 of per-key mean movement: 1.656 px;
- maximum individual vertex movement: 11.451 px;
- key area after/before: q05 0.979809, median 0.996166, q95 1.0;
- before/after key-shape IoU: median 0.995951, q05 0.979635, minimum 0.906606;
- no self-intersection was introduced;
- no row lost more than 0.05 IoU; 10 rows lost more than 0.01.

Of 103 tracks, 93 selected non-zero alpha. Alpha median was 0.066406, q95
0.214063, and maximum 0.449219. The original unconstrained pair-vote was
therefore useful as a direction, but most tracks could safely travel only a
small fraction of that direction.

The constrained search stage took 15.15 / 19.45 / 13.34 seconds for female,
male, and joint respectively. The full parallel run took 124.67 seconds
(188.58 video FPS), versus 253.84 FPS for best-v4. This exact grid search needs
optimization before Production use.

Machine-readable constrained result:

`output/phase2_best_v4_constrained_pair_vote_ablation_20260811/report.json`

## Fixed-key per-key constrained maximum-IoU follow-up

The next experiment keeps all 3,973 key identities and candidate paths fixed,
but replaces the single alpha shared by a whole track with one alpha per key:

```text
key_k = best_v4_k + alpha_k * (Production_pair_vote_k - best_v4_k)
0 <= alpha_k <= 1
```

Each coordinate update evaluates the two adjacent interpolation intervals with
the exact raster metric.  It maximizes their IoU sum and is accepted only when
every affected frame satisfies Recall >= 0.97.  There is still deliberately no
movement, area, temporal-smoothness, or local-tail penalty.  The fixed key
locations and key count are not optimization variables.

Full target-8 result:

| metric | best-v4 | per-key constrained | delta |
|---|---:|---:|---:|
| effective interval | 6.456072 | 6.456072 | unchanged |
| keyframes | 3,973 | 3,973 | unchanged |
| mean IoU | 0.871793 | 0.888143 | +0.016350 |
| q01 IoU | 0.516062 | 0.532742 | +0.016680 |
| q05 IoU | 0.683759 | 0.704593 | +0.020835 |
| minimum IoU | 0.228737 | 0.235654 | +0.006917 |
| minimum Recall | 0.941303 | 0.941303 | unchanged |
| Recall violations | 4 | 4 | no new violations |
| mean area/raw | 1.148152 | 1.115699 | -0.032453 |
| q95 area/raw | 1.442779 | 1.393939 | -0.048840 |
| maximum area/raw | 4.371832 | 4.243508 | -0.128324 |

The four Recall violations are the same known infeasible source streams; the
refinement introduced none.  Of 24,501 dense rows, 21,242 improved, 2,117
degraded, and 1,142 were unchanged.  No row degraded by more than 0.05 IoU;
140 degraded by more than 0.01 because the objective is the interval IoU sum,
not a per-frame lower-tail objective.

The optimizer changed 3,732 keys (93.93%).  Mean vertex movement was 2.772 px,
q95 of per-key mean movement was 9.349 px, and the largest individual vertex
movement was 123.686 px.  Key area after/before had q05 0.890678, median
0.977766, and q95 1.0.  No invalid or self-intersecting polygon was introduced.
These large tails are acceptable for this unconstrained-shape ablation, but
they are direct evidence that the next Production candidate needs temporal and
local-quality guardrails.

The result recovers 36.7% of the unconstrained pair-vote mean-IoU gain while
preserving the exact Recall contract.  This is much larger than the track-wide
alpha result (+0.003238), showing that the main restriction there was coupling
all keys in a track to one alpha.

A two-longest-track saturation check increased alternating coordinate sweeps
from two to four.  Mean IoU changed only from 0.944374574 to 0.944388279
(+0.000013705); only 22 of 205 keys changed further, with 0.0159 px average
vertex movement and no new Recall violation.  Two sweeps are therefore already
near coordinate-wise saturation at the current 1/128 alpha resolution.

This is a practical constrained maximum along one Production pair-vote
direction per key.  It is not a proof of the global optimum over arbitrary
polygon vertex positions.

Runtime remains experimental: the full exact-grid run took 169.37 seconds
(138.81 FPS), versus 92.62 seconds (253.84 FPS) for best-v4.  The per-key
refinement stages took 52.74 / 68.43 / 46.81 seconds for female, male, and joint.

Machine-readable results:

- `output/phase2_best_v4_per_key_pair_vote_ablation_20260811/report.json`
- `output/pair_vote_per_key_saturation4_vs2_smoke_20260811/report.json`

Machine-readable results:

`output/phase2_best_v4_pair_vote_ablation_20260811/report.json`
