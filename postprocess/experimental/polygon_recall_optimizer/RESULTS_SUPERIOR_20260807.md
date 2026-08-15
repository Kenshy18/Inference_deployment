# Superior polygon Pareto validation — 2026-08-07

All measurements in this document use SQLite polygon geometry only. No video
frame was opened, decoded, rendered, or uploaded.

## Validated configuration

- label: `男性器`
- observed masks: 6,499
- track/cut segments: 26
- frame range: 8,681–20,059
- minimum per-frame Recall: 0.97
- soft target mean keyframe interval: 10
- polygon points: 23
- core anchor states: 4
- additive Production pair-vote proposal: enabled
- screen-edge expansion: Production-compatible, clipped to the visible frame

Result artifacts are under
`output/superior_pareto_20260807/full_target10_pairvote_v5/`.

## Selected result

| Metric | Production interval 10 | Superior selected |
|---|---:|---:|
| keyframes | 664 | 686 |
| actual mean interval | 10.350 | 10.005 |
| mean IoU | 0.852660 | **0.861640** |
| minimum Recall | 0.023591 | **0.970000** |
| frames below Recall 0.97 | 1,571 | **0** |
| mean precision | 0.871123 | 0.870615 |
| mean excess-area ratio | 0.184251 | **0.181264** |

The target interval is a soft preference. Recall is never relaxed to hit it.
The selected result is 0.45% from the requested interval and also exceeds the
Production baseline mean IoU.

## Equal-or-lower key-budget Pareto comparison

| Production target | Production keys | Production IoU | New IoU at no more keys | Hard Recall 0.97 |
|---:|---:|---:|---:|---:|
| 1 | 2,448 | 0.927991 | **0.969044** | yes |
| 3 | 1,827 | 0.920660 | **0.958719** | yes |
| 5 | 1,317 | 0.900670 | **0.936605** | yes |
| 8 | 853 | 0.873290 | **0.890707** | yes |
| 10 | 664 | 0.852660 | **0.856869** | yes |
| 15 | 446 | **0.811933** | 0.789567 | yes |

The new front dominates Production throughout the practical interval 1–10
range while enforcing Recall 0.97. At the extreme 446-key point, Production's
minimum Recall is only 0.059122; that infeasible point is not dominated under
the added hard constraint. The new solver needs 503 keys to reach Production's
interval-15 mean IoU without violating Recall. This is expected under the
declared priority order `Recall > IoU > target interval`: the target is an
effort goal, not permission to leak.

The additive pair-vote version dominates every one of the preceding four-state
frontier's 5,205 points at the same or a lower key budget. It therefore inherits
the earlier solution set and can only improve it.

## Compatibility and integrity

- all 6,629 segment frames: editor `point_index` geometry equals Overlay
  geometry exactly; maximum symmetric-difference area is 0;
- input/output SQLite schema fingerprint is identical:
  `0d8ea3abcb7070ab9139e68fb9e360acfc943cb750283d06652ee294b1feff11`;
- stable public result contract remains schema v3, contract revision 5,
  `keyframe-primary-v3`;
- `PRAGMA integrity_check`: `ok`;
- foreign-key errors: 0;
- the real Overlay final-cache reader materialized 25,088 genital rows in
  5.08 seconds;
- postprocess tests: 110 passed plus 41 subtests;
- Overlay tests: 48 passed plus 2 subtests;
- experimental optimizer tests: 30 passed.

No schema table or column was added, removed, or changed. Optimized polygon
segments use the explicit `linear_polygon_index_v1` interpolation value so the
stored vertex correspondence used by editing software is also used by Overlay.

## Runtime and remaining engineering work

The initial exact five-state full-front reference run took 741.66 seconds in
the optimizer and 12:42 wall time including validation/export, with a 6.09 GB
peak RSS. Delayed path materialization, six-by-four process parallelism, and
longest-segment-first scheduling reduced the same exact frontier to 442.23
seconds and 7:37 wall time, with a 0.99 GB peak RSS. The complete frontier,
selected metrics, and selected-keyframe JSON were byte-for-byte identical.
Production's corresponding single-target male-polygon stage took 80.98 seconds.

The measured memory spike came from canonicalizing every local Pareto path.
The implementation now shares candidate anchors across the front and
canonicalizes only the globally selected path. Exact edge evaluation and a
complete 5,000-point global front are still slower than Production's
single-target solve; this experiment should not replace the default Production
stage until the target-only execution path is implemented and benchmarked.
