# Border-safe polygon Pareto implementation (2026-08-08)

## Scope and privacy

- Analysis and tests used SQLite mask geometry only.
- No source-video frames were decoded, opened, or uploaded.
- The production SQLite schema is unchanged.

## Implemented contract

For every reconstructed frame whose raw mask touches a video edge:

1. The normal whole-mask Recall floor remains mandatory.
2. Recall is also constrained inside the 24 px visible border strip.
3. The polygon must extend past the image boundary by Production's
   `_expand_polygon` extent (at least 6 px, at most 40 px).
4. These checks apply to every interpolated frame, not only keyframes.
5. The constrained visible border strip and invisible off-canvas portion are
   excluded from the IoU objective. The border strip is instead governed by
   its hard local-Recall constraint; the rest of the visible mask remains in
   the IoU objective.

The local border Recall defaults to the global Recall floor. It can be made
stricter with `--border-local-recall-floor`.

## Independent audits

The runner independently reconstructs the stored `point_index` polygons and
fails before export when any of the following occurs:

- whole-mask Recall violation;
- local border-Recall violation;
- missing off-canvas extent;
- missing reconstructed frame;
- in-memory versus exported-SQLite geometry mismatch.

## Full validation result

Input range: frames 8681..20059, 6,499 mask observations.

- Selected keyframes: 686
- Actual mean key interval: 10.0045 (target 10)
- Whole-mask minimum Recall: 0.9700000069
- Border-constrained frames: 3,149
- Constrained sides: 3,562
- Minimum local border Recall: 0.9700001261
- Local border-Recall violations: 0
- Off-canvas extent violations: 0
- Missing frames: 0
- Exported geometry differences: 0
- Optimizer time before the final exact low-level reuse: 343.04 s
- End-to-end test wall time: 407.08 s
- Peak reported RSS: 907,336 KiB

Output:

`output/superior_border_safety_20260808/full_interval10_v2/12月KPI動画_Superior_border_safe_目標間隔10_Recall097.sqlite`

Its `sqlite_master` schema hash exactly matches the source SQLite
(`9774b32c...b4d4`, 65 objects).

## Comparison at the same 686-key budget

| Metric | Old Superior | Border-safe Superior |
|---|---:|---:|
| Whole minimum Recall | 0.9700000 | 0.9700000 |
| Mean Recall | 0.9894164 | 0.9907193 |
| Minimum IoU | 0.1071118 | 0.1458494 |
| Mean IoU (border-domain split) | 0.8720581 | 0.8641053 |
| Minimum border Recall | 0.0000 | 0.9700001 |
| Border Recall violations | 2,013 | 0 |
| Off-canvas extent violations | 2,967 | 0 |

The hard border guarantee costs 0.795 percentage points of mean IoU at the
same key count, while improving the IoU tail, mean Recall, and all border
safety failures. This is an explicit safety/mean-IoU trade-off, not a hidden
quality non-regression claim.

## Exact speed optimization

Precomputing each frame's border-excluded quality geometry and reusing the
single-side strip intersection reduced the incident-track optimizer time from
136.37 s to 126.29 s (7.4%) and wall time from 157.04 s to 141.06 s (10.2%).
The selected-keyframe JSON files are byte-identical (same SHA-256:
`14f79aaf...03096a6`).

## Tests

- Experimental optimizer: 35 passed.
- Postprocess suite: 115 passed, 41 subtests passed.
- Overlay suite: 48 passed, 2 subtests passed.
