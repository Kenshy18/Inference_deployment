# 0809 polygon-keyframe experiments

## Phase 1: Production penalty DP, raw-only, hard Recall, no pair-vote

`phase1_runtime.py` loads the unchanged Production v22 implementation and
creates the clean baseline requested before candidate-shape work:

- one aligned Production `raw` shape state per prepared frame;
- every prepared observation/gap-filled frame is a key-position candidate;
- pair-vote is disabled;
- post-decode shape repair is disabled;
- every edge is densely evaluated and rejected if any frame has Recall below
  0.97; and
- the Production IoU/shape cost plus lambda-per-key shortest path is retained.

Run the full class/target matrix with:

```bash
PYTHONPATH=postprocess:overlay/src \
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
postprocess/experimental/0809/run_phase1.py
```

The default targets are `1,3,5,8,10,15`. Results are written to
`output/phase1_raw_hard_recall_no_pair_20260809_final_v6/`. See
`PHASE1_FINDINGS.md` for the conclusion, the Production comparison, and the
Phase 2 acceptance criteria.

On the 24-core reference machine, the three independent class jobs run in
parallel by default (`--label-workers 3`), with four DP workers in each class
(`--num-workers 4`). Use `--label-workers 1` to reproduce the former serial
class scheduling. Class parallelism changes scheduling only; the selected
keyframes, dense predictions, exact metrics, and their serialized artifacts
are byte-identical to the serial outputs.

The runner reuses the already materialized Production polygon inputs from the
interval-10 all-polygon baseline. This avoids re-running tracking, border
preparation, endpoint extension, or cut detection and makes every target use
byte-identical input geometry.

## Frozen Production candidate baseline v1

The fixed Phase-2 candidate-shape baseline is available under the profile
`production_candidate_baseline_v1`.  Its state order and geometry definitions
are frozen as:

```text
raw
C02_125
G02
G04
A06
F3_P1
D6_P1
```

The historical profile name `orthogonal_c02_125_endpoints` is an exact alias.
Pair-vote and post-decode shape repair remain disabled, and per-frame minimum
Recall is 0.97.  The profile changes neither Production code nor the final
SQLite schema.

Run the full interval-10 baseline with three independent class workers:

```bash
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
postprocess/experimental/0809/run_phase2.py \
  --profiles production_candidate_baseline_v1 \
  --target-interval 10 \
  --recall-floor 0.97 \
  --num-workers 1 \
  --label-workers 3 \
  --cuda-fast \
  --output-root output/phase2_production_candidate_baseline_v1
```

The optimized candidate builder preserves the former geometry exactly:

- selected keyframe JSON: byte-identical;
- dense interpolated JSON: byte-identical;
- exact metric CSV: byte-identical; and
- every `(frame, track_id, polygons)` prediction SQLite row: identical.

On the reference RTX 5090 system, the full three-class interval-10 run changed
from 906.99 seconds to 89.94 seconds with `--label-workers 3`: 261.39 video
frames/s for 23,510 frames.  Including Python startup and the outer report
writer, the measured command wall time was 93.14 seconds (252.42 frames/s).
Candidate construction itself changed from 756.21 seconds to 155.61 seconds in
the serial comparison.  The speedup comes from removing unused TSDF
transforms, using the exact odd-stack majority equivalence for the median zero
level set, reusing aligned temporal windows, and computing the point-count
predictor's eccentricity feature from binary image moments instead of
materializing every foreground pixel coordinate.  It does not change
candidate shapes, adaptive anchor counts, the optimizer objective, or any
emitted artifact; the 89.94-second run was byte-compared against the prior
124.78-second run.

## Best initial-shape research handoff (2026-08-11)

The class- and target-aware profile `production_candidate_best_v4` is the
selected no-pair-vote initial-shape palette.  On the full polygon source it
improves both effective interval and mean IoU over the frozen baseline at
requested intervals 5 and 8 while running at 252.64 and 253.84 video FPS.
It is not connected to Production yet: the next gates are pair-vote ablation,
local expansion/tail control, and exact revalidation of selected CUDA edges.

See `PRODUCTION_CANDIDATE_BEST_V4_20260811.md` for the palettes, complete
metrics, rejected families, known local regressions, deterministic artifact
checks, and the next work order.

The bounded interval-8 follow-up is documented in
`INTERVAL8_BOUNDED_CANDIDATE_SEARCH_20260811.md`. It adds deterministic
inverse-interpolation endpoint candidates, records why exact interval 8 is not
feasible on the difficult nonlinear-motion tracks without large inflation,
and packages the safest interval-oriented Pareto point for editor review. It
remains experimental and does not automatically replace best v4.

The path-locked Production pair-vote ablation is documented in
`PAIR_VOTE_ABLATION_20260811.md`. Original post-DP pair-vote moved 99.35% of
best-v4 keys and improved mean IoU, but removed Recall-oriented coverage and
created 15,984 per-frame Recall violations. It must not be enabled unchanged.
The same report includes an IoU-only constrained blend: it preserves the
existing Recall violations exactly while improving mean IoU by 0.003238, with
no movement or temporal penalty. Its exact grid search is still experimental
and slower than the 240-FPS target.

The fixed-key per-key refinement improves the full mean IoU by 0.016350 without
adding an exact Recall violation.  The frozen configuration immediately before
temporal/local-quality constraints, its deterministic rerun, runtime, exact
metrics, and software-facing SQLite are documented in
`PRE_TEMPORAL_BASELINE_20260811.md`.

## Earlier diagnostic matrix

The files below predate Phase 1 and remain useful as a historical baseline.
They compare unchanged Production with a *post-hoc* Recall guard; that guard is
not the hard-Recall DP used by `phase1_runtime.py`.

This experiment deliberately restarts the polygon-keyframe investigation from
the unchanged Production v22 implementation.

## Fixed contract

- Input: `data/12月KPI動画.sqlite`
- Classes: `女性器`, `男性器`, `結合部分`
- Shape mode: polygon for every class
- Requested keyframe intervals: 1, 3, 5, 8, 10
- Production DP state: one `raw` shape candidate per candidate frame
- Production pair-vote and exact-recall repair: unchanged
- Score threshold: 0.3
- Missing-mask interpolation limit: 15 frames
- Cut positions: reused from the previously validated cut artifact
- Video frames are never opened by the evaluator or the AI agent

Two variants are compared:

1. `production_raw`: unchanged Production behavior. Its Recall budget is an
   average/path budget, not a per-frame guarantee.
2. `production_raw_hard_recall`: the same completed Production result followed
   by a deliberately forceful dense guard. It retains every Production key,
   repairs an infeasible key shape, and inserts the fewest additional raw-shape
   keys needed to make every evaluated observation satisfy Recall >= 0.97.

The hard guard is intentionally a diagnostic baseline. It is not the proposed
final optimizer and is not connected to Production.

## Run

```bash
PYTHONPATH=postprocess:overlay/src \
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
postprocess/experimental/0809/run_matrix.py
```

Results are written under
`output/production_raw_only_0809_20260809/`. The runner is resumable and does
not modify the input SQLite, Production code, or SQLite schema.

Generate the aggregate and classwise tables, then run the fail-fast contract
audit:

```bash
PYTHONPATH=postprocess:overlay/src \
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
postprocess/experimental/0809/summarize.py

PYTHONPATH=postprocess:overlay/src \
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
postprocess/experimental/0809/audit.py
```

The audit requires all ten cells, one shared schema fingerprint, 100% geometry
coverage, polygon-only policies, raw-only Production state count, valid SQLite
foreign keys/integrity, preservation of the dataset's legitimate two-component
keyframe, and zero Recall violations in every hard-Recall cell.
