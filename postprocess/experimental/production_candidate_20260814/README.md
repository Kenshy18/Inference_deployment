# Production candidate 2026-08-14

`production_candidate_adaptive_vertices_v2` is the revised, end-to-end
postprocess candidate. It is intentionally **not** registered as
the default Production pipeline yet.

The package gives the approved algorithms one explicit entry point and one
immutable contract while reusing their validated implementations. It does not
copy the NMS geometry code, polygon fitter, DP runtime, pair-vote evaluator,
tracking implementation, or SQLite writer.

## Processing contract

The software-facing run performs these stages in order:

1. Normalize and score-filter inference detections, or consume an already
   canonical scored JSONL for parity testing.
2. Fill all true holes and delete owner-relative islands of at most 1%.
3. Run virtual-component adaptive Mask NMS. Bounding boxes are only a broad
   phase; native-pixel masks decide suppression. Main/main and island/island
   thresholds are 0.20, 0.10, and 0.05 for normal, small, and tiny objects.
   An island is removed against another main only at at least 80% coverage and
   at most 50% of the other main's area.
4. Track with the raw association geometry and remove tracks of at most 10
   frames. The cleaned mask remains the public output geometry.
5. Before border preparation, measure each track's continuous foreground-mask
   area q99.9 as a fraction of the real video frame. Assign 14/16/18/20 points
   at strict 3%/10%/25% crossings and keep that count fixed for the track.
6. Split tracks into 女性器, 男性器, and 結合部分. Apply border expansion with
   a 16 px maximum/influence band and explicit two-axis screen-corner support,
   followed by the five-frame endpoint extension independently per class.
7. Build track-consistent polygons with the assigned point count. Apply the
   same exact Recall repair at every count; point-count escalation is not a
   quality fallback and unresolved frames are audited rather than stopping.
8. Run the multistate CUDA-lazy-exact DP with exact per-frame Recall at least
   0.97 and a soft target interval of 6.
9. With DP key positions fixed, run two per-key pair-vote coordinate sweeps to
   maximize IoU under the same exact Recall and simple-polygon constraints.
10. Export the unchanged unified V3/revision-5 software SQLite schema.

The exact evaluator short-circuits an interval as soon as one frame proves
the hard Recall constraint impossible; feasible intervals still compute the
same complete IoU cost. On large graphs with a sustained exact-infeasible
ratio of at least 0.875, the runtime switches earlier to the same dense exact
fallback that the validated baseline eventually selected. Pair-vote uses
eight native threads, while the three class optimizers retain eight threads
each so total CPU concurrency matches this 24-core deployment target.

The preceding fixed-14 profile was validated at target interval 6 with 4,582
keyframes, minimum Recall 0.970000095, and zero violations. Those numbers are
kept only as the regression baseline: adaptive vertices and the revised edge
preparation intentionally change geometry and require a new KPI/full-V3
quality and runtime report before Production promotion.

Classes with no tracked instances are skipped by the optimizer and receive a
validated zero-row export artifact. A video therefore does not need to contain
all three genital classes. Class-specific preparation also removes unrelated
raw provenance and compacts its temporary SQLite files; on the KPI reference
this reduces each redundant copy from about 240 MB to 50–57 MB without changing
any mask row.

The exact constants and class-specific candidate roles live only in
[`config.py`](config.py) and
[`polygon/candidate_palette.py`](polygon/candidate_palette.py). The runtime
bridge fails closed if the approved polygon profile or its role palette
drifts.

## Layout

- `config.py`: immutable semantic and runtime contract.
- `nms/`: policy construction and streaming NMS stage.
- `polygon/spatial.py`: stable adaptive 14/16/18/20-point fitter boundary.
- `polygon/candidate_palette.py`: class-specific DP state roles.
- `polygon/dp.py`: exact minimum-Recall audit.
- `polygon/pair_vote.py`: exact pair-vote boundary.
- `polygon/topology.py`: simple-polygon hard guards.
- `polygon/preparation.py`: class split, border, and endpoint preparation.
- `polygon/engine.py`: quarantined bridge to the currently validated 0809
  optimizer runtime. This is the only intentionally legacy-facing module.
- `validation/`: canonical JSONL, SQLite schema, integrity, and semantic parity
  checks.
- `pipeline.py`: explicit stage assembly only.
- `run.py`: CLI only.

## Run

Use a new output directory. Existing non-empty directories are rejected so a
fresh result can never be mixed with stale artifacts.

```bash
cd /home/kenshin/inference_backend2
PY=/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10
PYTHONPATH=postprocess "$PY" -m experimental.production_candidate_20260814.run \
  --input-sqlite /absolute/path/to/inference.sqlite \
  --input-video /absolute/path/to/source.mp4 \
  --cuts-json /absolute/path/to/cuts.json \
  --target-interval 6 \
  --interval-evaluation native_exact \
  --output-root /absolute/path/to/new_candidate_run
```

`--target-interval` is the soft keyframe interval target. The validated
candidate supports positive integer targets; interval-specific role palettes
are selected deterministically while all quality constraints remain fixed.

`native_exact` is the default interval evaluator and evaluates every DP edge
exactly on CPU. `cuda_lazy_exact` remains available as an explicit accelerated
diagnostic; NMS, tracking, polygon candidates, pair-vote, topology gates, and
export are unchanged between the two modes.

The resumable full-V3 comparison used for interval 2 and 5 is:

```bash
PYTHONPATH=postprocess "$PY" -m \
  experimental.production_candidate_20260814.benchmark_v3_exact_vs_default \
  --intervals 2,5 \
  --modes default_cuda,cpu_exact \
  --output-root /absolute/path/to/new_v3_benchmark
```

It discovers the V3 raw-inference SQLite corpus, reads each source-video path
from the SQLite metadata, constructs one shared upstream geometry per video,
and writes one final software SQLite for every video × interval × evaluator.
Cut detection decodes 96x54 frames locally; no frame image is displayed or
sent outside this machine.

After the batch reports `status=complete`, run the geometry-only audit:

```bash
PYTHONPATH=postprocess:postprocess/experimental/0809/native_interval/build \
  "$PY" -m experimental.production_candidate_20260814.analyze_v3_exact_vs_default \
  --benchmark-root /absolute/path/to/new_v3_benchmark
```

This verifies all final SQLite files, measures runtime and exact Recall/IoU,
and raster-compares every matched dense output polygon between CUDA and CPU.
It reads no video or frame images.

For exact regression testing, add `--scored-jsonl` with the immutable scored
artifact. Otherwise `--score-min` defaults to 0.30.

The final SQLite path and every intermediate artifact are recorded in
`candidate_manifest.json`. `input_video` is required only for bounded endpoint
metadata (frame count); this pipeline does not decode video pixels for its
algorithm or parity checks.

## Validation and promotion

Do not use aggregate union Recall from empty-mask frames or the sparse final
SQLite's non-materialized dense evaluation as a quality oracle. The release
gates are:

- exact ordered NMS JSONL equality;
- per-class `runtime/exact/keyframe_exact_metrics.csv`, minimum Recall 0.97,
  and zero violations;
- equal key positions, roles, labels, track IDs, and polygon coordinates
  (absolute tolerance at most `1e-6` where serialization requires it);
- identical public SQLite schema fingerprint;
- `PRAGMA integrity_check = ok` and zero foreign-key errors;
- deterministic rerun.

The candidate remains under `experimental` until these gates pass on the
approved full KPI reference and the default GUI/Production registration is
changed in a separate promotion commit.
