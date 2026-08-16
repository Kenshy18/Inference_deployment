# Polygon candidate V4 experimental harness

This directory contains the reproducible harness used for the 2026-08-09
overnight polygon-keyframe experiments. It is intentionally disconnected from
the Production pipeline.

## Safety contract

- Read SQLite geometry only; do not open or upload video frames.
- Do not change the Production SQLite schema.
- Keep the stored polygon contract at 23 points with
  `linear_polygon_index_v1` correspondence.
- Enforce minimum per-frame Recall and border Recall before accepting output.
- Preserve unsupported multi-component segments unchanged.

## Recommended experimental configuration

- Candidate mode: `legacy_temporal_recall_interior_rdp_fallback`
- Temporal windows: short/medium/long Recall candidates
- Temporal candidates: interior frames only
- Minimum Recall: `0.97`
- Continuous refinement: 12 normal controls, at most 50 difficult keys
- RDP: fallback only when every legacy base candidate is infeasible

## Representative run

```bash
PYTHONPATH=postprocess \
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
postprocess/experimental/polygon_candidate_v4_20260809/run_full_v4.py \
  --source-sqlite INPUT.sqlite \
  --output-dir output/polygon_candidate_v4_20260809/example \
  --start-frame 40000 \
  --end-frame 49999 \
  --label 男性器 \
  --target-interval 5 \
  --candidate-mode legacy_temporal_recall_interior_rdp_fallback \
  --temporal-recall-quantile 0.97 \
  --workers 5 \
  --edge-processes 5 \
  --sequential-targets 50 \
  --normal-controls 12 \
  --rounds 3 \
  --export-sqlite
```

## Validation

Run the complete postprocess test suite:

```bash
PYTHONPATH=postprocess:overlay/src \
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/pytest -q \
postprocess/tests
```

Validate published artifact hashes:

```bash
sha256sum -c \
postprocess/experimental/polygon_candidate_v4_20260809/DELIVERABLES.sha256
```

See `RESULTS_20260809.md` for the complete ablation, holdout, tail, speed,
schema, interpolation, and determinism results.
