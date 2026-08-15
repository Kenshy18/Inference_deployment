# V3 topology/NMS four-arm ablation

`run_four_arm_v3.py` keeps the downstream polygon approximation, DP and
pair-vote out of the comparison and varies only the topology/NMS front-end.
Its NMS JSONL outputs can optionally be retained and passed to one fixed
downstream harness after the front-end audit.

The four arms are:

1. `legacy`: existing adaptive bbox-IoU/containment NMS.
2. `topology_then_legacy`: fill all holes and remove owner-relative islands
   up to 1%, then run the legacy NMS.
3. `mask_iou_only`: full-instance mask-IoU NMS at 0.70, without hole/island
   cleanup.
4. `component_candidate_v2`: fill holes, remove <=1% islands, mask-IoU NMS at
   0.70, then remove survivor islands only when coverage is >=80% and the
   island/other-main area ratio is <=50%.

## Full V3 invocation

Use the repository's Production Python environment:

```bash
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  postprocess/experimental/nms_ablation_20260813/run_four_arm_v3.py \
  --output-root output/nms_four_arm_v3_20260813
```

The five archived authoritative `scored.jsonl` inputs are reused. The other
V3 raw SQLite files are normalized and filtered at detector score 0.30 into
the output root, so the source lineage is explicit and reproducible.

Useful development modes:

```bash
# List the nine V3 inputs.
.../python3.10 postprocess/experimental/nms_ablation_20260813/run_four_arm_v3.py \
  --output-root /tmp/nms-audit --list-runs

# Small smoke test on one archived scored input.
.../python3.10 postprocess/experimental/nms_ablation_20260813/run_four_arm_v3.py \
  --output-root /tmp/nms-audit \
  --run-key v3__heyzo_3545_full --max-frames 200

# Also preserve full canonical NMS JSONL for a fixed downstream comparison.
.../python3.10 postprocess/experimental/nms_ablation_20260813/run_four_arm_v3.py \
  --output-root output/nms_four_arm_v3_20260813 --write-arm-jsonl
```

## Outputs

- `experiment_config.json`: frozen thresholds, selected inputs and Git revision.
- `summary.json`: complete machine-readable result.
- `run_arm_summary.csv`: timing, retention, topology and safety metrics per run.
- `aggregate_arm_summary.csv`: totals across selected runs.
- `pairwise_deltas.csv`: exact retained-ID deltas for every arm pair.
- `runs/<run>/retained_ids.jsonl.gz`: input and retained source detection IDs for
  every processed frame.
- `runs/<run>/safety_frames.csv`: union-mask recall/IoU and lost/added area on
  every frame where arms differ or topology cleanup fires.
- `runs/<run>/component_events.csv`: hole/tiny/redundant-island events.
- `runs/<run>/arm_outputs/*.jsonl`: optional canonical outputs for the common
  downstream pipeline.

The input videos are never decoded by this runner.
