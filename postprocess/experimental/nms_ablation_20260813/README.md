# Current NMS ablation

This experiment replays the production `AdaptiveNms` policy over the archived
raw segmentation detections in
`output/instance_mask_topology_20260806/topology.sqlite`.

It does not modify production code or source SQLite files.  Video frames are
decoded locally only when representative audit images are written.

Compared policies:

- `none`: retain every raw detection.
- `current`: exact production containment and adaptive bbox-IoU rules.
- `bbox_only`: production adaptive bbox-IoU thresholds without containment.
- `class_aware`: production rules, but only detections with the same final
  classification may suppress one another.

For every suppression made by `current`, the script reconstructs both binary
masks (including holes) and measures mask IoU and directional coverage.  The
`likely_beneficial`, `likely_harmful`, and `ambiguous` labels are audit
heuristics, not ground truth.

Run with the production Python environment:

```bash
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  postprocess/experimental/nms_ablation_20260813/run_ablation.py
```

## Fixed-downstream A/B harness

`run_fixed_downstream_ablation.py` is the causal comparison harness for the
four topology/NMS arms. All arms read exactly the same
`scored.jsonl` and `cuts.json`, then run the same tracking settings and the
same `polygon14_keyframe_v1` downstream (minimum Recall `0.97`, target
interval `6`, per-key pair-vote `2` sweeps).

- `legacy_production`: unchanged bbox/containment NMS, no topology cleanup
- `topology_legacy_nms`: all holes + <=1% islands, then legacy NMS
- `mask_nms_only`: raw topology with full-mask IoU 0.70 NMS
- `component_mask_v2`: full candidate including post-NMS 80%/50% island rule

Before polygon14, every arm is split into three schema-identical,
label-specific tracked SQLite files.  Each split receives its own border
expansion and endpoint extension, then its own classwise pipeline manifest.
This is required because the Phase-2 label environment selects candidate
roles but does not filter SQLite rows.  The harness validates that split row
counts sum exactly to the combined post-tracking row count before accepting
the output.

```bash
PYTHONPATH=/home/kenshin/inference_backend2/postprocess \
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
postprocess/experimental/nms_ablation_20260813/run_fixed_downstream_ablation.py \
  --scored-jsonl /absolute/path/to/v3/01_score_policy/scored.jsonl \
  --cuts-json /absolute/path/to/v3/03_cut_detection/cuts.json \
  --input-video /absolute/path/to/source.mp4 \
  --output-root /home/kenshin/inference_backend2/output/nms_fixed_downstream_v3
```

Use `--skip-polygon` for a fast NMS/tracking contract smoke test, or
`--max-tracks 1` for a bounded downstream smoke test.  The harness verifies
that tracked and per-class prediction SQLite schemas are structurally
identical across arms.  It intentionally stops at per-class optimizer
artifacts; final unified software-handoff SQLite assembly is outside this
controlled NMS experiment.

After a completed two-arm KPI run, generate the exact per-frame comparison
and independent SQLite integrity/schema audit with:

```bash
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
postprocess/experimental/nms_ablation_20260813/compare_fixed_downstream_kpi.py \
  --legacy-arm /absolute/path/to/legacy_production \
  --candidate-arm /absolute/path/to/component_mask_v2 \
  --output-dir /absolute/new/path/to/comparison
```
