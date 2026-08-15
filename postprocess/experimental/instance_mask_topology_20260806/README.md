# Instance-mask topology audit (2026-08-06)

This experiment measures whether one detector instance contains multiple
disconnected foreground components.  It deliberately distinguishes foreground
islands from holes before assessing the editable keyframe slot representation.

The production schema and production inference/postprocess code are not
modified.  Video pixels stay local; generated review overlays are intended for
human review and are not inspected by the agent.

## Matrix

- V3-lite (`dinov3_codino_mh0`, `tensorrt-fast`): all ten videos below.
- V3 (`dinov3_codino`, `tensorrt-fast`): seven non-duplicate full videos plus
  the first ten minutes of each long video.  The duplicated 15-minute HEYZO
  clip is omitted from V3 because it is contained in the full HEYZO input.

Run preparation and execution:

```bash
python run_inference_matrix.py --prepare
python run_inference_matrix.py --run
```

Analyze completed inference SQLite files:

```bash
python analyze_topology.py \
  --matrix ../../../../output/instance_mask_topology_20260806/matrix.json \
  --output ../../../../output/instance_mask_topology_20260806/topology.sqlite
```

Trace a completed production postprocess run by immutable source detection ID:

```bash
python analyze_postprocess.py \
  --topology ../../../../output/instance_mask_topology_20260806/topology.sqlite \
  --run-key v3lite__kpi_2025_12 \
  --postprocess-manifest ../../../../output/instance_mask_topology_20260806/postprocess/v3lite/kpi_polygon_k10_g15/pipeline_manifest.json \
  --output ../../../../output/instance_mask_topology_20260806/postprocess_trace.sqlite
```

The postprocess audit separates score filtering, NMS/unassigned detections,
short-track deletion and retained tracks.  It also records whether the exact
frame became an editable keyframe, how many slots were exported, and how long
multi-component observations persist within one retained track.

`make_review_artifacts.py` creates local side-by-side PNGs for large islands,
tiny islands and holes.  Raw foreground components use distinct colors; holes
use cyan.  The right panel shows the editable final-keyframe slots.  Video
pixels are decoded and written locally only.

## Validation and reporting

- `test_analyze_topology.py` checks island/hole nesting on synthetic polygons.
- `validate_raster_topology.py` rasterizes persisted contours and independently
  recounts foreground components and holes.
- `check_repeatability.py` compares the independently decoded HEYZO 30–45
  minute clip with the same frames inside the full video.
- `audit_artifacts.py` checks every inference manifest, SQLite `quick_check`,
  frame continuity, schema signature, segmentation cardinality and topology
  coverage.
- `audit_postprocess_results.py` checks both representative final-result
  SQLite files, foreign keys, editable geometry ownership and slot bounds.
- `summarize.py` writes the machine-readable aggregate and the compact Markdown
  report.  Paired model rates always use the same leading-frame window.
- `build_report_artifact.py` packages the reviewed aggregate, postprocess trace
  and validation evidence into the canonical self-contained HTML report.

The duplicated HEYZO 30–45 minute clip remains in the experiment as a
repeatability control, but is excluded from unique-video aggregate rates.
