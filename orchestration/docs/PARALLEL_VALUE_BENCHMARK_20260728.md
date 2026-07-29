# Parallel execution value benchmark — 2026-07-28

## Decision

Model concurrency is selected per segmentation model:

- compact `dinov3_codino_mh0` and Face DINO v2: enable concurrency with zero
  launch stagger;
- giant `dinov3_codino`, the legacy face detector, and single-model modes:
  reject the concurrency option and run sequentially;
- run high-precision CPU cut detection during GPU inference for either model.

Concurrency remains an explicit switch for the one approved model pair.
Cut-detection overlap is an independent switch.

## Controlled setup

- input: `data/codino_trt_3min_simple150_input.mp4`
- 5,290 frames, 1920×1080, 30000/1001 fps
- instance model: `dinov3_codino_mh0`, TensorRT fast
- face model: `face_dino_v2`, TensorRT fast
- postprocess: polygon, keyframe interval 3, no gap fill
- cut detector: `high_precision`
- overlay disabled so it cannot affect the comparison
- all three runs publish the new integrated `result.sqlite`

## Measurements

| Condition | Inference | Cut detection | Postprocess | External wall | Max RSS |
|---|---:|---:|---:|---:|---:|
| Models sequential, cut inline | 74.694 s | 3.477 s inline | 7.144 s | 88.33 s | 3,568,940 KiB |
| Models parallel, cut inline | 60.551 s | 3.487 s inline | 7.265 s | 77.31 s | 3,570,128 KiB |
| Models parallel, cut overlapped | 61.434 s | 4.034 s overlapped | 3.635 s | **71.70 s** | 3,667,040 KiB |

Model concurrency reduced inference by 18.9% and external end-to-end wall by
12.5%. CPU cut detection made concurrent inference 0.88 seconds slower, but
removed the entire cut stage from the subsequent postprocess critical path.
Compared with the parallel/inline-cut run, external wall improved by another
7.3%. The adopted combination is 18.8% faster than sequential models with an
inline cut detector.

The integrated SQLite publication itself took approximately 0.62 seconds and
produced an 82,993,152-byte file in each run.

## Giant Co-DINO controlled comparison

The same 5,290-frame input and Face DINO v2 configuration were also tested
with giant `dinov3_codino`. Postprocess and overlay were disabled so only the
combined inference stage was compared.

| Condition | Inference stage | External wall | Giant compute | Face compute |
|---|---:|---:|---:|---:|
| Models sequential | 261.969 s | 288.03 s | 23.857 FPS | 183.407 FPS |
| Models parallel | 254.041 s | 277.04 s | 21.378 FPS | 64.691 FPS |

Concurrency reduced the inference stage by only 7.928 seconds (3.0%) and
external wall by 10.99 seconds (3.8%). During the overlap interval, the giant
model ran at roughly 16 FPS before recovering after face inference completed.
The face model slowed to 35.3% of its sequential compute rate. This is expected
because giant Co-DINO already saturates the GPU.

Both outputs contained exactly 4,069 segmentations, 8,140 face observations,
and 40,700 face keypoints, with no instance class changes. The giant instance
output was not bitwise identical:

- mean raster mask IoU: 0.999262
- 1st percentile raster mask IoU: 0.985047
- minimum raster mask IoU: 0.969451
- masks below 0.90 IoU: 0
- maximum box-coordinate change: 2 px
- maximum classification-score change: 0.008735

Face output retained the same class/state/validity values. Four face
observations, five keypoints, and three probability masks differed
numerically; the maximum keypoint shift was 0.64 px horizontally and 1.18 px
vertically.

The speed gain is real but too small to justify exposing giant-model
concurrency as a production option. The current configuration validator rejects
this combination.

## Output comparison

All three integrated SQLite files passed `PRAGMA integrity_check` and contained:

- 5,290 frames
- 19,731 detections
- 4,579 segmentations
- 616,952 segmentation points
- 8,140 face observations and 40,700 face keypoints
- 4,371 tracked masks and 4,371 final masks
- cuts at frames `[2700, 4500, 4841]`

SHA-256 hashes over ordered rows matched exactly for detections, segmentations,
segmentation polygons and points, face probability masks, tracked masks, final
masks, raw tracking audit rows, and cuts.

TensorRT face geometry retained its known small run-to-run numerical
variation. One of 8,140 face-observation rows differed by `9.78e-6` in score,
and five of 40,700 keypoint rows shifted by at most 1.71 px horizontally and
1.95 px vertically. Keypoint class, state, and validity were unchanged. The
sequential and cut-overlap results happened to match each other exactly while
the intermediate parallel run differed, so this is normal TensorRT run
variation rather than a cut-overlap effect.

## Reproduction

- `orchestration/configs/benchmarks/parallel_value_3min_sequential.json`
- `orchestration/configs/benchmarks/parallel_value_3min_parallel.json`
- `orchestration/configs/benchmarks/parallel_value_3min_parallel_cut_overlap.json`
- `orchestration/configs/benchmarks/parallel_value_giant_3min_sequential.json`
- `orchestration/configs/benchmarks/parallel_value_giant_3min_parallel.json`
- outputs: `orchestration/output/parallel_value_3min_20260728`
- giant outputs: `orchestration/output/parallel_value_giant_3min_20260728`

The giant parallel configuration is retained only as exact historical
benchmark provenance. It is intentionally rejected by the current production
validator.
