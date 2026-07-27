# 3-minute pipeline optimization — 2026-07-28

## Result

The adopted exact-output configuration reduced external end-to-end wall time
from 120.78 seconds to 78.66 seconds on the 5,290-frame, 1920×1080,
30000/1001 fps test video. This is a 34.9% reduction from the original
baseline and an additional 11.7% reduction from the earlier 89.06-second
optimized result.

| Measurement | Original baseline | Earlier optimized | Final adopted |
|---|---:|---:|---:|
| External end-to-end wall | 120.78 s | 89.06 s | **78.66 s** |
| Inference stage | 89.02 s | 72.55 s | **62.57 s** |
| Postprocess stage | 20.93 s | 8.06 s | **6.54 s** |
| Final fast overlay stage | 3.82 s | 3.90 s | **3.96 s** |
| High-precision cut detection | 9.53 s | 3.73 s | **4.12 s, fully overlapped** |

The external measurement used `/usr/bin/time -v`; maximum resident set size
was 3,559,200 KiB. The stage manifest and process timer agree within their
expected launcher/finalization overhead.

The requested 50% target would be 60.39 seconds. It was not reached without
changing model precision, input resolution, detections, or output semantics.
The final inference stage alone is 62.57 seconds, before 6.54 seconds of
postprocessing, 3.96 seconds of overlay, and publication overhead. Reaching
60.39 seconds therefore requires approximately another 18.3 seconds from the
current flow, most plausibly through a materially different engine or a fused
multi-model runtime rather than more Python/SQLite scheduling changes.

## Final stage profile

### Inference

The fast Co-DINO MH0 and Face DINO v2 engines run concurrently in isolated
processes and publish into one schema-v3 SQLite database:

- 5,290 frames
- 19,731 detections
- 4,579 segmentations and 616,952 segmentation points
- 8,140 face observations and 40,700 face keypoints
- MH0 measured compute: 101.211 images/s while sharing the GPU
- Face DINO measured compute: 94.406 images/s while sharing the GPU
- combined inference-stage wall: 62.567 seconds

The final configuration starts both models together. The previously adopted
seven-second Face DINO head start was slower on repeated full-video tests.

### Postprocess

| Stage | Seconds |
|---|---:|
| Input normalization | 0.806 |
| Score filtering | 0.149 |
| NMS | 0.209 |
| Tracking | 0.454 |
| Ellipse approximation | 1.519 |
| Ellipse keyframes | 0.247 |
| Gap fill | 0.079 |
| Exact mask evaluation | 0.596 |
| Instance-mask export | 0.436 |
| Instance validation | 0.090 |
| Face privacy masks | 0.784 |
| Face/instance merge | 0.270 |
| Combined validation | 0.196 |
| Legacy export | 0.488 |

The measured total was 6.537 seconds. The three detected cuts were frames
`[2700, 4500, 4841]`. Cut detection took 4.119 seconds but ran concurrently
with inference, so it did not extend the critical path.

Outputs contained 4,371 tracked instance masks and 11,518 final masks,
including 7,012 face-eye privacy masks. The legacy-compatible SQLite export
also contained 11,518 masks.

### Overlay

The final overlay uses the C++/libav NVDEC + CUDA drawing + six segmented NVENC
workers at 8 Mbit/s:

- renderer wall: 3.820 seconds
- orchestration stage wall: 3.962 seconds
- aggregate renderer rate: 1,384.76 fps
- output size: 176,720,427 bytes

Full-stream validation passed:

- H.264, 1920×1080, yuv420p, BT.709
- 5,290 expected, encoded, packetized, and fully decoded frames
- 30000/1001 fps and 176.509667 seconds
- monotonic and uniform PTS/DTS
- maximum absolute PTS error: 0.000000333 seconds
- contiguous worker ranges
- all five split boundaries are keyframes
- no missing frames, timestamp discontinuities, or decode errors

Audio copying was intentionally disabled by this profiling configuration.

## Adopted changes

### Shared orchestration and I/O

- Run the segmentation and face engines concurrently in isolated processes.
- Start both engines without an artificial stagger on the measured RTX 5090.
- Precompute high-precision cuts during inference and pass the validated
  artifact into postprocess.
- Decode cut-detection input directly to 96×54 BGR in FFmpeg. SSIM,
  histogram, frame-difference thresholds, and decisions are unchanged.
- Keep final unified SQLite publication atomic while using the existing fast
  durability mode for intermediate writes.
- Use `orjson` for intermediate JSONL when installed, with a standard-library
  fallback.
- Cache artifact validation by artifact name, resolved path, size, and
  nanosecond mtime; changed artifacts are always revalidated.

### Fast Co-DINO MH0

- Capture the fixed-shape backbone/query/decoder portion in a CUDA Graph.
- Feed fused preprocessing directly into the persistent graph input, avoiding
  an additional device-to-device input copy.
- Keep variable-sized box and mask materialization eager so the external
  result contract is unchanged.
- Replace training-only DINOv3 construction imports with a lightweight
  TensorRT runtime shell and explicit checkpoint filtering.
- Vectorize preprocessing and output conversion while preserving the stored
  coordinate and polygon representation.

On a standalone full-video run:

| Measurement | Before latest MH0 work | Final |
|---|---:|---:|
| External process wall | 44.15 s | **40.69 s** |
| Measured model compute | 146.20 images/s | **150.34 images/s** |

All 5,290 frame rows, 4,579 segmentations, and 616,952 segmentation points
matched exactly against the earlier adopted output.

### Giant Co-DINO

- Replace the training framework construction path with a slim TensorRT
  deployment shell.
- Stream pinned host input directly into the captured static device input,
  eliminating one device-to-device copy per batch.
- Reduce graph warm-up from three iterations to the one iteration required
  by the fixed TensorRT execution path.

| Measurement | Original | Deployment shell | Final |
|---|---:|---:|---:|
| External full-video wall | 252.10 s | 244.99 s | **244.79 s** |
| Core pipeline | — | 227.13 s | **225.27 s** |
| Maximum RSS | 6.95 GB | 3.90 GB | **3.85 GB** |

The latest low-level change improves the core pipeline by 0.82%; external
wall time is effectively flat because startup and system variance dominate
that small difference. Detection count, frame/class order, boxes, detector
scores, and sampled masks were exact. Classification IDs were exact; the
largest observed classification-score delta was 0.001021, consistent with
normal GPU numerical variation.

### Face DINO v2

- Capture the detector in a zero-copy CUDA Graph.
- Reuse fixed input/output storage instead of allocating or copying per batch.
- Remove training-only imports and slim TensorRT startup.
- Vectorize output conversion.

Standalone measured compute improved from 137.99 to 180.63 images/s across
the combined optimizations. A controlled standalone comparison matched all
twelve SQLite tables and binary masks exactly.

### Postprocess

- Replace quadratic ellipse-keyframe boundary lookups with a linear pass.
- Auto-size independent K1 ellipse work to available CPUs.
- Keep K2 learned ellipse approximation on CUDA, with fixed batching and the
  existing exact-output settings.

## Output equivalence and reproducibility

The final inference database has the same row counts as the reference. Fast
Co-DINO segmentation output, cut decisions, tracked instance rows, and all
segmentation-derived final masks were exact.

Face DINO TensorRT is not bitwise deterministic when it shares the GPU with
the segmentation engine. In the final run:

- all 8,140 face observations were present
- all 7,012 stored face-mask BLOBs were exact
- 35 of 40,700 keypoint rows, belonging to 7 observations, differed slightly
- 6 of 11,518 derived eye masks consequently differed
- those six raster-mask IoUs had minimum 0.7530 and mean 0.9388

The lowest IoU occurred on a small eye mask where a several-pixel keypoint
shift has a proportionally large raster effect. This is run-to-run TensorRT
variation under concurrent kernel scheduling, not a changed postprocess
algorithm and not evidence that either run is more accurate against ground
truth. Serializing the engines removes the scheduling source but gives up a
large portion of the measured speed gain. The exact counts, head/face
observations, face-mask BLOBs, genital masks, tracks, and cuts remained stable.
This residual reproducibility risk is recorded rather than hidden by an
incorrect bitwise-determinism claim.

## Evaluated but not adopted

- A CUDA Graph that copied into a private MH0 input buffer was slower. Direct
  writing into the graph input was retained.
- Reordering MH0 mask device-to-host transfers reduced throughput from about
  176.5 to 168–169 images/s in the focused experiment.
- `CUDA_DEVICE_MAX_CONNECTIONS=1` made giant Co-DINO substantially slower;
  `32` did not produce a meaningful repeatable gain. The default is retained.
- Dual giant-Co-DINO CUDA Graphs invalidated the shared TensorRT execution
  context during second capture.
- A single Python process for MH0 and Face DINO failed because both engines
  register the same `MSDA_SM120` TensorRT plugin. Process isolation is required.
- CPU-affinity splitting changed the two-process benchmark by only 0.7%.
- Generic postprocess stage parallelization changed 6.57 to 6.53 seconds and
  added SQLite/Python contention; the simpler sequential dependency order was
  retained.
- NVDEC for inference reduced CPU decode load but changed decoded pixels/color
  enough to alter MH0 detections and polygons. It was rejected for the
  exact-output profile.
- Face DINO B16 changed threshold-adjacent detections and produced only a
  few-percent parallel speed improvement.
- Staggered model launches, including 3 and 7 seconds, were slower than
  simultaneous launch on the full-video benchmark.
- MPS was unavailable in this WSL/toolkit environment.

## Optional Face DINO B16 profile

A fixed-B16 TensorRT profile was evaluated on the full 528-image test split.
Relative to B8:

- bbox AP: +0.00038
- AP75: -0.00351
- end-to-end ellipse IoU: +0.00003
- point macro-F1 at NME 0.05: -0.00068

On the 3-minute video, B16 changed 8,140 face observations to 8,133 because of
threshold-adjacent numerical differences and improved fully parallel inference
by only a few percent. It remains available through explicit
`face_trt_bundle` selection but is not the default.

## Reproduction

- Baseline configuration:
  `orchestration/configs/profile_3min_baseline_20260728.json`
- Adopted configuration:
  `orchestration/configs/profile_3min_optimized_20260728.json`
- Original baseline output:
  `orchestration/output/profile_3min_optimization_20260728/00_baseline`
- Earlier adopted output:
  `orchestration/output/profile_3min_optimization_20260728/18_optimized_b8_final`
- Final output:
  `orchestration/output/autonomous_optimization_20260728/47_final_e2e`
- Final stage manifest:
  `orchestration/output/autonomous_optimization_20260728/47_final_e2e/run_manifest.json`
- Full overlay validation:
  `orchestration/output/autonomous_optimization_20260728/47_final_e2e/03_overlay/final.validation.json`

## Regression verification

The final committed source passed the complete repository-level suites used by
this workspace:

- instance inference: 61 passed, 2 environment-dependent tests skipped
- postprocess: 51 passed
- overlay Python renderer/contracts: 29 passed
- overlay native low-level modes: 1 passed
- orchestration: 20 passed

Total: 162 passed and 2 skipped. The full 5,290-frame E2E run and the separate
full-stream overlay validation are additional integration checks beyond these
unit/contract tests.
