# 3-minute pipeline optimization — 2026-07-28

## Result

The adopted B8 configuration reduced end-to-end wall time from
120.78 seconds to 89.06 seconds on the 5,290-frame 1080p test video. The best
observed optimized run was 88.34 seconds; the repeat below uses the final code
and corrected overlapping-stage timing:

| Measurement | Baseline | Optimized | Change |
|---|---:|---:|---:|
| End-to-end | 120.78 s | 89.06 s | -26.3% |
| Inference stage | 89.02 s | 72.55 s | -18.5% |
| Postprocess stage | 20.93 s | 8.06 s | -61.5% |
| Final fast overlay | 3.82 s | 3.90 s | within run variance |
| Cut detection | 9.53 s | 3.73 s | -60.9%, fully overlapped |

The requested 50% end-to-end target was not reached without changing model
precision, input resolution, or detections. In the optimized run, inference
alone takes about 72.6 seconds and therefore exceeds the 60.39-second target for
the entire flow. Concurrent B8 inference reaches the GPU software power cap;
further large gains require model/runtime changes rather than orchestration or
SQLite tuning.

## Adopted changes

- Run MH0 segmentation and Face DINO v2 in isolated processes concurrently.
- Start Face DINO v2 seven seconds before MH0 on the measured RTX 5090 to
  reduce peak contention.
- Keep final unified SQLite publication atomic while using the existing fast
  SQLite durability mode for intermediate writes.
- Decode cut-detection input directly to 96×54 BGR in FFmpeg. The detector
  thresholds, SSIM, histogram, and frame-difference decisions are unchanged.
- Precompute high-precision cuts during inference and pass the validated
  artifact into postprocess.
- Use `orjson` for intermediate JSONL when installed, retaining the standard
  library fallback.
- Cache artifact contract validation by artifact name, resolved path, size,
  and nanosecond mtime. New or changed artifacts are always revalidated.
- Auto-size the independent K1 ellipse solver to available CPUs. K2 remains on
  CUDA for masks routed to the learned approximation path.

## Output equivalence

Baseline and optimized B8 inference contain identical row counts:

- 5,290 frames
- 19,731 detections
- 4,579 segmentations and 616,952 segmentation points
- 8,140 face observations and 40,700 face keypoints

The first adopted optimized run matched every detection, segmentation, face
geometry/probability/mask, frame, and model row exactly after excluding run
metadata. A repeat with the same optimized configuration exposed limited
Face DINO TensorRT run-to-run numerical variation: 7 of 8,140 face
observations differed, resulting in 4 changed eye masks out of 11,518 final
masks. Their raster IoU against baseline was 0.9891 minimum and 0.9920 mean.
Segmentation, counts, cuts, and tracked masks remained exact. This is recorded
instead of claiming bitwise determinism from TensorRT.

Postprocess counts and segmentation-derived outputs were:

- cuts: `[2700, 4500, 4841]`
- tracked masks: 4,371, exact row-for-row
- final masks: 11,518, with only the 4 eye-mask variations described above
- legacy-compatible masks: 11,518, with the same 4 eye-mask variations

## Overlay QA

The optimized final overlay uses the existing C++/libav NVDEC + CUDA + six
NVENC segment path at 8 Mbit/s. Its renderer reported 1,405.4 fps in the final
run. FFprobe and full decode validation found:

- H.264, 1920×1080, yuv420p
- 5,290 encoded and decoded frames
- 30000/1001 fps and 176.509667 seconds
- strictly increasing timestamps
- step range 0.033366–0.033367 seconds
- no missing frames, large timestamp gaps, or decode errors

## Optional B16 profile

A fixed-B16 Face DINO v2 TensorRT profile was built and evaluated on the full
528-image test split. Relative to B8:

- bbox AP: +0.00038
- AP75: -0.00351
- end-to-end ellipse IoU: +0.00003
- point macro-F1 at NME 0.05: -0.00068

On the 3-minute video, B16 changed 8,140 face observations to 8,133 because of
threshold-adjacent numerical differences and improved fully parallel inference
by only a few percent. It is therefore available by explicit
`face_trt_bundle`, but is not the exact-output default.

## Reproduction

- Baseline:
  `orchestration/configs/profile_3min_baseline_20260728.json`
- Adopted exact-output configuration:
  `orchestration/configs/profile_3min_optimized_20260728.json`
- Output manifests:
  `orchestration/output/profile_3min_optimization_20260728/00_baseline` and
  `orchestration/output/profile_3min_optimization_20260728/18_optimized_b8_final`
