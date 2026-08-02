# Deployed speed comparison — 2026-08-02

## Scope

This comparison used the same Windows GUI executable and the same input file
while changing only the WSL distribution selected by the GUI:

- development backend: `Ubuntu-24.04`
- EXE-installed backend: `MaskPipelineProduction`

The reported 90 FPS case was reproduced with:

`C:\Users\kenke\Downloads\HEYZO-3548 新城由衣 しんしょうゆい 新城由衣に喉奥まてスッホリ咥えてもらいました - アタル.mp4`

The file is 1280x720 H.264 High, 24 FPS, 23,891 frames, and 995.458 seconds.

No SQLite schema or result-contract files were changed during this work.

## Three-minute deployment parity

The installed Windows GUI ran the same 5,290-frame 1920x1080 fixture twice
against each distribution. V3-lite and Face V2 were run concurrently with LIVE
preview enabled, followed by postprocess and fast overlay.

| Metric | Production average | Ubuntu average | Production delta |
|---|---:|---:|---:|
| Total wall time | 132.410 s | 131.642 s | +0.58% |
| V3-lite compute | 112.931 FPS | 112.361 FPS | +0.51% |
| V3-lite wall | 91.517 FPS | 91.004 FPS | +0.56% |
| Face V2 compute | 103.863 FPS | 104.680 FPS | -0.78% |
| Inference stage | 68.898 s | 68.471 s | +0.62% |
| Postprocess | 39.058 s | 39.205 s | -0.37% |
| Overlay | 5.346 s | 5.338 s | +0.15% |

The differences are normal run-to-run variation. The deployment image does not
introduce an inference, face, postprocess, or overlay regression.

A separate sequential, no-LIVE comparison measured V3-lite at 145.959 FPS in
Production and 146.905 FPS in Ubuntu. A detection-heavy three-minute segment
measured 146.884 FPS in Production and 146.315 FPS in Ubuntu.

## Reproduction of the reported slowdown

Running V3-lite alone on the complete reported file reproduced the progressive
slowdown after roughly frame 14,000:

- unmodified result: 98.372 compute FPS / 98.205 wall FPS
- process elapsed time: 310.26 s
- reserved VRAM: approximately 11,998 MiB to 32,131 MiB
- GPU clocks remained approximately 2.8 GHz and temperature remained below
  75 C, excluding thermal throttling

The original full GUI run finished V3-lite at 89.979 compute FPS because it also
carried the rest of the GUI workflow. Its characteristic decline was the same.

The cause was CUDA caching-allocator fragmentation from variable-size mask and
classifier RoIs. It was not a Windows/WSL distribution performance difference.

## Allocator fix and exact-output validation

V3-lite now defaults to PyTorch expandable CUDA segments before importing
PyTorch-heavy modules. An explicit operator-set `PYTORCH_CUDA_ALLOC_CONF` still
takes precedence.

The exact same complete source was rerun with only the embedded fix active:

- fixed result: 154.119 compute FPS / 153.695 wall FPS
- process elapsed time: 211.48 s
- reserved VRAM: approximately 11,315 MiB to 15,152 MiB
- compute FPS improvement: +56.67%
- end-to-end V3-only elapsed-time reduction: 31.84%

The following tables were compared between the unmodified and fixed SQLite
outputs and were exactly equal in row counts and stored result contents:

- `frames`
- `detections`
- `classifications`
- `classification_probabilities`
- `segmentations`
- `segmentation_polygons`
- `segmentation_points`

The SQLite schema was not changed.

## Cut-detection deployment difference

Production initially took an average of 13.460 seconds for cut detection while
Ubuntu took 4.109 seconds. The Production runtime lacked the development-only
FFmpeg path and silently fell back to OpenCV full-resolution decode.

Cut detection now locates the repository's bundled static FFmpeg. Direct runs
completed in 3.42–3.47 seconds and produced the same cuts. The patched Windows
GUI E2E run completed the cut stage in 4.111 seconds.

## Patched Windows GUI E2E result

The installed Windows GUI was rerun against `MaskPipelineProduction` after both
fixes:

- QA result: passed
- total wall time: 130.19 s
- V3-lite concurrent + LIVE: 112.629 compute FPS / 91.777 wall FPS
- Face V2: 103.833 compute FPS / 86.670 wall FPS
- cut stage: 4.111 s
- postprocess: 38.782 s
- overlay: 5.150 s (about 1,027 FPS at stage level)
- GUI heartbeat maximum: 339 ms; samples above 500 ms: 0
- rendered GUI errors: none
- final SQLite integrity: `ok`
- final schema SHA-256:
  `c66521848ec0f5fcc7f8846149ef1ac4b1f3d9436b63f36f714d08c6def8f5aa`
- overlay: 5,290 input frames and 5,290 decoded output frames

The schema hash and table counts match all pre-fix Ubuntu and Production E2E
runs.

## Automated checks

- InstanceSegmentation focused tests: 15 passed, 2 subtests passed
- Postprocess focused tests: 6 passed, 4 subtests passed
- Python compilation checks: passed

Benchmark artifacts and telemetry are stored under:

`D:\MaskPipelineSpeedComparison`

## Packaging status

The checked-out source tree and the currently installed
`MaskPipelineProduction` distribution contain the runtime fixes. The existing
`D:\MaskPipelineDeployment\LATEST\MaskPipelineDeployer.exe` was built before
these source changes and must not be described as containing the fixes until a
new production image and deployment package are created.
