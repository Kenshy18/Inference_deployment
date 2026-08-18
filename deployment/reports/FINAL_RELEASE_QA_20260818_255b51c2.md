# Mask Pipeline final release QA

## Decision

Release `mask-pipeline-20260818-223251-255b51c2` passed the short-form final QA
on the build/deployment PC. It is ready for the operator-run long-duration load
test. The long-duration test remains intentionally outside this report.

Only this release should be distributed. The earlier
`mask-pipeline-20260818-213718-1aa909b1` release was superseded after a live
preview shutdown fault was found and fixed.

## Release identity

- Source commit: `255b51c2c23eee4a07ff81bfa94d967bae9f2c01`
- Release folder:
  `D:\MaskPipelineDeployment\release\mask-pipeline-20260818-223251-255b51c2`
- Deployer:
  `MaskPipelineDeployer-20260818-223251-255b51c2.exe`
- Installed distribution:
  `MaskPipelineProduction-20260818-223251-255b51c2`
- Installed GUI:
  `Mask Pipeline Studio 20260818-223251-255b51c2.exe`
- Deployment report: `passed`
- GPU/driver observed by the deployer: NVIDIA GeForce RTX 5090 / 596.21
- Build profile: `all`

The Windows verifier recalculated every entry in `SHA256SUMS.txt`. The deployer,
PowerShell installer, GUI, backend archive, deployment manifest, and smoke video
all matched their recorded SHA-256 values.

## Functional checks

### Installed deployer and GUI

- The actual versioned deployer EXE installed the release while three older
  Mask Pipeline distributions were already present.
- The installer-created deployment report is `passed`.
- Desktop and Start Menu shortcuts both target the versioned GUI and pass its
  versioned `deployment-profile.json` explicitly.
- The install-time 120-frame GUI E2E completed in 14.635 seconds.
- A second 899-frame, detection-positive GUI E2E completed successfully after
  72.062 seconds with no renderer errors or preview shutdown abort.
- A two-item batch submitted through the installed GUI completed both queue
  entries and produced a SQLite plus overlay for each input.

### Inference models and modes

The installed final distribution executed all shipped model paths successfully:

| Case | Frames | Result |
|---|---:|---|
| Face V1 (`rtdetr_head_face`, PyTorch) | 1 | pass |
| Face V2 (`face_dino_v2`, TensorRT) | 16 | pass |
| Segmentation V1 (`eva02_cascade`, TensorRT backbone) | 1 | pass |
| Segmentation V2 (`dinov3_cascade`, TensorRT backbone) | 1 | pass |
| Segmentation V3 (`dinov3_codino`, TensorRT) | 16 | pass |
| Segmentation V3-lite (`dinov3_codino_mh0`, TensorRT) | 16 | pass |
| V3-lite + Face V2 parallel | 16 + 16 | pass |

The detection-positive 899-frame run used V3-lite + Face V2 and produced 867
instance segmentations, 1,275 face observations, and 6,375 face keypoints.

### New production postprocess and parameters

The 899-frame positive result was processed through the promoted production
NMS/topology, tracking, adaptive polygon approximation, hard-minimum-Recall DP,
pair-vote, face tracking/masks, and integrated exporter. The default score floor
was 0.6. Observed topology work included three filled holes and eighteen removed
tiny islands. The final default run contained 24 mask segments and 1,132 mask
keyframes.

The same positive input was reprocessed with target intervals 1, 3, and 6:

| Target | Actual weighted interval | Keys | Min Recall | Recall violations | Mean IoU | Min IoU | Max area ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.133 | 711 | 0.974754 | 0 | 0.984883 | 0.938351 | 1.0370 |
| 3 | 3.044 | 269 | 0.970004 | 0 | 0.956034 | 0.611697 | 1.5579 |
| 6 | 5.818 | 146 | 0.970006 | 0 | 0.921011 | 0.611697 | 1.5704 |

All three integrated result databases passed `PRAGMA integrity_check` and had
zero foreign-key violations. The larger worst-case area ratio at wider target
intervals is the expected Recall-versus-key-count trade-off, not a failed hard
Recall constraint.

### Media compatibility

The installed final EXE processed representative short inputs through inference,
postprocess, export, and overlay:

| Input | Result |
|---|---|
| 1280x720, 24p, H.264 + AAC, MP4 | pass |
| 1920x1080, 23.976p, H.264, MOV | pass |
| 1920x1080, 29.97p, H.264 + AAC, MP4 | pass |
| 1920x1080, 29.97i top-field-first, H.264, MP4 | pass; deinterlace path used |
| 1920x1080, 60p, HEVC, MKV | pass |
| 3840x2160, 30p, HEVC Main10, MP4 | pass; normalized 1080p processing/output policy used |
| 1920x1080 VFR H.264, MOV | intentionally rejected with an actionable CFR transcode message |

Every successful short case wrote the expected integrated SQLite and H.264 MP4
overlay with the exact processed frame count. All inspected SQLite files passed
integrity and foreign-key checks.

### Network/mapped drive

An installed-GUI E2E was run with both input and output under mapped Google Drive
`G:` while `/mnt/g` was initially unmounted. The GUI mounted `G:` as drvfs,
processed 12 frames, and wrote the SQLite, overlay, and logs back to `G:`. The
G-drive SQLite passed integrity and foreign-key checks and the overlay was
decodable with 12 frames. Raw UNC paths remain rejected with an actionable
instruction; mapped drive letters are the supported network-drive route.

### Output contract

The 899-frame result reports:

- schema: `video-mask-integrated-result`
- schema version: 3
- contract revision: 5
- compatibility profile: `keyframe-primary-v3`
- 899 frames, 3,285 detections, 867 segmentations
- 804 genital tracking assignments and 991 face tracking assignments
- 1,132 authoritative mask keyframes
- 2,620 native polygon points and 986 native face rectangles
- SQLite integrity: `ok`; foreign-key errors: 0
- overlay: 1920x1080 H.264 MP4, 899 frames, 29.97 fps

## Performance and resource behavior

For the 899-frame detection-positive GUI run:

- GUI wall time: 72.062 s
- inference stage: 19.588 s
- segmentation inference telemetry: 71.18 fps
- face inference telemetry: 69.24 fps
- postprocess/export stage: 47.326 s
- overlay: 2.595 s / 379.36 fps
- preview updates: 232; renderer errors: 0
- UI heartbeat samples over 500 ms: 0

This is consistent with the exact-CPU production postprocess and contains no
observed sudden performance regression. Short jobs include model and optimizer
startup costs, so these wall rates must not be extrapolated linearly to long
inputs.

During the 72-second run, system RAM peaked at 40.33 GiB and ended at 37.43 GiB;
VRAM peaked at 22.42 GiB and returned to 3.19 GiB. Neither series was monotonic,
and VRAM was released at completion. This short run found no linear leak, but it
cannot replace the planned long-duration test.

## Cache and coexistence

- Every GUI output manifest reported `work_removed: true`.
- `/tmp/mask-pipeline-studio` was 4 KiB after the test set.
- Versioned GUI user data was 3.17 MB after 19 QA jobs; each retained job record
  was approximately 14 KB and contained no `work` directory.
- Four old/new Mask Pipeline distributions all passed a backend-root probe.
- The pre-existing `20260818-203530-93b462bb` deployment-profile SHA and settings
  SHA remained exactly unchanged after the new installation.
- Old and new desktop shortcuts remain present and target their own versioned GUI,
  profile, WSL distribution, and user-data directory.

## Automated regression checks

- Production postprocess/NMS/polygon/architecture selection: 91 tests passed and
  7 subtests passed.
- GUI TypeScript typecheck: pass.
- GUI unit tests: 74 passed across 13 files.
- Source tree is clean at release commit before this evidence-only report.
- Runtime production code has no dependency on `postprocess/experimental`.

## Remaining operator check

Run the planned long-duration batch on the deployment PC and watch elapsed time,
RAM/VRAM plateau, disk free space, and final output opening in the editing
software. No short-form blocker was found. VFR source material must be converted
to CFR before processing.

The private positive clip was used only on this PC. No video frame was opened by
the language model or uploaded externally.
