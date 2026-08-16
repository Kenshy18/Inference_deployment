# Mask Pipeline deployment readiness — 2026-08-17

## Technical summary

**Commit `bfb125e` is ready for a controlled deployment on the validated RTX 5090 / WSL2 target.** The promoted postprocess is now the only default production path, the runtime image contains no retired or experimental postprocess implementation, the Windows GUI and runner expose the promoted arguments, and a release built from a clean WSL distribution was imported into a new QA distribution and passed its Windows GUI-to-WSL end-to-end test.

The deployable release is:

- Windows path: `D:\MaskPipelineQA\release-output-bfb125e\mask-pipeline-20260817-032049`
- release profile: `core`
- source, asset, GUI, and deployer commit: `bfb125eefd5db1bf0cdac63a3e04fe285d3ac4c9`
- backend SHA-256: `b25686f3660170e4e57d698fd73609255067e7670f13f5588834e50c534de796`
- GUI SHA-256: `bddec0eec82c9a76aa1af09c753cf0fb0488df025538668ac899bd9e2043c486`

The result is not a universal hardware certification. It is a release decision for the tested Windows 11 / WSL2 / Ubuntu 24.04 / RTX 5090 SM120 environment. Physical testing of a mapped network `O:` drive was not possible on this PC, and the Windows executables are not code-signed.

## The promoted implementation is now the production implementation

The refactor replaced the former mixed legacy/experimental layout with a production-owned postprocess runtime. The default pipeline resolves to these stages:

1. `preprocessing.normalize`
2. `preprocessing.score_policy`
3. `nms.production_v3`
4. `cut_detection.video`
5. `tracking.greedy`
6. `production.polygon_v3_cpu`
7. `evaluation.mask_iou`
8. `artifacts.validate`

The promoted polygon implementation is split by responsibility under `postprocess/production/polygon/`: input preparation, spatial approximation, candidate configuration, DP, pair-vote, topology validation, materialization, diagnostics, and the native interval evaluator are separate modules. The retired ellipse/keyframe/gap-fill packages and old benchmark profiles were removed from the production surface. The former 15,202-line vendor kernel is no longer the runtime implementation.

The release preflight confirmed that no retired option, retired stage, or forbidden experimental module was loaded. Image finalization records and removes eight development-only paths, including `postprocess/experimental`, `postprocess/tentative`, and test trees. This preserves experiment history in the source checkout while keeping deployment runtime-only.

Across the cleanup from the previous main checkpoint, 200 tracked files changed, with 22,339 inserted and 28,733 removed lines. The net reduction is intentional: duplicate implementations, retired configurations, and development-only paths were removed while the promoted runtime was modularized.

## GUI, runner, and release lifecycle are aligned

The GUI production contract and orchestration configuration now share the promoted postprocess defaults and argument names. The UI supports class-specific target intervals, the adaptive vertex policy, promoted NMS/topology processing, face settings, cut detection, and overlay controls without exposing retired implementations.

Progress is emitted from real polygon optimization work rather than a synthetic timer. Cancellation propagates through orchestration subprocesses, and the resource sampler follows the complete process tree. Frame-count discovery rejects container metadata that advertises frames which cannot actually be decoded. Interlaced inputs are normalized before inference, and the published overlay is progressive.

Release creation now starts from a clean Ubuntu 24.04 WSL distribution, pins the source commit, installs and verifies the selected asset profile, builds native evaluators and overlay runtimes, runs preflight, exports the validated image, builds the deployer, and verifies release hashes. Deployment is transactional: incompatible or corrupt payloads are rejected before they can replace an existing installation, and failed imports are rolled back.

A deployment-profile mismatch found during the first real import was fixed before this release. Manifest schema 2 now carries `profile: core`, and the deployer runs the same profile that the release builder validated instead of hard-coding the broader `all` profile.

## Verification results

| Area | Result | Evidence |
|---|---:|---|
| Postprocess tests | 237 passed + 44 subtests | Production profile, NMS/topology, polygon runtime, schema, classwise routing, progress, architecture |
| Orchestration tests | 47 passed + 14 subtests | Metadata, progress, cancellation, work cleanup, runner arguments |
| Overlay tests | 49 passed + 2 subtests | Keyframe interpolation, native/fast paths, end-frame behavior |
| Inference tests | 74 passed, 2 skipped + 2 subtests | Runtime contracts, SQLite, model routing; skips are platform-conditional |
| GUI tests on Linux | 69 passed | Defaults, queue, config generation, progress estimator |
| GUI clean Windows builder | 64 passed, 5 platform-skipped | Windows packaging and bridge tests |
| Deployment/unit safety | 10 passed | cleanup, resource sampler, release-profile contract |
| Release asset verification | 3,320 files; 15 groups passed | `core` assets in the exported image |
| Release payload hashes | 6 of 6 passed | independent `sha256sum -c` over release manifest |
| Negative deployment cases | 4 of 4 rejected safely | corrupt hash, incompatible GPU, existing distro, invalid VHD |
| Actual isolated deployment | passed | 8-stage import/preflight/GUI E2E/report lifecycle |

The actual deploy used a new `MaskPipelineQA-bfb125e` distribution and a new install root. It imported the 24,972,001,280-byte WSL archive, verified CUDA 12.9, Torch 2.11.0+cu129, TensorRT 10.13.0.35, RTX 5090 capability 12.0, the promoted stage list, the native polygon evaluator, and all overlay binaries. The Windows GUI E2E then completed 120 frames in 13.834 seconds with exit code 0, produced an integrity-clean schema-v3/revision-5 SQLite, and rendered a 120-frame fast NVENC overlay. The fixture intentionally has no positive genital or face detections, so this test proves integration and empty-result robustness rather than mask accuracy.

## Long input, resources, and accumulation

A 30-minute, 54,000-frame V3 postprocess soak completed successfully. It produced 379 track segments, 13,908 keyframes, and 208,676 polygon points. The promoted optimizer reported an effective interval of 5.477, mean IoU 0.900556, minimum recall 0.97, and zero recall violations.

The associated 766.39-second resource sampling window observed:

- peak process-tree PSS: 7,493.43 MiB
- peak process-tree RSS: 8,191.01 MiB
- peak file descriptors: 45
- swap range: 2,357.66–2,357.89 MiB (0.23 MiB span)
- minimum system memory available: 22,118.71 MiB
- GPU thermal slowdown: not active

Memory grew while the optimizer populated its working graph, then declined during the stable tail and returned to zero when the process tree exited. No monotonically accumulating tail, orphan inference/postprocess/overlay process, or increasing swap footprint was observed in this run. This is evidence against a leak in the measured workload, not a proof that every possible multi-hour workload is leak-free.

The output `logs/work` tree is removed after successful publication. During the real deployed E2E, the shared `/tmp/mask-pipeline-studio` directory remained only as an empty root and contained no job artifacts. After QA cleanup, the only registered WSL distributions were the user's existing `Ubuntu-24.04` and `MaskPipelineProduction`.

The packaged Windows executable was also exercised previously on a 3,000-frame MH0 path at 104.56 inference FPS, including the promoted postprocess. Portable and installed-GUI cancellation tests left no child processes. Queue/batch behavior is covered by GUI and orchestration regression tests; a final one-hour multi-item production batch was not rerun after the deployment-profile-only fix.

## Media and path compatibility

The local media matrix passed ten representative cases covering 720p, 1080p, 4K, portrait material, H.264, H.265 Main10, MP4, MKV, MOV, CFR, VFR, 60 fps, Unicode paths, audio/no-audio, and 1080i input. The interlaced case was deinterlaced before analysis, and the resulting overlay was progressive. For 16:9 sources, postprocess coordinates are normalized to the validated 1920×1080 workspace and then transformed back to source coordinates at publication.

Windows drive paths are translated through the GUI's WSL bridge. Local `C:` path translation was exercised in the real deploy. Missing mapped drives trigger a guarded `drvfs` mount attempt, and the bridge has regression coverage for mapped-drive success, failure, and path rejection. This PC did not expose a real mapped `O:` network drive, so authentication/session behavior for the customer's share remains an on-site acceptance item.

## Limitations and release conditions

- **Network drive:** a real `O:` share was unavailable. Before broad rollout, run one input/output job from the customer's mapped drive after Windows authentication.
- **Code signing:** `MaskPipelineDeployer.exe`, `Mask Pipeline Studio.exe`, and the PowerShell deployer are unsigned. SHA-256 protects payload integrity, but Windows publisher trust/SmartScreen requires signing.
- **Dependency audit:** packaged production GUI dependencies reported zero vulnerabilities. The local development/build toolchain reported four high-severity transitive advisories; those packages are not shipped in the runtime ASAR, but should be upgraded in a dedicated build-toolchain change rather than through an unreviewed automatic rewrite.
- **Hardware scope:** release compatibility is pinned to the validated RTX 5090, driver 596.21, compute capability 12.0 environment. Other NVIDIA generations need their own TensorRT engine/profile validation.
- **Smoke-fixture scope:** the final deployed 120-frame fixture validates integration with empty detections. Accuracy and recall evidence comes from the separate V3 corpus and 30-minute soak, not from that fixture.
- **Privacy:** no sensitive video frame was uploaded, shown to an external service, or visually inspected by the agent. Video probing and numerical mask/SQLite analysis stayed local.

## Recommended next steps

1. Preserve `mask-pipeline-20260817-032049` as the only release candidate and deploy it to one controlled target.
2. At that target, run one authenticated `O:`-drive input/output acceptance job and a one-hour multi-item batch.
3. Sign the GUI and deployer before distributing outside the controlled group.
4. Tag and push commit `bfb125e` only after the on-site drive check, since this session did not push.
5. Monitor process-tree PSS, swap, file descriptors, job work directories, and recall-violation counts during the first production batch.

## Further questions

- Does the deployment fleet use only RTX 5090/SM120, or must separate TensorRT engine profiles be packaged?
- Does the customer's mapped drive require credentials that are unavailable to non-interactive WSL sessions?
- Is code signing required before the first controlled deployment, or only before wider distribution?

