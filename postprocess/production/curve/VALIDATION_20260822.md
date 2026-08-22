# Production Catmull--Rom validation — 2026-08-22

## Decision

The CPU-only closed uniform Catmull--Rom path passed the Production gates on
the available V3 corpus at target interval 6.  It is suitable for exposure as
the `catmull_rom` alternative to the existing `polygon` geometry mode.

This validation uses tracked AI masks as references, not human ground truth.
The reported Recall and IoU therefore measure faithful, compact reproduction
of the post-tracking mask sequence.

## Corpus result

Authoritative report:

`output/catmull_rom_curve_v3_corpus_target6_frame_guard_v2_validation_20260822/corpus_summary.json`

The corpus contains nine completed V3 runs, 1,098,270 source-video frames and
434,375 evaluated component/frame observations.

| Metric | Result |
|---|---:|
| Target interval | 6 (soft target) |
| Output mask rows | 434,358 |
| Keyframe rows | 169,253 |
| Effective interval | 2.5663 |
| Mean IoU | 0.955109 |
| IoU q01 / q05 / minimum | 0.881625 / 0.904149 / 0.850072 |
| Minimum Recall | 0.970000 |
| Recall violations below 0.97 | 0 |
| Mean area ratio | 1.017458 |
| Area-ratio q95 / q99 / maximum | 1.076660 / 1.110981 / 1.168310 |
| Strict topology-invalid component frames | 0 |
| Local-quality violations (`IoU < 0.85` or `area > 1.2`) | 0 |
| Completion-envelope frames | 0 |
| Summed classwise output-mask throughput | 126.85 FPS |
| Maximum worker RSS | 1,949,900 KiB |
| Maximum retained phase-history frames | 1,198 |

The throughput is a diagnostic aggregate across separately executed runs.  It
is not a controlled hardware comparison because the two final repair runs
were deliberately restricted to four low-priority CPU cores while another
long inference workload was active.  CUDA was hidden for every curve run.

Track-level point-count policy was stable:

| P per component | Tracks | Component frames |
|---:|---:|---:|
| 14 | 1,629 | 306,981 |
| 16 | 101 | 98,601 |
| 18 | 26 | 25,817 |
| 20 | 8 | 2,976 |

## Rare spatial-outlier repair

The initial full-corpus audit found five bad output frames in HEYZO-3554 and
HEYZO-3560.  They were selected keyframes, not interpolation or DP failures.
A track-wide persistent P placement could not represent a brief highly
concave source contour; the old emergency isotropic repair then enlarged the
whole curve.

Production now uses the following strictly ordered fallback only inside the
already-rare emergency spatial path:

1. independently refit the source frame with the same P count;
2. cyclically align it to the persistent track point phase;
3. test exact-valid blends and choose maximum IoU, then minimum area;
4. let the normal DP add neighbouring support keys when needed;
5. retain the conservative envelope only as a non-stopping last resort.

Across the whole corpus this attempted seven component frames, accepted all
seven, added no editable points, and required no completion envelope.

| Run | Before | After |
|---|---:|---:|
| HEYZO-3554 minimum IoU | 0.5281 | 0.8504 |
| HEYZO-3554 maximum area ratio | 1.833 | 1.1679 |
| HEYZO-3554 local violations | 4 | 0 |
| HEYZO-3554 key rows | 32,052 | 32,051 |
| HEYZO-3560 minimum IoU | 0.6328 | 0.8512 |
| HEYZO-3560 maximum area ratio | 1.543 | 1.1645 |
| HEYZO-3560 local violations | 1 | 0 |

Exact affected-frame examples after the fix:

| Run / track / frame | IoU | Recall | Area ratio |
|---|---:|---:|---:|
| HEYZO-3554 / 197 / 45291 | 0.91856 | 0.97824 | 1.04320 |
| HEYZO-3554 / 198 / 45420 | 0.93625 | 0.97066 | 1.00742 |
| HEYZO-3554 / 198 / 45421 | 0.93964 | 0.97452 | 1.01163 |
| HEYZO-3554 / 198 / 45432 | 0.94199 | 0.97126 | 1.00234 |
| HEYZO-3560 / 115 / 59423 | 0.97445 | 0.98784 | 1.00159 |

On the same four-core restriction, HEYZO-3554 wall time changed from about
694.7 seconds to 699.9 seconds (about +0.7%).  The second timing was affected
by concurrent host CPU load and is not used as a speed regression verdict.

## Regression evidence

### Exact native Catmull--Rom evaluation fusion

The initial-fit scale lattice and point-refinement batches now accept
Catmull--Rom interpolation points directly.  Sampling, OpenCV-compatible
rasterization, and metric reduction execute in the existing C++ extension;
Python no longer materializes the much larger boundary tensor for these hot
paths.  Candidate counts, scale states, coordinate trials, Recall gates, and
selection order are unchanged.  Reusable native raster buffers also remove
per-candidate image allocation.

On the fixed 300-frame, 16-point track at target interval 3, three alternating
runs produced the following medians:

| Path | Engine seconds | Engine FPS | Fit seconds | Point-refine seconds | Max RSS KiB |
|---|---:|---:|---:|---:|---:|
| Materialized-boundary CPU | 2.6828 | 111.82 | 0.9711 | 0.7273 | 315,672 |
| Fused-control C++ CPU | 2.4621 | 121.85 | 0.8580 | 0.6176 | 257,592 |

This is about 9.0% higher end-to-end engine throughput and 18.4% lower peak
RSS.  Both `keyframes.sqlite` and `predictions.sqlite` were byte-identical.
All 56,492 initial-fit evaluations and 12,672 point-refinement trials were
retained.

The exact CUDA paths remained slower for this small, repeatedly launched
workload: the hybrid path measured 95.36 FPS and the all-CUDA path 39.04 FPS.
Both were also byte-identical, but launch/transfer overhead makes C++ CPU
fusion the selected default.

- Production curve tests: 32 passed.
- Changed GUI selector/configuration tests: 23 passed; TypeScript typecheck
  passed.
- Orchestration configuration tests: 23 passed plus 6 subtests.
- Overlay Catmull--Rom keyframe-cache tests: 5 passed.
- Native exact evaluator: 207 scalar parity cases, 120 batch edges, 2,592
  random exact-Recall edges, 111 cached-endpoint cases, 26 pair-vote cases,
  partial-cache parity, and lazy-topology contract all passed.
- Native evaluator micro-benchmark: 10.9x the Python exact reference for
  1,000 iterations on the validation host.
- Deployment test plan: 15 cases, 53 coverage tags, 480-minute budget, no
  static-plan issue.
- Deployment contract tests: 14 passed; one `/mnt/c` cleanup test was blocked
  solely because the managed validation sandbox mounts `/mnt/c` read-only.

Seven pre-existing polygon end-to-end tests were not rerun to completion in
this final pass because their default path requires CUDA and the validation
contract explicitly reserved the GPU for the user's long inference job.  The
other 85 tests in that invocation passed, and the polygon CUDA tests had
passed before GPU isolation.  Full GPU deployment preflight must be repeated
after the live inference job, without changing code or thresholds.

## Production invariants

- No import from `postprocess/experiments` is required at runtime.
- Curve optimization does not import or initialize CUDA, CuPy, Torch or ONNX.
- The public SQLite schema remains V3/revision 5.
- Editable variables are Catmull--Rom interpolation points P only; Bézier
  handles are always derived with factor `1/6` and tension `1.0`.
- The final interpolation identifier is
  `catmull_rom_uniform_tension_1_v1`.
- Recall, topology, local-quality, emergency repair and point-count details
  are persisted in manifests rather than inferred from UI state.
