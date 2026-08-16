# Trusted temporal-mask prototype results — 2026-08-06

All measurements used SQLite polygon geometry only. Video pixels were not
opened, decoded, inspected, or uploaded by the agent. Production code and the
public SQLite schema were not modified.

## Final experimental configuration

- independent `scene/track` segment fronts;
- asymmetric radius-2 temporal consensus;
- minimum trusted-reference Recall `0.97`;
- curvature-preserving polygon budget `<=23` points;
- three anchor states, at most `4%` candidate expansion;
- candidate-anchor relative IoU margin `0.15` against the trusted reference;
- lower-tail harmonic IoU plus soft normalized Hausdorff utility;
- requested mean interval `10` as an effort target;
- local diminishing-return and nearby minimum-IoU-jump selection; and
- two segment workers, each with twelve exact edge processes.

## Full 6,089-observation comparison

Metrics in this table use the temporally trusted reference.

| Mode | Keys | Mean gap | Min Recall | Mean IoU | Min IoU | CVaR 1% | CVaR 5% | HD q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Old Production | 636 | 10.157 | 0.2990 | 0.8803 | 0.1783 | 0.4900 | 0.6316 | 73.3 px |
| Prior raw-Recall Pareto | 689 | 9.347 | 0.3696 | 0.8834 | 0.2553 | 0.5058 | 0.6318 | 83.9 px |
| Trusted prototype v5 | 1,164 | 5.453 | **0.9700** | **0.9397** | **0.7739** | **0.8205** | **0.8491** | **44.3 px** |

The raw-observation Recall minimum of the trusted result is `0.4460`. This is
expected rather than a safety failure: its worst frame is a mixed temporal
outlier whose raw-vs-consensus IoU is `0.242`; the selected result has Recall
`0.993` and IoU `0.992` against the temporally supported reference. Blind raw
Recall would force that unsupported observation back into the final mask.

## Confirmed failure frames

At the same class/segment:

| Frame | Prior Pareto IoU | Trusted v5 raw IoU | Trusted v5 raw Recall | Trusted v5 boundary error |
|---|---:|---:|---:|---:|
| 13125 | 0.487 | **0.955** | **0.994** | **16.7 px** |
| 13143 | 0.968* | **0.953** | **0.996** | **26.1 px** |

`13143` demonstrates why area IoU alone is insufficient: the prior result had
high IoU but a visible narrow boundary defect and about `57.9 px` Hausdorff
error. The trusted result spends boundary utility to reduce that error.

## Runtime and determinism

- serial segment search: `307.94 s`;
- two-segment parallel search: `166.90 s`;
- speedup: `1.84x`;
- selected-key JSON SHA-256 was byte-identical between serial and parallel;
- parent RSS observed around `300–380 MB` without progressive growth; and
- GPU was not used.

The exact boundary-aware reference search is still slower than the earlier
mean-IoU Pareto (`112–151 s` in prior runs). This is acceptable for research,
not yet for Production. Boundary sampling/caching and scheduling remain future
optimization work.

## SQLite compatibility

The experimental export has:

- schema fingerprint unchanged before/after export;
- public schema `video-mask-integrated-result` v3, contract revision 5;
- schema signature unchanged;
- `PRAGMA integrity_check = ok`;
- zero foreign-key errors; and
- the full orchestration stable-result validator passing.

Exported validation SQLite:

`output/polygon_pareto_20260806/trusted_v5/full/12月KPI動画_trusted_v5_recall097_target10.sqlite`

## Remaining limitations

1. The trusted reference is inferred, not human GT. Its anomaly decisions need
   editor review on sampled contraction, expansion, and occlusion events.
2. Key count increased by `83%` versus Production. This is the measured price
   of the current quality priority; affine/non-rigid interpolation should
   recover part of that cost.
3. The SQLite/software interpolation remains polygon-linear. A better internal
   affine model can guide key placement, but exported keys must still recreate
   correctly under the existing reader without changing the schema.
4. The current symmetric boundary utility is soft and consensus-denoised, but
   a directed robust percentile should ultimately replace maximum Hausdorff.
5. No Production promotion or commit should occur until editor visual review
   confirms that temporally rejected raw observations are genuinely noise.
