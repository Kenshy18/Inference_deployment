# Orthogonal candidate CUDA exploration — 2026-08-10

## Scope and safety

- Only SQLite-derived polygon coordinates were read. No video frame was decoded or opened.
- Production postprocess code, the final SQLite schema, and overlay code were not changed.
- DP interval classification used the CUDA approximate-only path. The C++ exact
  interval graph was not used.
- The reported final Recall/IoU values are the existing independent CPU raster
  audit of the decoded result, not C++ edge revalidation.

## Implemented in this pass

- Shared translation-aligned +/-5 SDF/raster stack.
- Initial six states: `A07`, `A06`, `G02`, `G04`, `C02`, `E02`.
- Candidate state labels and selected state-pair counts in the audit output.
- `pi1` raw-union variants for the previously implemented consensus/local roles.
- Directional sweep area-cap ablation: `C02_115`, `C02_125`, `C02` (1.35).
- Geometry-only topology audit for keyframe and interpolated polygons.

This is the first screening stage, not a claim that all 46 base rules in the
proposal have been implemented. Six new orthogonal roles plus ten previously
implemented roles were screened. The results below determine which remaining
families are worth implementing next.

## Data

| Set | Observations | Tracks | Purpose |
|---|---:|---:|---|
| representative3 | 2,045 | 3 | fast ablation, one long track per class |
| half | 12,251 | 35 | generalisation check over whole tracks |

The half set contains 4,565 female, 3,250 male, and 4,436 joint observations.

## Representative3 results

All rows use minimum Recall 0.97. `infeasible` is the number of streams whose
independent final audit violated the constraint.

| Target | Profile | Effective interval | Mean IoU | Infeasible | Wall |
|---:|---|---:|---:|---:|---:|
| 1 | isotropic scale6 | 1.255 | 0.96782 | 0 | 12.7 s |
| 1 | orthogonal initial6 | 1.217 | 0.96203 | 0 | 29.6 s |
| 1 | C02-1.25 endpoints | 1.217 | 0.96152 | 0 | 32.4 s |
| 5 | isotropic scale6 | 4.732 | 0.89578 | 1 | 12.6 s |
| 5 | orthogonal initial6 | 4.606 | 0.89781 | 0 | 29.7 s |
| 5 | C02-1.25 endpoints | 4.335 | 0.91118 | 0 | 32.5 s |
| 5 | C02-1.35 endpoints | 4.626 | 0.89966 | 0 | 32.4 s |
| 10 | isotropic scale6 | 6.803 | 0.87101 | 1 | 12.9 s |
| 10 | orthogonal initial6 | 6.524 | 0.86883 | 0 | 30.1 s |
| 10 | C02-1.25 endpoints | 6.012 | 0.88229 | 0 | 32.8 s |
| 10 | C02-1.35 endpoints | 6.586 | 0.87038 | 0 | 32.6 s |

### Area-cap Pareto at target 5

The cap affects the difficult joint track; female and male are nearly
unchanged.

| C02 cap | Aggregate effective interval | Aggregate IoU | Joint interval | Joint IoU | Constraint |
|---:|---:|---:|---:|---:|---|
| 1.15 | 4.000 | 0.92220 | 2.909 | 0.84942 | failed (1 frame) |
| 1.25 | 4.335 | 0.91118 | 3.458 | 0.81807 | passed |
| 1.35 | 4.626 | 0.89966 | 4.027 | 0.78478 | passed |

This confirms the intended behaviour: a tighter directional-envelope cap
turns the Recall burden into extra keys rather than unrestricted endpoint
growth. The 1.25 and 1.35 profiles are distinct Pareto points; neither strictly
dominates the other.

## Half-set generalisation at target 5

| Profile | Effective interval | Mean IoU | Worst-class q01 IoU | Infeasible | Wall | Rows/s |
|---|---:|---:|---:|---:|---:|---:|
| isotropic scale6 | 4.802 | 0.90280 | 0.52795 | 3 | 52.2 s | 234.8 |
| orthogonal initial6 | 4.761 | 0.90152 | 0.56457 | 1 | 206.1 s | 59.5 |
| C02-1.25 endpoints | 4.662 | 0.90541 | 0.55670 | 1 | 232.0 s | 52.8 |

The remaining one infeasible stream is the same boundary case in both
orthogonal profiles: track 64/run1 contains one frame and two components. The
selected state is raw, but reducing both components to 11 vertices yields
Recall 0.9413. It is a point-budget/resampling issue, not a candidate-selection
or CUDA failure.

Compared with initial6, the 1.25 profile uses about 2.1% more keys and improves
mean IoU by 0.00389. Its worst-class q01 is lower by 0.00787, so it is not a
strict quality improvement at every quantile.

## Candidate attribution

Solo runs show that `C02` is the only initial state that independently makes
all representative streams feasible near the requested interval (effective
4.555, IoU 0.87372). `A06` reaches 3.837 but leaves one stream infeasible.
`A07`, `G02`, `G04`, and `E02` are primarily endpoint-quality states and remain
near interval 1.35 alone.

On the half-set 1.25 profile, keyframe selections were:

| State | Female | Male | Joint | Total |
|---|---:|---:|---:|---:|
| raw | 43 | 161 | 38 | 242 |
| C02_125 | 87 | 115 | 304 | 506 |
| G02 | 145 | 65 | 120 | 330 |
| G04 | 132 | 68 | 132 | 332 |
| A06 | 175 | 77 | 231 | 483 |
| F3_P1 | 202 | 111 | 165 | 478 |
| D6_P1 | 133 | 72 | 113 | 318 |

No slot is dead. C02 is concentrated in the joint class, while the local-normal
and robust-tube states contribute broadly.

## Topology audit

For the half-set 1.25 result:

- 2,690 keyframe polygons: 0 invalid, 0 winding flips.
- 12,402 interpolated polygons: 0 invalid/self-intersecting.
- 478 adjacent key edges would choose a non-zero cyclic shift if independently
  re-aligned by minimum point MSE; no reversal was needed.

The last item is a quality opportunity rather than a current corruption: all
rendered polygons are valid, but the proposal's edge-specific phase alignment
has not yet been integrated into the CUDA edge score and export together.
Changing export order alone is unsafe because it would make the stored
interpolation differ from the interpolation evaluated by DP.

## Performance

CUDA interval evaluation is active and fast. Runtime is currently dominated by
CPU candidate construction (per-frame rasterisation, SDF, contour extraction,
and the extra pi1 raster pass). The six-state orthogonal half run is 4.0-4.4x
slower than scale6 candidate construction even though both use the same CUDA
graph topology.

`num_workers > 1` currently fails on multi-track CUDA runs because a CuPy
backend object is deserialised through `ProcessPool`. The successful half runs
use one track worker. VRAM was not the limiter (about 29.9 GiB free at failure).

## Assessment

The orthogonal design is validated as a useful direction, but not yet ready for
Production:

1. It fixes more hard-Recall streams than the isotropic scale ladder.
2. C02 area caps expose a clean key-count/IoU Pareto and avoid forcing one fixed
   expansion level.
3. The 1.25 profile improves mean IoU on the half set but slightly worsens the
   worst-class q01, so further candidate selection is required.
4. Candidate generation needs shared-stack/contour caching before a large
   46-rule search is economical.
5. The single-frame multi-component point-budget failure and edge-specific
   cyclic phase evaluation must be fixed before any Production promotion.

## Reproducible artifacts

- Representative interval matrix:
  `output/phase2_orthogonal_c02_caps_representative3_cuda_interval5_20260810`
- Representative target 1/10 matrices:
  `output/phase2_orthogonal_c02_caps_representative3_cuda_interval1_20260810`
  and `...interval10_20260810`
- Half initial6 matrix:
  `output/phase2_orthogonal_capped_half_cuda_interval5_20260810`
- Half 1.25 matrix:
  `output/phase2_orthogonal_c02_125_half_cuda_interval5_20260810`
- Geometry audit:
  `output/phase2_orthogonal_c02_125_half_cuda_interval5_20260810/orthogonal_c02_125_endpoints_geometry_audit.json`
