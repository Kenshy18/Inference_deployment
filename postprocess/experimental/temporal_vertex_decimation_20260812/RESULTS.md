# Initial real-SQLite results

Date: 2026-08-12

The experiment read polygon coordinates from SQLite only. No video frame was
decoded, viewed, or uploaded.

## Configuration

- Dense starting contour: 48 vertices (32 in the final 60-frame smoke test)
- Hard minimum per-frame Recall: 0.97
- Candidate objective: mean IoU + q01 IoU + similarity-normalized temporal
  vertex displacement
- Curve-level objective additionally penalizes vertex count
- Local vertex refinement: disabled after its ablation doubled runtime and
  slightly worsened the minimum IoU
- Exact candidate screening: native C++/OpenMP, followed by a conservative
  Python raster audit before accepting a removal

## Results at the configured recommended point

| segment | frames | selected mode | vertices | comparison at same vertex count | result |
|---|---:|---|---:|---|---|
| track 47 | 456 | temporal-phase equal arc | 36 | current forward equal arc | identical mask IoU/Recall; temporal residual 0.017382 -> 0.016159 (-7.2%) |
| track 55 | 200 | trajectory decimation | 38 | current forward equal arc | mean IoU +0.000502; minimum IoU +0.001610; q01 IoU +0.001248; minimum Recall +0.000961; temporal residual -3.5% |

The selected counts reduce the 48-vertex cap by 25.0% and 20.8%, respectively.
Every output frame in each segment has the same vertex count. Exact final
audits found no self-intersection and both recommended solutions exceed Recall
0.97.

The trajectory-only method is not universally dominant. On track 47 it lowers
temporal displacement but loses about 0.001 mean IoU, so the hybrid correctly
falls back to temporally aligned equal-arc geometry. On track 55 the trajectory
candidate is better on all listed quality axes. This fallback is a required
part of the experimental design, not an optional cosmetic choice.

## Runtime

| segment | optimizer only | optimizer throughput | complete experiment wall time |
|---|---:|---:|---:|
| track 47, 456 frames, 48 initial vertices | 8.06 s | 56.6 FPS | 31.5 s |
| track 55, 200 frames, 48 initial vertices | 8.39 s | 23.9 FPS | 22.0 s |
| track 47 smoke, 60 frames, 32 initial vertices | 0.48 s | 124.3 FPS | 1.74 s |

The complete experiment time includes regenerating both comparison methods at
every reached vertex count. That repeated Pareto audit is not required in a
future Production execution.

## Validation

- 4 focused unit tests pass.
- Output SQLite `PRAGMA integrity_check` is `ok`.
- Output frame sequences are contiguous and row counts match the inputs.
- Vertex count is fixed across every frame in each output.
- Polygon orientation is consistent and final self-intersection count is zero.
- The public unified SQLite schema and all Production code remain unchanged.

## Current limitations

- The standalone runner deliberately accepts only one connected component in
  one contiguous track/cut segment. Multi-component slot handling belongs to a
  later integration experiment.
- Two real track segments are enough to expose useful failure modes but not to
  establish class-wide generalization.
- The end of the greedy deletion path is saved only as a diagnostic. It is not
  a globally minimal feasible vertex count and is not the recommended output.
