# Temporal vertex decimation experiment

This experiment replaces independent equal-arc polygon vertices with a fixed
set of vertex trajectories over one contiguous track/cut segment.

1. Resample the source contours densely.
2. Align cyclic phase from the segment centre in both temporal directions.
   Direction reversal is forbidden, and similarity motion is removed only for
   correspondence scoring.
3. Remove the same vertex index from every frame.
4. Gate every accepted removal on exact per-frame OpenCV Recall and zero
   self-intersections.
5. Screen candidates in an exact C++/OpenMP raster batch, then re-audit the
   selected candidate conservatively in Python to absorb float32 one-pixel
   boundary differences.
6. Rank feasible removals by exact IoU plus similarity-normalized temporal
   vertex residual.
7. Optionally apply a globally consistent dense-index refinement. It is off by
   default because the first real-data ablation improved mean IoU only
   marginally while worsening the lower tail and doubling runtime.

The report contains two controls at every vertex count: the current forward
equal-arc phase alignment and an equal-arc contour with the new centre-outward
temporal phase alignment. This separates phase/correspondence gains from the
effect of non-uniform vertex deletion.

The primary artifact is `recommended_hybrid.sqlite`. It selects the lower
configured objective between temporally aligned equal-arc geometry and
trajectory decimation. `temporal_vertices_greedy_terminal.sqlite` is only the
end of the greedy deletion path; it is not claimed to be the globally minimum
feasible vertex count or the recommended quality point.

The default candidate objective combines mean IoU, lower-tail (q01) IoU,
similarity-normalized per-index temporal displacement, and the vertex-count
penalty. Recall 0.97 and polygon simplicity remain hard constraints.

The output SQLite files use an isolated `masks` table for visual/metric review;
the public result schema and Production implementation are not modified.

Example:

```bash
python run_experiment.py \
  --input-sqlite ../../../output/phase2_engine_profile_medium_track_20260810/input_track47.sqlite \
  --track-id 47 \
  --output-dir ../../../output/temporal_vertex_decimation_20260812/track47
```

Only SQLite polygon geometry is read. Video pixels are never opened.

Measured results and limitations are recorded in [RESULTS.md](RESULTS.md).
