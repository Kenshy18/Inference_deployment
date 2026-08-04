# Original postprocess equivalence check

This directory contains isolated comparison tooling for the legacy
`Dinov3_postprocess` implementation and the current modular postprocess.
It does not modify either production pipeline or the public SQLite schema.

The comparison covers:

- score filtering, adaptive NMS, tracking, and short-track pruning;
- high-precision cut detection on a generated fixture;
- polygon and ellipse output on identical real tracked masks;
- ellipse keyframe selection with an exactly aligned metrics input.

Run from the `postprocess` directory:

```bash
python -m tentative.compare_original_postprocess \
  --source-tracked-sqlite /path/to/tracked.sqlite \
  --work-dir /path/to/isolated/output \
  --frames-per-track 48 \
  --tracks 3 \
  --cut-video /path/to/real-video.mp4 \
  --device cpu \
  --force
```

`--force` deletes only the explicitly supplied comparison work directory.
Use a directory under `output/`; do not point it at a repository root.

The command writes `comparison.json` for machine-readable evidence and
`REPORT.md` for the concise result. Generated artifacts intentionally live
outside this directory so the repository remains clean.

For a larger polygon-only timing and vertex-distribution comparison, run
`python -m tentative.benchmark_polygon_restore` with the same tracked SQLite.

To diagnose temporal keyframe allocation without reading any video frames,
split consecutive contour motion into translation, similarity, full affine,
and residual local deformation:

```bash
python -m tentative.analyze_temporal_geometry \
  --source-sqlite /path/to/tracked.sqlite \
  --keyframes-sqlite /path/to/keyframes.sqlite \
  --output /path/to/temporal_geometry.json
```

For the ellipse optimizer, pass its `final_keyframes.json` using
`--keyframes-json` instead. Multiple K2 slot rows at the same track/frame are
treated as one keyframe event for temporal-allocation diagnostics.

This diagnostic reads only mask coordinates from SQLite. It does not decode,
display, export, or otherwise inspect source video frame pixels.

The human-editability gap audit for the production polygon optimizer is in
`POLYGON_HUMAN_EDIT_GAP_AUDIT_20260804.md`. It documents the current objective,
temporal behavior, measurable differences from a human editing workflow, and a
quality-constrained replacement objective without changing the SQLite schema.
