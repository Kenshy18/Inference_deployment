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
