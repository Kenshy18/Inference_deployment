# Unreliable raw-mask review

This experiment ranks suspicious instance-segmentation observations using only
geometry and metadata stored in the V3 SQLite.  Video pixels are decoded only
by the local renderer when `--video` is supplied; they are never required for
candidate scoring.

The first pass is intentionally recall-oriented.  It separates sudden
expansion, sudden contraction, shape replacement, spatial jumps, short-lived
low-confidence detections, and border instability so a reviewer can label the
false positives before thresholds are tightened.

```bash
python -m postprocess.experimental.unreliable_raw_mask_review_20260807.review \
  --sqlite result.sqlite \
  --video source.mp4 \
  --output-dir output/raw_mask_review
```

Outputs:

- `all_observations.csv`: all scored raw genital observations.
- `review_candidates.csv`: temporally de-duplicated review candidates with
  empty human-review columns.
- `summary.json`: thresholds, counts, and distribution summaries.
- `unreliable_candidates_review.mp4`: local-only review video.  Red is the
  candidate raw mask; cyan is the translated neighbouring support.

For postprocess-retained observations only and chronological still images, add
`--kept-only --contact-sheets`.  Each contact sheet shows four frames before,
the target frame, and four frames after in a 3x3 grid.  The target mask is red;
the same raw track in neighbouring frames is yellow.
