# Production v22 minimum-Recall experiment

This directory leaves the validated Production implementation untouched and
runs an isolated semantic variant.

Production accumulates `1 - Recall` over a run and compares the total with
`frame_count * (1 - recall_floor)`.  Therefore `--recall-min 0.97` constrains
the mean Recall.  The experiment instead accumulates only
`max(recall_floor - Recall, 0)` and sets the permitted total to zero.  Exact
repair also compares candidates by minimum frame Recall rather than mean
Recall.

Everything else, including candidate construction, frame pooling, DP key
penalty, pair-vote refinement, and exported interpolation, is loaded from the
same Production source at runtime.

Example comparison:

```bash
python -m experimental.production_min_recall.compare \
  --input-sqlite /path/to/common/endpoint_extended.sqlite \
  --output-dir /path/to/output \
  --target-interval 10 \
  --recall-floor 0.97
```

The experiment only reads SQLite polygon geometry.  It does not open video
frames.
