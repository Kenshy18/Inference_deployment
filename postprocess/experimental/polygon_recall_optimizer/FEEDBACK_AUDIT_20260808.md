# Visual-feedback audit: screen borders and apparent periodic lag

This audit used SQLite geometry only. No video frame was opened, decoded, or
uploaded.

## Screen-border finding

The reported timecodes map to nominal 30-fps frame indices `10730`, `10737`,
`10745`, `14430`, and `14704`. Every raw mask touches the top, right, or bottom
border. Superior satisfies its whole-mask Recall floor at all five locations,
but that constraint does not prevent the permitted missing area from being
concentrated in the final 24-pixel border strip.

Across every border-touching male-mask frame in the audit range:

| Variant | Side | Border-strip Recall mean | q05 | Frames below 0.97 | At least 6 px off-canvas |
|---|---|---:|---:|---:|---:|
| Production 5 | right | 0.9905 | 0.9565 | 79 / 794 | 99.7% |
| Superior 5 | right | 0.9279 | 0.8036 | 533 / 794 | 46.9% |
| Production 5 | bottom | 0.9859 | 0.9486 | 293 / 1983 | 99.4% |
| Superior 5 | bottom | 0.8839 | 0.7279 | 1722 / 1983 | 30.4% |
| Production 10 | right | 0.9905 | 0.9483 | 79 / 794 | 99.9% |
| Superior 10 | right | 0.9631 | 0.8442 | 270 / 794 | 70.2% |
| Production 10 | bottom | 0.9839 | 0.9395 | 298 / 1983 | 99.5% |
| Superior 10 | bottom | 0.9073 | 0.7503 | 1460 / 1983 | 56.6% |

The current constraint geometry is clipped to the visible video rectangle.
Production-style off-canvas proposals are present in the anchor pool, but the
mean-IoU objective can reject them because invisible area lowers IoU and no
hard border-local coverage or off-canvas extent constraint exists. The visual
feedback is therefore confirmed.

## Temporal-direction finding

The solver is offline and bidirectional in the relevant sense:

- every intermediate mask is interpolated from a left and a future right key;
- every candidate edge is evaluated over the entire interval;
- dynamic programming selects a path using the complete segment; and
- pair-vote endpoints are least-squares fitted from all observations in their
  interval.

Although the DAG is calculated from left to right, this is not causal filtering
and cannot create a general forward-processing latency.

Two independent geometry-only lag checks found global lag zero for Production
and Superior at intervals 5 and 10:

1. cross-correlation of raw versus predicted centroid velocity; and
2. mean IoU between `prediction[t]` and `raw[t + lag]` for lags `-5..+5`.

For Superior interval 10, three 30-frame local windows matched past raw geometry
slightly better than the current frame:

| Window | Best past lag | IoU gain versus lag zero |
|---|---:|---:|
| `00:05:47:16..00:05:48:15` | 2 frames | 0.0112 |
| `00:05:50:16..00:05:51:15` | 4 frames | 0.0101 |
| `00:07:44:06..00:07:45:05` | 3 frames | 0.0131 |

No equivalent window above a 0.01 IoU gain was found for Superior interval 5.
The plausible local cause is sparse linear interpolation around direction
reversals, not one-way processing. A straight vertex trajectory between two
keys cannot reproduce a curved or reversing periodic trajectory, and the
current objective has no explicit phase, velocity, or turning-point term.

## Recommended correction order

1. Preserve the whole-mask minimum Recall constraint.
2. Add a separate hard border-strip Recall constraint for touched sides.
3. Require a small off-canvas extent on those sides without charging invisible
   area to the ordinary IoU objective.
4. Re-run the five reported incidents and all border-touch statistics.
5. Only after border behavior is restored, evaluate turning-point key proposals
   or a local trajectory term for interval 10. Interval 5 currently has no
   measured systematic or material local phase lag.
