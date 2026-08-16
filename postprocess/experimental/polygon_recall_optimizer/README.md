# Polygon recall optimizer experiment

This directory is deliberately isolated from the production stage registry. It
compares polygon keyframe strategies and contains a recall-constrained Pareto
DP. Production code and the SQLite schema are not modified.

The experiment reads only SQLite geometry.  It does not open, decode, export, or
inspect video frames.

Compared strategies include:

- the production keyframes as stored;
- production positions blended back toward their observed raw masks;
- raw anchors at the production positions;
- greedy minimax-recall key placement with the same key count;
- anchor-constrained pair-vote refinements;
- global and interval-local conservative scale repair.
- conservative anchor blending plus adaptive worst-frame key insertion.
- interval-free recall splitting, projected temporal smoothing, and dense
  recall revalidation (`lexicographic_recall_stability`).

The last strategy exposes the editability tradeoff directly:
`--repair-max-scale` limits mask expansion, while keys are added only when the
dense reconstructed mask would otherwise violate `--recall-floor`. A small
scale cap gives tighter masks and more keys; a larger cap gives fewer keys and
more over-mask. No fixed interval is used as a quality target.

## Production + raw-Recall guard

`run_production_guard.py` is the deliberately conservative Occam baseline. It
does not construct a temporal reference and does not replace Production's key
placement algorithm. It:

1. retains every Production keyframe position;
2. retains a Production key shape when it already passes the requested Recall;
3. minimally repairs only Production anchors that fail the floor;
4. evaluates every Production interpolation interval with the overlay reader;
5. inserts the fewest raw-observation anchors needed to make the interval safe;
6. uses raw-mask IoU only as the tie-breaker between equal-key paths; and
7. independently reloads the exported SQLite and repeats the dense validation.

Example:

```bash
PYTHONPATH=postprocess:overlay/src \
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  -m experimental.polygon_recall_optimizer.run_production_guard \
  --source-sqlite /path/to/raw-and-final.sqlite \
  --baseline-sqlite /path/to/production_interval10.sqlite \
  --output-dir /path/to/output \
  --output-sqlite /path/to/production_recall097.sqlite \
  --label 男性器 --start-frame 8681 --end-frame 20053 \
  --recall-floor 0.97 --guard-margin 0
```

This guarantees Recall only against the AI raw observation. It cannot repair a
region already missed by AI, and it can follow a raw false-positive expansion.
See `RESULTS_PRODUCTION_GUARD_20260806.md` for the measured keyframe, overlap,
stability, runtime, and SQLite-size tradeoff.

## Pareto DP

`pareto_dp.py` implements the current recommended experiment:

1. accept a fixed minimum **per-frame** recall value;
2. discard every interpolation edge that violates it at any observed frame;
3. retain the maximum cumulative IoU path for every reachable key count;
4. prune only solutions dominated in both key count and IoU;
5. combine independent track segments with a second Pareto DP; and
6. return the full front before selecting a knee, preference, key budget, soft
   target frequency/interval, or endpoint.

Unlike the production DP, a target keyframe interval is never mixed into the
optimization objective as a scalar penalty.  The complete Recall-safe Pareto
front is built first; a requested interval or frequency only chooses the
nearest point afterward. It is exact within the candidate-frame DAG and
configured edge horizon. Every raw observation is a candidate key position.
Candidate shapes are projected to the fixed recall floor before optimization,
and no unconstrained pair-vote mutation is applied after path decoding.

Example:

```bash
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  -m postprocess.experimental.polygon_recall_optimizer.run_pareto \
  --source-sqlite /path/to/source.sqlite \
  --baseline-sqlite /path/to/keyframe_interval_10.sqlite \
  --output-dir /path/to/output \
  --label 男性器 \
  --start-frame 8681 \
  --end-frame 20053 \
  --recall-floor 0.97 \
  --max-edge-span-frames 30 \
  --selection target_interval \
  --target-mean-key-interval 10 \
  --output-sqlite /path/to/result_recall097_interval10.sqlite
```

The command writes `pareto_frontier.json` and `selected_keyframes.json`. With
`--output-sqlite`, it also copies the baseline V3 result and transactionally
replaces only the selected segment keyframes. The SQL schema is fingerprinted
before and after and must be byte-for-byte identical. Capability row counts,
annotation revision, typed geometry, and provenance remain contract-valid.
The command independently reconstructs the selected solution and fails if its
Recall or IoU does not match the DP objective.

Available selection modes are `knee`, `preference`, `min_keys`, `max_iou`,
`key_budget`, `target_interval`, and `target_frequency`. Selection does not
alter the front or relax Recall. `target_interval` chooses the point nearest
`--target-mean-key-interval`; if the target is outside the feasible range, it
chooses the nearest boundary and records that fact in `target_status`.
`--quality-preference` ranges from 0 (fewer keys) to 1 (higher IoU).

### Multiple anchor-shape states

`--anchor-state-count` also lets the DP choose the polygon shape at every
selected keyframe. State 1 is the original highest-local-IoU feasible anchor.
Additional states retain geometrically distinct production/raw projections and
expansions up to `--anchor-expansion`. A locally wider
shape may have lower anchor IoU but support a longer recall-safe interpolation,
so it must remain a separate DP state rather than being locally pruned.

With `S` states, candidate edge work grows approximately as `S^2`: every left
shape must be tested against every right shape. States ending in different
shapes cannot be merged before their outgoing edges are evaluated. Segment
parallelism can reduce wall time later, but does not remove this fundamental
work increase.

The CLI now uses a CPU-aware exact-speed preset by default. On the 24-core
target workstation it resolves to six independent segment workers with four
forked edge workers per segment. Long segments are submitted first to avoid a
single-track tail, while results are restored to source order before global
combination. `--workers` and `--edge-processes` set these explicitly; zero keeps
automatic selection. The forked workers inherit the read-only geometry cache,
and result batches are merged in the original order, so serial and parallel
Pareto fronts are byte-for-byte identical.

Exact speedups also include cached anchor resampling/geometry, vectorized
roll/reversal alignment, cached raw areas/bounds, and a conservative bounding-
box intersection upper bound. GPU rasterization was rejected because it would
turn the hard continuous-polygon Recall constraint into an approximation. The
installed environment has no exact CUDA polygon-overlay library; GEOS remains
the correct backend for this reference objective.

## Earlier comparison harness

`run_comparison.py` retains the earlier fixed-budget and lexicographic methods
for reference. The lexicographic strategy treats minimum per-frame recall as a
hard constraint, uses fewer keys when possible, then improves overlap and
temporal stability without final mask dilation.

Example:

```bash
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  -m postprocess.experimental.polygon_recall_optimizer.run_comparison \
  --source-sqlite /path/to/source.sqlite \
  --baseline-sqlite /path/to/keyframe_interval_10.sqlite \
  --output-dir /path/to/output \
  --start-frame 11950 \
  --end-frame 14650 \
  --recall-floor 0.90 \
  --repair-max-scale 1.15
```

`summary.json` contains tail recall, violation counts, precision, IoU, excess
area, keyframe-anchor recall, adjacent-mask IoU, area flicker, and centroid
acceleration. `frame_metrics.csv` retains the paired per-frame comparison.

See `RESULTS_20260805.md` for the current full-dataset findings. These metrics
use the AI raw mask as a proxy reference, not a human-labelled semantic GT.

## Superior Production-compatible solver

`run_superior.py` is the strict successor experiment. It adds the safeguards
that are required before the Pareto implementation can replace Production:

1. Production segment/cut boundaries and endpoint coverage are inherited from
   the baseline V3 SQLite;
2. Production's exact border transform is retained for anchor construction,
   while invisible off-canvas area is excluded from Recall accounting;
3. minimum per-frame Recall is checked independently against both the visible
   border constraint and the original AI mask;
4. every candidate key uses one fixed vertex count, winding, and cyclic origin;
5. `linear_polygon_index_v1` makes Overlay use the same stored `point_index`
   interpolation as the editing software, including missing-observation gaps;
6. four default anchor states retain the precise 4% expansion candidate and
   add medium/large expansion candidates up to 30%, exposing the intended
   key-frequency versus over-mask tradeoff without sacrificing the high-key
   frontier;
7. Production pair-vote least-squares endpoints are additive candidate states
   before DP; they never replace the four core states and every edge using
   them is densely Recall-validated before selection;
8. the selected point must meet or exceed the Production baseline mean IoU;
   only then is closeness to the requested key interval considered; and
9. export copies the baseline and replaces keyframe rows transactionally. The
   schema fingerprint, integrity check, foreign keys, dense Recall, and saved
   geometry are all revalidated after writing.

The strict audit is `audit_superior.py`. It compares editor and Overlay
geometry on every segment frame (not only detection frames), reports
per-segment Recall, evaluates known incident frames, compares the complete new
front against multiple Production keyframe/IoU points, and rechecks schema
identity. Both commands operate on SQLite geometry only and never open video
pixels.

Measured full-range results and the remaining interval-15/runtime limitations
are documented in `RESULTS_SUPERIOR_20260807.md`.

Example:

```bash
PYTHONPATH=postprocess:overlay/src \
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  -m experimental.polygon_recall_optimizer.run_superior \
  --source-sqlite /path/to/raw-and-final.sqlite \
  --baseline-sqlite /path/to/production.sqlite \
  --output-dir /path/to/superior-output \
  --output-sqlite /path/to/superior.sqlite \
  --label 男性器 --start-frame 8681 --end-frame 20059 \
  --recall-floor 0.97 --target-mean-key-interval 10
```

## Trusted temporal-mask experiment

`run_trusted.py` addresses the main proxy-reference failure of the earlier DP:
an AI mask at one frame is a noisy observation, not ground truth. It therefore:

1. optimizes each cut/track segment independently;
2. motion-aligns neighbouring polygon observations without opening video;
3. keeps neighbour-supported area through a one-frame contraction;
4. rejects unsupported one-frame expansion but accepts persistent expansion;
5. enforces Recall against that trusted temporal reference, not blindly
   against every raw observation;
6. spends polygon vertices at corners/tips with `simplify_budget`;
7. optimizes a lower-tail IoU plus soft normalized-boundary utility;
8. excludes key-shape states that fall too far below the best trusted-anchor
   overlap; and
9. treats the requested key interval as an effort target, then adds keys only
   while their local robust-quality gain remains material.

Example research run:

```bash
PYTHONPATH=postprocess:overlay/src \
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  -m experimental.polygon_recall_optimizer.run_trusted \
  --source-sqlite /path/to/raw-and-final.sqlite \
  --baseline-sqlite /path/to/production.sqlite \
  --output-dir /path/to/trusted-output \
  --label 男性器 --start-frame 8681 --end-frame 20053 \
  --recall-floor 0.97 --target-mean-key-interval 10 \
  --quality-mode tail_boundary --consensus-radius 2 \
  --max-edge-span-frames 30 --point-count 23 \
  --anchor-state-count 3 --anchor-expansion 0.04 \
  --anchor-relative-iou-margin 0.15 \
  --workers 2 --edge-processes 12
```

The algorithm remains experimental. In particular, the trusted temporal mask
is a geometry-only proxy rather than human GT, and the current reference search
is intentionally exact and much slower than the Production postprocess. See
`RESULTS_TRUSTED_20260806.md` for measured quality, speed, and open issues.
