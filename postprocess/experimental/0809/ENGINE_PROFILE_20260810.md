# Phase-2 interval engine profile (2026-08-10)

This benchmark reads polygon geometry from SQLite only. No video frame was
opened. Production code and the final SQLite schema were not changed.

## Representative input

- class: `男性器`
- track: `47`
- observations: 456
- candidate states per frame: 3 (`raw`, `scale_104`, `scale_112`)
- target interval: 5
- hard minimum Recall: 0.97
- pair-vote: disabled
- DP workers: 1

## Baseline profile

- optimizer wall time: 39.38 s
- solve-DP stage: 38.40 s
- candidate construction: 0.15 s
- interval/state edges: 112,077
- primary interval frame evaluations: 1,719,735
- first decode (edge construction plus one DP scan): 36.62 s
- 33 cached lambda decodes: 1.76 s total
- Production cached interval rasterization: 22.81 s
- hard-Recall exact recheck: 13.34 s

The shortest-path traversal itself is about 4.6% of solve time. Edge mask
interpolation/rasterization and Recall/IoU evaluation are about 94%.

## Exact-recheck investigation

Of 38,711 cached-feasible edges, 326 became infeasible under the exact
recheck. The cached and exact rasterizers can produce different GT pixel areas
for the same polygon. Examples include:

- cached: GT area 8,044; intersection 7,831; Recall 0.973521
- exact: GT area 8,099; intersection 7,848; Recall 0.969009

Therefore the 13.3 s exact pass cannot simply be removed. Increasing the
cached ROI pad from 8 to 128 did not remove the discrepancy and increased
runtime to 60.7 s.

## Rejected quick experiments

- Four Python threads precomputing edges did not complete within 120 s,
  versus 38.8 s serial. Shared Python execution, private-mask allocations and
  eager evaluation outweighed OpenCV parallelism.
- Matching the interpolation arithmetic used by final output reduced mismatch
  edges only from 330 to 326, so interpolation rounding is not the main cause.

## Native-engine target

The native implementation should batch the first decode's edge construction,
while preserving both the current cached quality cost and the exact hard-
Recall decision until a single canonical rasterization contract is validated.
## Native build environment and first kernel

An isolated toolchain now exists under
`/home/kenshin/.local/share/mask-pipeline-native-build`; the Production Python
environment was not modified. It contains Python 3.10, NumPy 1.26.4, OpenCV
4.8.1, GCC 14.4, CMake, Ninja and pybind11.

The first C++ exact-raster kernel passes 207 generated parity cases and 455
adjacent real-mask pairs with bit-exact area/Recall/IoU results. A small-call
microbenchmark is about 9.4x faster in the isolated interpreter and 11.2x
faster when loaded by the Production Python interpreter. This is a kernel
number, not the end-to-end DP speedup. The next porting target is batched edge
interpolation and cached quality evaluation, which accounts for the larger
22.8-second share above.

On the representative 456-frame, three-state track, enabling only this native
exact kernel produced the following end-to-end optimizer result:

- hard-Recall exact recheck: 13.34 s -> 4.34 s (3.07x)
- DP solve: 38.40 s -> 28.97 s (1.33x)
- optimizer: 39.38 s -> 29.95 s (1.31x)
- interval evaluations: unchanged at 112,077
- final keyframe JSON: byte-identical (same SHA-256)
- prediction SQLite: byte-identical (same SHA-256)

The native exact path is opt-in through
`MASK_PIPELINE_PHASE1_NATIVE_EXACT=1`; it remains experimental and does not
affect Production by default.

## Full native interval batch

The cached interpolated raster, exact hard-Recall recheck, 2-D similarity
distance and all edge-state combinations were subsequently moved into one
OpenMP C++ batch. Eight threads are the default experimental setting: 16 was
only about 8% faster while consuming roughly another 120 MB and leaving fewer
cores for track-level workers.

Representative three-state track result:

- Production optimizer: 39.379 s
- native-batch optimizer: 4.405 s (8.94x, 88.8% less time)
- Production solve: 38.400 s
- native-batch solve: 3.452 s (11.12x)
- native batch precompute: about 1.45 s for 118,935 edges
- observed peak RSS: about 1.04 GB

Correctness checks:

- all 112,077 used cached edges compared against Python; maximum scalar cost
  difference was 1.78e-15 before the final batch integration;
- all 38,714 cached-feasible edges compared for exact hard-Recall
  classification; mismatch count was zero;
- exact boundary behavior preserves Production's separate float32 conversion
  of `alpha` and `1-alpha`;
- final keyframe JSON and prediction SQLite are byte-identical to the
  Production baseline and have the same SHA-256 hashes.

The full path additionally requires:

```bash
export MASK_PIPELINE_PHASE1_NATIVE_INTERVAL=1
export MASK_PIPELINE_PHASE2_NATIVE_BATCH=1
export MASK_PIPELINE_PHASE2_NATIVE_BATCH_THREADS=8
```

It remains opt-in and experimental. Production and the SQLite schema are
unchanged.

## Five-state C++ DP and bounded scratch memory

The lambda shortest-path scan was moved to C++ and the native raster workers
were changed from one scratch mask per video frame to one reusable maximum-ROI
mask pair per worker.  The Python and C++ DP outputs were checked with both
three and five states; final keyframe JSON and prediction SQLite were
byte-identical.

Representative five-state track (`raw`, 1.04, 1.08, 1.12, 1.16):

- native batch plus Python DP: 10.578 s
- native batch plus C++ DP, 8 threads: 4.968 s
- native batch plus C++ DP, 24 threads: 2.54--2.61 s when uncontended
- bounded-scratch peak RSS at 24 threads: about 0.95 GB

Full three-class five-state run, 8 threads per class and three classes in
parallel:

- profile wall: 345.208 s; external wall: 363.93 s
- observations: 24,501; effective mean interval: 4.891
- minimum Recall violations: 0
- mean IoU: 0.898761
- external peak RSS: 10.59 GB

The previous full three-state Python-heavy run required 941.421 s.  The new
five-state engine is therefore about 2.59x faster despite evaluating more
states.

## CUDA prefilter and lazy exact path evaluation

The CUDA scanline kernel evaluates the dense interval/state graph.  A safe
prefilter budget of 0.25 produced zero false-rejected feasible edges on four
representative tracks spanning all three classes.  Retained edges can still be
processed by the exact C++ batch for a conservative, byte-identical path.

The faster lazy mode uses CUDA costs to propose a DP path and evaluates every
edge on that path with the exact OpenCV cached and hard-Recall implementation.
If an edge fails, it is removed and DP is repeated.  Consequently all final
path edges obey the exact Recall constraint, while OpenCV rasterization is
performed for only a small fraction of the dense graph.  Boundary differences
can still select a slightly different feasible Pareto point, so this remains
an opt-in research mode.

Large 979-frame / 722,625-edge male track:

- conservative exact prefilter path: 95.33 s external
- CUDA lazy exact, uint16 prefix: 28.90 s external (3.30x)
- exact CPU edges evaluated lazily: 12,067 (1.67% of graph)
- keyframes: 196 in both modes
- minimum Recall: 0.97022
- mean IoU: 0.96613 versus 0.96620 exact (difference -0.00007)
- IoU 1st percentile: 0.92844 versus 0.92736 exact
- peak RSS: 5.69 GB versus about 8.4 GB before uint16 prefix packing

The uint16 row-prefix representation is lossless for ROI widths up to 65,535
pixels and produced byte-identical lazy-mode artifacts on both the 456-frame
and 979-frame validation tracks.

## Final adaptive five-state full run

Forced lazy evaluation exposed one pathological 605-frame track: a CUDA graph
with a 50.4% retained ratio repeatedly proposed exact-infeasible paths.  The
adaptive engine therefore routes tracks below a 60% retained ratio directly to
the conservative exact C++ batch.  The problematic track then selected the
same keyframe positions and states as the reference instead of spending more
than 100 seconds in lazy retries.

The final full run used the 60% router, three class workers and eight C++
threads per active class:

- external wall: 151.75 s; profile wall: 144.528 s
- conservative five-state external wall: 363.93 s (adaptive is 2.40x faster)
- prior three-state Python-heavy wall: 941.421 s (adaptive is 6.20x faster)
- observations: 24,501; throughput: 169.52 rows/s
- target interval: 5; effective interval: 4.8894
- keyframes: 5,213 versus 5,211 conservative
- minimum Recall: at least 0.97; violations: zero
- mean IoU: 0.898707 versus 0.898761 (difference -0.0000535)
- class-min IoU q01: identical at 0.467777
- class-min minimum IoU and maximum area ratio: identical
- routing: 29 lazy streams, 73 conservative dense streams, one trivial
  single-frame stream
- exact edges visited by lazy search: 149,122 out of 17,655,025 dense graph
  edges (0.845%; conservative-routed tracks are evaluated separately)
- maximum single-process RSS reported by `/usr/bin/time`: 7,733,320 KiB;
  observed total system use remained below available RAM and Swap did not grow

This is a practical five-state experimental result.  Production remains
unchanged and the mode must still be promoted through quality review before it
becomes a default.
