# Runtime floor stability experiments

This directory measures scheduling and transport changes that must preserve the
Production polygon result exactly.  Experiments are intentionally gated by
environment variables; Production defaults are not changed until the real-data
output, runtime, and memory audits pass.

`benchmark_worker_transport.py` compares the conservative `spawn` worker
transport with the existing Linux copy-on-write `fork` transport on the same
many-short-run source.  The parent executes variants serially and terminates a
variant if system `MemAvailable` falls below the configured floor.

The transport experiment found that `spawn` is not output-equivalent: the
Production optimizer installs its candidate adapters before forking, so spawned
children lose that inherited state.  Linux Production therefore remains on
`fork_inherited_index`; transport replacement is not a valid speed lever.

`benchmark_label_layout.py` compares equal per-label layouts, while
`benchmark_worker_allocation.py` measures unequal fixed-budget allocations such
as 4/1/1.  `adaptive_worker_allocation.py` is the deterministic, read-only row
count allocator being screened by those benchmarks.  It does not alter mask
geometry or the DP objective.

Current interval-3 results with a six-process budget:

| Data shape | Baseline | Row-proportional allocation | Optimizer FPS | Output |
| --- | --- | --- | ---: | --- |
| balanced KPI, 3 labels | 2/2/2 | 2/2/2 | 208.83 | exact match |
| highly skewed `white_axel`, 3 labels | 2/2/2: 106.74 FPS | 4/1/1 | 169.85 | exact match |
| one active label, joined male | 2 workers: 205.74 FPS | 6 workers | 282.57 | exact match |
| skewed KPI excerpt, 2 labels | 3/3: 199.15/203.36 FPS | 5/1 | 251.27/251.08 | exact match |

The skewed case improves 59.1% without changing any prediction row, keyframe
JSON byte, minimum Recall, or mean IoU.  The balanced case retains the fastest
tested layout.  The short two-label repeat differed by only 0.07% for the
adaptive allocation.  All long runs use a `MemAvailable` kill floor to avoid
repeating the prior WSL out-of-memory failure.

Budget screening on the 24-core / 30-GiB WSL host then found:

| Case | 6-process adaptive | 8-process adaptive | 9-process result |
| --- | ---: | ---: | --- |
| skewed `white_axel` | 169.85 FPS (4/1/1) | 199.38 FPS (6/1/1) | 203.05 FPS (7/1/1), only +1.8% over 8 |
| balanced KPI | 208.83 FPS (2/2/2) | 231.32 FPS (3/2/3) | stopped at 9.94-GiB `MemAvailable` |
| one active label | 282.57 FPS (6) | 281.08 FPS (8) | not useful |
| skewed KPI excerpt | 251.27 FPS (5/1) | 272.47 FPS (7/1) | not tested |

The screened policy is eight total optimizer processes for two or three active
labels, allocated in proportion to prepared observation rows, and six
processes for one active label.  Both values are hard caps: hosts exposing more
CPU cores do not receive a larger process budget without a separate memory
screen.  A caller-provided per-label worker cap is also respected.

Across these four scheduling strata, the screened policy changes the optimizer
timeline-FPS range from 106.74--282.57 (2.65x spread) to 199.38--282.57 (1.42x
spread).  The worst case rises 86.8%, while the mean rises 199.85 to 246.44 FPS;
the population coefficient of variation falls from 31.2% to 13.5%.  This is a
small, deliberately stratified runtime sample rather than a claim about every
possible video.

## Production integration validation

The policy is now the default in the Production polygon runtime.  It remains a
pure scheduling change: prepared SQLite row counts select the per-label worker
allocation, while candidate generation, DP costs, exact OpenCV validation,
pair-vote, and materialization are unchanged.  The legacy equal allocation can
still be selected diagnostically with
`MASK_PIPELINE_POLYGON_ADAPTIVE_WORKER_ALLOCATION=0`.

| End-to-end case | Old schedule | Adaptive schedule | Old full FPS | New full FPS | Polygon FPS | Quality |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `white_axel`, 18,000 frames | 2/2/2 | 6/1/1 | 93.08 | 156.98 | 103.29 -> 187.93 | exact geometry; Recall violations 0 |
| `white0210`, 54,000 frames | 2/2/2 | 4/3/1 | 144.80 | 194.77 | 170.15 -> 238.26 | all recorded quality metrics exact; Recall violations 0 |
| balanced KPI, 23,510 frames | 2/2/2 | 3/2/3 | 179.99 | 210.99 | 210.52 -> 250.07 | all recorded quality metrics exact; Recall violations 0 |
| compact KPI smoke, 2,400 frames | 2/2 | 7/1 | 128.82 | 216.86 | 141.96 -> 252.59 | Recall violations 0 |

For `white_axel`, all final geometry tables are exact between the old and new
runs.  The only final-SQLite differences are execution timestamps, artifact
paths, elapsed times, and worker-allocation provenance.  For `white0210`, all
recorded IoU, Recall, area-ratio, keyframe, run-count, and vertex-count metrics
are exactly equal.  Peak process-tree RSS rose from 8.09 to 11.28 GiB on
`white_axel` and from 15.62 to 18.54 GiB on `white0210`; minimum system
available memory remained 21.86 and 14.86 GiB respectively.

Across the three independent full V3 cases above, full-pipeline FPS changes
from 93.08--179.99 (1.93x spread) to 156.98--210.99 (1.34x spread).  The
population coefficient of variation falls from 25.6% to 12.1%, the measured
floor rises 68.6%, and mean FPS rises 34.7%.  The balanced KPI case retained
12.61 GiB minimum system-available memory.

The classwise GUI path was also exercised without environment overrides.  Two
independent class groups completed with the caller's three-worker cap, each
manifest recorded adaptive allocation, and the final integrated SQLite passed
validation.
