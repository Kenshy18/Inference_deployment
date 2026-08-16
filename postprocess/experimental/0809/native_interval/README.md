# Native interval evaluator bootstrap

This is an isolated CPU C++/OpenCV/pybind11 feasibility prototype for the
Phase 2 interval evaluator. It does not replace Production code and does not
change the SQLite schema.

The bootstrap pins micromamba 2.9.0 and verifies its SHA-256 before extracting
it with Python (so system `bzip2` is not required). The environment is installed under
`/home/kenshin/.local/share/mask-pipeline-native-build`, outside both the Git
working tree and the Production Python environment. Run:

```bash
bash postprocess/experimental/0809/native_interval/bootstrap_and_build.sh
```

The first module implements the exact polygon raster metrics only. The parity
test uses generated polygon coordinates and does not open videos or SQLite.
This provides a correctness boundary before porting interpolation and batched
edge evaluation into C++.

The experimental Phase 1/2 runtime can opt into the built module with:

```bash
export PYTHONPATH="$PWD/postprocess/experimental/0809/native_interval/build:$PYTHONPATH"
export MASK_PIPELINE_PHASE1_NATIVE_EXACT=1
export MASK_PIPELINE_PHASE1_NATIVE_INTERVAL=1
export MASK_PIPELINE_PHASE2_NATIVE_BATCH=1
export MASK_PIPELINE_PHASE2_NATIVE_BATCH_THREADS=8
export MASK_PIPELINE_PHASE2_NATIVE_DP=1
```

The module has also been loaded and tested directly from the existing
Production Python 3.10 environment. Its build RPATH points at the isolated
OpenCV runtime, so Production packages do not need to be modified.

The native batch is intentionally opt-in. It precomputes cached and exact
hard-Recall edge results in parallel, then leaves the existing Python penalty
search and artifact writers unchanged. For deep QA, setting
`MASK_PIPELINE_PHASE2_NATIVE_BATCH_EXACT_VERIFY=1` recomputes every used exact
edge through the Python reference path and reports classification mismatches.

`MASK_PIPELINE_PHASE2_NATIVE_DP=1` also moves each lambda shortest-path scan to
C++.  It changes neither the objective nor artifacts: the five-state reference
run produced byte-identical keyframe JSON and prediction SQLite compared with
the Python DP.

## Optional CUDA dense-graph prototype

CUDA dependencies are deliberately isolated from the Production environment:

```bash
bash postprocess/experimental/0809/native_interval/bootstrap_cuda_experiment.sh
```

The conservative CUDA mode only rejects clearly infeasible dense edges and
then runs the exact C++ evaluator on all retained edges:

```bash
export MASK_PIPELINE_PHASE2_CUDA_PREFILTER=1
export MASK_PIPELINE_PHASE2_CUDA_PREFILTER_BUDGET=0.25
```

The faster research mode uses the CUDA scanline result as the dense DP
heuristic and lazily re-evaluates every edge entering a candidate path with
the exact OpenCV implementation:

```bash
export MASK_PIPELINE_PHASE2_CUDA_LAZY_EXACT=1
```

Thus every accepted path edge still satisfies the exact hard-Recall check.
The dense CUDA raster differs slightly from OpenCV at polygon boundaries, so
the selected Pareto point can differ slightly.  This mode remains experimental
and is not enabled by Production defaults.

The lazy mode is adaptive.  If CUDA retains less than 60% of the graph, the
track is processed directly by the conservative dense C++ evaluator; this
avoids repeated proposals on graphs dominated by Recall failures.  A high
exact-infeasible-ratio guard provides a second content-based fallback.
Set `MASK_PIPELINE_PHASE2_CUDA_LAZY_MIN_RETAINED_RATIO=0` only when explicitly
profiling forced lazy behavior.
