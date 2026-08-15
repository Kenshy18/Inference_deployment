# Experimental new production

This freezes the accepted pre-temporal baseline:

- class-specific `production_candidate_best_v4` shape palettes;
- exact per-frame Recall floor 0.97;
- penalty-targeted key interval (not a hard key count);
- fixed-key, per-key pair-vote coordinate optimization;
- two alternating sweeps, 1/16 coarse and 1/128 local alpha grid;
- no temporal, movement, area, smoothness, or local-tail penalty;
- no post-decode expansion repair.

`new_production_v1` duplicates the palette rules explicitly rather than
changing the historical best-v4 profile. The optimized engine batches the
existing pair-vote alpha trials into exact C++/OpenCV calls. Candidate
generation, DP graph, alpha candidates, float32 interpolation, OpenCV raster
contract, Recall/IoU arithmetic, and tie-breaking remain unchanged.

The stable 24-core execution topology is three class jobs in parallel and one
CUDA-owning process per class. Pair-vote uses two OpenMP threads per class.
This avoids CUDA initialization failures from nested process pools.

Run the frozen matrix:

```bash
python postprocess/experimental/new_production/run.py \
  --engine optimized --intervals 1,3,6 \
  --label-workers 3 --num-workers 1 --pair-vote-threads 2
```

For output-parity benchmarking, run both engines and compare each interval:

```bash
python postprocess/experimental/new_production/run.py \
  --engine reference --intervals 1,3,6
python postprocess/experimental/new_production/compare_outputs.py \
  --reference output/new_production_benchmark_20260812/reference/interval_6 \
  --optimized output/new_production_benchmark_20260812/optimized/interval_6 \
  --output output/new_production_benchmark_20260812/parity_interval_6.json
```

Only SQLite polygon geometry is read; video pixels are never opened.

Measured results and the 20-minute projection are in
[`BENCHMARK_20260812.md`](BENCHMARK_20260812.md).
