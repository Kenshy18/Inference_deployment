# OpenCV-compatible CUDA raster validation (2026-08-22)

## Scope

The CUDA raster contract reproduces the Production subset of OpenCV 4.8.1
`fillPoly`: binary `uint8` masks, closed contours, `LINE_8`, and `shift=0`.
It preserves ties-to-even coordinate rounding, the LINE_8 boundary pixels,
the fixed-point scanline inclusion rule, and independent-component union.

The same implementation evaluates:

- ragged polygon boundaries;
- the canonical 96-point rotated-ellipse boundary;
- the closed uniform Catmull--Rom spline with fixed `1/6` handles, sampled at
  the Production density.

Metric output is exact integer `(reference area, prediction area,
intersection, union)` plus Recall, precision, and IoU derived from those
counts. The fused graph kernel also reproduces float64 interpolation, Recall
short-circuiting, loss accumulation order, and minimum Recall.

## Correctness gates

- Random and adversarial polygons from 3 through 20 points: pixel-identical.
- Canonical 96-point ellipses: pixel-identical.
- 224-point sampled Catmull--Rom curves: pixel-identical.
- Overlapping components filled independently then unioned: pixel-identical.
- Polygon, ellipse, and Catmull--Rom interval graphs: all five graph outputs
  bit-identical to `ExactDoubleRasterEvaluator`.
- Large-ROI forced tiling: bit-identical.
- Current post-processing test suite: 232 tests and 88 subtests passed.

On the 300-frame V3 Catmull--Rom acceptance slice at target interval 3, CPU,
CUDA, and hybrid runs produced byte-identical `keyframes.sqlite` and
`predictions.sqlite` files. The common output had 99 keys, effective interval
3.030303, mean IoU 0.9852734359, and minimum Recall 0.9706399892.

## Performance gate

Hardware: RTX 5090; CPU reference used the existing 8-thread native C++
OpenCV evaluator.

| Backend | DP seconds | Total seconds | Emitted FPS |
|---|---:|---:|---:|
| Exact CPU (default) | 0.390 | 2.642 | 113.53 |
| Exact CUDA | 0.446 | 7.720 | 38.86 |
| Exact hybrid | 0.811 | 3.170 | 94.63 |

The pure CUDA run is dominated by repeated small pair-vote and point-refine
batches. Against the native cached/OpenMP CPU evaluator, GPU launch and
transfer overhead is larger than their computation. A synthetic 38,784-edge
graph did favor fused CUDA by 1.15x (0.157 versus 0.181 seconds), but this did
not improve the complete real-data pipeline.

## Decision

The implementation proves that all three geometry types can use one exact
CUDA raster contract. It is retained as an explicit validation backend, with
an exact CPU fallback for unsupported topology or device limits. Production
continues to default to CPU because enabling CUDA would currently reduce
throughput without changing output quality.

Selection is explicit:

- `MASK_CURVE_EXACT_RASTER_BACKEND=cpu` (default)
- `MASK_CURVE_EXACT_RASTER_BACKEND=cuda`
- `MASK_CURVE_EXACT_RASTER_BACKEND=cuda_hybrid`

The selected backend is recorded in the curve manifest. Any future promotion
of CUDA must beat the native CPU evaluator on full-pipeline corpus timing while
retaining byte-identical SQLite output.
