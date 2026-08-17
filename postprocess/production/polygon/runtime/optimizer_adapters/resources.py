"""Memory and worker-lifecycle adapters for the Production optimizer."""

from __future__ import annotations

from types import ModuleType

from ..kernel import stream as kernel_stream


def install_resource_adapters(
    module: ModuleType,
    original_build_track_streams,
    original_apply_fixed_practical_defaults,
    original_get_context,
) -> None:
    def build_track_streams_releasing_predictor(*args, **kwargs):
        release_predictor_after_build = bool(
            kwargs.pop("_release_predictor_after_build", True)
        )
        predictor = kwargs.get("predictor")
        if predictor is None and len(args) >= 3:
            predictor = args[2]
        result = original_build_track_streams(*args, **kwargs)
        if predictor is not None and release_predictor_after_build:
            try:
                predictor.model.to("cpu")
            except Exception:
                pass
            try:
                if module.torch.cuda.is_available():
                    module.torch.cuda.synchronize()
                    module.torch.cuda.empty_cache()
            except Exception:
                pass
        return result

    def memory_bounded_build_frame_eval_contexts(run, args):
        import collections as collections_mod
        import os as os_mod

        default_cache = 512
        try:
            max_items = int(
                os_mod.environ.get(
                    "ATOSYORI_POLYGON_EVAL_CONTEXT_CACHE", str(default_cache)
                )
            )
        except ValueError:
            max_items = default_cache
        max_items = max(1, int(max_items))

        class LazyFrameEvalContexts:
            def __init__(self):
                self._cache = collections_mod.OrderedDict()

            def __len__(self):
                return int(len(run.frame_numbers))

            def _build_one(self, frame_idx):
                scale_factor = float(
                    module.np.clip(float(args.dp_eval_scale), 0.1, 1.0)
                )
                pad = int(max(0, int(args.dp_eval_pad)))
                raw_vector = module.flatten_contours(run.anchors[int(frame_idx)])
                (
                    gt_polygon_area,
                    gt_center,
                    gt_radii,
                    gt_mean_radius,
                ) = module.vector_proxy_stats(
                    raw_vector,
                    run.contour_count,
                    run.anchors_per_contour,
                )
                raw_polys = module.split_vector_to_polygons(
                    module.flatten_contours(run.anchors[int(frame_idx)]),
                    run.contour_count,
                    run.anchors_per_contour,
                )
                all_polys = [
                    module.np.asarray(poly, dtype=module.np.float32)
                    for poly in run.gt_polygons[int(frame_idx)] + raw_polys
                    if len(poly) >= 3
                ]
                if all_polys:
                    all_pts = module.np.concatenate(all_polys, axis=0)
                    min_xy = (
                        module.np.floor(all_pts.min(axis=0)).astype(module.np.int32)
                        - pad
                    )
                    max_xy = (
                        module.np.ceil(all_pts.max(axis=0)).astype(module.np.int32)
                        + pad
                    )
                else:
                    min_xy = module.np.asarray([0, 0], dtype=module.np.int32)
                    max_xy = module.np.asarray([4, 4], dtype=module.np.int32)
                shift_xy = min_xy.astype(module.np.float32)
                width = int(max_xy[0] - min_xy[0] + 1)
                height = int(max_xy[1] - min_xy[1] + 1)
                shape_hw = (
                    max(1, int(module.math.ceil(height * scale_factor))),
                    max(1, int(module.math.ceil(width * scale_factor))),
                )
                context = module.FrameEvalContext(
                    gt_mask=module.np.zeros(shape_hw, dtype=module.np.uint8),
                    gt_area=0,
                    shift_xy=shift_xy,
                    shape_hw=shape_hw,
                    scale_factor=scale_factor,
                    gt_center=module.np.asarray(gt_center, dtype=module.np.float32),
                    gt_radii=module.np.asarray(gt_radii, dtype=module.np.float32),
                    gt_mean_radius=float(gt_mean_radius),
                    gt_polygon_area=float(gt_polygon_area),
                )
                gt_mask = module.rasterize_mask_with_context(
                    run.gt_polygons[int(frame_idx)], context
                )
                return module.FrameEvalContext(
                    gt_mask=gt_mask,
                    gt_area=int(gt_mask.sum()),
                    shift_xy=shift_xy,
                    shape_hw=shape_hw,
                    scale_factor=scale_factor,
                    gt_center=module.np.asarray(gt_center, dtype=module.np.float32),
                    gt_radii=module.np.asarray(gt_radii, dtype=module.np.float32),
                    gt_mean_radius=float(gt_mean_radius),
                    gt_polygon_area=float(gt_polygon_area),
                    scratch_pred_mask=module.np.zeros(shape_hw, dtype=module.np.uint8),
                    scratch_intersection_mask=module.np.zeros(
                        shape_hw, dtype=module.np.uint8
                    ),
                )

            def __getitem__(self, frame_idx):
                idx = int(frame_idx)
                if idx < 0:
                    idx += int(len(run.frame_numbers))
                if idx < 0 or idx >= int(len(run.frame_numbers)):
                    raise IndexError(idx)
                cached = self._cache.get(idx)
                if cached is not None:
                    self._cache.move_to_end(idx)
                    return cached
                context = self._build_one(idx)
                self._cache[idx] = context
                if len(self._cache) > max_items:
                    self._cache.popitem(last=False)
                return context

        return LazyFrameEvalContexts()

    def apply_fixed_practical_defaults_with_worker_mode(args):
        args = original_apply_fixed_practical_defaults(args)
        module._fork_polygon_workers = True
        return args

    def stable_polygon_get_context(method=None):
        if method == "spawn" and bool(getattr(module, "_fork_polygon_workers", False)):
            try:
                return original_get_context("fork")
            except ValueError:
                return original_get_context(method)
        return original_get_context(method)

    class PolygonMultiprocessingProxy:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def get_context(self, method=None):
            return stable_polygon_get_context(method)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    module.build_track_streams = build_track_streams_releasing_predictor
    kernel_stream.build_track_streams = build_track_streams_releasing_predictor
    module.build_frame_eval_contexts = memory_bounded_build_frame_eval_contexts
    module.apply_fixed_practical_defaults = (
        apply_fixed_practical_defaults_with_worker_mode
    )
    module.multiprocessing = PolygonMultiprocessingProxy(module.multiprocessing)


__all__ = ("install_resource_adapters",)
