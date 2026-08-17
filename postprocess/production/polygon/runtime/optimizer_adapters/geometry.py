"""Small geometry adapters for the parity-frozen optimizer kernel."""

from __future__ import annotations

from types import ModuleType

from ..kernel import geometry as kernel_geometry
from ..kernel import stream as kernel_stream


def install_geometry_adapters(
    module: ModuleType, original_resample_closed_contour
) -> None:
    def safe_resample_closed_contour(poly, n_points):
        target = max(3, int(n_points))
        out = module.np.asarray(
            original_resample_closed_contour(poly, target), dtype=module.np.float32
        ).reshape(-1, 2)
        if len(out) == target:
            return out
        if len(out) == 0:
            return module.np.zeros((target, 2), dtype=module.np.float32)
        return module.np.repeat(out[:1], target, axis=0).astype(module.np.float32)

    def fast_align_polygon_phase(reference, poly):
        candidate = module.np.asarray(
            module.orient_ccw(poly), dtype=module.np.float32
        ).reshape(-1, 2)
        if reference is None:
            return candidate
        ref = module.np.asarray(reference, dtype=module.np.float32).reshape(-1, 2)
        if len(ref) != len(candidate):
            return safe_resample_closed_contour(candidate, len(ref))
        count = int(len(candidate))
        if count <= 1:
            return candidate

        shift_ids = module.np.arange(count, dtype=module.np.int32)
        gather = (shift_ids[None, :] + shift_ids[:, None]) % count

        def best_roll(variant):
            rolled = module.np.asarray(variant, dtype=module.np.float32)[gather]
            diff = rolled - ref[None, :, :]
            scores = module.np.mean(module.np.sum(diff * diff, axis=2), axis=1)
            best_idx = int(module.np.argmin(scores))
            return (
                float(scores[best_idx]),
                module.np.asarray(rolled[best_idx], dtype=module.np.float32).copy(),
            )

        best_score, best = best_roll(candidate)
        reverse_score, reverse_best = best_roll(candidate[::-1].copy())
        if reverse_score < best_score:
            return reverse_best
        return best

    module.resample_closed_contour = safe_resample_closed_contour
    module.align_polygon_phase = fast_align_polygon_phase
    # Functions moved to the geometry module resolve their own globals.
    kernel_geometry.resample_closed_contour = safe_resample_closed_contour
    kernel_geometry.align_polygon_phase = fast_align_polygon_phase
    kernel_stream.resample_closed_contour = safe_resample_closed_contour
    kernel_stream.align_polygon_phase = fast_align_polygon_phase


__all__ = ("install_geometry_adapters",)
