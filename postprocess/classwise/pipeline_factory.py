"""Nested geometry pipelines owned by class-aware post-processing."""

from __future__ import annotations

from common.config import PipelineConfig, StageSpec

from .policy import ClassPostprocessSettings


def build_nested_pipeline(
    settings: ClassPostprocessSettings,
    *,
    geometry_options: dict[str, object],
    geometry_mode: str,
    curve_cpu_threads: int,
    selected_track_ids: tuple[str, ...],
) -> PipelineConfig:
    """Build one disjoint class/track route without executing it."""

    if geometry_mode not in {"polygon", "catmull_rom"}:
        raise ValueError(f"unsupported classwise geometry: {geometry_mode}")
    stages = (
        StageSpec(
            (
                "polygon_optimization"
                if geometry_mode == "polygon"
                else "curve_optimization"
            ),
            (
                "production.polygon_v3_cpu"
                if geometry_mode == "polygon"
                else "production.curve_v1_cpu"
            ),
            {
                **geometry_options,
                "target_interval": settings.keyframe_interval,
                **(
                    {
                        "interval_evaluation": str(
                            geometry_options.get(
                                "interval_evaluation", "cuda_lazy_exact"
                            )
                        )
                    }
                    if geometry_mode == "polygon"
                    else {
                        "native_cpu_threads": int(curve_cpu_threads),
                        "selected_track_ids": list(selected_track_ids),
                    }
                ),
            },
        ),
        StageSpec(
            "exact_evaluation",
            "evaluation.mask_iou",
            (
                {"selected_track_ids": list(selected_track_ids)}
                if geometry_mode == "catmull_rom"
                else {}
            ),
        ),
        StageSpec("output_validation", "artifacts.validate"),
    )
    return PipelineConfig(
        name=f"classwise_{geometry_mode}_k{settings.keyframe_interval}",
        stages=stages,
    )


__all__ = ("build_nested_pipeline",)
