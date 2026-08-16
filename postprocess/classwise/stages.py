"""Composite stage that routes disjoint tracks through existing pipelines."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.config import PipelineConfig, StageSpec
from common.runner import PipelineRunner
from contracts.stages import StageContext, StageResult

from .policy import (
    ClassPostprocessSettings,
    load_class_postprocess_policy,
)
from .sqlite import (
    RoutedGroup,
    count_masks,
    filter_tracked_sqlite,
    merge_routed_outputs,
    read_track_labels,
)


def _nested_pipeline(
    settings: ClassPostprocessSettings,
    *,
    ellipse_options: dict[str, object],
    polygon_options: dict[str, object],
) -> PipelineConfig:
    if settings.shape_mode == "polygon":
        stages = (
            StageSpec(
                "polygon_optimization",
                "production.polygon_v3_cpu",
                {
                    **polygon_options,
                    "target_interval": settings.keyframe_interval,
                    "max_gap": settings.max_gap,
                    "interval_evaluation": "native_exact",
                },
            ),
            StageSpec("exact_evaluation", "evaluation.mask_iou"),
            StageSpec("output_validation", "artifacts.validate"),
        )
    else:
        stages = (
            StageSpec(
                "ellipse_approximation",
                "approximation.ellipse.production",
                dict(ellipse_options),
            ),
            StageSpec(
                "keyframe_selection",
                "keyframes.ellipse.dense",
                {
                    "target_ratio": 1.0 / settings.keyframe_interval,
                    "dense_recall_target": 0.96,
                },
            ),
            StageSpec(
                "mask_gap_fill",
                "gap_fill.ellipse.linear",
                {"max_gap": settings.max_gap},
            ),
            StageSpec("exact_evaluation", "evaluation.ellipse.exact"),
            StageSpec("sqlite_export", "artifacts.union_sqlite"),
            StageSpec("output_validation", "artifacts.validate"),
        )
    return PipelineConfig(
        name=(
            f"classwise_{settings.shape_mode}_"
            f"k{settings.keyframe_interval}_g{settings.max_gap}"
        ),
        stages=stages,
    )


@dataclass(frozen=True)
class ClasswisePostprocessStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "classwise_postprocess"
    requires: frozenset[str] = frozenset(
        {"tracked_sqlite", "class_postprocess_policy_json"}
    )
    provides: frozenset[str] = frozenset({"predictions_sqlite", "classwise_manifest"})

    def run(self, context: StageContext) -> StageResult:
        started = time.perf_counter()
        fallback_shape = str(self.options.get("default_shape_mode", "polygon"))
        fallback_gap_value = self.options.get("default_max_gap")
        fallback = ClassPostprocessSettings(
            shape_mode=fallback_shape,
            keyframe_interval=int(
                self.options.get(
                    "default_keyframe_interval",
                    6 if fallback_shape == "polygon" else 3,
                )
            ),
            max_gap=(
                int(fallback_gap_value)
                if fallback_gap_value is not None
                else 30
                if fallback_shape == "ellipse"
                else 15
            ),
        )
        policy = load_class_postprocess_policy(
            context.artifacts["class_postprocess_policy_json"],
            fallback=fallback,
        )
        tracked = context.artifacts["tracked_sqlite"]
        track_labels = read_track_labels(tracked)
        # The original production pipeline optimized each semantic class in an
        # independent branch.  Do not coalesce labels merely because their
        # settings happen to match: both ellipse and polygon optimizers have a
        # branch-global keyframe budget.
        tracks_by_group: dict[tuple[str, ClassPostprocessSettings], list[str]] = {}
        for track_id, label in sorted(track_labels.items()):
            settings = policy.resolve(label)
            tracks_by_group.setdefault((label, settings), []).append(track_id)

        ellipse_options = dict(self.options.get("ellipse_options", {}))
        polygon_options = dict(self.options.get("polygon_options", {}))
        routed: list[RoutedGroup] = []
        group_manifests: list[dict[str, object]] = []
        ordered_groups = sorted(
            tracks_by_group,
            key=lambda value: (
                value[1].shape_mode,
                value[1].keyframe_interval,
                value[1].max_gap,
                value[0],
            ),
        )
        for index, (label, settings) in enumerate(ordered_groups):
            group_started = time.perf_counter()
            group_id = (
                f"{index:02d}_{settings.shape_mode}_"
                f"k{settings.keyframe_interval}_g{settings.max_gap}"
            )
            group_root = context.stage_dir / "groups" / group_id
            projected = group_root / "tracked.sqlite"
            track_ids = tuple(tracks_by_group[(label, settings)])
            if set(track_ids) == set(track_labels):
                projected = Path(tracked)
                input_masks = count_masks(projected)
            else:
                input_masks = filter_tracked_sqlite(
                    tracked,
                    projected,
                    track_ids=track_ids,
                )
            nested_root = group_root / "pipeline"
            nested_inputs = {"tracked_sqlite": projected}
            if context.artifacts.get("input_video") is not None:
                nested_inputs["input_video"] = context.artifacts["input_video"]
            manifest = PipelineRunner(
                _nested_pipeline(
                    settings,
                    ellipse_options=ellipse_options,
                    polygon_options=polygon_options,
                ),
                nested_root,
            ).run(nested_inputs)
            predictions = (
                Path(str(manifest["artifacts"]["predictions_sqlite"]))
                .expanduser()
                .resolve()
            )
            routed.append(
                RoutedGroup(
                    group_id=group_id,
                    labels=(label,),
                    track_ids=track_ids,
                    settings=settings,
                    predictions_sqlite=predictions,
                )
            )
            output_masks = 0
            for stage in manifest["stages"]:
                if stage["id"] == "output_validation":
                    output_masks = int(stage["metadata"]["masks"])
                    break
            group_manifests.append(
                {
                    "id": group_id,
                    "labels": [label],
                    "track_ids": list(track_ids),
                    "settings": settings.as_dict(),
                    "input_masks": input_masks,
                    "output_masks": output_masks,
                    "pipeline_manifest": str(nested_root / "pipeline_manifest.json"),
                    "predictions_sqlite": str(predictions),
                    "elapsed_seconds": time.perf_counter() - group_started,
                }
            )

        output = context.stage_dir / "predictions.sqlite"
        merge_summary = merge_routed_outputs(
            tracked,
            tuple(routed),
            output,
            policy=policy,
            track_labels=track_labels,
        )
        elapsed = time.perf_counter() - started
        classwise_manifest = context.stage_dir / "classwise_manifest.json"
        manifest_value = {
            "schema_version": 1,
            "policy": policy.as_dict(),
            "policy_source": str(context.artifacts["class_postprocess_policy_json"]),
            "tracked_sqlite": str(tracked),
            "predictions_sqlite": str(output),
            "groups": group_manifests,
            "merge": merge_summary,
            "elapsed_seconds": elapsed,
        }
        classwise_manifest.write_text(
            json.dumps(manifest_value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return StageResult(
            {
                "predictions_sqlite": output,
                "classwise_manifest": classwise_manifest,
            },
            {
                "policy": policy.as_dict(),
                "groups": len(group_manifests),
                "group_summaries": group_manifests,
                **merge_summary,
                "elapsed_seconds": elapsed,
            },
        )


__all__ = ["ClasswisePostprocessStage"]
