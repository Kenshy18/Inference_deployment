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
                "polygon_approximation",
                "approximation.polygon.rdp",
                dict(polygon_options),
            ),
            StageSpec(
                "keyframe_selection",
                "keyframes.polygon.interval",
                {"interval_frames": settings.keyframe_interval},
            ),
            StageSpec(
                "mask_gap_fill",
                "gap_fill.polygon.linear",
                {"max_gap": settings.max_gap},
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
    provides: frozenset[str] = frozenset(
        {"predictions_sqlite", "classwise_manifest"}
    )

    def run(self, context: StageContext) -> StageResult:
        started = time.perf_counter()
        fallback_shape = str(self.options.get("default_shape_mode", "polygon"))
        fallback_gap_value = self.options.get("default_max_gap")
        fallback = ClassPostprocessSettings(
            shape_mode=fallback_shape,
            keyframe_interval=int(
                self.options.get("default_keyframe_interval", 3)
            ),
            max_gap=(
                int(fallback_gap_value)
                if fallback_gap_value is not None
                else 30
                if fallback_shape == "ellipse"
                else 0
            ),
        )
        policy = load_class_postprocess_policy(
            context.artifacts["class_postprocess_policy_json"],
            fallback=fallback,
        )
        tracked = context.artifacts["tracked_sqlite"]
        track_labels = read_track_labels(tracked)
        tracks_by_settings: dict[ClassPostprocessSettings, list[str]] = {}
        labels_by_settings: dict[ClassPostprocessSettings, set[str]] = {}
        for track_id, label in sorted(track_labels.items()):
            settings = policy.resolve(label)
            tracks_by_settings.setdefault(settings, []).append(track_id)
            labels_by_settings.setdefault(settings, set()).add(label)

        ellipse_options = dict(self.options.get("ellipse_options", {}))
        polygon_options = dict(self.options.get("polygon_options", {}))
        routed: list[RoutedGroup] = []
        group_manifests: list[dict[str, object]] = []
        ordered_settings = sorted(
            tracks_by_settings,
            key=lambda value: (
                value.shape_mode,
                value.keyframe_interval,
                value.max_gap,
            ),
        )
        for index, settings in enumerate(ordered_settings):
            group_started = time.perf_counter()
            group_id = (
                f"{index:02d}_{settings.shape_mode}_"
                f"k{settings.keyframe_interval}_g{settings.max_gap}"
            )
            group_root = context.stage_dir / "groups" / group_id
            projected = group_root / "tracked.sqlite"
            track_ids = tuple(tracks_by_settings[settings])
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
            manifest = PipelineRunner(
                _nested_pipeline(
                    settings,
                    ellipse_options=ellipse_options,
                    polygon_options=polygon_options,
                ),
                nested_root,
            ).run({"tracked_sqlite": projected})
            predictions = Path(
                str(manifest["artifacts"]["predictions_sqlite"])
            ).expanduser().resolve()
            routed.append(
                RoutedGroup(
                    group_id=group_id,
                    labels=tuple(sorted(labels_by_settings[settings])),
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
                    "labels": sorted(labels_by_settings[settings]),
                    "track_ids": list(track_ids),
                    "settings": settings.as_dict(),
                    "input_masks": input_masks,
                    "output_masks": output_masks,
                    "pipeline_manifest": str(
                        nested_root / "pipeline_manifest.json"
                    ),
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
            "policy_source": str(
                context.artifacts["class_postprocess_policy_json"]
            ),
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
