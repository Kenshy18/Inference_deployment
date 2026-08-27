"""Factories for feature-owned stage implementations.

Factories import a feature only when that stage is selected.  No algorithm is
implemented here.  Registration is explicit so importing :mod:`common` has no
process-global side effects.
"""

from collections.abc import Mapping
from typing import Any

from contracts.stages import PostprocessStage
from .registry import StageFactory, register_stage


def normalization(options: dict[str, Any]) -> PostprocessStage:
    from preprocessing.stages import NormalizationStage

    return NormalizationStage(options)


def raw_sqlite_normalization(options: dict[str, Any]) -> PostprocessStage:
    from preprocessing.stages import RawSqliteNormalizationStage

    return RawSqliteNormalizationStage(options)


def score_policy(options: dict[str, Any]) -> PostprocessStage:
    from preprocessing.stages import ScorePolicyStage

    return ScorePolicyStage(options)


def production_mask_nms(options: dict[str, Any]) -> PostprocessStage:
    from nms.production import ProductionMaskNmsStage

    return ProductionMaskNmsStage(options)


def cut_detection(options: dict[str, Any]) -> PostprocessStage:
    from cut_detection.stages import VideoCutDetectionStage

    return VideoCutDetectionStage(options)


def tracking(options: dict[str, Any]) -> PostprocessStage:
    from tracking.stages import TrackingStage

    return TrackingStage(options)


def polygon_production_v3_cpu(options: dict[str, Any]) -> PostprocessStage:
    from production.polygon import ProductionPolygonStage

    return ProductionPolygonStage(options)


def curve_production_v1_cpu(options: dict[str, Any]) -> PostprocessStage:
    from production.curve import ProductionCurveStage

    return ProductionCurveStage(options)


def mask_evaluation(options: dict[str, Any]) -> PostprocessStage:
    from evaluation.stages import MaskIouEvaluationStage

    return MaskIouEvaluationStage(options)


def union_sqlite(options: dict[str, Any]) -> PostprocessStage:
    from artifacts.stages import UnionSqliteExportStage

    return UnionSqliteExportStage(options)


def validate_output(options: dict[str, Any]) -> PostprocessStage:
    from artifacts.stages import OutputValidationStage

    return OutputValidationStage(options)


def legacy_sqlite(options: dict[str, Any]) -> PostprocessStage:
    from artifacts.stages import LegacySqliteExportStage

    return LegacySqliteExportStage(options)


def integrated_result_sqlite(options: dict[str, Any]) -> PostprocessStage:
    from artifacts.stages import IntegratedResultSqliteStage

    return IntegratedResultSqliteStage(options)


def face_privacy_masks(options: dict[str, Any]) -> PostprocessStage:
    from face_privacy.stages import FacePrivacyMaskStage

    return FacePrivacyMaskStage(options)


def face_privacy_merge(options: dict[str, Any]) -> PostprocessStage:
    from face_privacy.stages import FacePrivacyMergeStage

    return FacePrivacyMergeStage(options)


def classwise_postprocess(options: dict[str, Any]) -> PostprocessStage:
    from classwise.stages import ClasswisePostprocessStage

    return ClasswisePostprocessStage(options)


BUILTIN_STAGE_FACTORIES: Mapping[str, StageFactory] = {
    "preprocessing.normalize": normalization,
    "preprocessing.raw_sqlite": raw_sqlite_normalization,
    "preprocessing.score_policy": score_policy,
    "nms.production_v3": production_mask_nms,
    "cut_detection.video": cut_detection,
    "tracking.greedy": tracking,
    "production.polygon_v3_cpu": polygon_production_v3_cpu,
    "production.curve_v1_cpu": curve_production_v1_cpu,
    "evaluation.mask_iou": mask_evaluation,
    "artifacts.union_sqlite": union_sqlite,
    "artifacts.validate": validate_output,
    "artifacts.legacy_sqlite": legacy_sqlite,
    "artifacts.integrated_sqlite": integrated_result_sqlite,
    "face_privacy.masks": face_privacy_masks,
    "face_privacy.merge": face_privacy_merge,
    "classwise.production": classwise_postprocess,
}


def register_builtin_stages() -> None:
    """Install the immutable Production stage catalog into the registry."""

    for name, factory in BUILTIN_STAGE_FACTORIES.items():
        register_stage(name, factory)


__all__ = ("BUILTIN_STAGE_FACTORIES", "register_builtin_stages")
