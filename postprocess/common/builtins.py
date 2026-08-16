"""Register feature-owned stage implementations.

Factories import a feature only when that stage is selected.  No algorithm is
implemented here.
"""

from typing import Any

from contracts.stages import PostprocessStage
from .registry import register_stage


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


register_stage("preprocessing.normalize", normalization)
register_stage("preprocessing.raw_sqlite", raw_sqlite_normalization)
register_stage("preprocessing.score_policy", score_policy)
register_stage("nms.production_v3", production_mask_nms)
register_stage("cut_detection.video", cut_detection)
register_stage("tracking.greedy", tracking)
register_stage("production.polygon_v3_cpu", polygon_production_v3_cpu)
register_stage("evaluation.mask_iou", mask_evaluation)
register_stage("artifacts.union_sqlite", union_sqlite)
register_stage("artifacts.validate", validate_output)
register_stage("artifacts.legacy_sqlite", legacy_sqlite)
register_stage("artifacts.integrated_sqlite", integrated_result_sqlite)
register_stage("face_privacy.masks", face_privacy_masks)
register_stage("face_privacy.merge", face_privacy_merge)
register_stage("classwise.production", classwise_postprocess)
