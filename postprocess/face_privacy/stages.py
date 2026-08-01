"""Pipeline stages owned by face privacy postprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.stages import StageContext, StageResult

from .sqlite import export_face_masks, merge_face_masks
from .tracking import FaceTrackingConfig


@dataclass(frozen=True)
class FacePrivacyMaskStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "face_privacy_masks"
    requires: frozenset[str] = frozenset({"input_raw_sqlite"})
    provides: frozenset[str] = frozenset({"face_masks_sqlite"})

    def run(self, context: StageContext) -> StageResult:
        target = str(self.options.get("target", "eyes"))
        eye_shape = str(self.options.get("eye_shape", "ellipse"))
        minimum_eye_confidence = float(self.options.get("minimum_eye_confidence", 0.35))
        output = context.stage_dir / "face_masks.sqlite"
        summary = export_face_masks(
            context.artifacts["input_raw_sqlite"],
            output,
            target=target,
            eye_shape=eye_shape,
            minimum_eye_confidence=minimum_eye_confidence,
            face_detection_score_threshold=float(
                self.options.get("face_detection_score_threshold", 0.55)
            ),
            head_detection_score_threshold=float(
                self.options.get("head_detection_score_threshold", 0.55)
            ),
            tracking_config=FaceTrackingConfig(
                max_gap_frames=int(self.options.get("tracking_max_gap_frames", 5)),
                high_score_threshold=float(
                    self.options.get("tracking_high_score_threshold", 0.50)
                ),
                low_score_threshold=float(
                    self.options.get("tracking_low_score_threshold", 0.05)
                ),
                short_track_max_hits=int(self.options.get("short_track_max_hits", 2)),
                short_track_keep_score=float(
                    self.options.get("short_track_keep_score", 0.90)
                ),
            ),
            interpolation_max_gap=int(self.options.get("interpolation_max_gap", 3)),
            cuts_json=context.artifacts.get("cuts_json"),
        )
        return StageResult({"face_masks_sqlite": output}, summary)


@dataclass(frozen=True)
class FacePrivacyMergeStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "face_privacy_merge"
    requires: frozenset[str] = frozenset({"predictions_sqlite", "face_masks_sqlite"})
    provides: frozenset[str] = frozenset({"combined_predictions_sqlite"})

    def run(self, context: StageContext) -> StageResult:
        output = context.stage_dir / "predictions.with_faces.sqlite"
        summary = merge_face_masks(
            context.artifacts["predictions_sqlite"],
            context.artifacts["face_masks_sqlite"],
            output,
        )
        return StageResult({"combined_predictions_sqlite": output}, summary)
