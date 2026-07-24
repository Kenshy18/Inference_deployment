"""Pipeline stage owned by cut detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contracts.detections import CutList, write_cut_list
from contracts.stages import StageContext, StageResult

from .detector import DisabledCutDetector, create_cut_detector


@dataclass(frozen=True)
class VideoCutDetectionStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "cut_detection"
    requires: frozenset[str] = frozenset({"nms_jsonl"})
    provides: frozenset[str] = frozenset({"cuts_json"})

    def run(self, context: StageContext) -> StageResult:
        enabled = bool(self.options.get("enabled", True))
        if enabled:
            video_path = context.artifacts.get("input_video")
            if video_path is None or not Path(video_path).is_file():
                raise FileNotFoundError(
                    "cut_detection requires input_video; disable the stage "
                    "with options.enabled=false when no video is available"
                )
            detector = create_cut_detector(
                str(self.options.get("method", "high_precision"))
            )
            result = detector.detect(context.artifacts["nms_jsonl"], Path(video_path))
        else:
            detector = DisabledCutDetector()
            result = detector.detect(context.artifacts["nms_jsonl"], Path("."))
        output = context.stage_dir / "cuts.json"
        write_cut_list(
            output,
            CutList(
                tuple(result.frames),
                result.method,
                result.elapsed_seconds,
            ),
        )
        return StageResult(
            {"cuts_json": output},
            {
                "algorithm": result.method,
                "cuts": len(result.frames),
                "elapsed_seconds": result.elapsed_seconds,
            },
        )
