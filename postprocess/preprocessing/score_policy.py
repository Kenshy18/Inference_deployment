"""Class-aware confidence threshold policy."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contracts.detections import transform_detection_jsonl

SCORE_KEYS = (
    "raw_det_score_min",
    "confidence_min",
    "score_min",
    "min_score",
    "confidence",
)


def normalize_label(value: object) -> str:
    text = str(value).strip()
    return text if text else "unknown"


def _policy_score(policy: dict[str, object]) -> float | None:
    for key in SCORE_KEYS:
        try:
            value = float(policy[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


@dataclass(frozen=True)
class ScorePolicy:
    """Resolve the minimum detector score for each canonical detection."""

    default_min: float = 0.0
    by_label: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Path, *, fallback: float = 0.0) -> "ScorePolicy":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("class policy must be a JSON object")

        default_min = float(fallback)
        default = raw.get("default")
        if isinstance(default, dict):
            value = _policy_score(default)
            if value is not None:
                default_min = value

        classes = raw.get("classes")
        source = classes if isinstance(classes, dict) else raw
        by_label: dict[str, float] = {}
        for label, config in source.items():
            if label == "default" or not isinstance(config, dict):
                continue
            value = _policy_score(config)
            if value is not None:
                by_label[normalize_label(label)] = value
        return cls(default_min=default_min, by_label=by_label)

    def minimum_for(self, detection: dict[str, Any]) -> float:
        for key in ("class_name", "label"):
            label = normalize_label(detection.get(key, "unknown"))
            if label in self.by_label:
                return self.by_label[label]
        return self.default_min

    def accepts(self, detection: dict[str, Any]) -> bool:
        try:
            score = float(detection.get("score") or 0.0)
        except (TypeError, ValueError):
            return False
        return math.isfinite(score) and score >= self.minimum_for(detection)

    def apply(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [detection for detection in detections if self.accepts(detection)]


def apply_score_policy_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    policy: ScorePolicy,
) -> dict[str, int]:
    """Apply only the confidence policy to canonical detections."""

    def transform(record: dict[str, Any]) -> dict[str, Any]:
        output = dict(record)
        output["detections"] = policy.apply(list(record["detections"]))
        return output

    return transform_detection_jsonl(input_path, output_path, transform)
