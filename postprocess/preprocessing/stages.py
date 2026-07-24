"""Pipeline stages owned by preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.stages import StageContext, StageResult

from .normalization import normalize_detection_jsonl
from .raw_sqlite import normalize_raw_detection_sqlite
from .score_policy import ScorePolicy, apply_score_policy_jsonl


@dataclass(frozen=True)
class NormalizationStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "input_normalization"
    requires: frozenset[str] = frozenset({"input_jsonl"})
    provides: frozenset[str] = frozenset({"normalized_jsonl"})

    def run(self, context: StageContext) -> StageResult:
        output = context.stage_dir / "normalized.jsonl"
        stats = normalize_detection_jsonl(context.artifacts["input_jsonl"], output)
        return StageResult({"normalized_jsonl": output}, stats)


@dataclass(frozen=True)
class RawSqliteNormalizationStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "raw_sqlite_normalization"
    requires: frozenset[str] = frozenset({"input_raw_sqlite"})
    provides: frozenset[str] = frozenset({"normalized_jsonl"})

    def run(self, context: StageContext) -> StageResult:
        output = context.stage_dir / "normalized.jsonl"
        stats = normalize_raw_detection_sqlite(
            context.artifacts["input_raw_sqlite"], output
        )
        return StageResult({"normalized_jsonl": output}, stats)


@dataclass(frozen=True)
class ScorePolicyStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "score_policy"
    requires: frozenset[str] = frozenset({"normalized_jsonl"})
    provides: frozenset[str] = frozenset({"scored_jsonl"})

    def run(self, context: StageContext) -> StageResult:
        policy_path = context.artifacts.get("class_policy_json")
        default_min = float(self.options.get("score_min", 0.35))
        policy = (
            ScorePolicy.from_json(policy_path, fallback=default_min)
            if policy_path is not None
            else ScorePolicy(
                default_min=default_min,
                by_label={
                    str(label): float(value)
                    for label, value in dict(
                        self.options.get("score_min_by_label", {})
                    ).items()
                },
            )
        )
        output = context.stage_dir / "scored.jsonl"
        stats = apply_score_policy_jsonl(
            context.artifacts["normalized_jsonl"],
            output,
            policy=policy,
        )
        return StageResult(
            {"scored_jsonl": output},
            {
                **stats,
                "default_min": policy.default_min,
                "by_label": dict(sorted(policy.by_label.items())),
            },
        )
