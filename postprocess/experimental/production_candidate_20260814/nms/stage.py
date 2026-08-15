"""Streaming NMS stage for the consolidated candidate."""

from __future__ import annotations

import gzip
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts.detections import transform_detection_jsonl
from contracts.stages import StageContext, StageResult

from ..config import CANDIDATE, CandidateConfig
from .policy import build_policy


def run_nms_jsonl(
    source: Path,
    output: Path,
    *,
    trace_output: Path | None = None,
    config: CandidateConfig = CANDIDATE,
) -> dict[str, object]:
    """Apply hole/island cleanup and virtual-component Mask NMS per frame."""
    policy = build_policy(config)
    counters: Counter[str] = Counter()
    trace_handle = None
    if trace_output is not None:
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        trace_handle = gzip.open(trace_output, "wt", encoding="utf-8")

    def transform(record: dict[str, Any]) -> dict[str, Any]:
        retained, diagnostics, trace = policy.apply_with_trace(
            list(record["detections"])
        )
        counters.update(diagnostics.as_dict())
        if trace_handle is not None:
            frame = int(record["frame_index"])
            for event in trace:
                trace_handle.write(
                    json.dumps(
                        {"frame_index": frame, **event},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        result = dict(record)
        result["detections"] = retained
        return result

    started = time.perf_counter()
    try:
        stream = transform_detection_jsonl(source, output, transform)
    finally:
        if trace_handle is not None:
            trace_handle.close()
    elapsed = time.perf_counter() - started
    return {
        **stream,
        "implementation": policy.name,
        "candidate_profile": config.profile_id,
        "elapsed_seconds": elapsed,
        "frames_per_second": float(stream["frames"]) / max(elapsed, 1e-9),
        "diagnostics": dict(sorted(counters.items())),
    }


@dataclass(frozen=True)
class CandidateNmsStage:
    """Pipeline-compatible wrapper; intentionally not in the Production registry."""

    name: str = "production_candidate_20260814_nms"
    requires: frozenset[str] = frozenset({"scored_jsonl"})
    provides: frozenset[str] = frozenset({"nms_jsonl", "nms_trace_jsonl"})

    def run(self, context: StageContext) -> StageResult:
        output = context.stage_dir / "nms.jsonl"
        trace = context.stage_dir / "nms_trace.jsonl.gz"
        metadata = run_nms_jsonl(
            context.artifacts["scored_jsonl"], output, trace_output=trace
        )
        return StageResult({"nms_jsonl": output, "nms_trace_jsonl": trace}, metadata)
