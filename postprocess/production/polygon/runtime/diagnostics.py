"""Shared classification of optimizer and final exact-quality diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


StreamKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class StreamDiagnostics:
    final_exact_infeasible: frozenset[StreamKey]
    optimizer_fallback: frozenset[StreamKey]
    legacy_budget_diagnostic: frozenset[StreamKey]


def classify_streams(
    metric_rows: Sequence[Mapping[str, str]],
    stream_rows: Sequence[Mapping[str, str]],
    *,
    recall_floor: float,
    epsilon: float = 1e-10,
) -> StreamDiagnostics:
    """Separate final Recall failures from internal fallback diagnostics.

    A non-finite legacy scalar objective or a positive legacy average-budget
    diagnostic does not mean the published dense output violates the hard
    per-frame Recall floor. Only exact final rows define infeasibility.
    """

    def key(row: Mapping[str, str]) -> StreamKey:
        return str(row["track_id"]), int(row["run_id"])

    final_exact_infeasible = frozenset(
        key(row)
        for row in metric_rows
        if float(row["recall"]) + 1e-12 < float(recall_floor)
    )
    optimizer_fallback = frozenset(
        key(row) for row in stream_rows if not math.isfinite(float(row["objective"]))
    )
    legacy_budget_diagnostic = frozenset(
        key(row)
        for row in stream_rows
        if float(row["recall_budget_violation"]) > float(epsilon)
    )
    return StreamDiagnostics(
        final_exact_infeasible=final_exact_infeasible,
        optimizer_fallback=optimizer_fallback,
        legacy_budget_diagnostic=legacy_budget_diagnostic,
    )
