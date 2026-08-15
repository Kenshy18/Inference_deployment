"""Audit helpers for the minimum-Recall multistate DP result."""

from __future__ import annotations

import csv
from pathlib import Path

from ..config import CANDIDATE, CandidateConfig


def audit_exact_recall(
    metrics_csv: Path,
    config: CandidateConfig = CANDIDATE,
) -> dict[str, int | float]:
    """Read the exact CPU audit; aggregate summaries are not a valid gate."""
    config.validate()
    rows = 0
    violations = 0
    minimum = 1.0
    with Path(metrics_csv).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            recall = float(row["recall"])
            rows += 1
            minimum = min(minimum, recall)
            violations += int(recall + 1e-12 < float(config.temporal.recall_floor))
    if not rows:
        raise RuntimeError(f"exact Recall audit is empty: {metrics_csv}")
    return {
        "evaluated_rows": rows,
        "minimum_recall": minimum,
        "recall_violations": violations,
    }
