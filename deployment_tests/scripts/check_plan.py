#!/usr/bin/env python3
"""Validate the deployment GUI test plan without running product stages."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "cases.json"
PRIORITIES = {"P0", "P1", "P2"}


def load_plan() -> dict[str, Any]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def main() -> int:
    plan = load_plan()
    cases = plan.get("cases", [])
    issues: list[str] = []
    ids = [str(case.get("id", "")) for case in cases]
    duplicates = sorted(name for name, count in Counter(ids).items() if count > 1)
    if any(not name for name in ids):
        issues.append("case id must not be empty")
    if duplicates:
        issues.append(f"duplicate case ids: {duplicates}")

    case_minutes = 0
    covered: set[str] = set()
    priority_counts: Counter[str] = Counter()
    v3_stress_cases: list[str] = []
    for case in cases:
        case_id = str(case.get("id", "<unknown>"))
        priority = str(case.get("priority", ""))
        if priority not in PRIORITIES:
            issues.append(f"{case_id}: invalid priority {priority!r}")
        priority_counts[priority] += 1
        budget = case.get("budget_minutes")
        if not isinstance(budget, int) or budget <= 0:
            issues.append(f"{case_id}: budget_minutes must be a positive integer")
        else:
            case_minutes += budget
        tags = case.get("tags", [])
        if not isinstance(tags, list) or not tags:
            issues.append(f"{case_id}: at least one coverage tag is required")
        else:
            covered.update(str(tag) for tag in tags)

        model = case.get("segmentation_model")
        if model == "dinov3_codino":
            duration = float(case.get("source_duration_minutes", 0))
            max_frames = int(case.get("max_frames", 0))
            is_stress = any(str(tag).startswith("stress:") for tag in tags)
            if is_stress:
                v3_stress_cases.append(case_id)
                if case_id != "S00_v3_120m_single_load":
                    issues.append(f"{case_id}: only S00 may use V3 for stress/soak")
                if duration != 120 or max_frames != 216_000:
                    issues.append(
                        f"{case_id}: V3 stress must be exactly 120m/216000 frames"
                    )
                if case.get("face_model") is not None:
                    issues.append(f"{case_id}: V3 stress must not include face inference")
            else:
                if duration > 6:
                    issues.append(
                        f"{case_id}: non-stress V3 duration {duration}m exceeds 6m"
                    )
                if max_frames > 10_800:
                    issues.append(
                        f"{case_id}: non-stress V3 max_frames exceeds 10800"
                    )

        if any(str(tag).startswith("stress:") for tag in tags):
            if model not in {None, "dinov3_codino_mh0", "dinov3_codino"}:
                issues.append(
                    f"{case_id}: unsupported stress segmentation model {model}"
                )

    if v3_stress_cases != ["S00_v3_120m_single_load"]:
        issues.append(
            "exactly one V3 stress case is required: S00_v3_120m_single_load; "
            f"got {v3_stress_cases}"
        )

    overhead = plan.get("fixed_overhead_minutes", {})
    if not isinstance(overhead, dict):
        issues.append("fixed_overhead_minutes must be an object")
        overhead_minutes = 0
    else:
        invalid_overhead = {
            key: value
            for key, value in overhead.items()
            if not isinstance(value, int) or value < 0
        }
        if invalid_overhead:
            issues.append(f"invalid fixed overhead values: {invalid_overhead}")
        overhead_minutes = sum(
            value for value in overhead.values() if isinstance(value, int)
        )

    total_budget = int(plan.get("total_budget_minutes", 0))
    planned_minutes = case_minutes + overhead_minutes
    if total_budget != 480:
        issues.append(f"total_budget_minutes must be 480, got {total_budget}")
    if planned_minutes > total_budget:
        issues.append(
            f"planned {planned_minutes} minutes exceeds budget {total_budget}"
        )
    reserve = int(overhead.get("reserve", 0)) if isinstance(overhead, dict) else 0
    if total_budget - planned_minutes != 0:
        issues.append(
            "case budgets plus fixed overhead (including reserve) must exactly "
            f"fill 480 minutes; got {planned_minutes}"
        )
    if reserve < 30:
        issues.append(f"reserve must be at least 30 minutes, got {reserve}")

    required = {str(tag) for tag in plan.get("required_coverage", [])}
    missing = sorted(required - covered)
    if missing:
        issues.append(f"missing required coverage: {missing}")
    if priority_counts["P0"] == 0:
        issues.append("at least one P0 case is required")

    summary = {
        "plan": str(PLAN),
        "cases": len(cases),
        "priority_counts": dict(sorted(priority_counts.items())),
        "case_minutes": case_minutes,
        "fixed_overhead_minutes": overhead_minutes,
        "planned_minutes": planned_minutes,
        "total_budget_minutes": total_budget,
        "reserve_minutes": reserve,
        "coverage_tags": len(covered),
        "required_coverage_tags": len(required),
        "issues": issues,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
