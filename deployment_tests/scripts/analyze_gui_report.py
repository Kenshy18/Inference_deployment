#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


BUSY_STATUSES = {"starting", "running", "cancelling"}


def analyze_case(case: dict[str, object]) -> dict[str, object]:
    issues: list[str] = []
    warnings: list[str] = []
    samples = list(case.get("jobSamples", []))
    renderer_errors = list(case.get("rendererErrors", []))
    if renderer_errors:
        issues.append(f"renderer errors: {renderer_errors}")
    invisible = [sample for sample in samples if not sample.get("appVisible", False)]
    if invisible:
        issues.append(f"app invisible in {len(invisible)} samples")
    root_scrolled = [sample for sample in samples if int(sample.get("scrollY", 0)) != 0]
    if root_scrolled:
        issues.append(f"document root scrolled in {len(root_scrolled)} samples")

    phase_progress: dict[str, list[float]] = {}
    status_phase_mismatches = 0
    for sample in samples:
        phases = sample.get("phases", {})
        any_running = False
        for name, phase in phases.items():
            if phase.get("state") == "running":
                any_running = True
            progress = phase.get("progress")
            if progress is not None:
                phase_progress.setdefault(name, []).append(float(progress))
        if any_running and sample.get("status") not in BUSY_STATUSES:
            status_phase_mismatches += 1
    regressions = {}
    for name, values in phase_progress.items():
        count = sum(right + 1e-9 < left for left, right in zip(values, values[1:]))
        if count:
            regressions[name] = count
    if regressions and len(case.get("videos", [])) == 1:
        issues.append(f"phase progress regressed: {regressions}")
    elif len(case.get("videos", [])) > 1:
        regressions = {"not_evaluated_across_batch_job_boundary": sum(regressions.values())}
    if status_phase_mismatches:
        issues.append(
            f"busy phase paired with non-busy job status in {status_phase_mismatches} samples"
        )

    heartbeat = case.get("pageDiagnostics", {}).get("heartbeat") or {}
    if heartbeat.get("p95") is not None and float(heartbeat["p95"]) > 250:
        warnings.append(f"heartbeat p95 {heartbeat['p95']:.1f}ms > 250ms")
    if heartbeat.get("p99") is not None and float(heartbeat["p99"]) > 1000:
        issues.append(f"heartbeat p99 {heartbeat['p99']:.1f}ms > 1000ms")
    preview = case.get("pageDiagnostics", {}).get("preview") or {}
    if case.get("live") and int(preview.get("count", 0)) == 0:
        issues.append("LIVE enabled but no preview frames were observed")
    expected_status = "passed"
    if case.get("status") != expected_status:
        issues.append(f"case status is {case.get('status')!r}")
    return {
        "id": case.get("id"),
        "samples": len(samples),
        "status_phase_mismatches": status_phase_mismatches,
        "phase_progress_regressions": regressions,
        "heartbeat": heartbeat,
        "preview": preview,
        "issues": issues,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.report.read_text(encoding="utf-8-sig"))
    cases = [analyze_case(case) for case in source.get("cases", [])]
    report = {
        "schema_version": 1,
        "source_report": str(args.report),
        "cases": cases,
        "issue_count": sum(len(case["issues"]) for case in cases),
        "warning_count": sum(len(case["warnings"]) for case in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"issues": report["issue_count"], "warnings": report["warning_count"]}))
    return 1 if report["issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
