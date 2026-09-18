#!/usr/bin/env python3
"""Summarize the controlled runtime-driver factorial benchmark.

This experiment reports two complementary decompositions:

* a Shapley allocation of the measured fast-to-slow wall-time gap; and
* a balanced 2^3 factorial ANOVA, including factor interactions.

The reported percentages are conditional on this benchmark's deliberately
chosen ranges.  They are causal within the synthetic intervention, not a
claim about the frequency of those conditions in every production video.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


FACTORS = ("track_count", "area_factor", "vertices")
FAST = {"track_count": 6, "area_factor": 0.5, "vertices": 14}
SLOW = {"track_count": 1, "area_factor": 1.5, "vertices": 20}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--replicate", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[name] for name in FACTORS)


def _cell_key_from_state(state: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(state[name] for name in FACTORS)


def _percent_change(start: float, end: float) -> float:
    return 100.0 * (end / start - 1.0)


def _shapley(cell_times: dict[tuple[Any, ...], float]) -> dict[str, Any]:
    def value(changed: frozenset[str]) -> float:
        state = {
            factor: (SLOW[factor] if factor in changed else FAST[factor])
            for factor in FACTORS
        }
        return cell_times[_cell_key_from_state(state)]

    allocations = {factor: 0.0 for factor in FACTORS}
    permutations = list(itertools.permutations(FACTORS))
    for order in permutations:
        selected: frozenset[str] = frozenset()
        before = value(selected)
        for factor in order:
            selected = selected | {factor}
            after = value(selected)
            allocations[factor] += after - before
            before = after
    allocations = {
        factor: amount / len(permutations) for factor, amount in allocations.items()
    }
    fast_seconds = value(frozenset())
    slow_seconds = value(frozenset(FACTORS))
    gap = slow_seconds - fast_seconds
    return {
        "fast_seconds": fast_seconds,
        "slow_seconds": slow_seconds,
        "wall_time_gap_seconds": gap,
        "wall_time_increase_percent": _percent_change(fast_seconds, slow_seconds),
        "contribution_seconds": allocations,
        "contribution_percent": {
            factor: 100.0 * amount / gap for factor, amount in allocations.items()
        },
    }


def _factorial_anova(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Balanced +/-1 contrast coding.  With all eight cells represented equally,
    # the columns are orthogonal and each coefficient is a direct dot product.
    contrasts = {
        "track_count": lambda row: 1.0 if int(row["track_count"]) == 1 else -1.0,
        "area_factor": lambda row: 1.0 if float(row["area_factor"]) == 1.5 else -1.0,
        "vertices": lambda row: 1.0 if int(row["vertices"]) == 20 else -1.0,
    }
    terms = {
        "track_count": ("track_count",),
        "area_factor": ("area_factor",),
        "vertices": ("vertices",),
        "track_x_area": ("track_count", "area_factor"),
        "track_x_vertices": ("track_count", "vertices"),
        "area_x_vertices": ("area_factor", "vertices"),
        "track_x_area_x_vertices": FACTORS,
    }

    def analyze(transform: Any) -> dict[str, Any]:
        y = [float(transform(float(row["wall_seconds"]))) for row in rows]
        mean = statistics.fmean(y)
        columns: dict[str, list[float]] = {}
        for term, factors in terms.items():
            columns[term] = [
                math.prod(contrasts[factor](row) for factor in factors)
                for row in rows
            ]
        coefficients = {
            term: sum(value * yy for value, yy in zip(column, y))
            / sum(value * value for value in column)
            for term, column in columns.items()
        }
        fitted = [
            mean
            + sum(coefficients[term] * columns[term][index] for term in terms)
            for index in range(len(rows))
        ]
        ss = {
            term: sum(
                (coefficients[term] * value) ** 2 for value in columns[term]
            )
            for term in terms
        }
        residual = sum((yy - fit) ** 2 for yy, fit in zip(y, fitted))
        total = sum((yy - mean) ** 2 for yy in y)
        explained = sum(ss.values())
        return {
            "response": "log_wall_seconds" if transform is math.log else "wall_seconds",
            "total_sum_squares": total,
            "explained_sum_squares": explained,
            "residual_sum_squares": residual,
            "r_squared": 1.0 - residual / total if total else 1.0,
            "term_percent_of_total": {
                term: 100.0 * value / total for term, value in ss.items()
            },
            "residual_percent_of_total": 100.0 * residual / total if total else 0.0,
        }

    return {"wall_seconds": analyze(float), "log_wall_seconds": analyze(math.log)}


def _group_median(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    output = []
    for level, group in sorted(grouped.items()):
        wall = statistics.median(float(row["wall_seconds"]) for row in group)
        observations = int(group[0]["observations"])
        output.append(
            {
                key: level,
                "wall_seconds_median": wall,
                "throughput_fps_median": observations / wall,
                "samples": len(group),
            }
        )
    return output


def _response_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    series_to_factor = {
        "track_response": "track_count",
        "area_response": "area_factor",
        "vertex_response": "vertices",
    }
    result: dict[str, Any] = {}
    for series, factor in series_to_factor.items():
        curve = _group_median((row for row in rows if row["series"] == series), factor)
        result[series] = curve
        if curve:
            first, last = curve[0], curve[-1]
            result[f"{series}_endpoints"] = {
                "from": first[factor],
                "to": last[factor],
                "wall_time_change_percent": _percent_change(
                    first["wall_seconds_median"], last["wall_seconds_median"]
                ),
                "throughput_change_percent": _percent_change(
                    first["throughput_fps_median"], last["throughput_fps_median"]
                ),
                "throughput_ratio": (
                    last["throughput_fps_median"] / first["throughput_fps_median"]
                ),
            }
    return result


def main() -> None:
    args = _parser().parse_args()
    main_payload = _load(args.main)
    sources = [args.main, *args.replicate]
    all_rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        payload = _load(source)
        for row in payload["results"]:
            copied = dict(row)
            copied["source_file"] = str(source.resolve())
            copied["independent_run"] = source_index + 1
            all_rows.append(copied)

    output: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "benchmark_definition": {
            "observations": main_payload["observations"],
            "target_interval": main_payload["target_interval"],
            "source_track": main_payload["source_track"],
            "frame_dimensions": main_payload["frame_dimensions"],
            "factor_ranges": main_payload["factorial"],
            "shape_transform": main_payload["shape_transform"],
            "fast_corner": FAST,
            "slow_corner": SLOW,
        },
        "source_files": [str(path.resolve()) for path in sources],
        "geometries": {},
    }

    geometries = sorted({row["geometry"] for row in all_rows})
    for geometry in geometries:
        geometry_rows = [row for row in all_rows if row["geometry"] == geometry]
        factorial_rows = [row for row in geometry_rows if row["series"] == "factorial"]
        cells: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for row in factorial_rows:
            cells[_cell_key(row)].append(float(row["wall_seconds"]))
        cell_medians = {key: statistics.median(values) for key, values in cells.items()}
        run_shapley = []
        for source_file in sorted({row["source_file"] for row in factorial_rows}):
            run_rows = [row for row in factorial_rows if row["source_file"] == source_file]
            run_cells = {_cell_key(row): float(row["wall_seconds"]) for row in run_rows}
            if len(run_cells) == 8:
                run_shapley.append(
                    {
                        "source_file": source_file,
                        **_shapley(run_cells),
                    }
                )
        replicate_differences = []
        for key, values in sorted(cells.items()):
            if len(values) >= 2:
                replicate_differences.append(
                    {
                        "cell": dict(zip(FACTORS, key)),
                        "values_seconds": values,
                        "relative_range_percent": 100.0
                        * (max(values) - min(values))
                        / statistics.fmean(values),
                    }
                )
        response_rows = [
            row
            for row in main_payload["results"]
            if row["geometry"] == geometry and row["series"] != "factorial"
        ]
        output["geometries"][geometry] = {
            "factorial_cell_medians": [
                {
                    **dict(zip(FACTORS, key)),
                    "wall_seconds_median": value,
                    "throughput_fps_median": main_payload["observations"] / value,
                    "replicates": len(cells[key]),
                }
                for key, value in sorted(cell_medians.items())
            ],
            "shapley_wall_time_gap": _shapley(cell_medians),
            "shapley_independent_runs": run_shapley,
            "shapley_contribution_range_percent": {
                factor: {
                    "min": min(
                        run["contribution_percent"][factor] for run in run_shapley
                    ),
                    "max": max(
                        run["contribution_percent"][factor] for run in run_shapley
                    ),
                }
                for factor in FACTORS
            },
            "factorial_anova": _factorial_anova(factorial_rows),
            "response_curves": _response_summary(response_rows),
            "replicate_stability": {
                "cell_relative_ranges_percent": replicate_differences,
                "median_relative_range_percent": statistics.median(
                    item["relative_range_percent"] for item in replicate_differences
                ),
                "max_relative_range_percent": max(
                    item["relative_range_percent"] for item in replicate_differences
                ),
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
