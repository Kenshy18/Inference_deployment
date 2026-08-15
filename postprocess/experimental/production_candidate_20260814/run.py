#!/usr/bin/env python3
"""CLI for the complete pre-Production candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    INTERVAL_EVALUATION_MODES,
    with_interval_evaluation,
    with_target_interval,
)
from .pipeline import run_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the approved hole/island/Mask-NMS + polygon14 + minimum-"
            "Recall DP + constrained pair-vote candidate."
        )
    )
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--cuts-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument(
        "--scored-jsonl",
        type=Path,
        help="optional canonical score-filtered input used for exact parity runs",
    )
    parser.add_argument("--score-min", type=float, default=0.30)
    parser.add_argument(
        "--target-interval",
        type=int,
        default=6,
        help="soft keyframe interval target (default: 6)",
    )
    parser.add_argument(
        "--interval-evaluation",
        choices=INTERVAL_EVALUATION_MODES,
        default="cuda_lazy_exact",
        help=(
            "cuda_lazy_exact is the default accelerated evaluator; "
            "native_exact evaluates every DP edge exactly on CPU"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.score_min <= 1.0:
        raise ValueError("score-min must be in [0, 1]")
    config = with_interval_evaluation(
        args.interval_evaluation,
        with_target_interval(args.target_interval),
    )
    result = run_candidate(
        raw_input_sqlite=args.input_sqlite,
        cuts_json=args.cuts_json,
        output_root=args.output_root,
        input_video=args.input_video,
        scored_jsonl=args.scored_jsonl,
        score_min=args.score_min,
        config=config,
    )
    print(
        json.dumps(
            {
                "manifest": result["manifest"],
                "result_sqlite": result["artifacts"]["result_sqlite"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
