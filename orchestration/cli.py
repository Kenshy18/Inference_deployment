"""CLI for the repository-level workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import OrchestrationConfig, OrchestrationConfigError
from .runner import OrchestrationError, OrchestrationRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mask-workflow",
        description=(
            "Run instance segmentation, postprocess, and overlay through their "
            "public CLI/SQLite contracts."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="resume completed stages only when the resolved config is unchanged",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configuration and reused inputs, then print the execution plan",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = OrchestrationConfig.load(args.config)
        result = OrchestrationRunner(
            config,
            resume=args.resume,
            dry_run=bool(args.dry_run),
        ).run()
    except (
        FileNotFoundError,
        FileExistsError,
        OrchestrationConfigError,
        OrchestrationError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = ["build_parser", "main"]

