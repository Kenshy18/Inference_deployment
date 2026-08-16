"""CLI for projecting Production results into the legacy three-table schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .legacy_sqlite import export_legacy_sqlite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Project current predictions.sqlite into the former "
            "Dinov3_postprocess masks/tracks/cuts schema."
        )
    )
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--output-sqlite", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = export_legacy_sqlite(
        args.input_sqlite,
        args.output_sqlite,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
