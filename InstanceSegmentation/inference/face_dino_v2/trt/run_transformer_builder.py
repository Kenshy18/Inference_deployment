#!/usr/bin/env python3
"""Run a snapshotted Face DINO script with reviewed B8/B16 guards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--script", required=True, type=Path)
    args, forwarded = parser.parse_known_args()
    script = args.script.expanduser().resolve()
    source = script.read_text(encoding="utf-8")
    replacements = {
        (
            '    if args.batch_size != 8:\n'
            '        raise ValueError("The production transformer profile is fixed B8")\n'
        ): (
            "    if args.batch_size not in (8, 16):\n"
            '        raise ValueError("The supported transformer profiles are B8 and B16")\n'
        ),
        (
            "    if all(transformer_paths) and args.batch_size != 8:\n"
            '        raise ValueError("The TensorRT transformer is fixed to --batch-size 8")\n'
        ): (
            "    if all(transformer_paths) and args.batch_size not in (8, 16):\n"
            '        raise ValueError("The TensorRT transformer requires batch 8 or 16")\n'
        ),
    }
    matches = [original for original in replacements if source.count(original) == 1]
    if len(matches) != 1:
        raise RuntimeError(
            "snapshotted Face DINO batch guard changed; "
            "review the B16 compatibility wrapper"
        )
    original = matches[0]
    source = source.replace(original, replacements[original])
    sys.argv = [str(script), *forwarded]
    namespace = {
        "__name__": "__main__",
        "__file__": str(script),
        "__package__": None,
    }
    exec(compile(source, str(script), "exec"), namespace)


if __name__ == "__main__":
    main()
