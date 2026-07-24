#!/usr/bin/env python3
"""Build a DINOv3 backbone TensorRT engine from an ONNX model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

try:
    from .engine_build import build_engine_from_onnx
except ImportError:
    from engine_build import build_engine_from_onnx


def shape(value: str) -> tuple[int, int, int, int]:
    values = tuple(
        int(item) for item in value.lower().replace(",", "x").split("x") if item
    )
    if len(values) != 4:
        raise argparse.ArgumentTypeError("shape must be BxCxHxW")
    return values  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--min-shape", type=shape)
    parser.add_argument("--opt-shape", type=shape)
    parser.add_argument("--max-shape", type=shape)
    parser.add_argument("--force-layer-precision", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    engine = build_engine_from_onnx(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        min_shape=args.min_shape,
        opt_shape=args.opt_shape,
        max_shape=args.max_shape,
        workspace_bytes=int(args.workspace_gb * (1 << 30)),
        force_layer_precision=args.force_layer_precision,
    )
    payload = {
        "schema": "dinov3-backbone-engine-v1",
        "onnx": str(args.onnx.expanduser().resolve()),
        "engine": str(engine),
        "precision": args.precision,
        "min_shape": args.min_shape,
        "opt_shape": args.opt_shape,
        "max_shape": args.max_shape,
        "elapsed_sec": time.perf_counter() - started,
    }
    sidecar = engine.with_suffix(engine.suffix + ".json")
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
