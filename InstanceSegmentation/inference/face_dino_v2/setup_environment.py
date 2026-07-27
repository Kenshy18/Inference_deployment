#!/usr/bin/env python3
"""Validate the Face DINO v2 runtime source, checkpoint, and engine bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


FAMILY_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = FAMILY_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from face_dino_v2.model import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_TRT_BUNDLE,
    configure_source_root,
)
from face_dino_v2.trt.bundle import load_engine_bundle, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--trt-bundle", type=Path, default=DEFAULT_TRT_BUNDLE)
    parser.add_argument(
        "--verify",
        choices=("metadata", "engines"),
        default="engines",
    )
    args = parser.parse_args()
    runtime_python = args.runtime_python.expanduser().resolve()
    if not runtime_python.is_file():
        raise FileNotFoundError(runtime_python)
    source_root = configure_source_root(args.source_root)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    bundle = load_engine_bundle(args.trt_bundle, verify=args.verify)
    if sha256_file(checkpoint) != bundle.checkpoint_sha256:
        raise ValueError("checkpoint hash does not match the engine bundle")
    probe = subprocess.run(
        [
            str(runtime_python),
            "-c",
            (
                "import cv2, mmcv, mmdet, tensorrt, torch, torchvision;"
                "print(torch.__version__, tensorrt.__version__)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "runtime_python": str(runtime_python),
                "versions": probe.stdout.strip(),
                "source_root": str(source_root),
                "checkpoint": str(checkpoint),
                "bundle": str(bundle.manifest_path),
                "batch_size": bundle.batch_size,
                "input_shape": list(bundle.input_shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
