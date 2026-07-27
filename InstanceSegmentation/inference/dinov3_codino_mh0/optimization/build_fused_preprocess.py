#!/usr/bin/env python3
"""Build the fixed-736x1280 SM120 fused MH0 preprocessing extension."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--build-dir", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    environment_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get(
        "PATH", ""
    )
    os.environ.setdefault(
        "CUDA_HOME",
        str(Path(sys.executable).resolve().parents[1]),
    )
    from torch.utils.cpp_extension import load

    build_dir = args.build_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    native = MODULE_ROOT / "native"
    started = time.perf_counter()
    extension = load(
        name="mh0_preprocess_fused_sm120",
        sources=[
            str(native / "preprocess_fused.cpp"),
            str(native / "preprocess_fused.cu"),
        ],
        build_directory=str(build_dir),
        verbose=True,
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_120,code=sm_120",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(extension.__file__).resolve(), output)
    print(
        f"built={output} bytes={output.stat().st_size} "
        f"elapsed_seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
