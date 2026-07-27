"""Runtime loader for the local fused SM120 preprocessing kernel."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


_EXTENSION = None


def load_fused_preprocessor(path: Path):
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"fused preprocessing extension not found: {path}")
    name = "mh0_preprocess_fused_sm120"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fused preprocessing extension: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _EXTENSION = module
    return module


def preprocess_out(
    extension,
    source: torch.Tensor,
    output: torch.Tensor,
    *,
    resized_height: int,
    resized_width: int,
    pad_top: int,
    pad_left: int,
    stream: torch.cuda.Stream,
) -> None:
    extension.forward_out(
        source,
        output,
        int(resized_height),
        int(resized_width),
        int(pad_top),
        int(pad_left),
        int(stream.cuda_stream),
    )


__all__ = ["load_fused_preprocessor", "preprocess_out"]
