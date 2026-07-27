#!/usr/bin/env python3
"""Create the minimal state dictionary used by the TensorRT runtime shell."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch


RETAINED_PREFIXES = (
    "neck.",
    "query_head.",
    "mask_head.",
)
REPLACED_PREFIXES = (
    "query_head.transformer.encoder.",
    "query_head.transformer.decoder.",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    return value


def build_runtime_checkpoint(source: Path, output: Path) -> dict[str, object]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"source checkpoint not found: {source}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("state_dict"), dict
    ):
        raise ValueError("source checkpoint must contain a state_dict")
    source_state = payload["state_dict"]
    retained = {
        key: value
        for key, value in source_state.items()
        if key.startswith(RETAINED_PREFIXES)
        and not key.startswith(REPLACED_PREFIXES)
    }
    observed_roots = {key.split(".", 1)[0] for key in retained}
    expected_roots = {"neck", "query_head", "mask_head"}
    if observed_roots != expected_roots:
        raise RuntimeError(
            "TensorRT runtime checkpoint root drift: "
            f"expected={sorted(expected_roots)}, "
            f"observed={sorted(observed_roots)}"
        )
    metadata = dict(payload.get("meta") or {})
    metadata["trt_runtime_checkpoint"] = {
        "schema": "codino-trt-runtime-checkpoint-v1",
        "source_path": str(source),
        "source_size": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "source_state_keys": len(source_state),
        "retained_state_keys": len(retained),
        "retained_prefixes": list(RETAINED_PREFIXES),
        "replaced_prefixes": list(REPLACED_PREFIXES),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save({"meta": metadata, "state_dict": retained}, temporary)
    os.replace(temporary, output)
    print(
        f"source={source} source_keys={len(source_state)} "
        f"retained_keys={len(retained)} output={output} "
        f"bytes={output.stat().st_size}",
        flush=True,
    )
    return metadata["trt_runtime_checkpoint"]


def main() -> int:
    args = parser().parse_args()
    build_runtime_checkpoint(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
