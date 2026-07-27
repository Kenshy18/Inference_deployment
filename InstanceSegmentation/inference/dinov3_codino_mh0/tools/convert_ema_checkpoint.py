#!/usr/bin/env python3
"""Convert an MMCV ExpMomentumEMAHook checkpoint into a deploy checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


OMITTED_NONPERSISTENT = {"query_head.positional_encoding._dim_t"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_key(key: str) -> str:
    value = key
    while "._orig_mod." in value:
        value = value.replace("._orig_mod.", ".")
    if value.startswith("_orig_mod."):
        value = value[len("_orig_mod.") :]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {source}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("state_dict"), dict
    ):
        raise ValueError("checkpoint must contain a state_dict mapping")
    source_state = payload["state_dict"]
    ema_buffer_count = sum(str(key).startswith("ema_") for key in source_state)
    source_meta = payload.get("meta", {})
    already_deploy = (
        isinstance(source_meta, dict)
        and source_meta.get("weight_selection") == "EMA"
    )
    if not ema_buffer_count and not already_deploy:
        raise ValueError(
            "checkpoint has no ExpMomentumEMAHook buffers; "
            "EMA weight selection cannot be established"
        )

    state = {}
    normalized = 0
    omitted = []
    for raw_key, tensor in source_state.items():
        key = str(raw_key)
        if key.startswith("ema_"):
            continue
        clean = normalize_key(key)
        normalized += int(clean != key)
        if clean in OMITTED_NONPERSISTENT:
            omitted.append(clean)
            continue
        if clean in state:
            raise ValueError(f"normalized key collision: {clean}")
        state[clean] = tensor
    if not state:
        raise ValueError("no deploy tensors selected")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "meta": {
                "source_checkpoint": str(source),
                "source_sha256": sha256(source),
                "source_meta": source_meta,
                "weight_selection": "EMA",
                "ema_buffers_omitted": ema_buffer_count,
                "normalized_compile_keys": normalized,
                "omitted_nonpersistent_keys": sorted(omitted),
            },
            "state_dict": state,
        },
        output,
    )
    print(
        f"[PASS] wrote {output} tensors={len(state)} "
        f"normalized={normalized} omitted_ema_buffers={ema_buffer_count} "
        f"sha256={sha256(output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
