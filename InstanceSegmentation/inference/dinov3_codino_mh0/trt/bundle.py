"""Validate the fixed-B16 MH0 TensorRT artifact bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "dinov3-codino-mh0-trt-bundle-v1"
PROFILE = "fast-sm120-fixed-b16-v1"
ENGINE_FILES = {
    "backbone_neck": "engines/backbone_neck.engine",
    "query_encoder": "engines/query_encoder.engine",
    "decoder": "engines/decoder.engine",
    "mask_head": "engines/mask_head.engine",
}
PLUGIN_FILE = "plugins/codino_msda_direct_mh0_sm120.so"
PREPROCESS_PLUGIN_FILE = "plugins/mh0_preprocess_fused_sm120.so"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Mh0EngineBundle:
    manifest_path: Path
    engines: dict[str, Path]
    plugin: Path
    preprocess_plugin: Path
    batch_size: int


def _verify(path: Path, record: object, *, hashes: bool) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"missing artifact record for {path.name}")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"artifact not found: {path}")
    if path.stat().st_size != record.get("size"):
        raise ValueError(f"artifact size mismatch: {path}")
    if hashes and sha256_file(path) != record.get("sha256"):
        raise ValueError(f"artifact SHA-256 mismatch: {path}")


def load_engine_bundle(
    manifest_path: Path,
    *,
    verify: str = "engines",
) -> Mh0EngineBundle:
    if verify not in {"metadata", "engines"}:
        raise ValueError("verify must be 'metadata' or 'engines'")
    manifest = manifest_path.expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("profile") != PROFILE:
        raise ValueError("unsupported MH0 TensorRT bundle")
    if payload.get("status") != "complete" or payload.get("batch_size") != 16:
        raise ValueError("incomplete or non-B16 MH0 TensorRT bundle")
    records = payload.get("artifacts")
    if not isinstance(records, dict):
        raise ValueError("MH0 bundle artifact records are missing")
    engines = {}
    for name, relative in ENGINE_FILES.items():
        path = manifest.parent / relative
        _verify(path, records.get(name), hashes=verify == "engines")
        engines[name] = path.resolve()
    plugin = manifest.parent / PLUGIN_FILE
    _verify(plugin, records.get("plugin"), hashes=verify == "engines")
    preprocess_plugin = manifest.parent / PREPROCESS_PLUGIN_FILE
    _verify(
        preprocess_plugin,
        records.get("preprocess_plugin"),
        hashes=verify == "engines",
    )
    return Mh0EngineBundle(
        manifest,
        engines,
        plugin.resolve(),
        preprocess_plugin.resolve(),
        16,
    )


__all__ = [
    "ENGINE_FILES",
    "PLUGIN_FILE",
    "PREPROCESS_PLUGIN_FILE",
    "PROFILE",
    "SCHEMA",
    "Mh0EngineBundle",
    "load_engine_bundle",
    "sha256_file",
]
