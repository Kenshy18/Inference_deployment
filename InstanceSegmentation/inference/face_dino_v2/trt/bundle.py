"""Validate the fixed-B8 Face DINO TensorRT artifact bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "face-dino-v2-trt-bundle-v1"
PROFILE = "fast-sm120-fixed-b8-v1"
BATCH_SIZE = 8
INPUT_SHAPE = (8, 3, 736, 1280)
ENGINE_FILES = {
    "backbone_neck": "engines/backbone_neck.engine",
    "query_encoder": "engines/query_encoder.engine",
    "decoder": "engines/decoder.engine",
    "attribute": "engines/attribute.engine",
}
PLUGIN_FILES = {
    "msda_plugin": "plugins/codino_msda_direct_mh0_sm120.so",
    "preprocess_plugin": "plugins/face_preprocess_fused_sm120.so",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FaceDinoEngineBundle:
    manifest_path: Path
    engines: dict[str, Path]
    plugins: dict[str, Path]
    batch_size: int
    input_shape: tuple[int, int, int, int]
    checkpoint_sha256: str


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
) -> FaceDinoEngineBundle:
    if verify not in {"metadata", "engines"}:
        raise ValueError("verify must be 'metadata' or 'engines'")
    manifest = manifest_path.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"bundle manifest not found: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("profile") != PROFILE:
        raise ValueError("unsupported Face DINO TensorRT bundle")
    if payload.get("status") != "complete":
        raise ValueError("incomplete Face DINO TensorRT bundle")
    if payload.get("batch_size") != BATCH_SIZE:
        raise ValueError("Face DINO TensorRT bundle must use fixed B8")
    if tuple(payload.get("input_shape", ())) != INPUT_SHAPE:
        raise ValueError("Face DINO TensorRT input-shape mismatch")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict) or not checkpoint.get("sha256"):
        raise ValueError("checkpoint provenance is missing")
    records = payload.get("artifacts")
    if not isinstance(records, dict):
        raise ValueError("bundle artifact records are missing")
    engines: dict[str, Path] = {}
    for name, relative in ENGINE_FILES.items():
        path = manifest.parent / relative
        _verify(path, records.get(name), hashes=verify == "engines")
        engines[name] = path.resolve()
    plugins: dict[str, Path] = {}
    for name, relative in PLUGIN_FILES.items():
        path = manifest.parent / relative
        _verify(path, records.get(name), hashes=verify == "engines")
        plugins[name] = path.resolve()
    return FaceDinoEngineBundle(
        manifest_path=manifest,
        engines=engines,
        plugins=plugins,
        batch_size=BATCH_SIZE,
        input_shape=INPUT_SHAPE,
        checkpoint_sha256=str(checkpoint["sha256"]),
    )


__all__ = [
    "BATCH_SIZE",
    "ENGINE_FILES",
    "INPUT_SHAPE",
    "PLUGIN_FILES",
    "PROFILE",
    "SCHEMA",
    "FaceDinoEngineBundle",
    "load_engine_bundle",
    "sha256_file",
]
