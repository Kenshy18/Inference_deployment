"""Validated EVA-02 TensorRT-backbone bundle manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


MANIFEST_SCHEMA = "eva02-cascade-trt-backbone-bundle-v1"
PROFILE = "eva02-vit-dynamic-b1-20-fp16-v1"
VERIFY_MODES = frozenset({"metadata", "engine", "full"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, stored_path: str | None = None) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    return {
        "path": stored_path or str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _resolve_record(
    manifest_path: Path,
    record: object,
    *,
    label: str,
    verify_content: bool,
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} record must be an object")
    raw_path = record.get("path")
    expected_size = record.get("size")
    expected_hash = record.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path is missing")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise ValueError(f"{label} size must be a positive integer")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"{label} sha256 is invalid")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.stat().st_size != expected_size:
        raise ValueError(f"{label} size mismatch: {path}")
    if verify_content and sha256_file(path) != expected_hash:
        raise ValueError(f"{label} sha256 mismatch: {path}")
    return path


def _verify_bound_source(
    manifest_path: Path,
    record: object,
    supplied_path: Path | None,
    *,
    label: str,
) -> Path:
    if supplied_path is None:
        return _resolve_record(
            manifest_path,
            record,
            label=label,
            verify_content=True,
        )
    if not isinstance(record, dict):
        raise ValueError(f"{label} record must be an object")
    expected_size = record.get("size")
    expected_hash = record.get("sha256")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise ValueError(f"{label} size must be a positive integer")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"{label} sha256 is invalid")
    supplied = supplied_path.expanduser().resolve()
    if not supplied.is_file():
        raise FileNotFoundError(f"{label} not found: {supplied}")
    if supplied.stat().st_size != expected_size:
        raise ValueError(f"{label} does not match bundle size: {supplied}")
    if sha256_file(supplied) != expected_hash:
        raise ValueError(f"{label} does not match bundle sha256: {supplied}")
    return supplied


@dataclass(frozen=True, slots=True)
class Eva02TrtBundle:
    manifest_path: Path
    profile: str
    engine_path: Path
    precision: str
    target_size: int
    min_batch: int
    opt_batch: int
    max_batch: int
    input_name: str
    output_name: str
    validation: dict[str, object]


def load_trt_bundle(
    manifest_path: Path,
    *,
    verify: Literal["metadata", "engine", "full"] = "engine",
    checkpoint_path: Path | None = None,
    classifier_checkpoint: Path | None = None,
    config_path: Path | None = None,
) -> Eva02TrtBundle:
    """Load one complete bundle and reject mixed or tampered artifacts."""

    if verify not in VERIFY_MODES:
        raise ValueError(f"unsupported bundle verification mode: {verify!r}")
    manifest = manifest_path.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"TensorRT bundle manifest not found: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported EVA-02 bundle schema: {payload.get('schema')!r}")
    if payload.get("status") != "complete":
        raise ValueError("EVA-02 TensorRT bundle is not complete")
    if payload.get("profile") != PROFILE:
        raise ValueError(f"unsupported EVA-02 TensorRT profile: {payload.get('profile')!r}")

    shape = payload.get("shape_profile")
    if not isinstance(shape, dict):
        raise ValueError("shape_profile is missing")
    target_size = int(shape.get("target_size", 0))
    min_batch = int(shape.get("min_batch", 0))
    opt_batch = int(shape.get("opt_batch", 0))
    max_batch = int(shape.get("max_batch", 0))
    if target_size <= 0 or not (1 <= min_batch <= opt_batch <= max_batch):
        raise ValueError(f"invalid EVA-02 shape profile: {shape!r}")

    io = payload.get("io")
    if not isinstance(io, dict):
        raise ValueError("bundle io contract is missing")
    input_name = str(io.get("input_name", ""))
    output_name = str(io.get("output_name", ""))
    if not input_name or not output_name:
        raise ValueError("bundle io tensor names are missing")

    engine = _resolve_record(
        manifest,
        payload.get("engine"),
        label="EVA-02 TensorRT engine",
        verify_content=verify in {"engine", "full"},
    )
    if verify == "full":
        source = payload.get("source")
        if not isinstance(source, dict):
            raise ValueError("bundle source records are missing")
        _verify_bound_source(
            manifest,
            source.get("checkpoint"),
            checkpoint_path,
            label="detector checkpoint",
        )
        _verify_bound_source(
            manifest,
            source.get("classifier_checkpoint"),
            classifier_checkpoint,
            label="classifier checkpoint",
        )
        _verify_bound_source(
            manifest,
            source.get("config"),
            config_path,
            label="model config",
        )

    validation = payload.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != "pass":
        raise ValueError("bundle does not contain a passing validation report")
    return Eva02TrtBundle(
        manifest_path=manifest,
        profile=PROFILE,
        engine_path=engine,
        precision=str(payload.get("precision", "")),
        target_size=target_size,
        min_batch=min_batch,
        opt_batch=opt_batch,
        max_batch=max_batch,
        input_name=input_name,
        output_name=output_name,
        validation=validation,
    )


__all__ = [
    "MANIFEST_SCHEMA",
    "PROFILE",
    "Eva02TrtBundle",
    "file_record",
    "load_trt_bundle",
    "sha256_file",
]
