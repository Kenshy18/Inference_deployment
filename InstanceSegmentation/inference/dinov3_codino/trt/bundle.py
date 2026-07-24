"""Validation for one cohesive fixed-B2 Co-DINO TensorRT engine bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


MANIFEST_SCHEMA = "video-mask-codino-trt-bundle-v1"
PROFILE_PORTABLE = "portable-fixed-b2-v1"
PROFILE_FAST = "fast-sm120-fixed-b2-v1"
# Backward-compatible names used by the portable engine builder.
PROFILE = PROFILE_PORTABLE
BATCH_SIZE = 2
INPUT_SIZE = (736, 1280)
IMAGE_SIZE = (720, 1280)
QUERY_SHAPES = "184x320,92x160,46x80,23x40,12x20"
ENGINE_FILENAMES = {
    "backbone": (
        "codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed_bf16.engine"
    ),
    "query_encoder": (
        "codino_query_encoder_b2_736x1280_msda_plugin_sbc_fp16.engine"
    ),
    "decoder": "codino_decoder_b2_736x1280_msda_plugin_fp32.engine",
    "mask_head": "codino_mask_head_core_n1_736x1280_fp32.engine",
}
FAST_ENGINE_FILENAMES = {
    "backbone": "codino_dinov3_vitl_backbone_736x1280_b2_t090.engine",
    "query_encoder": "codino_query_encoder_736x1280_b2_sm120_t140.engine",
    "decoder": "codino_decoder_736x1280_b2_t090.engine",
    "mask_head": "codino_mask_head_core_n1_736x1280_d140_fp32.engine",
}
FAST_PLUGIN_FILENAME = "codino_msda_direct_sm120.so"
PRECISION_POLICY = {
    "backbone": "bf16-forced-public-fp32",
    "query_encoder": "fp16",
    "decoder": "fp32",
    "mask_head": "fp32-n1",
}
FAST_PRECISION_POLICY_RETAINED = {
    "backbone": "t090-minimal-fp32-protected-fp16",
    "query_encoder": "t140-sm120-msda-fp16-18-fp32-protected",
    "decoder": "t090-minimal-fp32-protected-fp16",
    "mask_head": "d140-fp32-n1",
}
FAST_PRECISION_POLICY_CLEAN = {
    "backbone": "t090-forced-bf16-public-fp32",
    "query_encoder": "t140-sm120-msda-fp16-18-fp32-protected",
    "decoder": "t090-fp32-msda-v2",
    "mask_head": "d140-fp32-n1",
}
# Compatibility name for the retained benchmark bundle.
FAST_PRECISION_POLICY = FAST_PRECISION_POLICY_RETAINED
FAST_PRECISION_POLICIES = (
    FAST_PRECISION_POLICY_RETAINED,
    FAST_PRECISION_POLICY_CLEAN,
)
VerifyMode = Literal["metadata", "engines", "full"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file(value: Path, *, label: str) -> Path:
    path = value.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be a regular file: {path}")
    return path


def _check_record(path: Path, record: object, *, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{label} record is missing")
    expected_size = record.get("size")
    expected_hash = record.get("sha256")
    if path.stat().st_size != expected_size:
        raise ValueError(f"{label} size mismatch: {path}")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")


def _source_path(
    *,
    supplied: Path | None,
    record: object,
    label: str,
) -> Path:
    if supplied is not None:
        return _required_file(supplied, label=label)
    if not isinstance(record, dict) or not record.get("path"):
        raise ValueError(f"{label} path is required for full verification")
    return _required_file(Path(str(record["path"])), label=label)


@dataclass(frozen=True, slots=True)
class CoDinoEngineBundle:
    manifest_path: Path
    profile: str
    batch_size: int
    query_shapes: str
    backbone_engine: Path
    query_encoder_engine: Path
    decoder_engine: Path
    mask_head_engine: Path
    runtime_profile: str
    query_plugin_extension: Path | None
    precision_policy: dict[str, str]

    @property
    def engines(self) -> dict[str, Path]:
        return {
            "backbone": self.backbone_engine,
            "query_encoder": self.query_encoder_engine,
            "decoder": self.decoder_engine,
            "mask_head": self.mask_head_engine,
        }


def load_engine_bundle(
    manifest_path: Path,
    *,
    verify: VerifyMode = "engines",
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    classifier_checkpoint: Path | None = None,
    runtime_python: Path | None = None,
) -> CoDinoEngineBundle:
    """Load and verify one supported, indivisible fixed-B2 engine group."""

    if verify not in {"metadata", "engines", "full"}:
        raise ValueError(f"unsupported verification mode: {verify}")
    manifest = _required_file(manifest_path, label="Co-DINO bundle manifest")
    payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Co-DINO bundle manifest must be an object")
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unknown Co-DINO bundle manifest schema")
    if payload.get("status") != "complete":
        raise ValueError("Co-DINO bundle is not complete")
    profile = str(payload.get("profile"))
    if profile == PROFILE_PORTABLE:
        engine_filenames = ENGINE_FILENAMES
        precision_policy = PRECISION_POLICY
        runtime_profile = "stable"
    elif profile == PROFILE_FAST:
        engine_filenames = FAST_ENGINE_FILENAMES
        observed_precision = payload.get("precision_policy")
        if observed_precision not in FAST_PRECISION_POLICIES:
            raise ValueError("Co-DINO fast precision policy mismatch")
        precision_policy = observed_precision
        runtime_profile = "fast-b2"
    else:
        raise ValueError(f"unsupported Co-DINO bundle profile: {profile}")
    if payload.get("fixed_batch") is not True:
        raise ValueError("Co-DINO portable bundle must use a fixed batch")
    if payload.get("batch_size") != BATCH_SIZE:
        raise ValueError(f"Co-DINO portable bundle requires batch {BATCH_SIZE}")
    if payload.get("input_tensor_size") != list(INPUT_SIZE):
        raise ValueError("Co-DINO bundle input tensor size mismatch")
    if payload.get("runtime_image_size") != list(IMAGE_SIZE):
        raise ValueError("Co-DINO bundle runtime image size mismatch")
    if payload.get("query_encoder_shapes") != QUERY_SHAPES:
        raise ValueError("Co-DINO query feature-shape contract mismatch")
    if payload.get("precision_policy") != precision_policy:
        raise ValueError("Co-DINO precision policy mismatch")
    execution_policy = payload.get("execution_policy")
    if execution_policy is not None:
        if (
            not isinstance(execution_policy, dict)
            or execution_policy.get("runtime_profile") != runtime_profile
        ):
            raise ValueError("Co-DINO execution policy mismatch")

    records = payload.get("engines")
    if not isinstance(records, dict) or set(records) != set(engine_filenames):
        raise ValueError("Co-DINO bundle must contain exactly four engine records")
    engines: dict[str, Path] = {}
    for name, filename in engine_filenames.items():
        record = records[name]
        if not isinstance(record, dict):
            raise ValueError(f"{name} engine record is invalid")
        if record.get("path") != f"engines/{filename}":
            raise ValueError(f"{name} engine path does not match the portable profile")
        path = _required_file(manifest.parent / str(record["path"]), label=f"{name} engine")
        if path.stat().st_size != record.get("size"):
            raise ValueError(f"{name} engine size mismatch: {path}")
        if verify in {"engines", "full"} and sha256_file(path) != record.get("sha256"):
            raise ValueError(f"{name} engine SHA-256 mismatch: {path}")
        engines[name] = path

    plugin: Path | None = None
    plugin_record = payload.get("query_plugin_extension")
    if profile == PROFILE_FAST:
        if not isinstance(plugin_record, dict):
            raise ValueError("fast Co-DINO bundle requires the SM120 plugin")
        expected_plugin_path = f"plugins/{FAST_PLUGIN_FILENAME}"
        if plugin_record.get("path") != expected_plugin_path:
            raise ValueError("fast Co-DINO plugin path mismatch")
        plugin = _required_file(
            manifest.parent / expected_plugin_path,
            label="SM120 MSDA plugin",
        )
        if plugin.stat().st_size != plugin_record.get("size"):
            raise ValueError(f"SM120 MSDA plugin size mismatch: {plugin}")
        if verify in {"engines", "full"}:
            if sha256_file(plugin) != plugin_record.get("sha256"):
                raise ValueError(f"SM120 MSDA plugin SHA-256 mismatch: {plugin}")
    elif plugin_record is not None:
        raise ValueError("portable Co-DINO bundle must not contain a query plugin")

    if verify == "full":
        source = payload.get("source")
        if not isinstance(source, dict):
            raise ValueError("Co-DINO bundle source records are missing")
        config = _source_path(
            supplied=config_path,
            record=source.get("config"),
            label="Co-DINO config",
        )
        checkpoint = _source_path(
            supplied=checkpoint_path,
            record=source.get("checkpoint"),
            label="Co-DINO checkpoint",
        )
        _check_record(config, source.get("config"), label="Co-DINO config")
        _check_record(
            checkpoint,
            source.get("checkpoint"),
            label="Co-DINO checkpoint",
        )
        classifier_record = source.get("classifier_checkpoint")
        if classifier_record is not None:
            classifier = _source_path(
                supplied=classifier_checkpoint,
                record=classifier_record,
                label="classifier checkpoint",
            )
            _check_record(
                classifier,
                classifier_record,
                label="classifier checkpoint",
            )
        runtime_record = payload.get("runtime_python")
        runtime = _source_path(
            supplied=runtime_python,
            record=runtime_record,
            label="runtime Python",
        )
        _check_record(runtime, runtime_record, label="runtime Python")

    return CoDinoEngineBundle(
        manifest_path=manifest,
        profile=profile,
        batch_size=BATCH_SIZE,
        query_shapes=QUERY_SHAPES,
        backbone_engine=engines["backbone"],
        query_encoder_engine=engines["query_encoder"],
        decoder_engine=engines["decoder"],
        mask_head_engine=engines["mask_head"],
        runtime_profile=runtime_profile,
        query_plugin_extension=plugin,
        precision_policy=dict(precision_policy),
    )


__all__ = [
    "BATCH_SIZE",
    "ENGINE_FILENAMES",
    "FAST_ENGINE_FILENAMES",
    "FAST_PLUGIN_FILENAME",
    "FAST_PRECISION_POLICY",
    "FAST_PRECISION_POLICIES",
    "FAST_PRECISION_POLICY_CLEAN",
    "FAST_PRECISION_POLICY_RETAINED",
    "IMAGE_SIZE",
    "INPUT_SIZE",
    "MANIFEST_SCHEMA",
    "PRECISION_POLICY",
    "PROFILE",
    "PROFILE_FAST",
    "PROFILE_PORTABLE",
    "QUERY_SHAPES",
    "CoDinoEngineBundle",
    "VerifyMode",
    "load_engine_bundle",
    "sha256_file",
]
