#!/usr/bin/env python3
"""Build the four fixed-batch TensorRT engines used by Co-DINO."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path


TRT_ROOT = Path(__file__).resolve().parent
FAMILY_ROOT = TRT_ROOT.parent
STANDALONE_CODINO_SOURCE = FAMILY_ROOT / ".runtime" / "src" / "codino"
STANDALONE_DINOV3_SOURCE = FAMILY_ROOT / ".runtime" / "src" / "dinov3_root"
if str(TRT_ROOT) not in sys.path:
    sys.path.insert(0, str(TRT_ROOT))
for _source_root in (STANDALONE_CODINO_SOURCE, STANDALONE_DINOV3_SOURCE):
    if _source_root.is_dir() and str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

try:
    from .bundle import (
        BATCH_SIZE,
        ENGINE_FILENAMES,
        IMAGE_SIZE,
        INPUT_SIZE,
        MANIFEST_SCHEMA,
        PRECISION_POLICY,
        PROFILE,
        QUERY_SHAPES,
        load_engine_bundle,
        sha256_file,
    )
except ImportError:
    from bundle import (
        BATCH_SIZE,
        ENGINE_FILENAMES,
        IMAGE_SIZE,
        INPUT_SIZE,
        MANIFEST_SCHEMA,
        PRECISION_POLICY,
        PROFILE,
        QUERY_SHAPES,
        load_engine_bundle,
        sha256_file,
    )

WORKSPACE_ROOT = (
    Path(os.environ.get("CODINO_HOME", "~/.local/share/dinov3-codino"))
    .expanduser()
    .resolve()
)
MODEL_ROOT = (
    Path(os.environ.get("CODINO_MODEL_ROOT", WORKSPACE_ROOT / "models"))
    .expanduser()
    .resolve()
)
CACHE_ROOT = (
    Path(os.environ.get("CODINO_CACHE_ROOT", WORKSPACE_ROOT / "cache"))
    .expanduser()
    .resolve()
)
TEMP_ROOT = (
    Path(os.environ.get("CODINO_TEMP_ROOT", WORKSPACE_ROOT / "tmp"))
    .expanduser()
    .resolve()
)

INPUT_HEIGHT, INPUT_WIDTH = INPUT_SIZE
IMAGE_HEIGHT, IMAGE_WIDTH = IMAGE_SIZE
FEATURE_SHAPES = QUERY_SHAPES


def require_standalone_sources() -> None:
    missing = [
        path
        for path in (STANDALONE_CODINO_SOURCE, STANDALONE_DINOV3_SOURCE)
        if not path.is_dir()
    ]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise RuntimeError(
            "standalone framework sources are missing; run "
            f"setup_environment.py first: {joined}"
        )


def _require_module_under(module: object, root: Path, *, name: str) -> None:
    module_file = Path(str(getattr(module, "__file__", ""))).resolve()
    if not module_file.is_relative_to(root.resolve()):
        raise RuntimeError(
            f"{name} resolved outside the standalone folder: {module_file}; "
            f"expected a module below {root.resolve()}"
        )


def prepare_codino_imports(*, patch_mmcv_stream: bool) -> None:
    """Register the bundled Co-DINO fork and its custom transform."""

    require_standalone_sources()
    import dinov3
    import mmdet
    import mmdet.models.backbones.dinov3_vit  # noqa: F401
    import projects.models as projects_models

    _require_module_under(mmdet, STANDALONE_CODINO_SOURCE, name="mmdet")
    _require_module_under(
        projects_models,
        STANDALONE_CODINO_SOURCE,
        name="projects.models",
    )
    _require_module_under(dinov3, STANDALONE_DINOV3_SOURCE, name="dinov3")

    codino_root = Path(mmdet.__file__).resolve().parent.parent
    augmentation = codino_root / "mmdet/datasets/pipelines/unified_letterbox_aug.py"
    if not augmentation.is_file():
        raise RuntimeError(
            "The installed Co-DINO fork is incomplete; missing " f"{augmentation}"
        )
    module_name = "codino_unified_letterbox_aug"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, augmentation)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load Co-DINO transform: {augmentation}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    if not patch_mmcv_stream:
        return
    import mmcv.parallel._functions as mmcv_parallel_functions
    import torch

    if getattr(mmcv_parallel_functions, "_codino_stream_patch", False):
        return
    original = mmcv_parallel_functions._get_stream

    def torch_compatible_get_stream(device):
        if isinstance(device, int):
            device = torch.device("cuda", device)
        return original(device)

    mmcv_parallel_functions._get_stream = torch_compatible_get_stream
    mmcv_parallel_functions._codino_stream_patch = True


def default_tensorrt_site_packages() -> Path | None:
    raw = os.environ.get("TENSORRT_SITE_PACKAGES")
    if raw:
        path = Path(raw).expanduser()
        return path if path.exists() else None
    for raw_path in sys.path:
        path = Path(raw_path)
        if (path / "tensorrt").exists():
            return path
    return None


def prepare_tensorrt(extra_site_packages: Path | None) -> None:
    if extra_site_packages is None or not extra_site_packages.exists():
        return
    extra_site = str(extra_site_packages)
    if extra_site not in sys.path:
        sys.path.append(extra_site)
    libraries = extra_site_packages / "tensorrt_libs"
    if not libraries.exists():
        return
    os.environ[
        "LD_LIBRARY_PATH"
    ] = f"{libraries}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    for name in (
        "libnvinfer.so.10",
        "libnvonnxparser.so.10",
        "libnvinfer_plugin.so.10",
    ):
        path = libraries / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def build_backbone_engine(
    *,
    onnx_path: Path,
    engine_path: Path,
    workspace_gb: int,
) -> None:
    try:
        from .runtime import _append_runtime_libs_to_env, _preload_vendor_libs
    except ImportError:
        from runtime import _append_runtime_libs_to_env, _preload_vendor_libs

    import tensorrt as trt

    _append_runtime_libs_to_env()
    _preload_vendor_libs()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    parsed = (
        parser.parse_from_file(str(onnx_path))
        if hasattr(parser, "parse_from_file")
        else parser.parse(onnx_path.read_bytes())
    )
    if not parsed:
        errors = "\n".join(
            str(parser.get_error(index)) for index in range(parser.num_errors)
        )
        raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gb) << 30,
    )
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    float_dtypes = {
        trt.DataType.FLOAT,
        trt.DataType.HALF,
        trt.DataType.BF16,
    }
    skipped_layers = {
        "LayerType.CAST",
        "LayerType.SHAPE",
        "LayerType.CONSTANT",
        "LayerType.FILL",
        "LayerType.DECONVOLUTION",
    }
    for layer_index in range(network.num_layers):
        layer = network.get_layer(layer_index)
        if str(layer.type) in skipped_layers:
            continue
        floating_outputs = []
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            shape_attribute = getattr(tensor, "is_shape_tensor", False)
            try:
                is_shape_tensor = bool(
                    shape_attribute() if callable(shape_attribute) else shape_attribute
                )
            except Exception:
                is_shape_tensor = False
            if (
                tensor is not None
                and not is_shape_tensor
                and tensor.dtype in float_dtypes
            ):
                floating_outputs.append(output_index)
        if not floating_outputs:
            continue
        try:
            layer.precision = trt.DataType.BF16
        except Exception:
            pass
        for output_index in floating_outputs:
            try:
                layer.set_output_type(output_index, trt.DataType.BF16)
            except Exception:
                pass
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT backbone engine build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))


def _run(runtime_python: Path, script: str, arguments: list[str]) -> None:
    command = [str(runtime_python), str(TRT_ROOT / script), *arguments]
    print("[build]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    stored = resolved.relative_to(relative_to) if relative_to is not None else resolved
    return {
        "path": stored.as_posix(),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _manifest(
    *,
    root: Path,
    config: Path,
    checkpoint: Path,
    classifier_checkpoint: Path | None,
    runtime_python: Path,
    environment_lock: Path | None,
) -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": PROFILE,
        "status": "complete",
        "production_registered": False,
        "performance_claim": None,
        "quality_claim": "requires-checkpoint-specific-gate",
        "fixed_batch": True,
        "batch_size": BATCH_SIZE,
        "input_tensor_size": list(INPUT_SIZE),
        "runtime_image_size": list(IMAGE_SIZE),
        "query_encoder_shapes": QUERY_SHAPES,
        "precision_policy": PRECISION_POLICY,
        "source": {
            "config": _file_record(config),
            "checkpoint": _file_record(checkpoint),
            "classifier_checkpoint": (
                None
                if classifier_checkpoint is None
                else _file_record(classifier_checkpoint)
            ),
            "builder_script": _file_record(Path(__file__)),
        },
        "runtime_python": _file_record(runtime_python),
        "engines": {
            name: _file_record(
                root / "engines" / filename,
                relative_to=root,
            )
            for name, filename in ENGINE_FILENAMES.items()
        },
        "environment_lock": (
            None if environment_lock is None else _file_record(environment_lock)
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--classifier-checkpoint", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workspace-gb", type=int, default=12)
    parser.add_argument("--verify-manifest", type=Path)
    parser.add_argument(
        "--verify",
        choices=("metadata", "engines", "full"),
        default="full",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runtime_python = args.runtime_python.expanduser().resolve()
    if not runtime_python.is_file():
        parser.error(f"runtime Python not found: {runtime_python}")
    if args.verify_manifest is not None:
        bundle = load_engine_bundle(
            args.verify_manifest,
            verify=args.verify,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            classifier_checkpoint=args.classifier_checkpoint,
            runtime_python=runtime_python,
        )
        print(f"[PASS] complete {bundle.profile} bundle: {bundle.manifest_path}")
        return 0
    missing = [
        name
        for name in ("config", "checkpoint", "output_dir")
        if getattr(args, name) is None
    ]
    if missing:
        switches = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        parser.error(f"build mode requires: {switches}")
    config = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    classifier_checkpoint = (
        None
        if args.classifier_checkpoint is None
        else args.classifier_checkpoint.expanduser().resolve()
    )
    environment_lock = (
        None
        if args.environment_lock is None
        else args.environment_lock.expanduser().resolve()
    )
    output = args.output_dir.expanduser().resolve()
    for label, path in (
        ("config", config),
        ("checkpoint", checkpoint),
    ):
        if not path.is_file():
            parser.error(f"{label} not found: {path}")
    for label, path in (
        ("classifier checkpoint", classifier_checkpoint),
        ("environment lock", environment_lock),
    ):
        if path is not None and not path.is_file():
            parser.error(f"{label} not found: {path}")
    if output.exists():
        parser.error(f"output already exists: {output}")
    if args.workspace_gb <= 0:
        parser.error("--workspace-gb must be positive")
    try:
        require_standalone_sources()
    except RuntimeError as exc:
        parser.error(str(exc))

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    onnx_dir = staging / "onnx"
    engine_dir = staging / "engines"
    onnx_dir.mkdir()
    engine_dir.mkdir()
    backbone_onnx = (
        onnx_dir
        / "codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed.onnx"
    )
    try:
        common = [
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--batch-size",
            str(BATCH_SIZE),
        ]
        _run(
            runtime_python,
            "export_backbone.py",
            [
                *common,
                "--output",
                str(backbone_onnx),
                "--engine",
                str(engine_dir / ENGINE_FILENAMES["backbone"]),
                "--workspace-gb",
                str(args.workspace_gb),
                "--height",
                str(INPUT_HEIGHT),
                "--width",
                str(INPUT_WIDTH),
                "--fixed-batch",
            ],
        )
        shared_partition_args = [
            *common,
            "--input-height",
            str(INPUT_HEIGHT),
            "--input-width",
            str(INPUT_WIDTH),
            "--img-height",
            str(IMAGE_HEIGHT),
            "--img-width",
            str(IMAGE_WIDTH),
            "--feature-shapes",
            FEATURE_SHAPES,
            "--workspace-gb",
            str(args.workspace_gb),
        ]
        _run(
            runtime_python,
            "export_query_encoder.py",
            [
                *shared_partition_args,
                "--onnx",
                str(
                    onnx_dir
                    / "codino_query_encoder_b2_736x1280_msda_trt_plugin_sbc.onnx"
                ),
                "--engine",
                str(engine_dir / ENGINE_FILENAMES["query_encoder"]),
                "--precision",
                "fp16",
            ],
        )
        _run(
            runtime_python,
            "export_decoder.py",
            [
                *shared_partition_args,
                "--onnx",
                str(
                    onnx_dir
                    / "codino_decoder_b2_736x1280_msda_trt_plugin.onnx"
                ),
                "--engine",
                str(engine_dir / ENGINE_FILENAMES["decoder"]),
                "--precision",
                "fp32",
            ],
        )
        _run(
            runtime_python,
            "export_mask_head.py",
            [
                *common,
                "--mode",
                "core",
                "--onnx",
                str(onnx_dir / "codino_mask_head_core_n1_736x1280.onnx"),
                "--engine",
                str(engine_dir / ENGINE_FILENAMES["mask_head"]),
                "--precision",
                "fp32",
                "--workspace-gb",
                str(args.workspace_gb),
            ],
        )
        manifest = _manifest(
            root=staging,
            config=config,
            checkpoint=checkpoint,
            classifier_checkpoint=classifier_checkpoint,
            runtime_python=runtime_python,
            environment_lock=environment_lock,
        )
        manifest_path = staging / "manifest.json"
        temporary_manifest = staging / ".manifest.json.tmp"
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
        load_engine_bundle(
            manifest_path,
            verify="full",
            config_path=config,
            checkpoint_path=checkpoint,
            classifier_checkpoint=classifier_checkpoint,
            runtime_python=runtime_python,
        )
        staging.rename(output)
    except BaseException:
        print(f"[failed] diagnostic files retained at {staging}", flush=True)
        raise
    final_manifest = output / "manifest.json"
    load_engine_bundle(
        final_manifest,
        verify="full",
        config_path=config,
        checkpoint_path=checkpoint,
        classifier_checkpoint=classifier_checkpoint,
        runtime_python=runtime_python,
    )
    print(f"[PASS] built and fully verified bundle: {final_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
