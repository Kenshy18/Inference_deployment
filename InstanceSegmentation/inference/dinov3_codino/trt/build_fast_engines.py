#!/usr/bin/env python3
"""Build the RTX 5090 fast fixed-B2 Co-DINO bundle from one checkpoint."""

from __future__ import annotations

import argparse
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
CODINO_SOURCE = FAMILY_ROOT / ".runtime" / "src" / "codino"
DINOV3_SOURCE = FAMILY_ROOT / ".runtime" / "src" / "dinov3_root"
if str(TRT_ROOT) not in sys.path:
    sys.path.insert(0, str(TRT_ROOT))

try:
    from .bundle import (
        BATCH_SIZE,
        FAST_ENGINE_FILENAMES,
        FAST_PLUGIN_FILENAME,
        FAST_PRECISION_POLICY_CLEAN,
        IMAGE_SIZE,
        INPUT_SIZE,
        MANIFEST_SCHEMA,
        PROFILE_FAST,
        QUERY_SHAPES,
        load_engine_bundle,
        sha256_file,
    )
except ImportError:
    from bundle import (
        BATCH_SIZE,
        FAST_ENGINE_FILENAMES,
        FAST_PLUGIN_FILENAME,
        FAST_PRECISION_POLICY_CLEAN,
        IMAGE_SIZE,
        INPUT_SIZE,
        MANIFEST_SCHEMA,
        PROFILE_FAST,
        QUERY_SHAPES,
        load_engine_bundle,
        sha256_file,
    )


EXPORTERS = {
    "backbone": TRT_ROOT / "export_backbone.py",
    "query_encoder": TRT_ROOT / "export_query_encoder.py",
    "decoder": TRT_ROOT / "export_decoder.py",
    "mask_head": TRT_ROOT / "export_mask_head.py",
}
WORKER = TRT_ROOT / "fast_engine_build.py"
NATIVE_DIR = TRT_ROOT / "native"
BUILDER_SOURCES = (
    Path(__file__).resolve(),
    WORKER,
    NATIVE_DIR / "msda_direct.cpp",
    NATIVE_DIR / "msda_direct.cu",
    *EXPORTERS.values(),
)
ONNX_FILENAMES = {
    "backbone": "codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed.onnx",
    "query_encoder": (
        "codino_query_encoder_b2_736x1280_msda_trt_plugin_sbc.onnx"
    ),
    "decoder": "codino_decoder_b2_736x1280_msda_trt_plugin.onnx",
    "mask_head": "codino_mask_head_core_n1_736x1280.onnx",
}


def _file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _executable(path: Path, label: str) -> Path:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"{label} is not executable: {path}")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--classifier-checkpoint", required=True, type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workspace-gb", type=int, default=12)
    return parser


def _environment(runtime_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    python_path = os.pathsep.join(
        str(path)
        for path in (CODINO_SOURCE, DINOV3_SOURCE, FAMILY_ROOT, TRT_ROOT)
        if path.is_dir()
    )
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_HOME": str(runtime_root),
            "CC": str(runtime_root / "bin/gcc"),
            "CXX": str(runtime_root / "bin/g++"),
            "PATH": f"{runtime_root / 'bin'}:{environment.get('PATH', '')}",
            "PYTHONPATH": python_path,
        }
    )
    return environment


def _run(command: list[str], environment: dict[str, str]) -> None:
    print("[CODINO-TRT]", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=FAMILY_ROOT, env=environment)


def _export_commands(
    python: Path,
    config: Path,
    checkpoint: Path,
    onnx_dir: Path,
) -> dict[str, list[str]]:
    common = ["--config", str(config), "--checkpoint", str(checkpoint)]
    outputs = {
        name: onnx_dir / filename for name, filename in ONNX_FILENAMES.items()
    }
    return {
        "backbone": [
            str(python),
            str(EXPORTERS["backbone"]),
            *common,
            "--output",
            str(outputs["backbone"]),
            "--height",
            "736",
            "--width",
            "1280",
            "--batch-size",
            "2",
            "--fixed-batch",
        ],
        "query_encoder": [
            str(python),
            str(EXPORTERS["query_encoder"]),
            *common,
            "--onnx",
            str(outputs["query_encoder"]),
            "--engine",
            str(onnx_dir / "unused-query.engine"),
            "--batch-size",
            "2",
            "--precision",
            "fp16",
            "--skip-build",
        ],
        "decoder": [
            str(python),
            str(EXPORTERS["decoder"]),
            *common,
            "--onnx",
            str(outputs["decoder"]),
            "--engine",
            str(onnx_dir / "unused-decoder.engine"),
            "--batch-size",
            "2",
            "--precision",
            "fp32",
            "--skip-build",
        ],
        "mask_head": [
            str(python),
            str(EXPORTERS["mask_head"]),
            *common,
            "--mode",
            "core",
            "--onnx",
            str(outputs["mask_head"]),
            "--engine",
            str(onnx_dir / "unused-mask.engine"),
            "--batch-size",
            "2",
            "--num-rois",
            "1",
            "--precision",
            "fp32",
            "--skip-build",
        ],
    }


def _worker_base(python: Path) -> list[str]:
    return [str(python), str(WORKER)]


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    resolved = _file(path, str(path))
    stored = (
        resolved.relative_to(relative_to.resolve())
        if relative_to is not None
        else resolved
    )
    return {
        "path": stored.as_posix(),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _write_manifest(
    *,
    root: Path,
    config: Path,
    checkpoint: Path,
    classifier_checkpoint: Path,
    runtime_python: Path,
    environment_lock: Path | None,
    build_report: Path,
) -> Path:
    payload = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": PROFILE_FAST,
        "status": "complete",
        "production_registered": False,
        "performance_claim": {
            "status": "requires-checkpoint-specific-video-gate",
            "build_report": "metadata/build-report.json",
            "build_report_sha256": sha256_file(build_report),
            "e2e_25fps_claimed": False,
        },
        "quality_claim": "requires-checkpoint-specific-eager-and-video-gate",
        "fixed_batch": True,
        "batch_size": BATCH_SIZE,
        "input_tensor_size": list(INPUT_SIZE),
        "runtime_image_size": list(IMAGE_SIZE),
        "query_encoder_shapes": QUERY_SHAPES,
        "precision_policy": FAST_PRECISION_POLICY_CLEAN,
        "execution_policy": {
            "runtime_profile": "fast-b2",
            "core": "stable-pointer-cuda-graph",
            "decode_preprocess": "bounded-opencv-lookahead",
            "tail": "double-buffered-worker-stream",
            "output_order": "source-order",
        },
        "source": {
            "config": _record(config),
            "checkpoint": _record(checkpoint),
            "classifier_checkpoint": _record(classifier_checkpoint),
            "builder_script": _record(Path(__file__)),
            "builder_sources": {
                str(path.relative_to(FAMILY_ROOT)): _record(path)
                for path in BUILDER_SOURCES
            },
        },
        "runtime_python": _record(runtime_python),
        "engines": {
            name: _record(
                root / "engines" / filename,
                relative_to=root,
            )
            for name, filename in FAST_ENGINE_FILENAMES.items()
        },
        "query_plugin_extension": _record(
            root / "plugins" / FAST_PLUGIN_FILENAME,
            relative_to=root,
        ),
        "environment_lock": (
            None if environment_lock is None else _record(environment_lock)
        ),
    }
    manifest = root / "manifest.json"
    temporary = root / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.workspace_gb <= 0:
        parser.error("--workspace-gb must be positive")
    python = _file(args.runtime_python, "runtime Python")
    config = _file(args.config, "Co-DINO config")
    checkpoint = _file(args.checkpoint, "Co-DINO checkpoint")
    classifier = _file(args.classifier_checkpoint, "classifier checkpoint")
    lock = (
        _file(args.environment_lock, "environment lock")
        if args.environment_lock is not None
        else None
    )
    for source in BUILDER_SOURCES:
        _file(source, "builder source")
    for source in (CODINO_SOURCE, DINOV3_SOURCE):
        if not source.is_dir():
            parser.error(
                f"standalone source missing: {source}; run setup_environment.py first"
            )
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError(f"fast bundle output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    stable_inputs = {
        str(path): sha256_file(path)
        for path in (config, checkpoint, python, *BUILDER_SOURCES)
    }
    stable_inputs[str(classifier)] = sha256_file(classifier)
    if lock is not None:
        stable_inputs[str(lock)] = sha256_file(lock)
    runtime_root = python.parent.parent
    for tool in ("gcc", "g++", "nvcc", "ninja"):
        _executable(runtime_root / "bin" / tool, f"fast-build tool {tool}")
    environment = _environment(runtime_root)
    onnx_dir = staging / "onnx"
    build_onnx_dir = staging / "build-onnx"
    engine_dir = staging / "engines"
    plugin = staging / "plugins" / FAST_PLUGIN_FILENAME
    metadata = staging / "metadata"
    try:
        onnx_dir.mkdir()
        _run(
            [
                *_worker_base(python),
                "--component",
                "preflight",
                "--report",
                str(metadata / "preflight.json"),
            ],
            environment,
        )
        for command in _export_commands(
            python, config, checkpoint, onnx_dir
        ).values():
            _run(command, environment)
        _run(
            [
                *_worker_base(python),
                "--component",
                "plugin",
                "--runtime-root",
                str(runtime_root),
                "--native-cpp",
                str(NATIVE_DIR / "msda_direct.cpp"),
                "--native-cuda",
                str(NATIVE_DIR / "msda_direct.cu"),
                "--native-build-dir",
                str(staging / "native-build"),
                "--plugin",
                str(plugin),
                "--report",
                str(metadata / "plugin.json"),
            ],
            environment,
        )
        for component in (
            "backbone",
            "query_encoder",
            "decoder",
            "mask_head",
        ):
            command = [
                *_worker_base(python),
                "--component",
                component,
                "--source-onnx",
                str(onnx_dir / ONNX_FILENAMES[component]),
                "--build-onnx",
                str(build_onnx_dir / ONNX_FILENAMES[component]),
                "--engine",
                str(engine_dir / FAST_ENGINE_FILENAMES[component]),
                "--timing-cache",
                str(metadata / f"{component}.timing-cache"),
                "--workspace-gb",
                str(args.workspace_gb),
                "--report",
                str(metadata / f"{component}.json"),
            ]
            if component == "query_encoder":
                command.extend(["--plugin", str(plugin)])
            _run(command, environment)
        reports = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(metadata.glob("*.json"))
        }
        build_report = metadata / "build-report.json"
        build_report.write_text(
            json.dumps(
                {
                    "schema": "video-mask-codino-fast-build-v1",
                    "status": "engines-loadable-quality-gate-required",
                    "checkpoint_sha256": stable_inputs[str(checkpoint)],
                    "components": reports,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        observed = {
            str(path): sha256_file(path) for path in map(Path, stable_inputs)
        }
        if observed != stable_inputs:
            raise RuntimeError("Co-DINO fast build inputs changed during the build")
        manifest = _write_manifest(
            root=staging,
            config=config,
            checkpoint=checkpoint,
            classifier_checkpoint=classifier,
            runtime_python=python,
            environment_lock=lock,
            build_report=build_report,
        )
        load_engine_bundle(
            manifest,
            verify="full",
            config_path=config,
            checkpoint_path=checkpoint,
            classifier_checkpoint=classifier,
            runtime_python=python,
        )
        staging.rename(output)
    except BaseException:
        print(f"[CODINO-TRT] failed staging retained: {staging}", flush=True)
        raise
    final_manifest = output / "manifest.json"
    load_engine_bundle(
        final_manifest,
        verify="full",
        config_path=config,
        checkpoint_path=checkpoint,
        classifier_checkpoint=classifier,
        runtime_python=python,
    )
    print(f"[PASS] built fast fixed-B2 bundle: {final_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
