#!/usr/bin/env python3
"""Build and package a supported fixed-batch Face DINO TensorRT bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    from .bundle import (
        ENGINE_FILES,
        PLUGIN_FILES,
        SCHEMA,
        SUPPORTED_PROFILES,
        sha256_file,
    )
except ImportError:
    package_parent = Path(__file__).resolve().parents[2]
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))
    from face_dino_v2.trt.bundle import (
        ENGINE_FILES,
        PLUGIN_FILES,
        SCHEMA,
        SUPPORTED_PROFILES,
        sha256_file,
    )


FAMILY_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_ROOT = FAMILY_ROOT.parent
DEFAULT_SOURCE_ROOT = Path("/home/kenshin/face_detection")
DEFAULT_CHECKPOINT = (
    DEFAULT_SOURCE_ROOT
    / "runs"
    / "face_dino_overnight_best_20260727"
    / "model_residual_v2.pth"
)
DEFAULT_REFERENCE_ROOT = INFERENCE_ROOT / "dinov3_codino_mh0"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="Existing SM120 Co-DINO implementation used for the MSDA builder.",
    )
    parser.add_argument("--batch-size", type=int, choices=(8, 16), default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workspace-gib", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-build-files", action="store_true")
    parser.add_argument(
        "--skip-runtime-snapshot",
        action="store_true",
        help="Do not copy source/checkpoint into this model package.",
    )
    return parser.parse_args()


def _record(path: Path, *, bundle_path: str | None = None) -> dict[str, object]:
    return {
        "path": bundle_path or str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    print("[build]", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
    )


def _ignore_runtime(directory: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".pytest_cache"}
    if Path(directory).name == "face_dino_v1":
        ignored.update({"tests"})
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo")))
    return ignored


def _snapshot_runtime(source_root: Path, checkpoint: Path) -> None:
    runtime_parent = FAMILY_ROOT / ".runtime" / "src"
    runtime_parent.mkdir(parents=True, exist_ok=True)
    destination = runtime_parent / "face_detection"
    staging = runtime_parent / (
        f".face_detection.staging-{os.getpid()}"
    )
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        shutil.copytree(
            source_root / "face_dino_v1",
            staging / "face_dino_v1",
            ignore=_ignore_runtime,
        )
        codino_destination = staging / "codino_face_detection"
        codino_destination.mkdir()
        shutil.copy2(
            source_root / "codino_face_detection" / "__init__.py",
            codino_destination / "__init__.py",
        )
        for name in ("models", "configs"):
            shutil.copytree(
                source_root / "codino_face_detection" / name,
                codino_destination / name,
                ignore=_ignore_runtime,
            )
        external = (
            source_root
            / "codino_face_detection"
            / ".runtime"
            / "external"
        )
        for name in ("codino", "dinov3"):
            shutil.copytree(
                external / name,
                codino_destination / ".runtime" / "external" / name,
                ignore=_ignore_runtime,
            )
        weights = (
            source_root
            / "Dino"
            / "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
        )
        if not weights.is_file():
            raise FileNotFoundError(f"DINOv3 ViT-S+ weights not found: {weights}")
        (staging / "Dino").mkdir()
        shutil.copy2(weights, staging / "Dino" / weights.name)
        source_manifest = {
            "source_root": str(source_root),
            "checkpoint_sha256": sha256_file(checkpoint),
            "dinov3_weights_sha256": sha256_file(weights),
        }
        (staging / "source_manifest.json").write_text(
            json.dumps(source_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    detector = FAMILY_ROOT / "artifacts" / "detector"
    detector.mkdir(parents=True, exist_ok=True)
    checkpoint_target = detector / "model_residual_v2.pth"
    checkpoint_staging = detector / (
        f".model_residual_v2.staging-{os.getpid()}.pth"
    )
    shutil.copy2(checkpoint, checkpoint_staging)
    os.replace(checkpoint_staging, checkpoint_target)
    (detector / "checkpoint_provenance.json").write_text(
        json.dumps(
            {
                "source": str(checkpoint),
                "packaged": str(checkpoint_target),
                "size": checkpoint_target.stat().st_size,
                "sha256": sha256_file(checkpoint_target),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = arguments()
    source_root = args.source_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()
    output = (
        args.output
        if args.output is not None
        else FAMILY_ROOT / "artifacts" / "trt" / SUPPORTED_PROFILES[args.batch_size]
    ).expanduser().resolve()
    required = (
        source_root / "face_dino_v1" / "scripts" / "build_backbone_tensorrt.py",
        source_root / "face_dino_v1" / "scripts" / "build_transformer_tensorrt.py",
        source_root / "face_dino_v1" / "scripts" / "build_attribute_tensorrt.py",
        source_root / "face_dino_v1" / "scripts" / "build_fused_preprocess.py",
        checkpoint,
        reference_root / "optimization" / "fast_engine_build.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required build inputs missing: " + ", ".join(missing))
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists; pass --overwrite to replace it: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    build = staging / "_build"
    build.mkdir()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MAX_JOBS"] = "1"
    environment_bin = str(Path(sys.executable).resolve().parent)
    environment["PATH"] = (
        environment_bin
        + os.pathsep
        + environment.get("PATH", "")
    )
    environment.setdefault(
        "CUDA_HOME",
        str(Path(sys.executable).resolve().parents[1]),
    )
    started = time.perf_counter()
    try:
        backbone = build / "backbone"
        transformer = build / "transformer"
        attribute = build / "attribute"
        preprocess = build / "preprocess"
        for directory in (backbone, transformer, attribute, preprocess):
            directory.mkdir()

        _run(
            [
                sys.executable,
                str(required[0]),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(backbone),
                "--height",
                "736",
                "--width",
                "1280",
                "--min-batch",
                "1",
                "--optimal-batch",
                str(args.batch_size),
                "--max-batch",
                str(args.batch_size),
                "--workspace-gib",
                str(args.workspace_gib),
                "--with-neck",
            ],
            cwd=source_root,
            environment=environment,
        )
        _run(
            [
                sys.executable,
                str(Path(__file__).with_name("run_transformer_builder.py")),
                "--script",
                str(required[1]),
                "--batch-size",
                str(args.batch_size),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(transformer),
                "--workspace-gib",
                str(args.workspace_gib),
                "--reference-root",
                str(reference_root),
            ],
            cwd=source_root,
            environment=environment,
        )
        _run(
            [
                sys.executable,
                str(required[2]),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(attribute),
                "--min-rois",
                "1",
                "--optimal-rois",
                "32",
                "--max-rois",
                "256",
                "--workspace-gib",
                str(min(args.workspace_gib, 8)),
            ],
            cwd=source_root,
            environment=environment,
        )
        preprocess_plugin = staging / PLUGIN_FILES["preprocess_plugin"]
        _run(
            [
                sys.executable,
                str(required[3]),
                "--build-dir",
                str(preprocess),
                "--output",
                str(preprocess_plugin),
            ],
            cwd=source_root,
            environment=environment,
        )

        sources = {
            "backbone_neck": (
                backbone
                / "backbone_neck_736x1280_normsoftmax_fp32.engine"
            ),
            "query_encoder": transformer / "engines" / "query_encoder.engine",
            "decoder": transformer / "engines" / "decoder.engine",
            "attribute": attribute / "attribute_fp16.engine",
            "msda_plugin": (
                transformer
                / "plugins"
                / "codino_msda_direct_mh0_sm120.so"
            ),
            "preprocess_plugin": preprocess_plugin,
        }
        missing_outputs = [
            str(path) for path in sources.values() if not path.is_file()
        ]
        if missing_outputs:
            raise RuntimeError(
                "engine build did not create: " + ", ".join(missing_outputs)
            )
        artifacts: dict[str, dict[str, object]] = {}
        for name, relative in {**ENGINE_FILES, **PLUGIN_FILES}.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = sources[name]
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            artifacts[name] = _record(
                destination,
                bundle_path=relative,
            )

        manifest = {
            "schema": SCHEMA,
            "profile": SUPPORTED_PROFILES[args.batch_size],
            "status": "complete",
            "batch_size": args.batch_size,
            "input_shape": [args.batch_size, 3, 736, 1280],
            "precision": {
                "backbone_neck": "mixed_fp16_fp32",
                "query_encoder": "mixed_fp16_fp32",
                "decoder": "fp32",
                "attribute": "fp16",
            },
            "checkpoint": {
                "source": str(checkpoint),
                "size": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
            },
            "artifacts": artifacts,
            "build": {
                "elapsed_seconds": time.perf_counter() - started,
                "python": sys.version,
                "platform": platform.platform(),
                "source_root": str(source_root),
                "reference_root": str(reference_root),
                "max_jobs": 1,
                "workspace_gib": args.workspace_gib,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not args.keep_build_files:
            shutil.rmtree(build)
        if not args.skip_runtime_snapshot:
            _snapshot_runtime(source_root, checkpoint)
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
        print(
            json.dumps(
                {
                    "bundle": str(output / "manifest.json"),
                    "checkpoint": str(
                        FAMILY_ROOT
                        / "artifacts"
                        / "detector"
                        / "model_residual_v2.pth"
                    ),
                    "runtime_source": str(
                        FAMILY_ROOT
                        / ".runtime"
                        / "src"
                        / "face_detection"
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
