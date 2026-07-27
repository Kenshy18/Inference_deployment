#!/usr/bin/env python3
"""Rebuild the fixed-B16 SM120 MH0 TensorRT bundle from a PyTorch checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from bundle import (
    ENGINE_FILES,
    PLUGIN_FILE,
    PROFILE,
    SCHEMA,
    load_engine_bundle,
    sha256_file,
)


TRT_ROOT = Path(__file__).resolve().parent
FAMILY_ROOT = TRT_ROOT.parent
OPT = FAMILY_ROOT / "optimization"
DEFAULT_CONFIG = FAMILY_ROOT / "artifacts" / "detector" / "resolved_config.py"
DEFAULT_CHECKPOINT = (
    FAMILY_ROOT
    / "artifacts"
    / "detector"
    / "video_pseudo_mh0_epoch6_ema_deploy.pth"
)
BUILD_SOURCES = (
    Path(__file__).resolve(),
    FAMILY_ROOT / "bootstrap.py",
    OPT / "build_trt_backbone.py",
    OPT / "export_trt_transformer.py",
    OPT / "export_trt_mask_head.py",
    OPT / "fast_engine_build.py",
    OPT / "native" / "msda_direct_mh0.cpp",
    OPT / "native" / "msda_direct_mh0.cu",
)


def _record(path: Path) -> dict[str, object]:
    return {
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _run(command: list[str], environment: dict[str, str]) -> None:
    print("[MH0-TRT]", " ".join(command), flush=True)
    subprocess.run(command, cwd=FAMILY_ROOT, env=environment, check=True)


def _write_manifest(
    output: Path,
    *,
    config: Path,
    checkpoint: Path,
) -> Path:
    artifacts = {
        name: _record(output / relative)
        for name, relative in ENGINE_FILES.items()
    }
    artifacts["plugin"] = _record(output / PLUGIN_FILE)
    payload = {
        "schema": SCHEMA,
        "profile": PROFILE,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_size": 16,
        "input_size": [16, 3, 736, 1280],
        "query_feature_shapes": [[92, 160], [46, 80], [23, 40]],
        "precision": {
            "backbone_neck": "BF16 with FP32 I/O",
            "query_encoder": "FP16 with protected FP32 reductions",
            "decoder": "FP32",
            "mask_head": "FP32",
        },
        "source": {
            "config": {
                "path": str(config),
                **_record(config),
            },
            "checkpoint": {
                "path": str(checkpoint),
                **_record(checkpoint),
            },
            "builder": {
                "path": str(Path(__file__).resolve()),
                **_record(Path(__file__).resolve()),
            },
            "builder_sources": {
                str(path.relative_to(FAMILY_ROOT)): _record(path)
                for path in BUILD_SOURCES
            },
            "runtime_source_archives": {
                name: _record(FAMILY_ROOT / "vendor" / name)
                for name in ("codino.tar.gz", "dinov3_root.tar.gz")
            },
        },
        "artifacts": artifacts,
    }
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    load_engine_bundle(manifest, verify="engines")
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    value.add_argument("--output-dir", required=True, type=Path)
    value.add_argument("--workspace-gb", type=int, default=12)
    value.add_argument(
        "--manifest-only",
        action="store_true",
        help="validate existing artifacts and write only manifest.json",
    )
    return value


def main() -> int:
    args = parser().parse_args()
    python = args.runtime_python.expanduser().resolve()
    config = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path, label in (
        (python, "runtime Python"),
        (config, "config"),
        (checkpoint, "checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.workspace_gb <= 0:
        raise ValueError("workspace must be positive")
    if args.manifest_only:
        manifest = _write_manifest(
            output,
            config=config,
            checkpoint=checkpoint,
        )
        print(f"[PASS] wrote {manifest}")
        return 0
    if output.exists():
        raise FileExistsError(
            f"output already exists: {output}; choose a new bundle directory"
        )

    onnx = output / "onnx"
    engines = output / "engines"
    plugins = output / "plugins"
    metadata = output / "metadata"
    for directory in (onnx, engines, plugins, metadata):
        directory.mkdir(parents=True, exist_ok=True)
    plugin = output / PLUGIN_FILE
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["CUDA_HOME"] = str(python.parent.parent)
    environment["PATH"] = (
        f"{python.parent}:{environment.get('PATH', '')}"
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(FAMILY_ROOT),
            str(FAMILY_ROOT / ".runtime" / "src" / "codino"),
            str(FAMILY_ROOT / ".runtime" / "src" / "dinov3_root"),
        ]
    )
    common = ["--config", str(config), "--checkpoint", str(checkpoint)]
    _run(
        [
            str(python),
            str(OPT / "build_trt_backbone.py"),
            *common,
            "--batch-size",
            "16",
            "--with-neck",
            "--omit-p6",
            "--onnx",
            str(onnx / "backbone_neck.onnx"),
            "--engine",
            str(engines / "backbone_neck.engine"),
            "--workspace-gb",
            str(args.workspace_gb),
        ],
        environment,
    )
    _run(
        [
            str(python),
            str(OPT / "export_trt_transformer.py"),
            *common,
            "--batch-size",
            "16",
            "--query-onnx",
            str(onnx / "query_encoder.onnx"),
            "--decoder-onnx",
            str(onnx / "decoder.onnx"),
        ],
        environment,
    )
    _run(
        [
            str(python),
            str(OPT / "export_trt_mask_head.py"),
            *common,
            "--num-rois",
            "16",
            "--onnx",
            str(onnx / "mask_head.onnx"),
        ],
        environment,
    )
    worker = OPT / "fast_engine_build.py"
    runtime_root = python.parent.parent
    _run(
        [
            str(python),
            str(worker),
            "--component",
            "plugin",
            "--runtime-root",
            str(runtime_root),
            "--native-cpp",
            str(OPT / "native" / "msda_direct_mh0.cpp"),
            "--native-cuda",
            str(OPT / "native" / "msda_direct_mh0.cu"),
            "--native-build-dir",
            str(output / "native-build"),
            "--plugin",
            str(plugin),
            "--report",
            str(metadata / "plugin.json"),
        ],
        environment,
    )
    for component in ("query_encoder", "decoder", "mask_head"):
        command = [
            str(python),
            str(worker),
            "--component",
            component,
            "--source-onnx",
            str(onnx / f"{component}.onnx"),
            "--build-onnx",
            str(onnx / f"{component}_trt.onnx"),
            "--engine",
            str(engines / f"{component}.engine"),
            "--timing-cache",
            str(metadata / f"{component}.cache"),
            "--workspace-gb",
            str(args.workspace_gb),
            "--report",
            str(metadata / f"{component}.json"),
        ]
        if component == "query_encoder":
            command.extend(["--plugin", str(plugin)])
        _run(command, environment)
    manifest = _write_manifest(output, config=config, checkpoint=checkpoint)
    print(f"[PASS] built {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
