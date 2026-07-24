#!/usr/bin/env python3
"""Build and validate one portable EVA-02 TensorRT-backbone bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

TRT_ROOT = Path(__file__).resolve().parent
FAMILY_ROOT = TRT_ROOT.parent
DEFAULT_CHECKPOINT = FAMILY_ROOT / "artifacts" / "detector" / "model_final.pth"
DEFAULT_CLASSIFIER = FAMILY_ROOT / "artifacts" / "classifier" / "best.pt"
DEFAULT_CONFIG = FAMILY_ROOT / "instance_segmentation" / "lazy_config.py"
DEFAULT_FRAMEWORK = FAMILY_ROOT / ".runtime" / "src" / "detectron2_root"
DEFAULT_LOCK = FAMILY_ROOT / ".runtime" / "environment-lock.json"
DEFAULT_OUTPUT = (
    FAMILY_ROOT / "artifacts" / "trt" / "eva02-vit-dynamic-b1-20-fp16-v1"
)

if str(FAMILY_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(FAMILY_ROOT.parent))

try:
    from .bundle import MANIFEST_SCHEMA, PROFILE, file_record, load_trt_bundle
except ImportError:
    from bundle import MANIFEST_SCHEMA, PROFILE, file_record, load_trt_bundle

EXPORTER = TRT_ROOT / "export_backbone.py"
ENGINE_BUILDER = TRT_ROOT / "engine_build.py"
VALIDATOR = TRT_ROOT / "validate_engine.py"
BUILDER_SOURCES = (
    Path(__file__).resolve(),
    TRT_ROOT / "bundle.py",
    TRT_ROOT / "runtime.py",
    EXPORTER,
    ENGINE_BUILDER,
    VALIDATOR,
)


def _regular_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--framework-source", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--environment-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-size", type=int, default=1280)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=12)
    parser.add_argument("--max-batch", type=int, default=20)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--optimization-level", type=int, default=2)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--drop-block-indices", default="19,21,22")
    parser.add_argument("--max-mean-abs", type=float, default=0.08)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    return parser


def _environment(framework: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": os.pathsep.join((str(framework), str(FAMILY_ROOT.parent))),
        }
    )
    return environment


def _run(command: list[str], environment: dict[str, str]) -> None:
    print("[EVA02-TRT]", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=FAMILY_ROOT, env=environment)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalize_retained_reports(
    export_report: dict[str, object],
    build_report: dict[str, object],
) -> None:
    for report in (export_report, build_report):
        onnx = report.get("onnx")
        if isinstance(onnx, dict):
            onnx["path"] = "temporary/eva02_backbone.onnx (not retained)"
    engine = build_report.get("engine")
    if isinstance(engine, dict):
        engine["path"] = "engines/eva02_backbone.engine"


def _write_manifest(
    *,
    root: Path,
    checkpoint: Path,
    classifier: Path,
    config: Path,
    runtime_python: Path,
    environment_lock: Path,
    export_report: dict[str, object],
    build_report: dict[str, object],
    validation_report: dict[str, object],
    target_size: int,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    drop_block_indices: str,
) -> Path:
    io = build_report.get("io")
    if not isinstance(io, dict):
        raise ValueError("TensorRT build report has no io contract")
    lock_snapshot = root / "metadata" / "environment-lock.json"
    lock_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(environment_lock, lock_snapshot)
    payload = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": PROFILE,
        "status": "complete",
        "production_registered": True,
        "precision": "fp16",
        "scope": {
            "tensorrt": "pruned EVA-02 ViT backbone",
            "pytorch": "SimpleFeaturePyramid, Cascade ROI heads, ROI classifier",
        },
        "shape_profile": {
            "target_size": target_size,
            "min_batch": min_batch,
            "opt_batch": opt_batch,
            "max_batch": max_batch,
        },
        "io": io,
        "drop_block_indices": [
            int(value) for value in drop_block_indices.split(",") if value
        ],
        "engine": file_record(
            root / "engines" / "eva02_backbone.engine",
            stored_path="engines/eva02_backbone.engine",
        ),
        "source": {
            "checkpoint": file_record(checkpoint),
            "classifier_checkpoint": file_record(classifier),
            "config": file_record(config),
            "builder_sources": {
                str(path.relative_to(FAMILY_ROOT)): file_record(path)
                for path in BUILDER_SOURCES
            },
        },
        "runtime_python": file_record(runtime_python),
        "environment_lock": file_record(
            lock_snapshot,
            stored_path="metadata/environment-lock.json",
        ),
        "reports": {
            "export": export_report,
            "build": build_report,
        },
        "validation": validation_report,
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
    if not (1 <= args.min_batch <= args.opt_batch <= args.max_batch):
        parser.error("batch profile must satisfy 1 <= min <= opt <= max")
    if args.target_size <= 0 or args.workspace_gb <= 0 or args.opset <= 0:
        parser.error("target size, workspace and opset must be positive")
    if not 0 <= args.optimization_level <= 5:
        parser.error("--optimization-level must be in [0, 5]")

    python = _regular_file(args.runtime_python, "runtime Python")
    checkpoint = _regular_file(args.checkpoint, "detector checkpoint")
    classifier = _regular_file(
        args.classifier_checkpoint, "classifier checkpoint"
    )
    config = _regular_file(args.config, "model config")
    lock = _regular_file(args.environment_lock, "environment lock")
    framework = args.framework_source.expanduser().resolve()
    if not (framework / "detectron2").is_dir():
        parser.error(f"framework source must contain detectron2/: {framework}")
    for source in BUILDER_SOURCES:
        _regular_file(source, "builder source")

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"bundle output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    work = Path(tempfile.mkdtemp(prefix=".eva02-onnx-", dir=output.parent))
    environment = _environment(framework)
    try:
        onnx = work / "eva02_backbone.onnx"
        export_report_path = staging / "reports" / "export.json"
        build_report_path = staging / "reports" / "build.json"
        validation_report_path = staging / "reports" / "validation.json"
        engine = staging / "engines" / "eva02_backbone.engine"
        common = [
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config),
            "--framework-source",
            str(framework),
            "--target-size",
            str(args.target_size),
            "--drop-block-indices",
            args.drop_block_indices,
        ]
        _run(
            [
                str(python),
                str(EXPORTER),
                *common,
                "--output",
                str(onnx),
                "--report",
                str(export_report_path),
                "--opset",
                str(args.opset),
            ],
            environment,
        )
        _run(
            [
                str(python),
                str(ENGINE_BUILDER),
                "--onnx",
                str(onnx),
                "--engine",
                str(engine),
                "--report",
                str(build_report_path),
                "--target-size",
                str(args.target_size),
                "--min-batch",
                str(args.min_batch),
                "--opt-batch",
                str(args.opt_batch),
                "--max-batch",
                str(args.max_batch),
                "--workspace-gb",
                str(args.workspace_gb),
                "--optimization-level",
                str(args.optimization_level),
            ],
            environment,
        )
        _run(
            [
                str(python),
                str(VALIDATOR),
                *common,
                "--engine",
                str(engine),
                "--report",
                str(validation_report_path),
                "--batches",
                f"{args.min_batch},{args.opt_batch},{args.max_batch}",
                "--max-mean-abs",
                str(args.max_mean_abs),
                "--min-cosine",
                str(args.min_cosine),
            ],
            environment,
        )
        export_report = _load_json(export_report_path)
        build_report = _load_json(build_report_path)
        validation_report = _load_json(validation_report_path)
        _normalize_retained_reports(export_report, build_report)
        _write_json(export_report_path, export_report)
        _write_json(build_report_path, build_report)
        manifest = _write_manifest(
            root=staging,
            checkpoint=checkpoint,
            classifier=classifier,
            config=config,
            runtime_python=python,
            environment_lock=lock,
            export_report=export_report,
            build_report=build_report,
            validation_report=validation_report,
            target_size=args.target_size,
            min_batch=args.min_batch,
            opt_batch=args.opt_batch,
            max_batch=args.max_batch,
            drop_block_indices=args.drop_block_indices,
        )
        load_trt_bundle(
            manifest,
            verify="full",
            checkpoint_path=checkpoint,
            classifier_checkpoint=classifier,
            config_path=config,
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"[PASS] EVA-02 TensorRT bundle: {output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
