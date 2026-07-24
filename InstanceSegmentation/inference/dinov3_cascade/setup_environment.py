#!/usr/bin/env python3
"""Prepare and verify the self-contained DINOv3 Cascade runtime folder."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

FAMILY_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = FAMILY_ROOT / ".runtime"
SOURCE_ROOT = RUNTIME_ROOT / "src"
DETECTRON2_SOURCE = SOURCE_ROOT / "detectron2_root"
DINOV3_SOURCE = SOURCE_ROOT / "dinov3_root"
VENDOR_ROOT = FAMILY_ROOT / "vendor"
SHARED_ROOT = RUNTIME_ROOT / "shared"
SHARED_ARCHIVE = VENDOR_ROOT / "inference_common.tar.gz"
LOCK_OUTPUT = RUNTIME_ROOT / "environment-lock.json"
REQUIREMENTS = FAMILY_ROOT / "requirements.txt"
ARTIFACTS = {
    "detector": FAMILY_ROOT / "artifacts" / "detector" / "model_final.pth",
    "backbone": FAMILY_ROOT
    / "artifacts"
    / "backbone"
    / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "classifier": FAMILY_ROOT / "artifacts" / "classifier" / "best.pt",
    "trt_backbone": FAMILY_ROOT
    / "artifacts"
    / "trt"
    / "dinov3_backbone_fp32_720x1280_dynamic_bf16_forced_b1_8_8.engine",
}
MODULES = (
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "onnx",
    "tensorrt",
    "detectron2",
    "dinov3",
    "contracts",
    "mask_geometry",
    "video",
    "persistence",
    "pipelines",
)
PROBE = r"""
import importlib, json, sys
for source in reversed(sys.argv[1:]):
    sys.path.insert(0, source)
modules, failures = {}, {}
for name in __MODULES__:
    try:
        module = importlib.import_module(name)
        modules[name] = {
            "version": str(getattr(module, "__version__", "unknown")),
            "file": str(getattr(module, "__file__", "")),
        }
    except BaseException as exc:
        failures[name] = f"{type(exc).__name__}: {exc}"
torch = importlib.import_module("torch") if "torch" not in failures else None
print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "modules": modules,
    "failures": failures,
    "cuda_available": bool(torch is not None and torch.cuda.is_available()),
    "gpu": None if torch is None or not torch.cuda.is_available() else torch.cuda.get_device_name(0),
}))
raise SystemExit(0 if not failures and torch.cuda.is_available() else 3)
""".replace("__MODULES__", repr(MODULES))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    result.add_argument("--detectron2-source", type=Path)
    result.add_argument("--dinov3-source", type=Path)
    result.add_argument("--clone-to", type=Path)
    result.add_argument("--install", action="store_true")
    result.add_argument("--force-source", action="store_true")
    result.add_argument("--lock-output", type=Path, default=LOCK_OUTPUT)
    return result


def clone_environment(source_python: Path, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(f"clone destination already exists: {destination}")
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError("conda is required by --clone-to")
    subprocess.run(
        [
            conda,
            "--no-plugins",
            "create",
            "--yes",
            "--offline",
            "--prefix",
            str(destination),
            "--clone",
            str(source_python.parent.parent),
        ],
        check=True,
    )
    return destination / "bin/python"


def extract_source(name: str, destination: Path, *, force: bool) -> Path:
    if destination.is_dir() and not force:
        return destination
    archive = VENDOR_ROOT / f"{name}.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(f"bundled framework archive not found: {archive}")
    if destination.exists():
        shutil.rmtree(destination)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        root = SOURCE_ROOT.resolve()
        for member in bundle.getmembers():
            target = (root / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        bundle.extractall(SOURCE_ROOT)
    if not destination.is_dir():
        raise RuntimeError(f"archive did not create expected source: {destination}")
    return destination


def extract_shared(*, force: bool) -> Path:
    if (SHARED_ROOT / "contracts").is_dir() and not force:
        return SHARED_ROOT
    if not SHARED_ARCHIVE.is_file():
        raise FileNotFoundError(f"shared runtime archive not found: {SHARED_ARCHIVE}")
    if SHARED_ROOT.exists():
        shutil.rmtree(SHARED_ROOT)
    SHARED_ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(SHARED_ARCHIVE, "r:gz") as bundle:
        root = SHARED_ROOT.resolve()
        for member in bundle.getmembers():
            target = (root / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        bundle.extractall(SHARED_ROOT)
    return SHARED_ROOT


def validate_artifacts() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, path in ARTIFACTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"required artifact not found: {path}")
        if path.suffix != ".engine":
            with path.open("rb") as stream:
                modern_archive = stream.read(4).startswith(b"PK")
            if modern_archive:
                try:
                    with zipfile.ZipFile(path):
                        pass
                except zipfile.BadZipFile as exc:
                    raise RuntimeError(f"corrupt checkpoint: {path}") from exc
        result[name] = {"path": str(path.resolve()), "size": path.stat().st_size}
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    runtime_python = args.runtime_python.expanduser().resolve()
    if not runtime_python.is_file():
        parser().error(f"runtime Python not found: {runtime_python}")
    if args.clone_to is not None:
        runtime_python = clone_environment(
            runtime_python, args.clone_to.expanduser().resolve()
        )
    if args.install:
        subprocess.run(
            [str(runtime_python), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
            check=True,
        )
    detectron2_source = (
        args.detectron2_source.expanduser().resolve()
        if args.detectron2_source is not None
        else extract_source(
            "detectron2_root", DETECTRON2_SOURCE, force=args.force_source
        )
    )
    dinov3_source = (
        args.dinov3_source.expanduser().resolve()
        if args.dinov3_source is not None
        else extract_source("dinov3_root", DINOV3_SOURCE, force=args.force_source)
    )
    if not (detectron2_source / "detectron2").is_dir():
        parser().error(
            f"Detectron2 source must contain detectron2/: {detectron2_source}"
        )
    if not (dinov3_source / "dinov3").is_dir():
        parser().error(f"DINOv3 source must contain dinov3/: {dinov3_source}")
    shared = extract_shared(force=args.force_source)
    completed = subprocess.run(
        [
            str(runtime_python),
            "-I",
            "-c",
            PROBE,
            str(detectron2_source),
            str(dinov3_source),
            str(shared),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if not completed.stdout.strip():
        raise RuntimeError(completed.stderr.strip() or "environment probe failed")
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    payload.update(
        {
            "schema": "dinov3-cascade-standalone-v2",
            "family_root": str(FAMILY_ROOT),
            "sources": [str(detectron2_source), str(dinov3_source)],
            "shared_runtime": str(shared),
            "artifacts": validate_artifacts(),
        }
    )
    output = args.lock_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if completed.returncode != 0:
        raise RuntimeError(f"environment probe failed; see {output}")
    print(f"[PASS] standalone environment: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
