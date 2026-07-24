#!/usr/bin/env python3
"""Prepare and verify the self-contained RT-DETR Head/Face runtime folder."""

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
SOURCE_ROOT = RUNTIME_ROOT / "src" / "RT-DETRv4"
SOURCE_ARCHIVE = FAMILY_ROOT / "vendor" / "RT-DETRv4.tar.gz"
LOCAL_SITE_PACKAGES = RUNTIME_ROOT / "site-packages"
DEPENDENCY_ARCHIVE = FAMILY_ROOT / "vendor" / "faster_coco_eval_cp310_linux_x86_64.tar.gz"
SHARED_ROOT = RUNTIME_ROOT / "shared"
SHARED_ARCHIVE = FAMILY_ROOT / "vendor" / "inference_common.tar.gz"
LOCK_OUTPUT = RUNTIME_ROOT / "environment-lock.json"
REQUIREMENTS = FAMILY_ROOT / "requirements.txt"
CHECKPOINT = (
    FAMILY_ROOT / "artifacts" / "detector" / "head_face_best_stg1.pth"
)
MODULES = (
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "scipy",
    "yaml",
    "faster_coco_eval",
    "engine.core",
    "contracts",
    "video",
    "persistence",
    "pipelines",
)
PROBE = r"""
import importlib, json, sys, types
for source in reversed(sys.argv[1:]):
    sys.path.insert(0, source)
try:
    import calflops
except ModuleNotFoundError:
    calflops = types.ModuleType("calflops")
    def unavailable(*args, **kwargs):
        raise RuntimeError("calflops is unavailable in the inference-only runtime")
    calflops.calculate_flops = unavailable
    sys.modules["calflops"] = calflops
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
    result.add_argument("--framework-source", type=Path)
    result.add_argument("--extra-site-packages", action="append", default=[], type=Path)
    result.add_argument("--clone-to", type=Path)
    result.add_argument("--install", action="store_true")
    result.add_argument(
        "--install-local",
        action="store_true",
        help="Install Python requirements into .runtime/site-packages.",
    )
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


def extract_source(*, force: bool) -> Path:
    if SOURCE_ROOT.is_dir() and not force:
        return SOURCE_ROOT
    if not SOURCE_ARCHIVE.is_file():
        raise FileNotFoundError(
            f"bundled framework archive not found: {SOURCE_ARCHIVE}"
        )
    if SOURCE_ROOT.exists():
        shutil.rmtree(SOURCE_ROOT)
    SOURCE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(SOURCE_ARCHIVE, "r:gz") as bundle:
        root = SOURCE_ROOT.parent.resolve()
        for member in bundle.getmembers():
            target = (root / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        bundle.extractall(SOURCE_ROOT.parent)
    if not (SOURCE_ROOT / "engine").is_dir():
        raise RuntimeError(f"archive did not create expected source: {SOURCE_ROOT}")
    return SOURCE_ROOT


def extract_local_dependencies(*, force: bool) -> Path:
    package = LOCAL_SITE_PACKAGES / "faster_coco_eval"
    if package.is_dir() and not force:
        return LOCAL_SITE_PACKAGES
    if not DEPENDENCY_ARCHIVE.is_file():
        raise FileNotFoundError(
            f"bundled inference dependency archive not found: {DEPENDENCY_ARCHIVE}"
        )
    if force and package.exists():
        shutil.rmtree(package)
    LOCAL_SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
    with tarfile.open(DEPENDENCY_ARCHIVE, "r:gz") as bundle:
        root = LOCAL_SITE_PACKAGES.resolve()
        for member in bundle.getmembers():
            target = (root / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        bundle.extractall(LOCAL_SITE_PACKAGES)
    if not package.is_dir():
        raise RuntimeError(
            f"dependency archive did not create expected package: {package}"
        )
    return LOCAL_SITE_PACKAGES


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


def validate_checkpoint() -> dict[str, object]:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"required checkpoint not found: {CHECKPOINT}")
    with CHECKPOINT.open("rb") as stream:
        modern_archive = stream.read(4).startswith(b"PK")
    if modern_archive:
        try:
            with zipfile.ZipFile(CHECKPOINT):
                pass
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"corrupt checkpoint: {CHECKPOINT}") from exc
    return {"path": str(CHECKPOINT.resolve()), "size": CHECKPOINT.stat().st_size}


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    runtime_python = args.runtime_python.expanduser().resolve()
    if not runtime_python.is_file():
        parser().error(f"runtime Python not found: {runtime_python}")
    if args.clone_to is not None:
        runtime_python = clone_environment(
            runtime_python, args.clone_to.expanduser().resolve()
        )
    if args.install and args.install_local:
        parser().error("--install and --install-local are mutually exclusive")
    if args.install:
        subprocess.run(
            [str(runtime_python), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
            check=True,
        )
    if args.install_local:
        LOCAL_SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(runtime_python),
                "-m",
                "pip",
                "install",
                "--target",
                str(LOCAL_SITE_PACKAGES),
                "-r",
                str(REQUIREMENTS),
            ],
            check=True,
        )
    source = (
        args.framework_source.expanduser().resolve()
        if args.framework_source is not None
        else extract_source(force=args.force_source)
    )
    if not (source / "engine").is_dir() or not (source / "configs").is_dir():
        parser().error(f"framework source must contain engine/ and configs/: {source}")
    extract_local_dependencies(force=args.force_source)
    shared = extract_shared(force=args.force_source)
    extras = [path.expanduser().resolve() for path in args.extra_site_packages]
    if LOCAL_SITE_PACKAGES.is_dir():
        extras.append(LOCAL_SITE_PACKAGES.resolve())
    completed = subprocess.run(
        [
            str(runtime_python),
            "-I",
            "-c",
            PROBE,
            str(source),
            str(shared),
            *map(str, extras),
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
            "schema": "rtdetr-head-face-standalone-v2",
            "family_root": str(FAMILY_ROOT),
            "framework_source": str(source),
            "extra_site_packages": [str(path) for path in extras],
            "shared_runtime": str(shared),
            "artifacts": {"detector": validate_checkpoint()},
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
