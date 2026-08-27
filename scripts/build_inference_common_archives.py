#!/usr/bin/env python3
"""Build deterministic standalone-model snapshots of ``inference_core``."""

from __future__ import annotations

import argparse
from io import BytesIO
import gzip
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_ROOT = ROOT / "InstanceSegmentation" / "inference"
COMMON_ROOT = INFERENCE_ROOT / "inference_core"
INCLUDED = (
    Path("__init__.py"),
    Path("contracts"),
    Path("live_preview.py"),
    Path("mask_geometry"),
    Path("persistence"),
    Path("pipelines"),
    Path("progress_protocol.py"),
    Path("registry.py"),
    Path("video"),
)
ARCHIVES = tuple(
    INFERENCE_ROOT / family / "vendor" / "inference_common.tar.gz"
    for family in (
        "dinov3_cascade",
        "dinov3_codino",
        "dinov3_codino_mh0",
        "eva02_cascade",
        "rtdetr_head_face",
    )
)


def _source_files() -> list[Path]:
    files: set[Path] = set()
    for relative in INCLUDED:
        source = COMMON_ROOT / relative
        if source.is_file():
            files.add(source)
            continue
        if not source.is_dir():
            raise FileNotFoundError(f"common runtime source not found: {source}")
        files.update(
            path
            for path in source.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    return sorted(files, key=lambda path: path.relative_to(INFERENCE_ROOT).as_posix())


def build_archive_bytes() -> bytes:
    compressed = BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0) as gzip_stream:
        with tarfile.open(fileobj=gzip_stream, mode="w", format=tarfile.PAX_FORMAT) as bundle:
            for source in _source_files():
                archive_name = source.relative_to(INFERENCE_ROOT).as_posix()
                info = bundle.gettarinfo(str(source), arcname=archive_name)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                with source.open("rb") as stream:
                    bundle.addfile(info, stream)
    return compressed.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if a checked-in archive differs from the source tree",
    )
    args = parser.parse_args()
    payload = build_archive_bytes()
    stale: list[Path] = []
    for archive in ARCHIVES:
        if args.check:
            if not archive.is_file() or archive.read_bytes() != payload:
                stale.append(archive)
            continue
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(payload)
        print(f"wrote {archive.relative_to(ROOT)} ({len(payload)} bytes)")
    if stale:
        rendered = "\n".join(str(path.relative_to(ROOT)) for path in stale)
        parser.error(f"stale inference common archive(s):\n{rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
