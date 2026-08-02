"""Shared helpers for the external production asset contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


MANIFEST = Path(__file__).with_name("assets.production.json")
STAGES = {"source": 0, "runtime": 1}


def load_manifest() -> dict[str, Any]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported asset manifest: {MANIFEST}")
    return payload


def selected_artifacts(profile: str, stage: str) -> list[dict[str, Any]]:
    payload = load_manifest()
    if profile not in payload["profiles"]:
        raise ValueError(f"unknown profile {profile!r}")
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    return [
        item
        for item in payload["artifacts"]
        if profile in item["profiles"] and STAGES[item["stage"]] <= STAGES[stage]
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_included_files(root: Path, include: Iterable[str] | None) -> Iterable[Path]:
    if include is None:
        yield from sorted(path for path in root.rglob("*") if path.is_file())
        return
    seen: set[Path] = set()
    for relative in include:
        target = root / relative
        candidates = [target] if target.is_file() else sorted(target.rglob("*"))
        for path in candidates:
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def verify_artifact(root: Path, item: dict[str, Any], *, full_hash: bool) -> list[str]:
    path = root / item["path"]
    errors: list[str] = []
    if item.get("type", "file") == "directory":
        if not path.is_dir():
            return [f"{item['id']}: directory not found: {path}"]
        files = list(iter_included_files(path, item.get("include")))
        if not files:
            errors.append(f"{item['id']}: directory has no required files: {path}")
        return errors
    if not path.is_file() or path.is_symlink():
        return [f"{item['id']}: regular file not found: {path}"]
    if "size" in item and path.stat().st_size != item["size"]:
        errors.append(f"{item['id']}: size mismatch: {path}")
    if full_hash and item.get("sha256") and sha256_file(path) != item["sha256"]:
        errors.append(f"{item['id']}: SHA-256 mismatch: {path}")
    return errors
