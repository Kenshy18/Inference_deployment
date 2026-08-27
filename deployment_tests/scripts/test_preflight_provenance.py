from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployment.preflight import resolve_source_commit


def test_release_metadata_replaces_git_in_finalized_runtime(tmp_path: Path) -> None:
    root = tmp_path / "runtime-source"
    root.mkdir()
    metadata = tmp_path / "release.json"
    metadata.write_text(
        json.dumps({"release_commit": "0123456789abcdef" * 2 + "01234567"}),
        encoding="utf-8",
    )

    assert resolve_source_commit(root, metadata) == (
        "0123456789abcdef" * 2 + "01234567",
        "release-metadata",
    )


@pytest.mark.parametrize(
    "payload",
    [None, {}, {"release_commit": ""}, {"release_commit": 123}],
)
def test_finalized_runtime_requires_valid_release_provenance(
    tmp_path: Path,
    payload: object,
) -> None:
    root = tmp_path / "runtime-source"
    root.mkdir()
    metadata = tmp_path / "release.json"
    if payload is not None:
        metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError):
        resolve_source_commit(root, metadata)
