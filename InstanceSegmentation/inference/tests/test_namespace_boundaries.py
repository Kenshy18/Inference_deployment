from __future__ import annotations

from pathlib import Path
import tarfile


def test_shared_runtime_has_one_explicit_namespace() -> None:
    inference_root = Path(__file__).resolve().parents[1]
    retired_top_level_names = (
        "contracts",
        "mask_geometry",
        "orchestration",
        "persistence",
        "pipelines",
        "video",
    )
    assert not [
        name for name in retired_top_level_names if (inference_root / name).exists()
    ]


def test_repository_and_inference_orchestration_do_not_alias() -> None:
    import inference_core.execution
    import orchestration

    inference_path = Path(inference_core.execution.__file__).resolve()
    repository_path = Path(orchestration.__file__).resolve()
    assert inference_path != repository_path
    assert "inference_core/execution" in inference_path.as_posix()
    assert repository_path.parent.name == "orchestration"


def test_standalone_archives_publish_the_namespaced_runtime() -> None:
    inference_root = Path(__file__).resolve().parents[1]
    archives = sorted(inference_root.glob("*/vendor/inference_common.tar.gz"))
    assert len(archives) == 5
    expected_payload: bytes | None = None
    for archive in archives:
        payload = archive.read_bytes()
        if expected_payload is None:
            expected_payload = payload
        assert payload == expected_payload
        with tarfile.open(archive, "r:gz") as bundle:
            members = set(bundle.getnames())
        assert "inference_core/__init__.py" in members
        assert "inference_core/contracts/__init__.py" in members
        assert "contracts/__init__.py" not in members
