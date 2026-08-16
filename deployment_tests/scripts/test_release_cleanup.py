from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deployment" / "cleanup_release_work.sh"
WINDOWS_TEST_ROOT = Path("/mnt/c/MaskPipelineQA/release-cleanup-unit-tests")


def _run(target: Path, allowed_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), str(target), str(allowed_root)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_cleanup_accepts_only_direct_mask_pipeline_child() -> None:
    if not Path("/mnt/c").is_dir():
        pytest.skip("WSL Windows mount is unavailable")
    WINDOWS_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=WINDOWS_TEST_ROOT) as directory:
        allowed_root = Path(directory) / "custom-work-root"
        target = allowed_root / "mask-pipeline-test-release"
        target.mkdir(parents=True)
        (target / "marker.txt").write_text("test\n", encoding="utf-8")

        result = _run(target, allowed_root)

        assert result.returncode == 0, result.stderr
        assert not target.exists()
        assert allowed_root.is_dir()


@pytest.mark.parametrize(
    ("target", "allowed_root"),
    [
        (Path("/mnt/c/safe/not-a-release"), Path("/mnt/c/safe")),
        (Path("/mnt/c/other/mask-pipeline-x"), Path("/mnt/c/safe")),
        (Path("/tmp/mask-pipeline-x"), Path("/tmp")),
        (Path("/mnt/c/mask-pipeline-x"), Path("/mnt/c")),
    ],
)
def test_cleanup_rejects_unsafe_scope(target: Path, allowed_root: Path) -> None:
    result = _run(target, allowed_root)

    assert result.returncode == 2
    assert "refusing" in result.stderr
