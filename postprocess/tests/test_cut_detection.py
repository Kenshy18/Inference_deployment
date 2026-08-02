from __future__ import annotations

from pathlib import Path

from cut_detection import detector


def test_resolve_ffmpeg_uses_repository_runtime_when_external_tools_are_absent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "repo" / "postprocess" / "cut_detection" / "detector.py"
    repository_ffmpeg = (
        tmp_path
        / "repo"
        / "overlay"
        / ".runtime"
        / "ffmpeg-nvenc-btbn-8.1"
        / "bin"
        / "ffmpeg"
    )
    repository_ffmpeg.parent.mkdir(parents=True)
    repository_ffmpeg.touch()

    monkeypatch.delenv("VIDEO_MASK_FFMPEG", raising=False)
    monkeypatch.setattr(detector.shutil, "which", lambda _name: None)
    monkeypatch.setattr(detector, "__file__", str(module_path))
    monkeypatch.setattr(detector.sys, "executable", str(tmp_path / "python"))

    assert detector._resolve_ffmpeg() == repository_ffmpeg
