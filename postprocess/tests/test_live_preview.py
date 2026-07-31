from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from common.live_preview import PostprocessPreviewSink, PreviewGeometry


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        30.0,
        (320, 180),
    )
    assert writer.isOpened()
    for index in range(12):
        image = np.full((180, 320, 3), (index * 11) % 255, np.uint8)
        writer.write(image)
    writer.release()


def test_postprocess_preview_is_exact_size_and_emits_stage_metadata(
    tmp_path: Path,
    capsys,
) -> None:
    video = tmp_path / "source.avi"
    _video(video)
    sink = PostprocessPreviewSink(
        tmp_path / "preview" / "latest.jpg",
        video,
        width=160,
        height=90,
        max_fps=100.0,
    )
    sink.submit(
        PreviewGeometry(
            5,
            "tracking",
            "tracking + short-track filter",
            polygons=(((20.0, 20.0), (100.0, 20.0), (60.0, 80.0)),),
            boxes=((20.0, 20.0, 100.0, 80.0),),
            track_id="7",
            detail="active 1",
        )
    )
    deadline = time.monotonic() + 2.0
    images: list[Path] = []
    while time.monotonic() < deadline:
        images = list((tmp_path / "preview").glob("latest-*.jpg"))
        if images:
            break
        time.sleep(0.01)
    sink.close()
    assert images
    rendered = cv2.imread(str(images[0]))
    assert rendered.shape[:2] == (90, 160)
    line = next(
        value
        for value in capsys.readouterr().out.splitlines()
        if value.startswith("[live-preview] ")
    )
    payload = json.loads(line.split(" ", 1)[1])
    assert payload["phase"] == "postprocess"
    assert payload["stage"] == "tracking"
    assert payload["frame_index"] == 5
    assert payload["detail"] == "active 1"


def test_preview_queue_coalesces_by_stage(tmp_path: Path) -> None:
    video = tmp_path / "source.avi"
    _video(video)
    control = tmp_path / "enabled"
    control.write_text("1", encoding="utf-8")
    sink = PostprocessPreviewSink(
        tmp_path / "latest.jpg",
        video,
        max_fps=100.0,
        control_path=control,
    )
    for frame in range(1000):
        sink.submit(PreviewGeometry(frame % 12, "nms", "NMS"))
    # The contract is one pending value per stage, independent of video length.
    assert len(sink._pending) <= 1
    assert sink._dropped >= 900
    control.unlink()
    sink.close()
