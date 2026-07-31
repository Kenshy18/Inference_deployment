from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from contracts import (
    ColorSpace,
    Frame,
    ModelDescriptor,
    TaskType,
    segmentation_frame_from_rows,
)
from live_preview import LivePreviewSink, render_preview
import live_preview


DESCRIPTOR = ModelDescriptor(
    model_id="preview-test",
    task=TaskType.INSTANCE_SEGMENTATION,
    implementation="test",
)


def sample(index: int = 0):
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :, 1] = 30
    frame = Frame(
        index=index,
        timestamp_sec=index / 30,
        image=image,
        color_space=ColorSpace.BGR,
    )
    result = segmentation_frame_from_rows(
        model=DESCRIPTOR,
        frame=frame,
        rows=[
            {
                "category_id": 0,
                "class_name": "foreground",
                "detector_score": 0.9,
                "bbox_xyxy": (100, 80, 300, 260),
                "polygons": ((100, 80, 300, 80, 300, 260, 100, 260),),
            }
        ],
    )
    return frame, result


def test_render_preview_has_exact_dimensions_and_overlay() -> None:
    frame, result = sample()
    rendered = render_preview(frame, result, width=960, height=540)

    assert rendered.shape == (540, 960, 3)
    assert rendered.dtype == np.uint8
    assert np.any(rendered[:, :, 2] > rendered[:, :, 1])


def test_sink_keeps_a_three_file_ring_and_skips_non_interval_frames(
    tmp_path: Path,
) -> None:
    sink = LivePreviewSink(
        tmp_path / "latest.jpg",
        phase="segmentation_inference",
        interval_frames=5,
        width=960,
        height=540,
        jpeg_quality=85,
        control_path=None,
    )
    for index in (1, 5, 10, 15):
        frame, result = sample(index)
        sink.submit(frame, result)
    sink.close()

    outputs = sorted(tmp_path.glob("latest-*.jpg"))
    assert len(outputs) <= 3
    assert outputs
    decoded = cv2.imread(str(outputs[-1]))
    assert decoded is not None
    assert decoded.shape == (540, 960, 3)


def test_sink_rate_gate_runs_before_resize(tmp_path: Path, monkeypatch) -> None:
    resize_calls = 0
    original = live_preview._fit_canvas

    def counted_fit_canvas(*args, **kwargs):
        nonlocal resize_calls
        resize_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(live_preview, "_fit_canvas", counted_fit_canvas)
    sink = LivePreviewSink(
        tmp_path / "latest.jpg",
        phase="face_inference",
        interval_frames=5,
        width=320,
        height=180,
        jpeg_quality=75,
        max_fps=1.0,
    )
    for index in (5, 10, 15, 20, 25):
        frame, result = sample(index)
        sink.submit(frame, result)
    sink.close()

    # Fast model batches may offer many eligible frames at once. Rejected
    # offers must not resize/copy a full source frame on the inference thread.
    assert resize_calls == 1
