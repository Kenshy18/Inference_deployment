from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from inference_core.contracts import (
    ColorSpace,
    Frame,
    ModelDescriptor,
    TaskType,
    segmentation_frame_from_rows,
)
from inference_core.live_preview import LivePreviewSink, render_preview
from inference_core import live_preview


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


def test_sink_caches_control_file_state_on_the_inference_hot_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    control = tmp_path / "preview.enabled"
    control.write_text("1\n", encoding="utf8")
    checks = 0
    original = Path.is_file

    def counted_is_file(path: Path) -> bool:
        nonlocal checks
        checks += 1
        return original(path)

    monkeypatch.setattr(Path, "is_file", counted_is_file)
    sink = LivePreviewSink(
        tmp_path / "latest.jpg",
        phase="segmentation_inference",
        interval_frames=1,
        width=320,
        height=180,
        jpeg_quality=75,
        max_fps=10.0,
        control_path=control,
    )
    for index in range(100):
        frame, result = sample(index)
        sink.submit(frame, result)
    sink.close()

    assert checks == 1


def test_sink_observes_control_file_changes_within_one_preview_period(
    tmp_path: Path,
) -> None:
    control = tmp_path / "preview.enabled"
    control.write_text("1\n", encoding="utf8")
    sink = LivePreviewSink(
        tmp_path / "latest.jpg",
        phase="segmentation_inference",
        interval_frames=1,
        width=320,
        height=180,
        jpeg_quality=75,
        max_fps=10.0,
        control_path=control,
    )
    try:
        assert sink._preview_is_enabled(0.0)
        control.unlink()
        assert sink._preview_is_enabled(0.099)
        assert not sink._preview_is_enabled(0.1)
        control.write_text("1\n", encoding="utf8")
        assert not sink._preview_is_enabled(0.199)
        assert sink._preview_is_enabled(0.2)
    finally:
        sink.close()
