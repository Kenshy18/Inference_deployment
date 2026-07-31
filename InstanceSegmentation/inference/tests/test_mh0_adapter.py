from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from contracts import Frame, FrameBatch
from dinov3_codino_mh0.adapter import Mh0Adapter
from dinov3_codino_mh0.pipeline import run_mh0_video_inference
from persistence import SqliteWriter
from pipelines import run_video_inference
from video import VideoMetadata


def _mask(
    *,
    shape: tuple[int, int] = (12, 16),
    regions: tuple[tuple[slice, slice], ...] = (),
) -> np.ndarray:
    value = np.zeros(shape, dtype=np.uint8)
    for rows, columns in regions:
        value[rows, columns] = 1
    return value


def _raw_result(
    boxes: list[list[float]],
    masks: list[np.ndarray],
):
    return (
        [np.asarray(boxes, dtype=np.float32)],
        [masks],
    )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        backend="pytorch",
        device="cpu",
        fixed_batch_size=2,
        model=None,
        classifier=None,
        class_names=("女性器", "男性器", "結合部分"),
        class_ids=(1, 2, 3),
    )


class Mh0AdapterTest(unittest.TestCase):
    def test_preserves_foreground_detector_and_three_class_result(self) -> None:
        frames = FrameBatch.from_sequence(
            [
                Frame(
                    index=0,
                    timestamp_sec=0.0,
                    image=np.zeros((12, 16, 3), dtype=np.uint8),
                )
            ]
        )
        raw_results = [
            _raw_result(
                [[1, 2, 8, 10, 0.9, 1, 0.8, 0.1, 0.8, 0.1]],
                [_mask(regions=((slice(2, 10), slice(1, 8)),))],
            )
        ]
        adapter = Mh0Adapter(_runtime(), score_threshold=0.5)
        with patch(
            "dinov3_codino_mh0.adapter.infer",
            return_value=raw_results,
        ):
            result = adapter.predict(frames)[0].instances[0].detection
        self.assertEqual((result.class_id, result.class_name), (0, "foreground"))
        self.assertIsNotNone(result.classification)
        assert result.classification is not None
        self.assertEqual(
            (result.classification.class_id, result.classification.class_name),
            (2, "男性器"),
        )
        self.assertAlmostEqual(result.classification.score, 0.8, places=6)
        self.assertEqual(len(result.classification.probabilities or ()), 3)

    def test_normalizes_order_clipping_threshold_and_masks(self) -> None:
        frames = FrameBatch.from_sequence(
            [
                Frame(
                    index=7,
                    timestamp_sec=0.25,
                    image=np.zeros((12, 16, 3), dtype=np.uint8),
                ),
                Frame(
                    index=9,
                    timestamp_sec=0.5,
                    image=np.zeros((12, 16, 3), dtype=np.uint8),
                ),
            ]
        )
        disconnected = _mask(
            regions=(
                (slice(1, 4), slice(1, 4)),
                (slice(7, 11), slice(10, 15)),
            )
        )
        raw_results = [
            _raw_result(
                [[-4, -2, 20, 18, 0.9]],
                [disconnected],
            ),
            _raw_result(
                [
                    [2, 2, 8, 8, 0.8],
                    [0, 0, 4, 4, 0.49],
                ],
                [
                    _mask(),
                    _mask(regions=((slice(0, 4), slice(0, 4)),)),
                ],
            ),
        ]
        adapter = Mh0Adapter(_runtime(), score_threshold=0.5)

        with patch(
            "dinov3_codino_mh0.adapter.infer",
            return_value=raw_results,
        ):
            results = adapter.predict(frames)

        self.assertEqual([result.frame.index for result in results], [7, 9])
        self.assertEqual([len(result.instances) for result in results], [1, 1])
        clipped = results[0].instances[0].detection.bbox
        self.assertEqual(
            (clipped.x1, clipped.y1, clipped.x2, clipped.y2),
            (0.0, 0.0, 16.0, 12.0),
        )
        self.assertEqual(
            len(results[0].instances[0].segmentation.polygons),
            2,
        )
        self.assertEqual(
            results[1].instances[0].segmentation.polygons,
            (),
        )
        self.assertAlmostEqual(
            results[1].instances[0].detection.score,
            0.8,
            places=6,
        )

    def test_common_pipeline_persists_fake_runtime_results(self) -> None:
        batch = FrameBatch.from_sequence(
            [
                Frame(
                    index=0,
                    timestamp_sec=0.0,
                    image=np.zeros((12, 16, 3), dtype=np.uint8),
                ),
                Frame(
                    index=1,
                    timestamp_sec=1 / 30,
                    image=np.zeros((12, 16, 3), dtype=np.uint8),
                ),
            ]
        )
        raw_results = [
            _raw_result(
                [[1, 1, 8, 8, 0.9]],
                [
                    _mask(
                        regions=(
                            (slice(1, 4), slice(1, 4)),
                            (slice(7, 10), slice(10, 14)),
                        )
                    )
                ],
            ),
            _raw_result(
                [[2, 2, 6, 6, 0.75]],
                [_mask()],
            ),
        ]

        class FakeDecoder:
            def __init__(self, *args, **kwargs) -> None:
                del args, kwargs
                self.metadata = VideoMetadata(
                    frames=2,
                    fps=30.0,
                    width=16,
                    height=12,
                )

            def __iter__(self):
                yield batch

            def close(self) -> None:
                return None

        adapter = Mh0Adapter(_runtime(), score_threshold=0.5)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mh0.sqlite"
            writer = SqliteWriter(output)
            with (
                patch(
                    "dinov3_codino_mh0.adapter.infer",
                    return_value=raw_results,
                ),
                patch(
                    "pipelines.inference.AsyncVideoDecoder",
                    FakeDecoder,
                ),
            ):
                summary = run_video_inference(
                    input_path=Path("fake.mp4"),
                    adapter=adapter,
                    writer=writer,
                    batch_size=2,
                    max_frames=None,
                    warmup_frames=0,
                )

            self.assertEqual(summary.processed_frames, 2)
            self.assertEqual(summary.result_items, 2)
            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM frames"
                    ).fetchone()[0],
                    2,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM detections"
                    ).fetchone()[0],
                    2,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM segmentations"
                    ).fetchone()[0],
                    2,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM segmentation_polygons"
                    ).fetchone()[0],
                    2,
                )
                self.assertEqual(
                    [
                        row[0]
                        for row in connection.execute(
                            "SELECT frame_index FROM frames ORDER BY frame_index"
                        )
                    ],
                    [0, 1],
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM metadata WHERE key='model_id'"
                    ).fetchone()[0],
                    "dinov3_codino_mh0_pytorch",
                )
                self.assertEqual(
                    connection.execute(
                        "PRAGMA integrity_check"
                    ).fetchone()[0],
                    "ok",
                )

    def test_overlapped_pipeline_preserves_source_order(self) -> None:
        batches = tuple(
            FrameBatch.from_sequence(
                [
                    Frame(
                        index=index,
                        timestamp_sec=index / 30,
                        image=np.zeros((12, 16, 3), dtype=np.uint8),
                    )
                ]
            )
            for index in range(3)
        )
        raw_by_frame = {
            index: [
                _raw_result(
                    [[index, 1, index + 3, 5, 0.9]],
                    [
                        _mask(
                            regions=(
                                (slice(1, 5), slice(index, index + 3)),
                            )
                        )
                    ],
                )
            ]
            for index in range(3)
        }

        class FakeDecoder:
            def __init__(self, *args, **kwargs) -> None:
                del args, kwargs
                self.metadata = VideoMetadata(
                    frames=3,
                    fps=30.0,
                    width=16,
                    height=12,
                )

            def __iter__(self):
                yield from batches

            def close(self) -> None:
                return None

        adapter = Mh0Adapter(_runtime(), score_threshold=0.5)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mh0-overlapped.sqlite"
            writer = SqliteWriter(output)

            def fake_infer(_runtime, images):
                index = int(images[0][0, 0, 0])
                return raw_by_frame[index]

            for index, batch in enumerate(batches):
                batch.frames[0].image[0, 0, 0] = index
            with (
                patch(
                    "dinov3_codino_mh0.adapter.infer",
                    side_effect=fake_infer,
                ),
                patch(
                    "dinov3_codino_mh0.pipeline.AsyncVideoDecoder",
                    FakeDecoder,
                ),
            ):
                summary = run_mh0_video_inference(
                    input_path=Path("fake.mp4"),
                    adapter=adapter,
                    writer=writer,
                    batch_size=1,
                    max_frames=None,
                    warmup_frames=0,
                    output_queue_batches=2,
                )

            self.assertEqual(summary.processed_frames, 3)
            self.assertEqual(summary.result_items, 3)
            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    [
                        row[0]
                        for row in connection.execute(
                            "SELECT frame_index FROM frames ORDER BY rowid"
                        )
                    ],
                    [0, 1, 2],
                )
                self.assertEqual(
                    connection.execute(
                        "PRAGMA integrity_check"
                    ).fetchone()[0],
                    "ok",
                )


if __name__ == "__main__":
    unittest.main()
