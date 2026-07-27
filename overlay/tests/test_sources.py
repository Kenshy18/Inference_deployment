from __future__ import annotations

import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

from overlay_renderer.sources import (
    inspect_inference_source,
    inspect_mask_source,
    iter_face_frames,
    iter_mask_frames,
    iter_raw_segmentation_frames,
)

from helpers import (
    create_mask_sqlite,
    create_rich_face_sqlite,
    create_unified_sqlite,
)


class SourceTests(unittest.TestCase):
    def test_unified_inference_roles_are_read_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = create_unified_sqlite(Path(temporary) / "inference.sqlite")

            raw_info = inspect_inference_source(path, "instance_segmentation")
            face_info = inspect_inference_source(path, "face_detection")
            raw = list(iter_raw_segmentation_frames(path))
            faces = list(iter_face_frames(path))

            self.assertEqual(1, raw_info.item_count)
            self.assertEqual(1, face_info.item_count)
            self.assertEqual(0, raw[0].frame_index)
            self.assertEqual("sample", raw[0].items[0].label)
            self.assertEqual(1, faces[0].frame_index)
            self.assertEqual((18.0, 10.0, 42.0, 34.0), faces[0].items[0].box)

    def test_postprocess_mask_contract_is_stage_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = create_mask_sqlite(Path(temporary) / "masks.sqlite")

            info = inspect_mask_source(path)
            frames = list(iter_mask_frames(path))

            self.assertEqual("postprocess-mask-sqlite", info.schema)
            self.assertEqual(1, info.item_count)
            self.assertEqual("7", frames[0].items[0].track_id)
            self.assertEqual(4, len(frames[0].items[0].polygons[0]))

    def test_schema_v3_faces_use_exact_ellipse_and_keypoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = create_rich_face_sqlite(Path(temporary) / "rich.sqlite")

            frames = list(iter_face_frames(path))

            self.assertEqual(1, len(frames))
            self.assertEqual(1, frames[0].frame_index)
            face = next(item for item in frames[0].items if item.label == "Face")
            head = next(item for item in frames[0].items if item.label == "Head")
            self.assertIsNone(face.box)
            self.assertEqual(
                (30.0, 22.0, 12.0, 8.0, 0.25),
                face.ellipse,
            )
            self.assertEqual(5, len(face.keypoints))
            self.assertEqual(4, sum(point.valid for point in face.keypoints))
            self.assertIsNotNone(face.face_mask)
            assert face.face_mask is not None
            self.assertEqual(16, len(face.face_mask.probabilities))
            self.assertEqual((12.0, 5.0, 48.0, 42.0), head.box)

    def test_rich_face_components_can_be_disabled_without_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = create_rich_face_sqlite(Path(temporary) / "rich.sqlite")

            frames = list(
                iter_face_frames(
                    path,
                    include_ellipses=False,
                    include_keypoints=False,
                    include_probability_masks=False,
                )
            )

            face = next(item for item in frames[0].items if item.label == "Face")
            self.assertEqual((18.0, 10.0, 42.0, 34.0), face.box)
            self.assertIsNone(face.ellipse)
            self.assertEqual((), face.keypoints)
            self.assertIsNone(face.face_mask)

    def test_rich_face_masks_are_decompressed_lazily(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = create_rich_face_sqlite(Path(temporary) / "rich.sqlite")
            with sqlite3.connect(path) as connection:
                for observation_id, frame_id, detection_id in (
                    (2, 3, 30),
                    (3, 4, 40),
                ):
                    connection.executemany(
                        """
                        INSERT INTO detections(
                            id, frame_id, model_execution_id, class_name, score,
                            x1, y1, x2, y2, group_id
                        ) VALUES (?, ?, 2, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            (
                                detection_id,
                                frame_id,
                                "Face",
                                0.9,
                                18,
                                10,
                                42,
                                34,
                                observation_id,
                            ),
                            (
                                detection_id + 1,
                                frame_id,
                                "Head",
                                0.95,
                                12,
                                5,
                                48,
                                42,
                                observation_id,
                            ),
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO face_observations VALUES(
                            ?, ?, ?, ?, 0.9, 1, 'ellipse',
                            30.0, 22.0, 12.0, 8.0, 0.25
                        )
                        """,
                        (
                            observation_id,
                            detection_id + 1,
                            detection_id + 1,
                            detection_id,
                        ),
                    )
                    payload = (
                        zlib.compress(bytes(range(16)))
                        if observation_id == 2
                        else b"not-zlib"
                    )
                    connection.execute(
                        """
                        INSERT INTO face_masks VALUES(
                            ?, 'zlib-u8-probability-v1', 4, 4,
                            18.0, 10.0, 42.0, 34.0, ?
                        )
                        """,
                        (observation_id, payload),
                    )

            frames = iter_face_frames(path)
            first = next(frames)
            self.assertEqual(1, first.frame_index)
            with self.assertRaisesRegex(ValueError, "corrupt face mask"):
                list(frames)


if __name__ == "__main__":
    unittest.main()
