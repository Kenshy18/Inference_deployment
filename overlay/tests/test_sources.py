from __future__ import annotations

import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

from overlay_renderer.sources import (
    OverlayContractError,
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

    def test_integrated_result_selects_tracked_snapshot_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = create_mask_sqlite(Path(temporary) / "result.sqlite")
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    CREATE TABLE tracked_masks(
                        frame INTEGER NOT NULL,
                        track_id TEXT NOT NULL,
                        polygons TEXT NOT NULL,
                        label TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO tracked_masks
                    SELECT frame, 'tracked', polygons, label
                    FROM masks
                    """
                )

            final = list(iter_mask_frames(path))
            tracked = list(iter_mask_frames(path, prefer_tracked=True))

            self.assertEqual("7", final[0].items[0].track_id)
            self.assertEqual("tracked", tracked[0].items[0].track_id)

    def test_mask_domain_excludes_face_privacy_from_genital_preset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = create_mask_sqlite(Path(temporary) / "result.sqlite")
            with sqlite3.connect(path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE tracks(
                        track_id TEXT PRIMARY KEY,
                        label TEXT,
                        domain TEXT NOT NULL
                    );
                    INSERT INTO tracks VALUES
                        ('7', 'sample', 'genital'),
                        ('face:eyes:1', 'Eyes', 'face_privacy');
                    INSERT INTO masks(frame, track_id, polygons, label)
                    SELECT frame, 'face:eyes:1', polygons, 'Eyes'
                    FROM masks WHERE track_id='7';
                    """
                )

            all_masks = list(iter_mask_frames(path))
            genital = list(iter_mask_frames(path, mask_domain="genital"))

            self.assertEqual(2, len(all_masks[0].items))
            self.assertEqual(
                ["7"],
                [item.track_id for item in genital[0].items],
            )

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

    def test_fixed_v3_schema_uses_legacy_boxes_when_rich_face_is_unsupported(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = create_rich_face_sqlite(Path(temporary) / "legacy-v3.sqlite")
            with sqlite3.connect(path) as connection:
                connection.executescript(
                    """
                    DELETE FROM face_keypoint_state_probabilities;
                    DELETE FROM face_keypoint_class_probabilities;
                    DELETE FROM face_keypoints;
                    DELETE FROM face_masks;
                    DELETE FROM face_observations;
                    CREATE TABLE result_capabilities(
                        name TEXT PRIMARY KEY,
                        available INTEGER NOT NULL
                    );
                    INSERT INTO result_capabilities
                    VALUES ('rich_face_geometry', 0);
                    """
                )

            frames = list(iter_face_frames(path, display_style="detailed"))

            self.assertEqual([1], [frame.frame_index for frame in frames])
            self.assertEqual(2, len(frames[0].items))
            self.assertEqual(
                {"Face", "Head"},
                {item.label for item in frames[0].items},
            )
            self.assertTrue(all(item.box is not None for item in frames[0].items))
            self.assertTrue(all(item.ellipse is None for item in frames[0].items))
            self.assertTrue(all(item.keypoints == () for item in frames[0].items))
            with self.assertRaisesRegex(
                OverlayContractError,
                "face privacy masks require schema-v3 rich face",
            ):
                list(iter_face_frames(path, require_privacy_geometry=True))

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

    def test_detailed_rich_face_overlay_uses_postprocessed_track_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = create_rich_face_sqlite(Path(temporary) / "rich.sqlite")
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    CREATE TABLE face_tracking_assignments(
                        observation_id INTEGER PRIMARY KEY,
                        raw_track_id TEXT NOT NULL,
                        final_track_id TEXT,
                        removed_by_short_track INTEGER NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO face_tracking_assignments
                    VALUES (1, 'face:raw:0:7', 'face:0:7', 0)
                    """
                )

            frames = list(iter_face_frames(path, display_style="detailed"))

            self.assertEqual(1, len(frames))
            self.assertEqual("face:0:7", frames[0].items[0].track_id)
            self.assertEqual("OBSERVED", frames[0].items[0].provenance)

    def test_face_threshold_keeps_accepted_head_without_face_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = create_rich_face_sqlite(Path(temporary) / "rich.sqlite")
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE detections SET score=0.40 WHERE id=20")
                connection.execute(
                    "UPDATE face_observations SET face_score=0.40 WHERE id=1"
                )

            detailed = list(iter_face_frames(path, display_style="detailed"))
            simple = list(iter_face_frames(path, display_style="simple"))
            legacy = list(iter_face_frames(path, display_style="legacy"))

            self.assertEqual(1, len(detailed))
            self.assertEqual(1, len(detailed[0].items))
            head = detailed[0].items[0]
            self.assertEqual("Head", head.label)
            self.assertEqual(0.40, head.face_score)
            self.assertFalse(head.face_present)
            self.assertIsNotNone(head.box)
            self.assertIsNone(head.ellipse)
            self.assertEqual((), head.keypoints)
            self.assertIsNone(head.face_mask)
            self.assertEqual([], simple)
            self.assertEqual(["Head"], [item.label for item in legacy[0].items])

    def test_head_threshold_rejects_entire_rich_face_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = create_rich_face_sqlite(Path(temporary) / "rich.sqlite")
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE detections SET score=0.40 WHERE id=21")

            self.assertEqual(
                [],
                list(iter_face_frames(path, display_style="detailed")),
            )
            self.assertEqual(
                [],
                list(iter_face_frames(path, display_style="simple")),
            )

    def test_face_threshold_is_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = create_rich_face_sqlite(Path(temporary) / "rich.sqlite")
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE detections SET score=0.40 WHERE id=20")
                connection.execute(
                    "UPDATE face_observations SET face_score=0.40 WHERE id=1"
                )

            frames = list(
                iter_face_frames(
                    path,
                    display_style="simple",
                    face_detection_score_threshold=0.35,
                )
            )

            self.assertEqual(1, len(frames))
            self.assertIsNotNone(frames[0].items[0].ellipse)

    def test_detailed_face_overlay_marks_removed_and_interpolated_tracks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = create_rich_face_sqlite(Path(temporary) / "rich.sqlite")
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    CREATE TABLE face_tracking_assignments(
                        observation_id INTEGER PRIMARY KEY,
                        raw_track_id TEXT NOT NULL,
                        final_track_id TEXT,
                        removed_by_short_track INTEGER NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO face_tracking_assignments
                    VALUES (1, 'face:raw:0:9', NULL, 1)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE face_track_interpolations(
                        frame INTEGER NOT NULL,
                        final_track_id TEXT NOT NULL,
                        head_x1 REAL NOT NULL,
                        head_y1 REAL NOT NULL,
                        head_x2 REAL NOT NULL,
                        head_y2 REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO face_track_interpolations
                    VALUES (2, 'face:0:7', 13, 6, 49, 43)
                    """
                )

            frames = list(iter_face_frames(path, display_style="detailed"))

            self.assertEqual([1, 2], [frame.frame_index for frame in frames])
            removed = frames[0].items[0]
            interpolated = frames[1].items[0]
            self.assertEqual("face:raw:0:9", removed.track_id)
            self.assertEqual("REMOVED_SHORT_TRACK", removed.provenance)
            self.assertEqual("face:0:7", interpolated.track_id)
            self.assertEqual("INTERPOLATED", interpolated.provenance)
            self.assertEqual((13.0, 6.0, 49.0, 43.0), interpolated.box)
            self.assertEqual([], list(iter_face_frames(path, display_style="simple")))

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

            bounded = list(iter_face_frames(path, start_frame=1, end_frame=1))
            self.assertEqual([1], [frame.frame_index for frame in bounded])

            frames = iter_face_frames(path)
            first = next(frames)
            self.assertEqual(1, first.frame_index)
            with self.assertRaisesRegex(ValueError, "corrupt face mask"):
                list(frames)


if __name__ == "__main__":
    unittest.main()
