from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from overlay_renderer.sources import (
    inspect_inference_source,
    inspect_mask_source,
    iter_face_frames,
    iter_mask_frames,
    iter_raw_segmentation_frames,
)

from helpers import create_mask_sqlite, create_unified_sqlite


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


if __name__ == "__main__":
    unittest.main()

