from __future__ import annotations

import math
import unittest

from overlay_renderer.face_privacy import (
    derive_eye_privacy_mask,
    derive_face_privacy_mask,
)
from overlay_renderer.models import FaceKeypointOverlay


def point(
    x: float,
    y: float,
    class_name: str,
    *,
    confidence: float = 0.95,
    valid: bool = True,
) -> FaceKeypointOverlay:
    return FaceKeypointOverlay(
        x=x,
        y=y,
        class_name=class_name,
        state=2,
        state_name="visible",
        confidence=confidence,
        valid=valid,
    )


class FacePrivacyTests(unittest.TestCase):
    def test_face_mask_is_the_exact_detector_ellipse(self) -> None:
        mask = derive_face_privacy_mask((100.0, 80.0, 40.0, 20.0, 0.0))
        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual("face-ellipse", mask.derivation)
        self.assertEqual(96, len(mask.polygon))
        xs = [value[0] for value in mask.polygon]
        ys = [value[1] for value in mask.polygon]
        self.assertAlmostEqual(60.0, min(xs), places=6)
        self.assertAlmostEqual(140.0, max(xs), places=6)
        self.assertAlmostEqual(60.0, min(ys), places=6)
        self.assertAlmostEqual(100.0, max(ys), places=6)

    def test_eye_ellipse_follows_two_valid_eye_keypoints(self) -> None:
        keypoints = (
            point(70.0, 62.0, "Eye", confidence=0.91),
            point(130.0, 72.0, "Eye", confidence=0.87),
            point(102.0, 94.0, "Nose"),
            point(100.0, 120.0, "Mouth"),
        )
        mask = derive_eye_privacy_mask(
            (100.0, 90.0, 60.0, 50.0, math.pi / 2.0),
            keypoints,
            shape="ellipse",
        )
        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual("eye-keypoints", mask.derivation)
        self.assertEqual("ellipse", mask.shape)
        self.assertAlmostEqual(0.87, mask.confidence)
        center_x = sum(value[0] for value in mask.polygon) / len(mask.polygon)
        center_y = sum(value[1] for value in mask.polygon) / len(mask.polygon)
        self.assertAlmostEqual(100.0, center_x, places=6)
        self.assertAlmostEqual(67.0, center_y, places=6)

    def test_eye_rectangle_is_rotated_and_has_four_points(self) -> None:
        mask = derive_eye_privacy_mask(
            (100.0, 90.0, 60.0, 50.0, math.pi / 2.0),
            (
                point(70.0, 62.0, "Eye"),
                point(130.0, 72.0, "Eye"),
                point(100.0, 115.0, "Mouth"),
            ),
            shape="rectangle",
        )
        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(4, len(mask.polygon))
        self.assertNotAlmostEqual(mask.polygon[0][1], mask.polygon[1][1])

    def test_eye_masks_include_outer_corner_and_vertical_padding(self) -> None:
        keypoints = (
            point(70.0, 65.0, "Eye"),
            point(130.0, 65.0, "Eye"),
            point(100.0, 95.0, "Nose"),
            point(100.0, 120.0, "Mouth"),
        )
        for shape in ("ellipse", "rectangle"):
            with self.subTest(shape=shape):
                mask = derive_eye_privacy_mask(
                    (100.0, 90.0, 60.0, 50.0, math.pi / 2.0),
                    keypoints,
                    shape=shape,
                )
                self.assertIsNotNone(mask)
                assert mask is not None
                xs = [value[0] for value in mask.polygon]
                ys = [value[1] for value in mask.polygon]
                self.assertAlmostEqual(41.8, min(xs), places=5)
                self.assertAlmostEqual(158.2, max(xs), places=5)
                self.assertAlmostEqual(40.7, min(ys), places=5)
                self.assertAlmostEqual(89.3, max(ys), places=5)

    def test_missing_eyes_use_oriented_face_ellipse_fallback(self) -> None:
        mask = derive_eye_privacy_mask(
            (100.0, 100.0, 60.0, 40.0, math.pi / 2.0),
            (
                point(100.0, 112.0, "Nose"),
                point(100.0, 130.0, "Mouth"),
            ),
            shape="rectangle",
        )
        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual("ellipse-fallback", mask.derivation)
        center_y = sum(value[1] for value in mask.polygon) / len(mask.polygon)
        self.assertLess(center_y, 100.0)
        self.assertEqual(0.0, mask.confidence)

    def test_invalid_ellipse_never_emits_a_privacy_mask(self) -> None:
        self.assertIsNone(derive_face_privacy_mask((0.0, 0.0, 0.0, 4.0, 0.0)))
        self.assertIsNone(
            derive_eye_privacy_mask(
                (0.0, 0.0, math.nan, 4.0, 0.0),
                (),
            )
        )


if __name__ == "__main__":
    unittest.main()
