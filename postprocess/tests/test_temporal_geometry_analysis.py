from __future__ import annotations

import unittest

import numpy as np

from tentative.analyze_temporal_geometry import (
    _nearest_keyframe_offset,
    _transition,
)


class TemporalGeometryAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        x = np.linspace(-20.0, 20.0, 48)
        self.points = np.column_stack((x, 0.03 * x * x + 2.0 * np.sin(x)))

    def test_translation_is_not_reported_as_local_deformation(self) -> None:
        current = self.points + np.asarray([12.0, -7.0])
        metrics = _transition(self.points, current)
        self.assertAlmostEqual(np.hypot(12.0, 7.0), metrics["translation_px"])
        self.assertLess(metrics["local_deformation_px"], 1e-10)
        self.assertLess(metrics["after_similarity_px"], 1e-10)

    def test_full_affine_is_separated_from_local_deformation(self) -> None:
        transform = np.asarray([[1.25, 0.18], [-0.08, 0.73]])
        current = self.points @ transform + np.asarray([4.0, 9.0])
        metrics = _transition(self.points, current)
        self.assertGreater(metrics["after_similarity_px"], 0.1)
        self.assertGreater(metrics["affine_component_px"], 0.1)
        self.assertGreater(metrics["affine_anisotropy_log"], 0.1)
        self.assertLess(metrics["local_deformation_px"], 1e-10)

    def test_non_affine_change_remains_as_local_deformation(self) -> None:
        current = self.points.copy()
        current[:, 1] += 4.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, len(current)))
        metrics = _transition(self.points, current)
        self.assertGreater(metrics["local_deformation_px"], 1.0)

    def test_nearest_keyframe_offset_sign_is_stable(self) -> None:
        keys = {8, 13}
        self.assertEqual(-2, _nearest_keyframe_offset(10, keys))
        self.assertEqual(1, _nearest_keyframe_offset(12, keys))
        self.assertIsNone(_nearest_keyframe_offset(12, set()))


if __name__ == "__main__":
    unittest.main()
