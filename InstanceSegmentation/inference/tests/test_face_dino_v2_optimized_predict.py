from __future__ import annotations

import types
import unittest
from types import SimpleNamespace

import torch

from face_dino_v2.optimized_predict import (
    _features_and_detections_prefiltered,
    install_prefiltered_predict,
)


class _QueryHead:
    num_classes = 1
    _face_original_get_bboxes = object()

    def simple_test(self, features, metas, **kwargs):
        del features, metas, kwargs
        return (
            [
                (
                    torch.tensor(
                        [
                            [1.0, 2.0, 3.0, 4.0, 0.9],
                            [5.0, 6.0, 7.0, 8.0, 0.8],
                        ]
                    ),
                    torch.zeros(2, dtype=torch.long),
                )
            ],
            None,
        )


class _Detector:
    def __init__(self) -> None:
        self.query_head = _QueryHead()
        self.query_level_indices = None

    def _metas(self, images, image_sizes):
        del images
        return [{"img_shape": size} for size in image_sizes]

    def extract_feat(self, images, metas):
        del metas
        return (images,)


class FaceDinoOptimizedPredictTest(unittest.TestCase):
    def test_one_class_features_keep_prefiltered_row_order(self) -> None:
        model = SimpleNamespace(
            detector=_Detector(),
            detector_backend=None,
        )
        pyramid, detections = _features_and_detections_prefiltered(
            model,
            torch.zeros((1, 3, 8, 8)),
            [(8, 8)],
        )
        self.assertEqual(len(pyramid), 1)
        self.assertEqual(
            detections[0]["boxes"].tolist(),
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        )
        torch.testing.assert_close(
            detections[0]["scores"],
            torch.tensor([0.9, 0.8]),
        )

    def test_installer_retains_original_predict_for_threshold_fallback(self) -> None:
        model = SimpleNamespace(
            detector=_Detector(),
            detector_backend=None,
        )

        def original_predict(self, images, image_sizes, **kwargs):
            del self, images, image_sizes
            return [kwargs["score_threshold"]]

        def original_features(self, images, image_sizes):
            del self, image_sizes
            return [images], []

        model.predict = types.MethodType(original_predict, model)
        model._features_and_detections = types.MethodType(
            original_features,
            model,
        )
        install_prefiltered_predict(model, score_threshold=0.30)
        result = model.predict(
            torch.zeros((1, 3, 8, 8)),
            [(8, 8)],
            score_threshold=0.40,
        )
        self.assertEqual(result, [0.40])
        self.assertIs(
            model._face_original_features_and_detections.__func__,
            original_features,
        )

    def test_installer_rejects_non_face_multiclass_detector(self) -> None:
        model = SimpleNamespace(detector=_Detector())
        model.detector.query_head.num_classes = 2
        with self.assertRaisesRegex(ValueError, "one detector class"):
            install_prefiltered_predict(model, score_threshold=0.30)


if __name__ == "__main__":
    unittest.main()
