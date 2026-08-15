from __future__ import annotations

import importlib
import pickle
import unittest

from approximation.ellipse import runtime_fst


RUNTIME_MODULES = (
    "contracts.detections",
    "contracts.mask_sqlite",
    "common.config",
    "common.registry",
    "common.runner",
    "preprocessing.normalization",
    "preprocessing.raw_sqlite",
    "preprocessing.score_policy",
    "preprocessing.stages",
    "nms.adaptive",
    "nms.component_aware",
    "nms.component_virtual",
    "nms.mask_adaptive",
    "nms.stages",
    "cut_detection.detector",
    "cut_detection.stages",
    "tracking.association",
    "tracking.builder",
    "tracking.stages",
    "approximation.ellipse.inference",
    "approximation.ellipse.stages",
    "approximation.polygon.rdp",
    "approximation.polygon.stages",
    "keyframes.ellipse.stages",
    "keyframes.polygon.interval",
    "keyframes.polygon.stages",
    "gap_fill.polygon.interpolate",
    "gap_fill.polygon.stages",
    "gap_fill.ellipse.interpolate",
    "gap_fill.ellipse.stages",
    "evaluation.mask_iou",
    "evaluation.stages",
    "artifacts.sqlite",
    "artifacts.stages",
    "visualization.overlay",
    "run_pipeline",
)


class EngineImportTests(unittest.TestCase):
    def test_all_runtime_modules_import(self) -> None:
        for module_name in RUNTIME_MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_registered_fst_worker_is_pickleable(self) -> None:
        for worker in (
            runtime_fst.fst._solve_k1_row_worker,
            runtime_fst.fst._solve_k1_payload_worker,
        ):
            with self.subTest(worker=worker.__name__):
                self.assertIs(pickle.loads(pickle.dumps(worker)), worker)


if __name__ == "__main__":
    unittest.main()
