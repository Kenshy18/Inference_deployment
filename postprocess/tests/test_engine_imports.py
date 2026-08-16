from __future__ import annotations

import importlib
import pickle
import subprocess
import sys
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
    "nms.components",
    "nms.component_virtual",
    "nms.mask_geometry",
    "nms.mask_adaptive",
    "nms.production",
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
    "production.config",
    "production.polygon.materialize",
    "production.polygon.preparation",
    "production.polygon.runtime_bridge",
    "production.polygon.runtime.candidate_config",
    "production.polygon.runtime.engine",
    "production.polygon.runtime.phase1_runtime",
    "production.polygon.runtime.phase2_runtime",
    "production.polygon.runtime.run_phase2",
    "production.polygon.stage",
    "production.polygon.vertex_policy",
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

    def test_production_nms_does_not_load_historical_policies(self) -> None:
        script = (
            "import sys; import nms.production; "
            "forbidden={'nms.adaptive','nms.component_aware','nms.stages'}; "
            "loaded=forbidden.intersection(sys.modules); "
            "assert not loaded, sorted(loaded)"
        )
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
