from __future__ import annotations

import importlib
import subprocess
import sys
import unittest


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
    "production.polygon.input_geometry",
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
    "production.polygon.runtime.diagnostics",
    "production.polygon.runtime.engine",
    "production.polygon.runtime.optimizer_factory",
    "production.polygon.runtime.optimizer_kernel",
    "production.polygon.runtime.kernel.artifacts",
    "production.polygon.runtime.kernel.candidates",
    "production.polygon.runtime.kernel.defaults",
    "production.polygon.runtime.kernel.evaluation",
    "production.polygon.runtime.kernel.geometry",
    "production.polygon.runtime.kernel.interpolation",
    "production.polygon.runtime.kernel.model",
    "production.polygon.runtime.kernel.solver",
    "production.polygon.runtime.kernel.stream",
    "production.polygon.runtime.kernel.types",
    "production.polygon.runtime.optimizer_adapters.artifacts",
    "production.polygon.runtime.optimizer_adapters.geometry",
    "production.polygon.runtime.optimizer_adapters.native_dp",
    "production.polygon.runtime.optimizer_adapters.python_dp",
    "production.polygon.runtime.optimizer_adapters.resources",
    "production.polygon.runtime.phase1_runtime",
    "production.polygon.runtime.phase2_candidates",
    "production.polygon.runtime.phase2_config",
    "production.polygon.runtime.phase2_hard_dp",
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

    def test_refactored_composition_roots_keep_compatibility_symbols(self) -> None:
        expected = {
            "production.polygon.runtime.optimizer_kernel": {
                "build_track_streams",
                "run_multistate_penalty_path",
                "repair_keyframe_vectors_for_exact_recall",
                "process_single_run",
            },
            "production.polygon.runtime.phase2_runtime": {
                "_class_role_state_profile",
                "_patch_phase2_candidates",
                "_build_dense_edge_array",
                "main",
            },
        }
        for module_name, symbols in expected.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(
                    [],
                    sorted(symbol for symbol in symbols if not hasattr(module, symbol)),
                )

    def test_production_nms_does_not_load_historical_policies(self) -> None:
        script = (
            "import sys; import nms.production; "
            "forbidden={'nms.adaptive','nms.component_aware','nms.stages'}; "
            "loaded=forbidden.intersection(sys.modules); "
            "assert not loaded, sorted(loaded)"
        )
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_retired_genital_algorithms_are_not_registered(self) -> None:
        from common.registry import stage_implementations

        retired = {
            "approximation.ellipse.production",
            "approximation.polygon.rdp",
            "keyframes.ellipse.dense",
            "keyframes.polygon.interval",
            "gap_fill.ellipse.linear",
            "gap_fill.polygon.linear",
            "evaluation.ellipse.exact",
        }
        self.assertTrue(retired.isdisjoint(stage_implementations()))


if __name__ == "__main__":
    unittest.main()
