from __future__ import annotations

import ast
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from common.registry import create_stage
from contracts.mask_sqlite import MaskRow, write_mask_sqlite
from production.config import PRODUCTION
from production.polygon.materialize import materialize_outputs
from production.polygon.runtime_bridge import build_runtime_config
from production.polygon.stage import ProductionPolygonStage


def _reference(path: Path) -> Path:
    rows = [
        MaskRow(
            frame=0,
            track_id="7",
            polygons="[[[0,0],[10,0],[10,10],[0,10]]]",
            label="女性器",
        )
    ]
    output = write_mask_sqlite(path, rows)
    with sqlite3.connect(output) as connection:
        connection.execute("CREATE TABLE cuts(frame INTEGER PRIMARY KEY)")
    return output


class PromotedProductionProfileTests(unittest.TestCase):
    def test_production_tree_never_imports_experimental_code(self) -> None:
        root = Path(__file__).resolve().parents[1] / "production"
        offenders: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                module = ""
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("experimental"):
                            offenders.append(str(path.relative_to(root)))
                if module.startswith("experimental"):
                    offenders.append(str(path.relative_to(root)))
        self.assertEqual([], offenders)

    def test_native_exact_runtime_is_shipped_with_reproducible_sources(self) -> None:
        root = Path(__file__).resolve().parents[1]
        native = root / "production/polygon/runtime/native_interval"
        required = {
            "CMakeLists.txt",
            "bootstrap_and_build.sh",
            "environment.yml",
            "native_interval_metrics.cpp",
            "test_native_interval_metrics.py",
        }
        self.assertEqual(required, {path.name for path in native.iterdir()} & required)
        setup = (root.parent / "deployment/setup_phase2.sh").read_text(encoding="utf-8")
        self.assertIn(
            "production/polygon/runtime/native_interval/bootstrap_and_build.sh",
            setup,
        )
        package_config = (root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"production",', package_config)
        self.assertIn('"vendor" = "vendor"', package_config)
        coordinator = (root / "production/polygon/runtime/run_phase2.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("MASK_PIPELINE_CUDA_EXPERIMENT_SITE", coordinator)
        self.assertNotIn("mask-pipeline-cuda-experiment", coordinator)

    def test_public_contract_forces_native_exact_cpu(self) -> None:
        PRODUCTION.validate()
        runtime = build_runtime_config(PRODUCTION)
        self.assertEqual("native_exact", runtime.runtime.interval_evaluation)
        self.assertTrue(runtime.spatial.adaptive_vertex_policy)
        self.assertEqual(
            (14, 16, 18, 20),
            runtime.spatial.allowed_vertices_per_component,
        )
        self.assertEqual(0.999, runtime.spatial.track_area_quantile)
        self.assertEqual(
            (0.03, 0.10, 0.25),
            runtime.spatial.screen_occupancy_thresholds,
        )
        self.assertEqual(16.0, runtime.preparation.border_max_expand_px)
        self.assertEqual(16.0, runtime.preparation.border_influence_px)
        self.assertTrue(runtime.preparation.border_corner_support)
        self.assertFalse(hasattr(runtime.spatial, "vertex_fallbacks"))
        self.assertEqual(1.05, runtime.spatial.recall_repair_max_scale)
        self.assertEqual(0.97, runtime.temporal.recall_floor)

    def test_registered_stages_use_stable_names(self) -> None:
        nms = create_stage("nms.production_v3", {})
        polygon = create_stage("production.polygon_v3_cpu", {})
        self.assertEqual("production_virtual_component_mask_nms_v1", nms.name)
        self.assertEqual(
            "production_polygon_adaptive_recall_cpu_exact_v3", polygon.name
        )

    def test_nms_thresholds_cannot_be_changed_from_pipeline_json(self) -> None:
        stage = create_stage("nms.production_v3", {"mask_iou_threshold": 0.99})
        with tempfile.TemporaryDirectory() as temporary:
            from contracts.stages import StageContext

            root = Path(temporary)
            source = root / "scored.jsonl"
            source.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "thresholds are frozen"):
                stage.run(
                    StageContext(
                        pipeline_name="test",
                        stage_id="nms",
                        output_dir=root,
                        stage_dir=root / "nms",
                        artifacts={"scored_jsonl": source},
                    )
                )

    def test_polygon_stage_rejects_cuda_evaluator(self) -> None:
        with self.assertRaisesRegex(ValueError, "CPU native_exact"):
            ProductionPolygonStage({"interval_evaluation": "cuda_lazy_exact"})._config()

    def test_materialized_keyframes_declare_index_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracked = _reference(root / "tracked.sqlite")
            phase2 = root / "phase2"
            runtime_profile = build_runtime_config(PRODUCTION).polygon_profile_id
            runtime = phase2 / runtime_profile / "女性器" / "runtime"
            prediction = runtime / "pred/predictions.sqlite"
            prediction.parent.mkdir(parents=True)
            with sqlite3.connect(prediction) as connection:
                connection.execute(
                    "CREATE TABLE masks(frame INTEGER,track_id TEXT,polygons TEXT)"
                )
                connection.execute(
                    "INSERT INTO masks VALUES(0,'7',?)",
                    ("[[[0,0],[10,0],[10,10],[0,10]]]",),
                )
            keyframes = runtime / "opt/final_keyframes.json"
            keyframes.parent.mkdir(parents=True)
            keyframes.write_text(
                json.dumps(
                    [
                        {
                            "frame": 0,
                            "track_id": "7",
                            "polygons": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "predictions.sqlite"
            keys = root / "keyframes.sqlite"
            summary = materialize_outputs(
                phase2,
                tracked,
                output,
                keys,
                config=PRODUCTION,
                runtime_profile=runtime_profile,
            )
            self.assertEqual(1, summary["prediction_rows"])
            self.assertEqual(1, summary["keyframes"])
            with sqlite3.connect(keys) as connection:
                value = connection.execute(
                    "SELECT value FROM polygon_keyframe_metadata "
                    "WHERE key='interpolation_method'"
                ).fetchone()
                fallbacks = connection.execute(
                    "SELECT value FROM polygon_keyframe_metadata "
                    "WHERE key='vertex_fallbacks'"
                ).fetchone()
                maximum = connection.execute(
                    "SELECT value FROM polygon_keyframe_metadata "
                    "WHERE key='maximum_vertices_per_component'"
                ).fetchone()
                allowed = connection.execute(
                    "SELECT value FROM polygon_keyframe_metadata "
                    "WHERE key='allowed_vertices_per_component'"
                ).fetchone()
                policy = connection.execute(
                    "SELECT value FROM polygon_keyframe_metadata "
                    "WHERE key='vertices_per_component'"
                ).fetchone()
            self.assertEqual(("linear_polygon_index_v1",), value)
            self.assertIsNone(fallbacks)
            self.assertEqual(("20",), maximum)
            self.assertEqual(("[14, 16, 18, 20]",), allowed)
            self.assertEqual(("adaptive_by_track",), policy)


if __name__ == "__main__":
    unittest.main()
