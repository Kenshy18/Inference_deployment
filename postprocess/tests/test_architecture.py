from __future__ import annotations

import ast
import json
import sqlite3
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from contracts.detections import CutList, write_cut_list
from contracts.stages import StageContext, StageResult
from common.config import (
    default_mask_pipeline,
    default_polygon_pipeline,
    load_pipeline_config,
)
from common.registry import create_stage
from run_pipeline import build_parser, run_pipeline


FEATURES = {
    "preprocessing",
    "nms",
    "cut_detection",
    "tracking",
    "evaluation",
    "artifacts",
    "face_privacy",
    "classwise",
    "visualization",
}


@dataclass(frozen=True)
class FixedCutStage:
    name: str = "fixed_cut"
    requires: frozenset[str] = frozenset({"nms_jsonl"})
    provides: frozenset[str] = frozenset({"cuts_json"})

    def run(self, context: StageContext) -> StageResult:
        output = context.stage_dir / "cuts.json"
        write_cut_list(output, CutList((3,), self.name, 0.125))
        return StageResult({"cuts_json": output})


class ArchitectureTests(unittest.TestCase):
    def test_retired_research_tree_is_absent(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "experimental").exists())

    def test_production_runtime_has_no_experimental_dependency(self) -> None:
        root = Path(__file__).resolve().parents[1]
        violations: list[str] = []
        for path in (root / "production").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                for module in modules:
                    if module in {"experimental", "experiments"} or module.startswith(
                        ("experimental.", "experiments.")
                    ):
                        violations.append(f"{path.relative_to(root)} -> {module}")
        self.assertEqual([], violations)

    def test_large_composition_roots_stay_split_by_responsibility(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        limits = {
            "postprocess/production/polygon/runtime/optimizer_factory.py": 150,
            "postprocess/production/polygon/runtime/optimizer_kernel.py": 1200,
            "postprocess/production/polygon/runtime/optimizer_process.py": 1100,
            "postprocess/production/polygon/runtime/optimizer_adapters/native_dp.py": 700,
            "postprocess/artifacts/unified_sqlite.py": 1250,
            "postprocess/classwise/stages.py": 500,
            "postprocess/classwise/curve_parallel.py": 260,
            "orchestration/runner.py": 1100,
            "orchestration/config.py": 100,
            "gui/src/App.tsx": 800,
            "gui/src/components/MonitorPanel.tsx": 700,
            "gui/src/components/InspectorPanel.tsx": 100,
        }
        for relative, limit in limits.items():
            with self.subTest(module=relative):
                line_count = len(
                    (repository / relative).read_text(encoding="utf-8").splitlines()
                )
                self.assertLessEqual(line_count, limit)

        required_modules = [
            "postprocess/production/polygon/runtime/kernel/geometry.py",
            "postprocess/production/polygon/runtime/kernel/stream.py",
            "postprocess/production/polygon/runtime/kernel/candidates.py",
            "postprocess/production/polygon/runtime/kernel/interpolation.py",
            "postprocess/production/polygon/runtime/kernel/solver.py",
            "postprocess/production/polygon/runtime/candidate_generation.py",
            "postprocess/production/polygon/runtime/hard_recall_dp.py",
            "postprocess/production/polygon/runtime/optimizer_adapters/native_dp_kernel.cpp",
            "postprocess/production/curve/config.py",
            "postprocess/production/curve/engine.py",
            "postprocess/production/curve/stage.py",
            "postprocess/production/curve/runtime/model.py",
            "postprocess/production/curve/runtime/multistate_dp.py",
            "postprocess/production/curve/runtime/native_cpu.py",
            "postprocess/classwise/curve_parallel.py",
            "postprocess/classwise/pipeline_factory.py",
            "postprocess/contracts/integrated_result.py",
            "postprocess/contracts/result_schema.py",
            "orchestration/config_loader.py",
            "orchestration/config_validation.py",
            "orchestration/runner_media.py",
            "orchestration/runner_commands.py",
            "gui/src/components/inspector/InferenceSection.tsx",
            "gui/src/components/inspector/PostprocessSection.tsx",
            "gui/src/components/inspector/OverlaySection.tsx",
            "gui/src/components/inspector/RuntimeSection.tsx",
            "gui/src/components/monitor/flow.ts",
            "gui/src/hooks/useInspectorActions.ts",
        ]
        self.assertEqual(
            [],
            [
                relative
                for relative in required_modules
                if not (repository / relative).is_file()
            ],
        )

    def test_feature_packages_do_not_import_each_other(self) -> None:
        root = Path(__file__).resolve().parents[1]
        violations: list[str] = []
        for feature in sorted(FEATURES):
            for path in (root / feature).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    imported: list[str] = []
                    if isinstance(node, ast.Import):
                        imported = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0:
                        imported = [node.module or ""]
                    for module in imported:
                        dependency = module.split(".", 1)[0]
                        if dependency in FEATURES and dependency != feature:
                            violations.append(f"{path.relative_to(root)} -> {module}")
        self.assertEqual([], violations)

    def test_contracts_do_not_import_artifact_implementations(self) -> None:
        root = Path(__file__).resolve().parents[1]
        violations: list[str] = []
        for path in (root / "contracts").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                imported: list[str] = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    imported = [node.module or ""]
                for module in imported:
                    if module == "artifacts" or module.startswith("artifacts."):
                        violations.append(f"{path.relative_to(root)} -> {module}")
        self.assertEqual([], violations)

    def test_root_entrypoint_does_not_import_feature_implementations(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "run_pipeline.py").read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imports.add((node.module or "").split(".", 1)[0])
        self.assertEqual(set(), imports & FEATURES)

    def test_common_package_import_has_no_stage_registration_side_effect(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "common" / "__init__.py").read_text(encoding="utf-8"))
        imported_modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("builtins", imported_modules)

    def test_deployed_role_generators_match_the_frozen_palettes(self) -> None:
        from production.curve.runtime.role_states import curve_role_ids
        from production.polygon.runtime.candidate_config import CANDIDATE, LABELS
        from production.polygon.runtime.candidate_palette import role_ids
        from production.polygon.runtime.role_candidates import PRODUCTION_ROLE_IDS

        selected = {
            value.removesuffix("_P1")
            for label in LABELS
            for value in (
                *role_ids(label, CANDIDATE.temporal.target_interval),
                *curve_role_ids(label),
            )
        }
        self.assertEqual(selected, set(PRODUCTION_ROLE_IDS))

    def test_default_raw_pipeline_has_one_stage_per_feature(self) -> None:
        implementations = [
            stage.implementation
            for stage in default_polygon_pipeline(include_preprocess=True).stages
        ]
        self.assertEqual(
            [
                "preprocessing.normalize",
                "preprocessing.score_policy",
                "nms.production_v3",
                "cut_detection.video",
                "tracking.greedy",
                "production.polygon_v3_cpu",
                "evaluation.mask_iou",
                "artifacts.validate",
            ],
            implementations,
        )

    def test_default_curve_pipeline_replaces_only_geometry_stage(self) -> None:
        polygon = default_mask_pipeline(
            include_preprocess=True,
            geometry_mode="polygon",
        )
        curve = default_mask_pipeline(
            include_preprocess=True,
            geometry_mode="catmull_rom",
        )
        polygon_implementations = [stage.implementation for stage in polygon.stages]
        curve_implementations = [stage.implementation for stage in curve.stages]
        self.assertEqual(
            polygon_implementations[:5],
            curve_implementations[:5],
        )
        self.assertEqual(
            polygon_implementations[6:],
            curve_implementations[6:],
        )
        self.assertEqual("production.polygon_v3_cpu", polygon_implementations[5])
        self.assertEqual("production.curve_v1_cpu", curve_implementations[5])

    def test_shipped_pipeline_configs_have_valid_artifact_chains(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / "configs" / "pipelines").glob("*.json")):
            with self.subTest(config=path.name):
                config = load_pipeline_config(path)
                available = (
                    {"input_jsonl", "input_video"}
                    if path.stem.endswith("from_jsonl")
                    else {"tracked_sqlite"}
                )
                for spec in config.stages:
                    if not spec.enabled:
                        continue
                    stage = create_stage(spec.implementation, spec.options)
                    self.assertEqual(set(), stage.requires - available)
                    available.update(stage.provides)
                self.assertIn("predictions_sqlite", available)
                self.assertIn("validation_report", available)

    def test_cut_detection_can_be_replaced_in_raw_to_output_e2e(self) -> None:
        default = default_polygon_pipeline(include_preprocess=True)
        stages = [
            (
                {
                    "id": stage.id,
                    "implementation": f"{__name__}:FixedCutStage",
                }
                if stage.id == "cut_detection"
                else {
                    "id": stage.id,
                    "implementation": stage.implementation,
                    "options": (
                        {"remove_short_tracks_max_frames": 0}
                        if stage.id == "tracking"
                        else stage.options
                    ),
                    "enabled": stage.enabled,
                }
            )
            for stage in default.stages
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_jsonl = root / "raw.jsonl"
            records = []
            for frame in range(6):
                x = float(frame)
                polygon = [
                    [x, 0.0],
                    [x + 10.0, 0.0],
                    [x + 10.0, 10.0],
                    [x, 10.0],
                ]
                records.append(
                    {
                        "frame_idx": frame,
                        "instances": [
                            {
                                "label": "男性器",
                                "score": 0.9,
                                "bbox": [x, 0.0, 10.0, 10.0],
                                "segmentation": [polygon],
                            }
                        ],
                    }
                )
            input_jsonl.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            config_path = root / "pipeline.json"
            config_path.write_text(
                json.dumps({"name": "replace_cut_e2e", "stages": stages}),
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                [
                    "--input-jsonl",
                    str(input_jsonl),
                    "--output-dir",
                    str(root / "output"),
                    "--pipeline-config",
                    str(config_path),
                ]
            )
            manifest = run_pipeline(args)

            self.assertTrue(manifest["complete"])
            self.assertEqual(
                "fixed_cut",
                manifest["stages"][3]["name"],
            )
            self.assertTrue(Path(manifest["artifacts"]["predictions_sqlite"]).is_file())
            self.assertTrue(Path(manifest["artifacts"]["validation_report"]).is_file())
            with sqlite3.connect(
                manifest["artifacts"]["predictions_sqlite"]
            ) as connection:
                self.assertEqual(
                    [(3,)],
                    connection.execute(
                        "SELECT frame FROM cuts ORDER BY frame"
                    ).fetchall(),
                )
                self.assertEqual(
                    [
                        (
                            "fixed_cut",
                            0.125,
                            1,
                            "first_frame_of_new_scene",
                        )
                    ],
                    connection.execute(
                        """
                        SELECT method, elapsed_seconds, cut_count,
                               frame_semantics
                        FROM cut_detection_metadata
                        """
                    ).fetchall(),
                )


if __name__ == "__main__":
    unittest.main()
