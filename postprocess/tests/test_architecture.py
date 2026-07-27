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
from common.config import default_polygon_pipeline, load_pipeline_config
from common.registry import create_stage
from run_pipeline import build_parser, run_pipeline


FEATURES = {
    "preprocessing",
    "nms",
    "cut_detection",
    "tracking",
    "approximation",
    "keyframes",
    "gap_fill",
    "evaluation",
    "artifacts",
    "face_privacy",
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

    def test_default_raw_pipeline_has_one_stage_per_feature(self) -> None:
        implementations = [
            stage.implementation
            for stage in default_polygon_pipeline(include_preprocess=True).stages
        ]
        self.assertEqual(
            [
                "preprocessing.normalize",
                "preprocessing.score_policy",
                "nms.adaptive",
                "cut_detection.video",
                "tracking.greedy",
                "approximation.polygon.rdp",
                "keyframes.polygon.interval",
                "gap_fill.polygon.linear",
                "evaluation.mask_iou",
                "artifacts.validate",
            ],
            implementations,
        )

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
                                "label": "target",
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
