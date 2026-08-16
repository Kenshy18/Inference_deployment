from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from common.config import PipelineConfig, StageSpec
from common.runner import PipelineRunner
from contracts.stages import StageContext, StageResult
from run_pipeline import _configured_pipeline, build_parser, run_pipeline
from tests.helpers import write_sample_sqlite


@dataclass(frozen=True)
class MarkerStage:
    message: str = "inserted"
    name: str = "test_marker"
    requires: frozenset[str] = frozenset({"predictions_sqlite"})
    provides: frozenset[str] = frozenset({"marker"})

    def run(self, context: StageContext) -> StageResult:
        output = context.stage_dir / "marker.txt"
        output.write_text(self.message, encoding="utf-8")
        return StageResult({"marker": output})


@dataclass(frozen=True)
class MalformedCutStage:
    name: str = "malformed_cut"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset({"cuts_json"})

    def run(self, context: StageContext) -> StageResult:
        output = context.stage_dir / "cuts.json"
        output.write_text("not-json", encoding="utf-8")
        return StageResult({"cuts_json": output})


class PipelineTests(unittest.TestCase):
    def test_cli_defaults_to_promoted_production_polygon(self) -> None:
        args = build_parser().parse_args(
            ["--input-jsonl", "input.jsonl", "--output-dir", "output"]
        )
        config = _configured_pipeline(args)
        implementations = [stage.implementation for stage in config.stages]
        self.assertIn("nms.production_v3", implementations)
        self.assertIn("production.polygon_v3_cpu", implementations)
        self.assertNotIn("nms.adaptive", implementations)
        self.assertNotIn("approximation.polygon.production_v22", implementations)

    def test_artifact_validation_is_cached_until_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.bin"
            path.write_bytes(b"first")
            runner = PipelineRunner(PipelineConfig("empty", ()), Path(temporary))
            with patch("common.runner.validate_artifact") as validate:
                runner._validate_once("marker", path)
                runner._validate_once("marker", path)
                self.assertEqual(1, validate.call_count)
                path.write_bytes(b"changed-size")
                runner._validate_once("marker", path)
                self.assertEqual(2, validate.call_count)

    def test_precomputed_cuts_replace_video_cut_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = build_parser().parse_args(
                [
                    "--input-jsonl",
                    str(root / "input.jsonl"),
                    "--input-video",
                    str(root / "input.mp4"),
                    "--output-dir",
                    str(root / "output"),
                    "--precomputed-cuts-json",
                    str(root / "cuts.json"),
                ]
            )
            config = _configured_pipeline(args)
            self.assertNotIn(
                "cut_detection.video",
                [stage.implementation for stage in config.stages],
            )
            tracking = next(
                stage
                for stage in config.stages
                if stage.implementation == "tracking.greedy"
            )
            self.assertTrue(tracking.enabled)

    def test_pipeline_options_are_preserved_until_cli_explicitly_overrides(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "pipeline.json"
            config_path.write_text(
                json.dumps(
                    {
                        "name": "configured",
                        "stages": [
                            {
                                "id": "score",
                                "implementation": "preprocessing.score_policy",
                                "options": {"score_min": 0.91},
                            },
                            {
                                "id": "cut",
                                "implementation": "cut_detection.video",
                                "options": {
                                    "enabled": False,
                                    "method": "frame_diff",
                                },
                            },
                            {
                                "id": "production_polygon",
                                "implementation": "production.polygon_v3_cpu",
                                "options": {"target_interval": 99},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            base_arguments = [
                "--input-jsonl",
                str(root / "input.jsonl"),
                "--output-dir",
                str(root / "output"),
                "--pipeline-config",
                str(config_path),
            ]
            configured = _configured_pipeline(build_parser().parse_args(base_arguments))
            self.assertEqual(0.91, configured.stages[0].options["score_min"])
            self.assertFalse(configured.stages[1].options["enabled"])
            self.assertEqual("frame_diff", configured.stages[1].options["method"])
            self.assertEqual(99, configured.stages[2].options["target_interval"])

            overridden = _configured_pipeline(
                build_parser().parse_args(
                    [
                        *base_arguments,
                        "--score-min",
                        "0.2",
                        "--cut-detect",
                        "--cut-method",
                        "high_precision",
                        "--keyframe-interval",
                        "4",
                    ]
                )
            )
            self.assertEqual(0.2, overridden.stages[0].options["score_min"])
            self.assertTrue(overridden.stages[1].options["enabled"])
            self.assertEqual("high_precision", overridden.stages[1].options["method"])
            self.assertEqual(4, overridden.stages[2].options["target_interval"])

    def test_polygon_pipeline_supports_external_stage_insertion(self) -> None:
        config = PipelineConfig(
            "test_polygon",
            (
                StageSpec("production_polygon", "production.polygon_v3_cpu"),
                StageSpec(
                    "custom_filter",
                    "tests.test_pipeline:MarkerStage",
                    {"message": "after promoted production"},
                ),
                StageSpec("validation", "artifacts.validate"),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_sample_sqlite(root / "source.sqlite", frames=8)

            manifest = PipelineRunner(config, root / "output").run(
                {"tracked_sqlite": source}
            )

            self.assertTrue(manifest["complete"])
            self.assertEqual(
                [
                    "production_polygon",
                    "custom_filter",
                    "validation",
                ],
                [stage["id"] for stage in manifest["stages"]],
            )
            marker = Path(manifest["artifacts"]["marker"])
            self.assertEqual(
                "after promoted production",
                marker.read_text(encoding="utf-8"),
            )
            with sqlite3.connect(manifest["artifacts"]["predictions_sqlite"]) as db:
                self.assertEqual(
                    8, db.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
                )

    def test_missing_stage_artifact_fails_before_execution(self) -> None:
        config = PipelineConfig(
            "invalid",
            (StageSpec("custom", "tests.test_pipeline:MarkerStage"),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "predictions_sqlite"):
                PipelineRunner(config, Path(temporary)).run({})

    def test_malformed_declared_artifact_is_rejected_at_stage_boundary(self) -> None:
        config = PipelineConfig(
            "invalid_contract",
            (
                StageSpec(
                    "malformed",
                    "tests.test_pipeline:MalformedCutStage",
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "invalid artifact"):
                PipelineRunner(config, Path(temporary)).run({})

    def test_runtime_packages_do_not_import_removed_layers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        feature_roots = (
            "preprocessing",
            "nms",
            "cut_detection",
            "tracking",
            "approximation",
            "keyframes",
            "gap_fill",
            "evaluation",
            "artifacts",
            "visualization",
            "common",
            "contracts",
        )
        offenders: list[str] = []
        for relative in feature_roots:
            for path in (root / relative).rglob("*.py"):
                source = path.read_text(encoding="utf-8")
                if (
                    "atosyori_postprocess" in source
                    or "from workflow" in source
                    or "import workflow" in source
                ):
                    offenders.append(str(path.relative_to(root)))
        self.assertEqual([], offenders)

    def test_manifest_is_machine_readable(self) -> None:
        config = PipelineConfig(
            "single",
            (StageSpec("validation", "artifacts.validate"),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_sample_sqlite(root / "source.sqlite", frames=2)
            PipelineRunner(config, root / "output").run({"predictions_sqlite": source})
            manifest = json.loads(
                (root / "output" / "pipeline_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, manifest["schema_version"])
            self.assertTrue(manifest["complete"])

    def test_default_run_uses_modular_polygon_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_sample_sqlite(root / "source.sqlite", frames=6, label="男性器")
            args = build_parser().parse_args(
                [
                    "--input-sqlite",
                    str(source),
                    "--output-dir",
                    str(root / "output"),
                    "--keyframe-interval",
                    "2",
                ]
            )
            run_pipeline(args)
            manifest = json.loads(
                (root / "output" / "pipeline_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual("polygon_modular", manifest["pipeline"])
            self.assertEqual(
                "production.polygon_v3_cpu",
                manifest["stages"][0]["implementation"],
            )
            self.assertIn("keyframes_sqlite", manifest["artifacts"])

    def test_polygon_pipeline_accepts_relative_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_sample_sqlite(root / "source.sqlite", frames=6, label="男性器")
            args = build_parser().parse_args(
                [
                    "--input-sqlite",
                    str(source),
                    "--output-dir",
                    "relative-output",
                    "--keyframe-interval",
                    "2",
                ]
            )
            previous = Path.cwd()
            try:
                os.chdir(root)
                manifest = run_pipeline(args)
            finally:
                os.chdir(previous)
            result = Path(str(manifest["artifacts"]["predictions_sqlite"]))
            self.assertTrue(result.is_absolute())
            self.assertTrue(result.is_file())

    def test_retired_genital_geometry_cli_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "--input-sqlite",
                    "source.sqlite",
                    "--output-dir",
                    "output",
                    "--device",
                    "cpu",
                ]
            )


if __name__ == "__main__":
    unittest.main()
