from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from orchestration.config import OrchestrationConfig, OrchestrationConfigError
from orchestration.runner import OrchestrationRunner

from helpers import create_video


class ConfigTests(unittest.TestCase):
    def test_three_overlay_execution_modes_select_expected_engines(self) -> None:
        cases = (
            ("cpu", "python_opencv", "h264", False),
            ("nvenc", "python_opencv", "h264_nvenc", True),
            (
                "fast",
                "native",
                "h264_nvenc",
                True,
            ),
        )
        for (
            execution_mode,
            backend,
            codec,
            uses_nvenc,
        ) in cases:
            with self.subTest(execution_mode=execution_mode):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    video = create_video(root / "input.avi")
                    sqlite = root / "input.sqlite"
                    sqlite.touch()
                    overlay: dict[str, object] = {
                        "enabled": True,
                        "execution_mode": execution_mode,
                        "raw": True,
                        "tracked": False,
                        "final": False,
                    }
                    if execution_mode == "fast":
                        overlay["target_bitrate_mbps"] = 8.0
                    config_path = root / "config.json"
                    config_path.write_text(
                        json.dumps(
                            {
                                "input_video": str(video),
                                "output_root": str(root / "output"),
                                "execution": {"runtime_python": sys.executable},
                                "inference": {
                                    "enabled": False,
                                    "input_sqlite": str(sqlite),
                                    "mode": "segmentation",
                                },
                                "postprocess": {"enabled": False},
                                "overlay": overlay,
                            }
                        ),
                        encoding="utf-8",
                    )
                    config = OrchestrationConfig.load(config_path)
                    self.assertEqual(
                        config.overlay.execution_mode,
                        execution_mode,
                    )
                    self.assertEqual(config.overlay.backend, backend)
                    self.assertEqual(config.overlay.codec, codec)
                    self.assertEqual(
                        config.overlay.uses_nvenc,
                        uses_nvenc,
                    )
                    command = OrchestrationRunner(
                        config,
                        dry_run=True,
                    ).overlay_command(
                        mode="raw",
                        source_sqlite=sqlite,
                        output=root / "output" / "raw.mp4",
                    )
                    self.assertNotIn("segmented.py", " ".join(command))
                    self.assertEqual(
                        command[command.index("--execution-mode") + 1],
                        execution_mode,
                    )
                    if execution_mode == "fast":
                        self.assertNotIn("--codec", command)
                        self.assertEqual(
                            command[command.index("--cpu-workers") + 1],
                            "0",
                        )
                    elif execution_mode == "cpu":
                        self.assertEqual(
                            command[command.index("--h264-crf") + 1],
                            "18",
                        )
                        self.assertEqual(
                            command[command.index("--h264-preset") + 1],
                            "veryfast",
                        )
                    else:
                        self.assertEqual(
                            command[command.index("--nvenc-cq") + 1],
                            "18",
                        )

    def test_dry_run_builds_inference_command_without_executing_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "input_video": str(video),
                        "output_root": str(root / "output"),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": True,
                            "mode": "segmentation-face",
                            "segmentation_model": "dinov3_codino",
                            "segmentation_backend": "tensorrt-fast",
                            "device": "cuda:0",
                        },
                        "postprocess": {"enabled": True, "device": "cpu"},
                        "overlay": {
                            "enabled": True,
                            "faces": True,
                            "final_include_faces": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = OrchestrationConfig.load(config_path)
            plan = OrchestrationRunner(config, dry_run=True).plan()
            inference = plan["stages"][0]
            command = inference["command"]
            self.assertIn("--segmentation-model", command)
            self.assertIn("dinov3_codino", command)
            self.assertIn("--face-model", command)
            self.assertTrue(inference["uses_gpu"])
            self.assertFalse((root / "output").exists())

    def test_cpu_stage_cannot_be_overridden_by_extra_args(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            sqlite = root / "input.sqlite"
            sqlite.touch()
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(root / "output"),
                        "inference": {
                            "enabled": False,
                            "input_sqlite": str(sqlite),
                            "mode": "segmentation",
                        },
                        "postprocess": {
                            "enabled": True,
                            "device": "cpu",
                            "extra_args": ["--device", "cuda"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                OrchestrationConfigError,
                "must not override",
            ):
                OrchestrationConfig.load(config_path)

    def test_nvenc_overlay_is_planned_as_gpu_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            sqlite = root / "input.sqlite"
            sqlite.touch()
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(root / "output"),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": False,
                            "input_sqlite": str(sqlite),
                            "mode": "segmentation",
                        },
                        "postprocess": {"enabled": False},
                        "overlay": {
                            "enabled": True,
                            "raw": True,
                            "tracked": False,
                            "final": False,
                            "codec": "h264_nvenc",
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = OrchestrationConfig.load(config_path)
            plan = OrchestrationRunner(config, dry_run=True).plan()
            overlay = next(
                stage for stage in plan["stages"] if stage["stage"] == "overlay"
            )
            self.assertTrue(config.overlay.uses_nvenc)
            self.assertTrue(overlay["uses_gpu"])
            command = OrchestrationRunner(config).overlay_command(
                mode="raw",
                source_sqlite=sqlite,
                output=root / "output" / "raw.mp4",
            )
            self.assertEqual(
                command[command.index("--nvenc-preset") + 1],
                "p5",
            )
            self.assertEqual(
                command[command.index("--nvenc-gpu") + 1],
                "0",
            )

    def test_overlay_quality_settings_are_typed_and_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            sqlite = root / "input.sqlite"
            sqlite.touch()
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(root / "output"),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": False,
                            "input_sqlite": str(sqlite),
                            "mode": "segmentation",
                        },
                        "postprocess": {"enabled": False},
                        "overlay": {
                            "enabled": True,
                            "execution_mode": "cpu",
                            "raw": True,
                            "tracked": False,
                            "final": False,
                            "h264_crf": 16,
                            "h264_preset": "fast",
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = OrchestrationConfig.load(config_path)
            command = OrchestrationRunner(config, dry_run=True).overlay_command(
                mode="raw",
                source_sqlite=sqlite,
                output=root / "output" / "raw.mp4",
            )
            self.assertEqual(command[command.index("--h264-crf") + 1], "16")
            self.assertEqual(command[command.index("--h264-preset") + 1], "fast")

    def test_legacy_experimental_backend_maps_to_fast_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            sqlite = root / "input.sqlite"
            sqlite.touch()
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(root / "output"),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": False,
                            "input_sqlite": str(sqlite),
                            "mode": "segmentation",
                        },
                        "postprocess": {"enabled": False},
                        "overlay": {
                            "enabled": True,
                            "backend": "experimental_cpp",
                            "raw": True,
                            "tracked": False,
                            "final": False,
                            "codec": "h264_nvenc",
                            "workers": 6,
                            "cpu_workers": 3,
                            "copy_audio": True,
                            "target_bitrate_mbps": 8.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = OrchestrationConfig.load(config_path)
            runner = OrchestrationRunner(config, dry_run=True)
            command = runner.overlay_command(
                mode="raw",
                source_sqlite=sqlite,
                output=root / "output" / "03_overlay" / "raw.mp4",
            )
            self.assertEqual(config.overlay.execution_mode, "fast")
            self.assertEqual(config.overlay.backend, "native")
            self.assertIn("overlay_renderer", " ".join(command))
            self.assertEqual(
                command[command.index("--execution-mode") + 1],
                "fast",
            )
            self.assertEqual(command[command.index("--workers") + 1], "6")
            self.assertEqual(command[command.index("--cpu-workers") + 1], "3")
            self.assertIn("--copy-audio", command)

    def test_fast_overlay_rejects_invalid_worker_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            sqlite = root / "input.sqlite"
            sqlite.touch()
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(root / "output"),
                        "inference": {
                            "enabled": False,
                            "input_sqlite": str(sqlite),
                            "mode": "segmentation",
                        },
                        "postprocess": {"enabled": False},
                        "overlay": {
                            "backend": "experimental_cpp",
                            "raw": True,
                            "tracked": False,
                            "final": False,
                            "codec": "h264_nvenc",
                            "workers": 3,
                            "cpu_workers": 4,
                            "target_bitrate_mbps": 8.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                OrchestrationConfigError,
                "cpu_workers",
            ):
                OrchestrationConfig.load(config_path)


if __name__ == "__main__":
    unittest.main()
