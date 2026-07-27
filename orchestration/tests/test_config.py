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

    def test_experimental_overlay_builds_segmented_gpu_command(self) -> None:
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
            self.assertIn("benchmark_segmented.py", " ".join(command))
            self.assertIn("--gpu-pipeline", command)
            self.assertEqual(command[command.index("--workers") + 1], "6")
            self.assertEqual(command[command.index("--cpu-workers") + 1], "3")
            self.assertIn("--copy-audio", command)

    def test_experimental_overlay_rejects_invalid_worker_split(self) -> None:
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
