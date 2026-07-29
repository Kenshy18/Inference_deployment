from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from orchestration.config import OrchestrationConfig, OrchestrationConfigError
from orchestration.runner import OrchestrationRunner

from helpers import create_video


class ConfigTests(unittest.TestCase):
    def test_mode_aware_defaults_keep_face_only_and_raw_only_configs_valid(
        self,
    ) -> None:
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
                        "output_root": str(root / "face-output"),
                        "inference": {
                            "enabled": False,
                            "input_sqlite": str(sqlite),
                            "mode": "face",
                            "face_model": "face_dino_v2",
                        },
                    }
                ),
                encoding="utf-8",
            )
            face = OrchestrationConfig.load(config_path)
            self.assertFalse(face.postprocess.enabled)
            self.assertFalse(face.overlay.raw)
            self.assertFalse(face.overlay.tracked)
            self.assertFalse(face.overlay.final)
            self.assertTrue(face.overlay.faces)

            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(root / "raw-output"),
                        "inference": {
                            "enabled": False,
                            "input_sqlite": str(sqlite),
                            "mode": "segmentation",
                        },
                        "postprocess": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            raw = OrchestrationConfig.load(config_path)
            self.assertTrue(raw.overlay.raw)
            self.assertFalse(raw.overlay.tracked)
            self.assertFalse(raw.overlay.final)
            self.assertFalse(raw.overlay.faces)

    def test_face_only_privacy_mask_is_packaged_without_segmentation_pipeline(
        self,
    ) -> None:
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
                            "mode": "face",
                            "face_model": "face_dino_v2",
                        },
                        "postprocess": {
                            "face_mask_target": "eyes",
                            "eye_mask_shape": "rectangle",
                        },
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            config = OrchestrationConfig.load(config_path)
            self.assertFalse(config.postprocess.enabled)
            command = OrchestrationRunner(config).package_result_command(
                inference_sqlite=sqlite,
                output=root / "result.sqlite",
            )
            self.assertEqual(
                "eyes",
                command[command.index("--face-mask-target") + 1],
            )
            self.assertEqual(
                "rectangle",
                command[command.index("--eye-mask-shape") + 1],
            )

    def test_parallel_models_is_available_only_for_compact_dino_and_new_face(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            config_path = root / "config.json"

            def write_config(
                *,
                mode: str,
                segmentation_model: str,
                face_model: str,
            ) -> None:
                config_path.write_text(
                    json.dumps(
                        {
                            "input_video": str(video),
                            "output_root": str(root / "output"),
                            "execution": {"runtime_python": sys.executable},
                            "inference": {
                                "enabled": True,
                                "mode": mode,
                                "segmentation_model": segmentation_model,
                                "face_model": face_model,
                                "parallel_models": True,
                            },
                            "postprocess": {"enabled": False},
                            "overlay": {"enabled": False},
                        }
                    ),
                    encoding="utf-8",
                )

            write_config(
                mode="segmentation-face",
                segmentation_model="dinov3_codino_mh0",
                face_model="face_dino_v2",
            )
            approved = OrchestrationConfig.load(config_path)
            self.assertIn(
                "--parallel-models",
                OrchestrationRunner(approved).inference_command(
                    root / "inference.sqlite"
                ),
            )

            invalid = (
                ("segmentation-face", "dinov3_codino", "face_dino_v2"),
                (
                    "segmentation-face",
                    "dinov3_codino_mh0",
                    "rtdetr_head_face",
                ),
                ("segmentation", "dinov3_codino_mh0", "face_dino_v2"),
            )
            for mode, segmentation_model, face_model in invalid:
                with self.subTest(
                    mode=mode,
                    segmentation_model=segmentation_model,
                    face_model=face_model,
                ):
                    write_config(
                        mode=mode,
                        segmentation_model=segmentation_model,
                        face_model=face_model,
                    )
                    with self.assertRaisesRegex(
                        OrchestrationConfigError,
                        "mode=segmentation-face",
                    ):
                        OrchestrationConfig.load(config_path)

    def test_class_postprocess_policy_is_typed_and_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            policy = root / "class-postprocess.json"
            policy.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "default": {
                            "shape_mode": "polygon",
                            "keyframe_interval": 3,
                            "max_gap": 0,
                        },
                        "classes": {
                            "target": {
                                "shape_mode": "ellipse",
                                "keyframe_interval": 2,
                                "max_gap": 12,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(root / "output"),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": True,
                            "mode": "segmentation",
                            "segmentation_model": "dinov3_codino",
                        },
                        "postprocess": {
                            "enabled": True,
                            "shape_mode": "polygon",
                            "class_postprocess_policy_json": str(policy),
                            "keyframe_interval": 4,
                            "max_gap": 9,
                            "device": "cuda:0",
                        },
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            config = OrchestrationConfig.load(config_path)
            command = OrchestrationRunner(config).postprocess_command(
                root / "inference.sqlite"
            )

            self.assertEqual(
                policy.resolve(),
                config.postprocess.class_postprocess_policy_json,
            )
            self.assertTrue(config.postprocess.uses_gpu)
            self.assertEqual(
                str(policy.resolve()),
                command[command.index("--class-postprocess-policy-json") + 1],
            )
            self.assertEqual(
                "9",
                command[command.index("--max-gap") + 1],
            )

    def test_invalid_class_postprocess_policy_is_rejected_early(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            policy = root / "class-postprocess.json"
            policy.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "classes": {
                            "target": {
                                "shape_mode": "spline",
                                "keyframe_interval": 0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(root / "output"),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": True,
                            "mode": "segmentation",
                            "segmentation_model": "dinov3_codino",
                        },
                        "postprocess": {
                            "enabled": True,
                            "class_postprocess_policy_json": str(policy),
                        },
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                OrchestrationConfigError,
                "shape_mode must be polygon or ellipse",
            ):
                OrchestrationConfig.load(config_path)

    def test_cut_precompute_records_its_own_elapsed_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(root / "output"),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": True,
                            "mode": "segmentation",
                            "segmentation_model": "dinov3_codino",
                        },
                        "postprocess": {
                            "enabled": True,
                            "device": "cpu",
                            "cut_detect": True,
                            "cut_method": "high_precision",
                            "precompute_cuts_during_inference": True,
                        },
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            runner = OrchestrationRunner(OrchestrationConfig.load(config_path))
            stage = runner._start_cut_precompute()
            self.assertIsNotNone(stage)
            assert stage is not None
            stage.waiter.join(timeout=10.0)
            self.assertFalse(stage.waiter.is_alive())

            # Delay collection just as a long inference stage would. The cut
            # elapsed time must remain the subprocess duration, while the
            # overlap window includes the delayed collection.
            time.sleep(0.1)
            cuts = runner._finish_background(stage)
            self.assertTrue(cuts.is_file())
            record = next(
                item
                for item in runner.manifest["stages"]
                if item["name"] == "cut_precompute"
            )
            self.assertLess(
                record["elapsed_seconds"] + 0.05,
                record["overlap_window_seconds"],
            )

    def test_cut_precompute_overlap_is_typed_and_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(root / "output"),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": True,
                            "mode": "segmentation",
                            "segmentation_model": "dinov3_codino",
                        },
                        "postprocess": {
                            "enabled": True,
                            "device": "cpu",
                            "cut_detect": True,
                            "cut_method": "high_precision",
                            "precompute_cuts_during_inference": True,
                        },
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            config = OrchestrationConfig.load(config_path)
            runner = OrchestrationRunner(config, dry_run=True)
            plan = runner.plan()
            postprocess = next(
                stage for stage in plan["stages"] if stage["stage"] == "postprocess"
            )
            command = postprocess["command"]
            self.assertIn("--precomputed-cuts-json", command)
            self.assertTrue(config.postprocess.precompute_cuts_during_inference)

    def test_face_only_cut_precompute_is_consumed_by_result_packaging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(root / "output"),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": True,
                            "mode": "face",
                            "face_model": "rtdetr_head_face",
                        },
                        "postprocess": {
                            "enabled": False,
                            "cut_detect": True,
                            "cut_method": "high_precision",
                            "precompute_cuts_during_inference": True,
                            "face_mask_target": "none",
                        },
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )

            config = OrchestrationConfig.load(config_path)
            plan = OrchestrationRunner(config, dry_run=True).plan()

            self.assertEqual("cut_precompute", plan["stages"][0]["stage"])
            packaging = next(
                stage
                for stage in plan["stages"]
                if stage["stage"] == "result_packaging"
            )
            self.assertIn("--precomputed-cuts-json", packaging["command"])

    def test_face_trt_bundle_is_typed_and_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            bundle = root / "face-b16-manifest.json"
            bundle.write_text("{}", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(root / "output"),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": True,
                            "mode": "face",
                            "face_model": "face_dino_v2",
                            "face_trt_bundle": str(bundle),
                        },
                        "postprocess": {"enabled": False},
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            config = OrchestrationConfig.load(config_path)
            command = OrchestrationRunner(config).inference_command(
                root / "output.sqlite"
            )
            self.assertEqual(bundle.resolve(), config.inference.face_trt_bundle)
            self.assertEqual(
                str(bundle.resolve()),
                command[command.index("--face-trt-bundle") + 1],
            )

    def test_face_privacy_postprocess_options_are_typed_and_forwarded(self) -> None:
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
                            "mode": "segmentation-face",
                            "face_model": "face_dino_v2",
                        },
                        "postprocess": {
                            "enabled": True,
                            "device": "cpu",
                            "face_mask_target": "eyes",
                            "eye_mask_shape": "rectangle",
                            "minimum_eye_confidence": 0.4,
                        },
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )

            config = OrchestrationConfig.load(config_path)
            command = OrchestrationRunner(
                config,
                dry_run=True,
            ).postprocess_command(sqlite)

            self.assertEqual("eyes", config.postprocess.face_mask_target)
            self.assertEqual("rectangle", config.postprocess.eye_mask_shape)
            self.assertEqual(
                "eyes",
                command[command.index("--face-mask-target") + 1],
            )
            self.assertEqual(
                "rectangle",
                command[command.index("--eye-mask-shape") + 1],
            )
            self.assertEqual(
                "0.4",
                command[command.index("--minimum-eye-confidence") + 1],
            )

    def test_face_privacy_requires_face_dino_v2_inference(self) -> None:
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
                        "postprocess": {
                            "enabled": True,
                            "face_mask_target": "eyes",
                        },
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                OrchestrationConfigError,
                "requires face inference",
            ):
                OrchestrationConfig.load(config_path)

    def test_ellipse_postprocess_accepts_cuda_and_is_planned_on_gpu(self) -> None:
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
                        "postprocess": {
                            "enabled": True,
                            "shape_mode": "ellipse",
                            "device": "cuda:0",
                        },
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            config = OrchestrationConfig.load(config_path)
            self.assertTrue(config.postprocess.uses_gpu)
            plan = OrchestrationRunner(config, dry_run=True).plan()
            postprocess = next(
                stage for stage in plan["stages"] if stage["stage"] == "postprocess"
            )
            self.assertTrue(postprocess["uses_gpu"])
            self.assertEqual(
                "cuda:0",
                postprocess["command"][postprocess["command"].index("--device") + 1],
            )

    def test_invalid_postprocess_device_is_rejected(self) -> None:
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
                            "shape_mode": "ellipse",
                            "device": "gpu",
                        },
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                OrchestrationConfigError,
                "must be cpu, auto, cuda",
            ):
                OrchestrationConfig.load(config_path)

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
                            "face_model": "face_dino_v2",
                            "face_backend": "tensorrt-fast",
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
            self.assertEqual(
                "face_dino_v2",
                command[command.index("--face-model") + 1],
            )
            self.assertEqual(
                "tensorrt-fast",
                command[command.index("--face-backend") + 1],
            )
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

    def test_typed_k2_and_ffmpeg_options_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            sqlite = root / "input.sqlite"
            sqlite.touch()
            ffmpeg = root / "ffmpeg"
            ffmpeg.touch()
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
                        "postprocess": {
                            "enabled": True,
                            "shape_mode": "ellipse",
                            "device": "cuda:0",
                            "k2_batch_size": 128,
                            "k2_prep_workers": 4,
                            "k2_precision": "fp16",
                            "k2_forward_mode": "states_only",
                            "k2_profile_stages": False,
                            "k2_cudnn_benchmark": "on",
                            "k2_tf32": "off",
                        },
                        "overlay": {
                            "enabled": True,
                            "execution_mode": "fast",
                            "raw": True,
                            "tracked": False,
                            "final": False,
                            "ffmpeg_bin": str(ffmpeg),
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = OrchestrationConfig.load(config_path)
            postprocess = OrchestrationRunner(config).postprocess_command(sqlite)
            expected = {
                "--k2-batch-size": "128",
                "--k2-prep-workers": "4",
                "--k2-precision": "fp16",
                "--k2-forward-mode": "states_only",
                "--k2-cudnn-benchmark": "on",
                "--k2-tf32": "off",
            }
            for flag, value in expected.items():
                self.assertEqual(postprocess[postprocess.index(flag) + 1], value)
            self.assertIn("--no-k2-profile-stages", postprocess)

            overlay = OrchestrationRunner(config).overlay_command(
                mode="raw",
                source_sqlite=sqlite,
                output=root / "output" / "raw.mp4",
            )
            self.assertEqual(
                overlay[overlay.index("--ffmpeg-bin") + 1],
                str(ffmpeg),
            )

    def test_legacy_sqlite_export_is_typed_and_forwarded(self) -> None:
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
                        "postprocess": {
                            "enabled": True,
                            "export_legacy_sqlite": True,
                            "device": "cpu",
                        },
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            config = OrchestrationConfig.load(config_path)
            self.assertTrue(config.postprocess.export_legacy_sqlite)
            command = OrchestrationRunner(config).postprocess_command(sqlite)
            self.assertIn("--export-legacy-sqlite", command)

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
