from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import cv2

from orchestration.config import OrchestrationConfig
from orchestration.runner import OrchestrationRunner

from helpers import (
    create_mask_sqlite,
    create_rich_face_unified_sqlite,
    create_unified_sqlite,
    create_video,
    keep_only_inference_role,
)


class WorkflowTests(unittest.TestCase):
    def test_overlay_defaults_to_fast_and_exposes_typed_presets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi", frames=1)
            inference = create_rich_face_unified_sqlite(root / "inference.sqlite")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(root / "run"),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": False,
                            "input_sqlite": str(inference),
                            "mode": "segmentation-face",
                            "face_model": "face_dino_v2",
                        },
                        "postprocess": {"enabled": False},
                        "overlay": {
                            "enabled": False,
                            "presets": [
                                "genital-simple",
                                "face-detailed",
                                "combined-simple",
                            ],
                            "face_mask_target": "eyes",
                            "eye_mask_shape": "rectangle",
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = OrchestrationConfig.load(config_path)
            command = OrchestrationRunner(config).overlay_command(
                mode=None,
                source_sqlite=inference,
                output=root / "combined.mp4",
                preset="combined-simple",
            )

            self.assertEqual("fast", config.overlay.execution_mode)
            self.assertEqual("native", config.overlay.backend)
            self.assertEqual(8.0, config.overlay.target_bitrate_mbps)
            self.assertIn("combined-simple", command)
            self.assertIn("--genital-source", command)
            self.assertIn("--face-mask-target", command)
            self.assertIn("rectangle", command)

    def test_face_only_privacy_mask_uses_the_same_final_masks_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi", frames=1)
            inference = keep_only_inference_role(
                create_rich_face_unified_sqlite(root / "inference.sqlite"),
                "face_detection",
            )
            output = root / "run"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(output),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": False,
                            "input_sqlite": str(inference),
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

            manifest = OrchestrationRunner(OrchestrationConfig.load(config_path)).run()
            result = Path(manifest["artifacts"]["result_sqlite"])
            with sqlite3.connect(result) as connection:
                self.assertEqual(
                    [("Eyes",)],
                    connection.execute(
                        """
                        SELECT label FROM tracks
                        WHERE domain='face_privacy'
                        """
                    ).fetchall(),
                )
                self.assertEqual(
                    [("eyes",)],
                    connection.execute(
                        """
                        SELECT mask_kind FROM mask_provenance
                        """
                    ).fetchall(),
                )
                capabilities = dict(
                    connection.execute(
                        """
                        SELECT name, available FROM result_capabilities
                        WHERE name IN (
                            'instance_segmentation',
                            'face_detection',
                            'tracking_assignments',
                            'face_tracking',
                            'final_annotations',
                            'face_privacy_masks'
                        )
                        """
                    )
                )
                self.assertEqual(
                    {
                        "instance_segmentation": 0,
                        "face_detection": 1,
                        "tracking_assignments": 0,
                        "face_tracking": 1,
                        "final_annotations": 1,
                        "face_privacy_masks": 1,
                    },
                    capabilities,
                )
                self.assertEqual(
                    [("face_privacy", "rectangle", 0)],
                    connection.execute(
                        """
                        SELECT domain, geometry_type, keyframe_index
                        FROM editable_keyframe_components
                        """
                    ).fetchall(),
                )
                self.assertEqual(
                    1,
                    connection.execute(
                        "SELECT COUNT(*) FROM keyframe_rectangles"
                    ).fetchone()[0],
                )

    def test_face_privacy_is_merged_with_reused_final_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi", frames=1)
            inference = create_rich_face_unified_sqlite(root / "inference.sqlite")
            final = create_mask_sqlite(root / "final.sqlite")
            output = root / "run"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(output),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": False,
                            "input_sqlite": str(inference),
                            "mode": "segmentation-face",
                            "face_model": "face_dino_v2",
                        },
                        "postprocess": {
                            "enabled": False,
                            "final_sqlite": str(final),
                            "face_mask_target": "eyes",
                        },
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )

            manifest = OrchestrationRunner(OrchestrationConfig.load(config_path)).run()
            result = Path(manifest["artifacts"]["result_sqlite"])
            with sqlite3.connect(result) as connection:
                self.assertEqual(
                    [("Eyes", 1), ("target", 1)],
                    connection.execute(
                        """
                        SELECT label, COUNT(*) FROM tracks
                        GROUP BY label ORDER BY label
                        """
                    ).fetchall(),
                )
                self.assertEqual(
                    1,
                    connection.execute(
                        """
                        SELECT available FROM result_capabilities
                        WHERE name='face_privacy_masks'
                        """
                    ).fetchone()[0],
                )

    def test_reused_tracked_and_final_sqlites_are_promoted_to_stable_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi", frames=1)
            inference = keep_only_inference_role(
                create_unified_sqlite(root / "inference.sqlite", frames=1),
                "instance_segmentation",
            )
            tracked = create_mask_sqlite(root / "tracked.sqlite")
            final = create_mask_sqlite(root / "final.sqlite")
            output = root / "run"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(output),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": False,
                            "input_sqlite": str(inference),
                            "mode": "segmentation",
                        },
                        "postprocess": {
                            "enabled": False,
                            "tracked_sqlite": str(tracked),
                            "final_sqlite": str(final),
                        },
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )

            manifest = OrchestrationRunner(OrchestrationConfig.load(config_path)).run()
            result = Path(manifest["artifacts"]["result_sqlite"])
            with sqlite3.connect(result) as connection:
                self.assertEqual(
                    (0, 1),
                    (
                        connection.execute(
                            "SELECT COUNT(*) FROM tracking_assignments"
                        ).fetchone()[0],
                        connection.execute(
                            "SELECT COUNT(*) FROM mask_keyframes"
                        ).fetchone()[0],
                    ),
                )
                self.assertEqual(
                    [("final_annotations", 1), ("tracking_assignments", 0)],
                    connection.execute(
                        """
                        SELECT name, available
                        FROM result_capabilities
                        WHERE name IN (
                            'tracking_assignments', 'final_annotations'
                        )
                        ORDER BY name
                        """
                    ).fetchall(),
                )

    def test_face_privacy_is_merged_into_software_final_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi", frames=1)
            inference = create_rich_face_unified_sqlite(root / "inference.sqlite")
            output = root / "run"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(output),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": False,
                            "input_sqlite": str(inference),
                            "mode": "segmentation-face",
                            "face_model": "face_dino_v2",
                        },
                        "postprocess": {
                            "enabled": True,
                            "export_legacy_sqlite": True,
                            "shape_mode": "polygon",
                            "cut_detect": False,
                            "remove_short_tracks_max_frames": 0,
                            "device": "cpu",
                            "face_mask_target": "eyes",
                            "eye_mask_shape": "rectangle",
                        },
                        "overlay": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )

            manifest = OrchestrationRunner(OrchestrationConfig.load(config_path)).run()

            final = Path(manifest["artifacts"]["result_sqlite"])
            legacy = Path(manifest["artifacts"]["legacy_final_sqlite"])
            with sqlite3.connect(final) as connection:
                labels = connection.execute(
                    "SELECT label, COUNT(*) FROM tracks GROUP BY label"
                ).fetchall()
                self.assertIn(("target", 1), labels)
                self.assertIn(("Eyes", 1), labels)
                self.assertEqual(
                    [
                        (
                            "eyes",
                            "ellipse-fallback",
                            "face-privacy-geometry-v1",
                        )
                    ],
                    connection.execute(
                        """
                        SELECT mask_kind, derivation, algorithm_version
                        FROM mask_provenance
                        """
                    ).fetchall(),
                )
            with sqlite3.connect(legacy) as connection:
                self.assertEqual(
                    [("Eyes", 1), ("target", 1)],
                    connection.execute(
                        """
                        SELECT label, COUNT(*) FROM masks
                        GROUP BY label ORDER BY label
                        """
                    ).fetchall(),
                )

    def test_reused_face_dino_v2_sqlite_renders_rich_face_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi", frames=1)
            inference = create_rich_face_unified_sqlite(root / "inference.sqlite")
            output = root / "run"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_video": str(video),
                        "output_root": str(output),
                        "execution": {"runtime_python": sys.executable},
                        "inference": {
                            "enabled": False,
                            "input_sqlite": str(inference),
                            "mode": "segmentation-face",
                            "face_model": "face_dino_v2",
                        },
                        "postprocess": {"enabled": False},
                        "overlay": {
                            "enabled": True,
                            "execution_mode": "cpu",
                            "raw": False,
                            "tracked": False,
                            "final": False,
                            "faces": True,
                            "progress_every": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = OrchestrationRunner(OrchestrationConfig.load(config_path)).run()

            self.assertEqual("complete", manifest["status"])
            self.assertEqual(
                1,
                manifest["validation"]["result_sqlite"]["inference"][
                    "face_observations"
                ],
            )
            self.assertNotIn("inference_sqlite", manifest["artifacts"])
            self.assertIn("result_sqlite", manifest["artifacts"])
            overlay = Path(manifest["artifacts"]["overlay_faces"])
            self.assertTrue(overlay.is_file())
            capture = cv2.VideoCapture(str(overlay))
            self.assertTrue(capture.isOpened())
            self.assertEqual(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            capture.release()

    def test_public_result_contract_is_stable_across_inference_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi", frames=1)
            inputs = {
                "segmentation": keep_only_inference_role(
                    create_unified_sqlite(root / "segmentation.sqlite", frames=1),
                    "instance_segmentation",
                ),
                "face-old": keep_only_inference_role(
                    create_unified_sqlite(root / "face-old.sqlite", frames=1),
                    "face_detection",
                ),
                "face-new": keep_only_inference_role(
                    create_rich_face_unified_sqlite(root / "face-new.sqlite"),
                    "face_detection",
                ),
                "combined": create_rich_face_unified_sqlite(root / "combined.sqlite"),
            }
            settings = {
                "segmentation": ("segmentation", "rtdetr_head_face"),
                "face-old": ("face", "rtdetr_head_face"),
                "face-new": ("face", "face_dino_v2"),
                "combined": ("segmentation-face", "face_dino_v2"),
            }
            schemas: dict[str, dict[str, tuple[str, ...]]] = {}
            capabilities: dict[str, dict[str, int]] = {}
            for name, source in inputs.items():
                mode, face_model = settings[name]
                output = root / f"run-{name}"
                config_path = root / f"{name}.json"
                config_path.write_text(
                    json.dumps(
                        {
                            "input_video": str(video),
                            "output_root": str(output),
                            "execution": {"runtime_python": sys.executable},
                            "inference": {
                                "enabled": False,
                                "input_sqlite": str(source),
                                "mode": mode,
                                "face_model": face_model,
                            },
                            "postprocess": {"enabled": False},
                            "overlay": {"enabled": False},
                        }
                    ),
                    encoding="utf-8",
                )
                manifest = OrchestrationRunner(
                    OrchestrationConfig.load(config_path)
                ).run()
                result = Path(manifest["artifacts"]["result_sqlite"])
                with sqlite3.connect(result) as connection:
                    stable_tables = (
                        "schema_info",
                        "videos",
                        "video_streams",
                        "runs",
                        "run_metadata",
                        "model_executions",
                        "model_metadata",
                        "frames",
                        "detections",
                        "classifications",
                        "classification_probabilities",
                        "segmentations",
                        "segmentation_polygons",
                        "segmentation_points",
                        "result_schema_info",
                        "result_capabilities",
                        "result_components",
                        "processing_runs",
                        "processing_stage_runs",
                        "face_observations",
                        "face_keypoints",
                        "face_masks",
                        "face_keypoint_class_probabilities",
                        "face_keypoint_state_probabilities",
                        "annotation_state",
                        "tracking_assignments",
                        "face_tracks",
                        "face_tracking_assignments",
                        "face_track_interpolations",
                        "tracks",
                        "cuts",
                        "cut_detection_metadata",
                        "raw_tracks",
                        "class_postprocess_policies",
                        "mask_postprocess_provenance",
                        "mask_provenance",
                        "mask_track_segments",
                        "mask_keyframes",
                        "keyframe_components",
                        "keyframe_ellipses",
                        "keyframe_rectangles",
                        "keyframe_polygon_rings",
                        "keyframe_polygon_points",
                        "mask_geometry_provenance",
                    )
                    schemas[name] = {
                        table: tuple(
                            str(row[1])
                            for row in connection.execute(
                                f'PRAGMA table_info("{table}")'
                            )
                        )
                        for table in stable_tables
                    }
                    schemas[name].update(
                        {
                            f"view:{view}": (
                                connection.execute(
                                    """
                                    SELECT sql FROM sqlite_master
                                    WHERE type='view' AND name=?
                                    """,
                                    (view,),
                                ).fetchone()[0],
                            )
                            for view in (
                                "editable_keyframe_components",
                                "editable_polygon_vertices",
                            )
                        }
                    )
                    capabilities[name] = {
                        str(capability): int(available)
                        for capability, available in connection.execute(
                            """
                            SELECT name, available FROM result_capabilities
                            """
                        )
                    }

            reference = schemas["segmentation"]
            self.assertTrue(all(schema == reference for schema in schemas.values()))
            self.assertEqual(
                {
                    "instance_segmentation": 1,
                    "face_detection": 0,
                    "rich_face_geometry": 0,
                    "tracking_assignments": 0,
                    "final_annotations": 0,
                },
                {
                    key: capabilities["segmentation"][key]
                    for key in (
                        "instance_segmentation",
                        "face_detection",
                        "rich_face_geometry",
                        "tracking_assignments",
                        "final_annotations",
                    )
                },
            )
            self.assertEqual(1, capabilities["face-old"]["face_detection"])
            self.assertEqual(0, capabilities["face-old"]["rich_face_geometry"])
            self.assertEqual(1, capabilities["face-new"]["face_detection"])
            self.assertEqual(1, capabilities["face-new"]["rich_face_geometry"])
            self.assertEqual(1, capabilities["combined"]["instance_segmentation"])
            self.assertEqual(1, capabilities["combined"]["face_detection"])

    def test_reused_inference_runs_real_postprocess_and_all_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            inference = create_unified_sqlite(root / "inference.sqlite")
            output = root / "run"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "input_video": str(video),
                        "output_root": str(output),
                        "execution": {
                            "runtime_python": sys.executable,
                            "resume": False,
                        },
                        "inference": {
                            "enabled": False,
                            "input_sqlite": str(inference),
                            "mode": "segmentation-face",
                        },
                        "postprocess": {
                            "enabled": True,
                            "export_legacy_sqlite": True,
                            "shape_mode": "polygon",
                            "cut_detect": False,
                            "remove_short_tracks_max_frames": 0,
                            "device": "cpu",
                        },
                        "overlay": {
                            "enabled": True,
                            "execution_mode": "cpu",
                            "raw": True,
                            "tracked": True,
                            "final": True,
                            "faces": True,
                            "final_include_faces": True,
                            "progress_every": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = OrchestrationConfig.load(config_path)
            manifest = OrchestrationRunner(config).run()

            self.assertEqual("complete", manifest["status"])
            self.assertNotIn("inference_sqlite", manifest["artifacts"])
            self.assertNotIn("tracked_sqlite", manifest["artifacts"])
            self.assertNotIn("final_sqlite", manifest["artifacts"])
            result = Path(manifest["artifacts"]["result_sqlite"])
            self.assertTrue(result.is_file())
            legacy = Path(manifest["artifacts"]["legacy_final_sqlite"])
            self.assertTrue(legacy.is_file())
            with sqlite3.connect(result) as connection:
                self.assertEqual(
                    [("disabled", 0, "first_frame_of_new_scene")],
                    connection.execute(
                        """
                        SELECT method, cut_count, frame_semantics
                        FROM cut_detection_metadata
                        """
                    ).fetchall(),
                )
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                self.assertIn("tracking_assignments", tables)
                self.assertNotIn("raw_tracked_masks", tables)
                self.assertNotIn("tracked_masks", tables)
                self.assertNotIn("masks", tables)
                self.assertEqual(
                    "video-mask-integrated-result",
                    connection.execute(
                        """
                        SELECT value FROM result_schema_info
                        WHERE key='schema_name'
                        """
                    ).fetchone()[0],
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM segmentations").fetchone()[
                        0
                    ],
                    connection.execute(
                        "SELECT COUNT(*) FROM tracking_assignments"
                    ).fetchone()[0],
                )
                self.assertEqual(
                    [("orchestration",), ("postprocess",)],
                    connection.execute(
                        """
                        SELECT kind FROM processing_runs
                        WHERE kind IN ('orchestration', 'postprocess')
                        ORDER BY kind
                        """
                    ).fetchall(),
                )
                self.assertGreater(
                    connection.execute(
                        "SELECT COUNT(*) FROM processing_stage_runs"
                    ).fetchone()[0],
                    0,
                )
            with sqlite3.connect(legacy) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                }
                self.assertEqual({"masks", "tracks", "cuts"}, tables)
                self.assertEqual(
                    0,
                    connection.execute("SELECT COUNT(*) FROM cuts").fetchone()[0],
                )
            for mode in ("raw", "tracked", "final", "faces"):
                path = Path(manifest["artifacts"][f"overlay_{mode}"])
                self.assertTrue(path.is_file())
                capture = cv2.VideoCapture(str(path))
                self.assertTrue(capture.isOpened())
                self.assertEqual(8, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                capture.release()
            cpu_stages = [
                stage
                for stage in manifest["stages"]
                if stage["name"] == "postprocess"
                or stage["name"].startswith("overlay_")
            ]
            self.assertTrue(cpu_stages)
            self.assertTrue(all(stage["cpu_only"] for stage in cpu_stages))

            resumed = OrchestrationRunner(config, resume=True).run()
            self.assertEqual("complete", resumed["status"])


if __name__ == "__main__":
    unittest.main()
