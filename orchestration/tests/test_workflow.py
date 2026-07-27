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

from helpers import create_rich_face_unified_sqlite, create_unified_sqlite, create_video


class WorkflowTests(unittest.TestCase):
    def test_face_privacy_is_merged_into_software_final_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi", frames=1)
            inference = create_rich_face_unified_sqlite(
                root / "inference.sqlite"
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

            manifest = OrchestrationRunner(
                OrchestrationConfig.load(config_path)
            ).run()

            final = Path(manifest["artifacts"]["final_sqlite"])
            legacy = Path(manifest["artifacts"]["legacy_final_sqlite"])
            with sqlite3.connect(final) as connection:
                labels = connection.execute(
                    "SELECT label, COUNT(*) FROM masks GROUP BY label"
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
                manifest["validation"]["inference_sqlite"]["face_observations"],
            )
            overlay = Path(manifest["artifacts"]["overlay_faces"])
            self.assertTrue(overlay.is_file())
            capture = cv2.VideoCapture(str(overlay))
            self.assertTrue(capture.isOpened())
            self.assertEqual(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            capture.release()

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
            self.assertTrue(Path(manifest["artifacts"]["tracked_sqlite"]).is_file())
            self.assertTrue(Path(manifest["artifacts"]["final_sqlite"]).is_file())
            legacy = Path(manifest["artifacts"]["legacy_final_sqlite"])
            self.assertTrue(legacy.is_file())
            with sqlite3.connect(manifest["artifacts"]["final_sqlite"]) as connection:
                self.assertEqual(
                    [("disabled", 0, "first_frame_of_new_scene")],
                    connection.execute(
                        """
                        SELECT method, cut_count, frame_semantics
                        FROM cut_detection_metadata
                        """
                    ).fetchall(),
                )
                self.assertIn(
                    "raw_tracked_masks",
                    {
                        row[0]
                        for row in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )
                    },
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
