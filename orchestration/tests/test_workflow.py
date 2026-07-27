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

from helpers import create_unified_sqlite, create_video


class WorkflowTests(unittest.TestCase):
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
