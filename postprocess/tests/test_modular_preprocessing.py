from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from contracts.detections import (
    CutList,
    transform_detection_jsonl,
    write_cut_list,
)
from nms.adaptive import AdaptiveNms
from preprocessing.score_policy import ScorePolicy, apply_score_policy_jsonl
from tracking import build_tracked_sqlite


class PassthroughNms:
    name = "passthrough_test_nms"

    def __init__(self) -> None:
        self.calls = 0

    def apply(self, detections: list[dict[str, object]]) -> list[dict[str, object]]:
        self.calls += 1
        return detections


def detection(x: float, *, score: float = 0.9) -> dict[str, object]:
    polygon = [[x, 0], [x + 10, 0], [x + 10, 10], [x, 10]]
    return {
        "class_name": "target",
        "label": "target",
        "score": score,
        "bbox_xyxy": [x, 0, x + 10, 10],
        "polygons": [polygon],
    }


class ModularPreprocessingTests(unittest.TestCase):
    def test_adaptive_nms_is_a_real_implementation(self) -> None:
        retained = AdaptiveNms().apply(
            [detection(0, score=0.9), detection(0, score=0.8)]
        )
        self.assertEqual(1, len(retained))
        self.assertEqual(0.9, retained[0]["score"])

    def test_each_raw_feature_writes_and_consumes_its_own_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jsonl_path = root / "detections.jsonl"
            sqlite_path = root / "tracked.sqlite"
            scored_path = root / "scored.jsonl"
            nms_path = root / "nms.jsonl"
            cuts_path = root / "cuts.json"
            records = [
                {
                    "frame_index": frame,
                    "detections": [detection(float(frame % 2))],
                }
                for frame in range(4)
            ]
            jsonl_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            nms_policy = PassthroughNms()

            apply_score_policy_jsonl(
                jsonl_path,
                scored_path,
                policy=ScorePolicy(default_min=0.1),
            )

            def apply_nms(record: dict[str, object]) -> dict[str, object]:
                output = dict(record)
                output["detections"] = nms_policy.apply(list(record["detections"]))
                return output

            transform_detection_jsonl(scored_path, nms_path, apply_nms)
            write_cut_list(
                cuts_path,
                CutList((2,), "fixed_test_detector", 0.001),
            )
            summary = build_tracked_sqlite(
                nms_path,
                sqlite_path,
                cuts_path,
                remove_short_tracks_max_frames=0,
            )

            self.assertEqual(4, nms_policy.calls)
            self.assertEqual("fixed_test_detector", summary["cut_detection_method"])
            self.assertEqual(2, summary["tracks_after_prune"])
            with sqlite3.connect(sqlite_path) as connection:
                self.assertEqual(
                    [(2,)], connection.execute("SELECT frame FROM cuts").fetchall()
                )
                self.assertEqual(
                    [
                        (
                            1,
                            "fixed_test_detector",
                            0.001,
                            1,
                            "first_frame_of_new_scene",
                        )
                    ],
                    connection.execute(
                        """
                        SELECT schema_version, method, elapsed_seconds,
                               cut_count, frame_semantics
                        FROM cut_detection_metadata
                        """
                    ).fetchall(),
                )
                self.assertEqual(
                    [("1",), ("2",)],
                    connection.execute(
                        "SELECT track_id FROM tracks ORDER BY CAST(track_id AS INTEGER)"
                    ).fetchall(),
                )
                self.assertEqual(
                    4, connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
                )
                self.assertEqual(
                    [],
                    connection.execute(
                        """
                        SELECT name FROM sqlite_master
                        WHERE name LIKE '%_staging'
                        """
                    ).fetchall(),
                )

    def test_tracking_builder_does_not_import_approximation_monolith(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "tracking" / "builder.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("approximation.ellipse.inference", source)


if __name__ == "__main__":
    unittest.main()
