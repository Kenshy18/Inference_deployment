from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from contracts.detections import dumps_json_line
from experimental.nms_ablation_20260813.run_four_arm_v3 import (
    ARM_NAMES,
    InputLineage,
    RunInput,
    _run_one,
)


def rectangle(x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def detection(
    detection_id: int, score: float, polygons: list[list[list[float]]]
) -> dict[str, object]:
    xs = [point[0] for polygon in polygons for point in polygon]
    ys = [point[1] for polygon in polygons for point in polygon]
    box = [min(xs), min(ys), max(xs), max(ys)]
    return {
        "source_detection_id": detection_id,
        "score": score,
        "class_name": "foreground",
        "bbox_xyxy": box,
        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
        "polygons": polygons,
        "segmentation": polygons,
    }


class FourArmRunnerTests(unittest.TestCase):
    def test_writes_traceable_four_arm_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scored.jsonl"
            main = rectangle(0, 0, 100, 100)
            island = rectangle(120, 0, 140, 20)
            covering = rectangle(115, -5, 155, 35)
            outer = rectangle(0, 0, 50, 50)
            hole = rectangle(10, 10, 20, 20)
            records = [
                {
                    "frame_index": 0,
                    "detections": [
                        detection(1, 0.9, [main, island]),
                        detection(2, 0.8, [covering]),
                    ],
                },
                {
                    "frame_index": 1,
                    "detections": [detection(3, 0.9, [outer, hole])],
                },
            ]
            with source.open("wb") as handle:
                for record in records:
                    handle.write(dumps_json_line(record))

            run = RunInput("v3__test", "test", root / "raw.sqlite", root / "x.mp4", 2, 3)
            lineage = InputLineage(
                "v3__test",
                source,
                "unit_test",
                root / "raw.sqlite",
                None,
                0.3,
            )
            payload = _run_one(
                run,
                lineage,
                output_root=root / "output",
                start_frame=0,
                max_frames=None,
                max_union_pixels=1_000_000,
                safety_mode="changed",
                write_arm_jsonl=True,
            )

            self.assertEqual(2, payload["frames_processed"])
            self.assertEqual(list(ARM_NAMES), [row["arm"] for row in payload["arms"]])
            run_root = root / "output/runs/v3__test"
            self.assertTrue((run_root / "summary.json").is_file())
            self.assertTrue((run_root / "component_events.csv").is_file())
            for arm in ARM_NAMES:
                self.assertTrue((run_root / "arm_outputs" / f"{arm}.jsonl").is_file())
            with gzip.open(
                run_root / "retained_ids.jsonl.gz", "rt", encoding="utf-8"
            ) as handle:
                decisions = [json.loads(line) for line in handle]
            self.assertEqual([0, 1], [row["frame_index"] for row in decisions])
            self.assertEqual(set(ARM_NAMES), set(decisions[0]["retained_ids"]))


if __name__ == "__main__":
    unittest.main()
