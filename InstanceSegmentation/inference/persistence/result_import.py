"""Copy detection, classification, and mask rows into unified schema v2."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImportedResultCounts:
    detections: int
    classifications: int
    segmentations: int


def import_candidate_results(
    target: sqlite3.Connection,
    candidate: sqlite3.Connection,
    *,
    execution_id: int,
    frame_ids: Mapping[int, int],
) -> ImportedResultCounts:
    probabilities: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in candidate.execute(
        """
        SELECT detection_id, class_index, probability
        FROM classification_probabilities
        ORDER BY detection_id, class_index
        """
    ):
        probabilities[int(row["detection_id"])].append(
            (int(row["class_index"]), float(row["probability"]))
        )
    segmentations = {
        int(row["detection_id"]): str(row["encoding"])
        for row in candidate.execute(
            "SELECT detection_id, encoding FROM segmentations"
        )
    }
    polygons: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in candidate.execute(
        """
        SELECT id, detection_id, polygon_index
        FROM segmentation_polygons
        ORDER BY detection_id, polygon_index
        """
    ):
        polygons[int(row["detection_id"])].append(row)
    points: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in candidate.execute(
        """
        SELECT polygon_id, point_index, x, y
        FROM segmentation_points
        ORDER BY polygon_id, point_index
        """
    ):
        points[int(row["polygon_id"])].append(row)

    detection_count = 0
    classification_count = 0
    segmentation_count = 0
    for row in candidate.execute("SELECT * FROM detections ORDER BY id"):
        old_detection_id = int(row["id"])
        frame_index = int(row["frame_index"])
        if frame_index not in frame_ids:
            raise ValueError(f"detection references missing frame {frame_index}")
        detection_id = _insert_detection(
            target,
            row=row,
            frame_id=frame_ids[frame_index],
            execution_id=execution_id,
        )
        detection_count += 1
        if row["classifier_class_id"] is not None:
            _insert_classification(
                target,
                row=row,
                detection_id=detection_id,
                probabilities=probabilities.get(old_detection_id, ()),
            )
            classification_count += 1
        elif old_detection_id in probabilities:
            raise ValueError(
                "classification probabilities exist without classification"
            )
        if old_detection_id in segmentations:
            _insert_segmentation(
                target,
                detection_id=detection_id,
                encoding=segmentations[old_detection_id],
                source_detection_id=old_detection_id,
                polygons=polygons,
                points=points,
            )
            segmentation_count += 1
    return ImportedResultCounts(
        detections=detection_count,
        classifications=classification_count,
        segmentations=segmentation_count,
    )


def _insert_detection(
    target: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    frame_id: int,
    execution_id: int,
) -> int:
    cursor = target.execute(
        """
        INSERT INTO detections(
            frame_id, model_execution_id,
            class_id, class_name, score,
            x1, y1, x2, y2, track_id, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            frame_id,
            execution_id,
            int(row["class_id"]),
            str(row["class_name"]),
            float(row["score"]),
            float(row["x1"]),
            float(row["y1"]),
            float(row["x2"]),
            float(row["y2"]),
            None if row["track_id"] is None else int(row["track_id"]),
            str(row["source"]),
        ),
    )
    return int(cursor.lastrowid)


def _insert_classification(
    target: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    detection_id: int,
    probabilities,
) -> None:
    target.execute(
        """
        INSERT INTO classifications(
            detection_id, class_id, class_name, score
        ) VALUES (?, ?, ?, ?)
        """,
        (
            detection_id,
            int(row["classifier_class_id"]),
            str(row["classifier_class_name"]),
            float(row["classifier_score"]),
        ),
    )
    target.executemany(
        """
        INSERT INTO classification_probabilities(
            detection_id, class_index, probability
        ) VALUES (?, ?, ?)
        """,
        [
            (detection_id, class_index, probability)
            for class_index, probability in probabilities
        ],
    )


def _insert_segmentation(
    target: sqlite3.Connection,
    *,
    detection_id: int,
    encoding: str,
    source_detection_id: int,
    polygons: Mapping[int, list[sqlite3.Row]],
    points: Mapping[int, list[sqlite3.Row]],
) -> None:
    target.execute(
        """
        INSERT INTO segmentations(detection_id, encoding)
        VALUES (?, ?)
        """,
        (detection_id, encoding),
    )
    for polygon in polygons.get(source_detection_id, ()):
        cursor = target.execute(
            """
            INSERT INTO segmentation_polygons(
                detection_id, polygon_index
            ) VALUES (?, ?)
            """,
            (detection_id, int(polygon["polygon_index"])),
        )
        polygon_id = int(cursor.lastrowid)
        target.executemany(
            """
            INSERT INTO segmentation_points(
                polygon_id, point_index, x, y
            ) VALUES (?, ?, ?, ?)
            """,
            [
                (
                    polygon_id,
                    int(point["point_index"]),
                    float(point["x"]),
                    float(point["y"]),
                )
                for point in points.get(int(polygon["id"]), ())
            ],
        )


__all__ = ["ImportedResultCounts", "import_candidate_results"]
