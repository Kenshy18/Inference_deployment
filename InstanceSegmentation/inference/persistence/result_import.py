"""Copy model rows, including rich faces, into unified schema v3."""

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
    face_observations: int
    face_keypoints: int


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
        for row in candidate.execute("SELECT detection_id, encoding FROM segmentations")
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
    detection_ids: dict[int, int] = {}
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
        detection_ids[old_detection_id] = detection_id
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
    face_observation_count, face_keypoint_count = _insert_face_results(
        target,
        candidate,
        detection_ids=detection_ids,
    )
    return ImportedResultCounts(
        detections=detection_count,
        classifications=classification_count,
        segmentations=segmentation_count,
        face_observations=face_observation_count,
        face_keypoints=face_keypoint_count,
    )


def _insert_detection(
    target: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    frame_id: int,
    execution_id: int,
) -> int:
    group_id = row["group_id"] if "group_id" in row.keys() else None
    cursor = target.execute(
        """
        INSERT INTO detections(
            frame_id, model_execution_id,
            class_id, class_name, score,
            x1, y1, x2, y2, track_id, source, group_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            None if group_id is None else int(group_id),
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


def _insert_face_results(
    target: sqlite3.Connection,
    candidate: sqlite3.Connection,
    *,
    detection_ids: Mapping[int, int],
) -> tuple[int, int]:
    tables = {
        str(row[0])
        for row in candidate.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if "face_observations" not in tables:
        return 0, 0
    required = {
        "face_keypoints",
        "face_masks",
        "face_keypoint_class_probabilities",
        "face_keypoint_state_probabilities",
    }
    missing = required - tables
    if missing:
        raise ValueError(f"incomplete rich face candidate tables: {sorted(missing)}")
    keypoints: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in candidate.execute(
        "SELECT * FROM face_keypoints ORDER BY observation_id, point_index"
    ):
        keypoints[int(row["observation_id"])].append(row)
    class_probabilities: dict[tuple[int, int], list[sqlite3.Row]] = defaultdict(list)
    for row in candidate.execute(
        """
        SELECT * FROM face_keypoint_class_probabilities
        ORDER BY observation_id, point_index, class_index
        """
    ):
        class_probabilities[
            (int(row["observation_id"]), int(row["point_index"]))
        ].append(row)
    state_probabilities: dict[tuple[int, int], list[sqlite3.Row]] = defaultdict(list)
    for row in candidate.execute(
        """
        SELECT * FROM face_keypoint_state_probabilities
        ORDER BY observation_id, point_index, state_index
        """
    ):
        state_probabilities[
            (int(row["observation_id"]), int(row["point_index"]))
        ].append(row)
    masks = {
        int(row["observation_id"]): row
        for row in candidate.execute("SELECT * FROM face_masks")
    }

    observation_count = 0
    keypoint_count = 0
    for row in candidate.execute("SELECT * FROM face_observations ORDER BY id"):
        old_observation_id = int(row["id"])

        def mapped_detection(column: str) -> int | None:
            value = row[column]
            if value is None:
                return None
            old_id = int(value)
            if old_id not in detection_ids:
                raise ValueError(
                    f"face observation references missing detection {old_id}"
                )
            return detection_ids[old_id]

        cursor = target.execute(
            """
            INSERT INTO face_observations(
                anchor_detection_id, head_detection_id, face_detection_id,
                face_score, face_present, geometry_type,
                ellipse_cx, ellipse_cy,
                ellipse_major_radius, ellipse_minor_radius,
                ellipse_theta_radians
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mapped_detection("anchor_detection_id"),
                mapped_detection("head_detection_id"),
                mapped_detection("face_detection_id"),
                float(row["face_score"]),
                int(row["face_present"]),
                row["geometry_type"],
                row["ellipse_cx"],
                row["ellipse_cy"],
                row["ellipse_major_radius"],
                row["ellipse_minor_radius"],
                row["ellipse_theta_radians"],
            ),
        )
        observation_id = int(cursor.lastrowid)
        observation_count += 1
        mask = masks.get(old_observation_id)
        if mask is not None:
            target.execute(
                """
                INSERT INTO face_masks(
                    observation_id, encoding, width, height,
                    box_x1, box_y1, box_x2, box_y2, data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    str(mask["encoding"]),
                    int(mask["width"]),
                    int(mask["height"]),
                    float(mask["box_x1"]),
                    float(mask["box_y1"]),
                    float(mask["box_x2"]),
                    float(mask["box_y2"]),
                    bytes(mask["data"]),
                ),
            )
        for point in keypoints.get(old_observation_id, ()):
            point_index = int(point["point_index"])
            target.execute(
                """
                INSERT INTO face_keypoints(
                    observation_id, point_index,
                    class_id, class_name, x, y,
                    state, state_name, confidence, valid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    point_index,
                    int(point["class_id"]),
                    str(point["class_name"]),
                    float(point["x"]),
                    float(point["y"]),
                    int(point["state"]),
                    str(point["state_name"]),
                    float(point["confidence"]),
                    int(point["valid"]),
                ),
            )
            target.executemany(
                """
                INSERT INTO face_keypoint_class_probabilities(
                    observation_id, point_index, class_index, probability
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        observation_id,
                        point_index,
                        int(probability["class_index"]),
                        float(probability["probability"]),
                    )
                    for probability in class_probabilities.get(
                        (old_observation_id, point_index), ()
                    )
                ],
            )
            target.executemany(
                """
                INSERT INTO face_keypoint_state_probabilities(
                    observation_id, point_index, state_index, probability
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        observation_id,
                        point_index,
                        int(probability["state_index"]),
                        float(probability["probability"]),
                    )
                    for probability in state_probabilities.get(
                        (old_observation_id, point_index), ()
                    )
                ],
            )
            keypoint_count += 1
    return observation_count, keypoint_count


__all__ = ["ImportedResultCounts", "import_candidate_results"]
