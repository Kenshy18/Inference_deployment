"""Copy model rows, including rich faces, into unified schema v3.

Large child tables are merged with ordered cursors.  At most one detection or
face observation worth of child rows is retained in Python memory.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass


class _OrderedRows:
    """Expose equal-key groups from an already ordered SQLite cursor."""

    def __init__(
        self,
        rows: Iterable[sqlite3.Row],
        key: Callable[[sqlite3.Row], object],
        *,
        description: str,
    ) -> None:
        self._rows: Iterator[sqlite3.Row] = iter(rows)
        self._key = key
        self._description = description
        self._current = next(self._rows, None)

    def take(self, expected: object) -> Iterator[sqlite3.Row]:
        current = self._current
        if current is not None and self._key(current) < expected:
            raise ValueError(
                f"{self._description} references missing parent "
                f"{self._key(current)!r}"
            )
        while current is not None and self._key(current) == expected:
            yield current
            current = next(self._rows, None)
            self._current = current

    def require_exhausted(self) -> None:
        if self._current is not None:
            raise ValueError(
                f"{self._description} references missing parent "
                f"{self._key(self._current)!r}"
            )


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
    """Merge candidate results without materializing full child tables."""

    probability_rows = _OrderedRows(
        candidate.execute(
            """
            SELECT detection_id, class_index, probability
            FROM classification_probabilities
            ORDER BY detection_id, class_index
            """
        ),
        lambda row: int(row["detection_id"]),
        description="classification probability",
    )
    segmentation_rows = _OrderedRows(
        candidate.execute(
            """
            SELECT detection_id, encoding
            FROM segmentations
            ORDER BY detection_id
            """
        ),
        lambda row: int(row["detection_id"]),
        description="segmentation",
    )
    geometry_rows = _OrderedRows(
        candidate.execute(
            """
            SELECT sp.detection_id,
                   sp.id AS source_polygon_id,
                   sp.polygon_index,
                   pt.point_index,
                   pt.x,
                   pt.y
            FROM segmentation_polygons sp
            LEFT JOIN segmentation_points pt ON pt.polygon_id=sp.id
            ORDER BY sp.detection_id, sp.polygon_index, pt.point_index
            """
        ),
        lambda row: int(row["detection_id"]),
        description="segmentation polygon",
    )

    target.execute(
        """
        CREATE TABLE imported_detection_ids_staging(
            source_detection_id INTEGER PRIMARY KEY,
            target_detection_id INTEGER NOT NULL
        ) WITHOUT ROWID
        """
    )

    detection_count = 0
    classification_count = 0
    segmentation_count = 0
    try:
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
            target.execute(
                """
                INSERT INTO imported_detection_ids_staging(
                    source_detection_id, target_detection_id
                ) VALUES (?, ?)
                """,
                (old_detection_id, detection_id),
            )
            detection_count += 1

            probabilities = probability_rows.take(old_detection_id)
            if row["classifier_class_id"] is not None:
                _insert_classification(
                    target,
                    row=row,
                    detection_id=detection_id,
                    probabilities=probabilities,
                )
                classification_count += 1
            elif next(probabilities, None) is not None:
                # Consume the rest before raising so the cursor lifecycle is simple.
                for _unused in probabilities:
                    pass
                raise ValueError(
                    "classification probabilities exist without classification"
                )

            segmentation = tuple(segmentation_rows.take(old_detection_id))
            if segmentation:
                if len(segmentation) != 1:
                    raise ValueError(
                        f"multiple segmentations for detection {old_detection_id}"
                    )
                _insert_segmentation(
                    target,
                    detection_id=detection_id,
                    encoding=str(segmentation[0]["encoding"]),
                    geometry_rows=geometry_rows.take(old_detection_id),
                )
                segmentation_count += 1
            elif next(geometry_rows.take(old_detection_id), None) is not None:
                raise ValueError("segmentation polygons exist without segmentation")

        probability_rows.require_exhausted()
        segmentation_rows.require_exhausted()
        geometry_rows.require_exhausted()
        face_observation_count, face_keypoint_count = _insert_face_results(
            target,
            candidate,
        )
    finally:
        target.execute("DROP TABLE IF EXISTS imported_detection_ids_staging")

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
    probabilities: Iterable[sqlite3.Row],
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
        (
            (
                detection_id,
                int(probability["class_index"]),
                float(probability["probability"]),
            )
            for probability in probabilities
        ),
    )


def _insert_segmentation(
    target: sqlite3.Connection,
    *,
    detection_id: int,
    encoding: str,
    geometry_rows: Iterable[sqlite3.Row],
) -> None:
    target.execute(
        """
        INSERT INTO segmentations(detection_id, encoding)
        VALUES (?, ?)
        """,
        (detection_id, encoding),
    )
    source_polygon_id: int | None = None
    polygon_id: int | None = None
    point_batch: list[tuple[int, int, float, float]] = []

    def flush_points() -> None:
        if not point_batch:
            return
        target.executemany(
            """
            INSERT INTO segmentation_points(
                polygon_id, point_index, x, y
            ) VALUES (?, ?, ?, ?)
            """,
            point_batch,
        )
        point_batch.clear()

    for point in geometry_rows:
        observed_polygon_id = int(point["source_polygon_id"])
        if observed_polygon_id != source_polygon_id:
            flush_points()
            cursor = target.execute(
                """
                INSERT INTO segmentation_polygons(
                    detection_id, polygon_index
                ) VALUES (?, ?)
                """,
                (detection_id, int(point["polygon_index"])),
            )
            polygon_id = int(cursor.lastrowid)
            source_polygon_id = observed_polygon_id
        if point["point_index"] is not None:
            assert polygon_id is not None
            point_batch.append(
                (
                    polygon_id,
                    int(point["point_index"]),
                    float(point["x"]),
                    float(point["y"]),
                )
            )
            if len(point_batch) >= 1024:
                flush_points()
    flush_points()


def _mapped_detection(
    target: sqlite3.Connection,
    value: object,
) -> int | None:
    if value is None:
        return None
    old_id = int(value)
    mapped = target.execute(
        """
        SELECT target_detection_id
        FROM imported_detection_ids_staging
        WHERE source_detection_id=?
        """,
        (old_id,),
    ).fetchone()
    if mapped is None:
        raise ValueError(f"face observation references missing detection {old_id}")
    return int(mapped[0])


def _insert_face_results(
    target: sqlite3.Connection,
    candidate: sqlite3.Connection,
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

    keypoint_rows = _OrderedRows(
        candidate.execute(
            "SELECT * FROM face_keypoints ORDER BY observation_id, point_index"
        ),
        lambda row: int(row["observation_id"]),
        description="face keypoint",
    )
    mask_rows = _OrderedRows(
        candidate.execute("SELECT * FROM face_masks ORDER BY observation_id"),
        lambda row: int(row["observation_id"]),
        description="face mask",
    )
    class_probability_rows = _OrderedRows(
        candidate.execute(
            """
            SELECT * FROM face_keypoint_class_probabilities
            ORDER BY observation_id, point_index, class_index
            """
        ),
        lambda row: (int(row["observation_id"]), int(row["point_index"])),
        description="face keypoint class probability",
    )
    state_probability_rows = _OrderedRows(
        candidate.execute(
            """
            SELECT * FROM face_keypoint_state_probabilities
            ORDER BY observation_id, point_index, state_index
            """
        ),
        lambda row: (int(row["observation_id"]), int(row["point_index"])),
        description="face keypoint state probability",
    )

    observation_count = 0
    keypoint_count = 0
    for row in candidate.execute("SELECT * FROM face_observations ORDER BY id"):
        old_observation_id = int(row["id"])
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
                _mapped_detection(target, row["anchor_detection_id"]),
                _mapped_detection(target, row["head_detection_id"]),
                _mapped_detection(target, row["face_detection_id"]),
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

        masks = tuple(mask_rows.take(old_observation_id))
        if len(masks) > 1:
            raise ValueError(
                f"multiple face masks for observation {old_observation_id}"
            )
        if masks:
            mask = masks[0]
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

        for point in keypoint_rows.take(old_observation_id):
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
            probability_key = (old_observation_id, point_index)
            target.executemany(
                """
                INSERT INTO face_keypoint_class_probabilities(
                    observation_id, point_index, class_index, probability
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (
                        observation_id,
                        point_index,
                        int(probability["class_index"]),
                        float(probability["probability"]),
                    )
                    for probability in class_probability_rows.take(probability_key)
                ),
            )
            target.executemany(
                """
                INSERT INTO face_keypoint_state_probabilities(
                    observation_id, point_index, state_index, probability
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (
                        observation_id,
                        point_index,
                        int(probability["state_index"]),
                        float(probability["probability"]),
                    )
                    for probability in state_probability_rows.take(probability_key)
                ),
            )
            keypoint_count += 1

    keypoint_rows.require_exhausted()
    mask_rows.require_exhausted()
    class_probability_rows.require_exhausted()
    state_probability_rows.require_exhausted()
    return observation_count, keypoint_count


__all__ = ["ImportedResultCounts", "import_candidate_results"]
