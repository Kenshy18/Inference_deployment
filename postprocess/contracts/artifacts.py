"""Executable contracts for named pipeline artifacts."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .detections import CutList, iter_detection_records
from .detector_sqlite import validate_detector_input_sqlite
from .ellipses import canonicalize_ellipse


class ArtifactContractError(ValueError):
    """Raised when a named artifact does not satisfy its public schema."""


ArtifactValidator = Callable[[Path], None]
_validators: dict[str, ArtifactValidator] = {}


def register_artifact_contract(
    name: str,
    validator: ArtifactValidator,
    *,
    replace: bool = False,
) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("artifact contract name must not be empty")
    if key in _validators and not replace:
        raise ValueError(f"artifact contract already registered: {key}")
    _validators[key] = validator


def artifact_contract_names() -> tuple[str, ...]:
    return tuple(sorted(_validators))


def validate_artifact(name: str, path: Path) -> None:
    source = Path(path)
    if not source.is_file():
        raise ArtifactContractError(f"{name}: file not found: {source}")
    validator = _validators.get(str(name))
    if validator is None:
        return
    try:
        validator(source)
    except ArtifactContractError:
        raise
    except Exception as exc:
        raise ArtifactContractError(
            f"{name}: invalid artifact {source}: {exc}"
        ) from exc


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactContractError(f"{path}: invalid JSON: {exc}") from exc


def _validate_json_object(path: Path) -> None:
    if not isinstance(_load_json(path), dict):
        raise ArtifactContractError(f"{path}: expected a JSON object")


def _validate_raw_jsonl(path: Path) -> None:
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                frame = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ArtifactContractError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(frame, dict):
                raise ArtifactContractError(
                    f"{path}:{line_number}: frame must be an object"
                )
            frame_index = frame.get("frame_index", frame.get("frame_idx"))
            if frame_index is None:
                raise ArtifactContractError(
                    f"{path}:{line_number}: frame_index/frame_idx is required"
                )
            int(frame_index)
            detections = frame.get("detections", frame.get("instances", []))
            if not isinstance(detections, list):
                raise ArtifactContractError(
                    f"{path}:{line_number}: detections/instances must be a list"
                )
            for detection in detections:
                if not isinstance(detection, dict):
                    raise ArtifactContractError(
                        f"{path}:{line_number}: detection must be an object"
                    )
                masks = detection.get("polygons", detection.get("segmentation"))
                if not isinstance(masks, list) or not masks:
                    raise ArtifactContractError(
                        f"{path}:{line_number}: detection mask is required"
                    )
            records += 1
    if records == 0:
        raise ArtifactContractError(f"{path}: no frame records")


def _validate_canonical_jsonl(path: Path) -> None:
    records = 0
    for record in iter_detection_records(path):
        for detection in record["detections"]:
            if not isinstance(detection, dict):
                raise ArtifactContractError(f"{path}: detection must be an object")
            masks = detection.get("polygons")
            if not isinstance(masks, list) or not masks:
                raise ArtifactContractError(
                    f"{path}: canonical detection requires polygons"
                )
        records += 1
    if records == 0:
        raise ArtifactContractError(f"{path}: no frame records")


def _validate_cuts(path: Path) -> None:
    CutList.read(path)


def _validate_mask_sqlite(path: Path) -> None:
    with sqlite3.connect(str(path)) as connection:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(masks)")
        }
        required = {"frame", "track_id", "polygons"}
        missing = required - columns
        if missing:
            raise ArtifactContractError(
                f"{path}: masks is missing columns: {sorted(missing)}"
            )
        for frame, track_id, polygons in connection.execute(
            "SELECT frame, track_id, polygons FROM masks"
        ):
            if frame is None or int(frame) < 0:
                raise ArtifactContractError(f"{path}: invalid frame {frame!r}")
            if track_id is None or not str(track_id):
                raise ArtifactContractError(f"{path}: empty track_id")
            decoded = json.loads(str(polygons))
            if not isinstance(decoded, list):
                raise ArtifactContractError(f"{path}: polygons must decode to a list")


def _validate_face_masks_sqlite(path: Path) -> None:
    _validate_mask_sqlite(path)
    with sqlite3.connect(str(path)) as connection:
        info = dict(connection.execute("SELECT key, value FROM schema_info"))
        schema_version = str(info.get("schema_version"))
        if (
            info.get("schema_name") != "face-privacy-mask-sqlite"
            or schema_version not in {"1", "2"}
        ):
            raise ArtifactContractError(f"{path}: unsupported face mask schema")
        geometry_columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(face_mask_geometries)"
            )
        }
        required_geometry = {
            "frame",
            "track_id",
            "geometry_type",
            "cx",
            "cy",
            "half_width",
            "half_height",
            "theta_radians",
        }
        missing_geometry = required_geometry - geometry_columns
        if missing_geometry:
            raise ArtifactContractError(
                f"{path}: face_mask_geometries columns missing: "
                f"{sorted(missing_geometry)}"
            )
        provenance_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(mask_provenance)")
        }
        required = {
            "frame",
            "track_id",
            "mask_kind",
            "source_observation_id",
            "derivation",
            "confidence",
            "algorithm_version",
        }
        if schema_version == "2":
            required.update(
                {
                    "source_observation_id_end",
                    "is_interpolated",
                }
            )
        missing = required - provenance_columns
        if missing:
            raise ArtifactContractError(
                f"{path}: mask_provenance columns missing: {sorted(missing)}"
            )
        mask_count = int(
            connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
        )
        provenance_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM mask_provenance"
            ).fetchone()[0]
        )
        geometry_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM face_mask_geometries"
            ).fetchone()[0]
        )
        if mask_count != provenance_count or mask_count != geometry_count:
            raise ArtifactContractError(
                f"{path}: every face mask must have one native geometry "
                "and provenance row"
            )
        for (
            frame,
            track_id,
            kind,
            observation_id,
            observation_id_end,
            is_interpolated,
            derivation,
            confidence,
            algorithm_version,
        ) in connection.execute(
            f"""
            SELECT frame, track_id, mask_kind, source_observation_id,
                   {
                       "source_observation_id_end"
                       if schema_version == "2"
                       else "source_observation_id"
                   },
                   {
                       "is_interpolated"
                       if schema_version == "2"
                       else "0"
                   },
                   derivation, confidence, algorithm_version
            FROM mask_provenance
            """
        ):
            derivation = str(derivation)
            is_interpolated = int(is_interpolated)
            provenance_is_consistent = (
                (
                    derivation == "interpolated-linear"
                    and schema_version == "2"
                    and is_interpolated == 1
                    and int(observation_id_end) != int(observation_id)
                )
                or (
                    derivation
                    in {"face-ellipse", "eye-keypoints", "ellipse-fallback"}
                    and is_interpolated == 0
                    and int(observation_id_end) == int(observation_id)
                )
            )
            if (
                int(frame) < 0
                or not str(track_id).startswith("face:")
                or str(kind) not in {"face", "eyes"}
                or int(observation_id) < 1
                or int(observation_id_end) < 1
                or is_interpolated not in {0, 1}
                or not provenance_is_consistent
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
                or str(algorithm_version) != "face-privacy-geometry-v1"
            ):
                raise ArtifactContractError(
                    f"{path}: invalid face mask provenance for {track_id}"
                )


def _validate_combined_mask_sqlite(path: Path) -> None:
    _validate_mask_sqlite(path)
    with sqlite3.connect(str(path)) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "mask_provenance" not in tables:
            raise ArtifactContractError(
                f"{path}: combined masks require mask_provenance"
            )
        if "face_mask_geometries" not in tables:
            raise ArtifactContractError(
                f"{path}: combined masks require native face_mask_geometries"
            )
        missing_face_provenance = connection.execute(
            """
            SELECT 1
            FROM masks m
            LEFT JOIN mask_provenance p
              ON p.frame=m.frame AND p.track_id=m.track_id
            WHERE m.track_id LIKE 'face:%'
              AND p.track_id IS NULL
            LIMIT 1
            """
        ).fetchone()
        missing_face_geometry = connection.execute(
            """
            SELECT 1
            FROM masks m
            LEFT JOIN face_mask_geometries g
              ON g.frame=m.frame AND g.track_id=m.track_id
            WHERE m.track_id LIKE 'face:%'
              AND g.track_id IS NULL
            LIMIT 1
            """
        ).fetchone()
        orphan_provenance = connection.execute(
            """
            SELECT 1
            FROM mask_provenance p
            LEFT JOIN masks m
              ON m.frame=p.frame AND m.track_id=p.track_id
            WHERE m.track_id IS NULL
               OR p.track_id NOT LIKE 'face:%'
            LIMIT 1
            """
        ).fetchone()
        if (
            missing_face_provenance is not None
            or missing_face_geometry is not None
            or orphan_provenance is not None
        ):
            raise ArtifactContractError(
                f"{path}: face masks, native geometry, and provenance "
                "must have a 1:1 relation"
            )


def _validate_legacy_mask_sqlite(path: Path) -> None:
    with sqlite3.connect(str(path)) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        expected_tables = {"masks", "tracks", "cuts"}
        if tables != expected_tables:
            raise ArtifactContractError(
                f"{path}: legacy tables must be exactly "
                f"{sorted(expected_tables)}, got {sorted(tables)}"
            )
        expected_columns = {
            "masks": [
                "frame",
                "track_id",
                "polygons",
                "shape_type",
                "dilate_px",
                "feather_px",
                "mosaic_block",
                "mosaic_alias",
                "label",
            ],
            "tracks": ["track_id", "label"],
            "cuts": ["frame"],
        }
        for table, expected in expected_columns.items():
            actual = [
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{table}")')
            ]
            if actual != expected:
                raise ArtifactContractError(
                    f"{path}: legacy {table} columns must be {expected}, "
                    f"got {actual}"
                )
        _validate_mask_sqlite(path)
        for (frame,) in connection.execute("SELECT frame FROM cuts"):
            if frame is None or int(frame) < 0:
                raise ArtifactContractError(
                    f"{path}: invalid legacy cut frame {frame!r}"
                )


def _validate_metrics_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"frame", "track_id", "ellipse_params"}
        missing = required - fields
        if missing:
            raise ArtifactContractError(
                f"{path}: metrics CSV is missing columns: {sorted(missing)}"
            )
        for row in reader:
            int(row["frame"])
            if not row["track_id"]:
                raise ArtifactContractError(f"{path}: empty track_id")
            ellipses = json.loads(row["ellipse_params"])
            if not isinstance(ellipses, list):
                raise ArtifactContractError(f"{path}: ellipse_params must be a list")
            for ellipse in ellipses:
                canonicalize_ellipse(ellipse)


def _validate_ellipse_rows(path: Path, *, keyframes: bool) -> None:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise ArtifactContractError(f"{path}: expected a JSON list")
    for row in rows:
        if not isinstance(row, dict):
            raise ArtifactContractError(f"{path}: row must be an object")
        int(row["frame"])
        if not str(row["track_id"]):
            raise ArtifactContractError(f"{path}: empty track_id")
        if keyframes:
            canonicalize_ellipse(row["ellipse"])
        else:
            ellipses = row.get("ellipse_params")
            if not isinstance(ellipses, list):
                raise ArtifactContractError(f"{path}: ellipse_params must be a list")
            for ellipse in ellipses:
                canonicalize_ellipse(ellipse)


def _validate_keyframes_json(path: Path) -> None:
    _validate_ellipse_rows(path, keyframes=True)


def _validate_union_json(path: Path) -> None:
    _validate_ellipse_rows(path, keyframes=False)


def _validate_integrated_result_sqlite(path: Path) -> None:
    from artifacts.unified_sqlite import validate_integrated_result

    validate_integrated_result(path)


for _name in ("input_jsonl",):
    register_artifact_contract(_name, _validate_raw_jsonl)
register_artifact_contract("input_raw_sqlite", validate_detector_input_sqlite)
for _name in ("normalized_jsonl", "scored_jsonl", "nms_jsonl"):
    register_artifact_contract(_name, _validate_canonical_jsonl)
register_artifact_contract("class_policy_json", _validate_json_object)
register_artifact_contract(
    "class_postprocess_policy_json",
    _validate_json_object,
)
register_artifact_contract("cuts_json", _validate_cuts)
for _name in (
    "tracked_sqlite",
    "approximated_sqlite",
    "keyframes_sqlite",
    "predictions_sqlite",
):
    register_artifact_contract(_name, _validate_mask_sqlite)
register_artifact_contract(
    "combined_predictions_sqlite",
    _validate_combined_mask_sqlite,
)
register_artifact_contract("face_masks_sqlite", _validate_face_masks_sqlite)
register_artifact_contract(
    "legacy_predictions_sqlite",
    _validate_legacy_mask_sqlite,
)
register_artifact_contract(
    "combined_legacy_predictions_sqlite",
    _validate_legacy_mask_sqlite,
)
register_artifact_contract(
    "result_sqlite",
    _validate_integrated_result_sqlite,
)
for _name in ("approximation_metrics_csv", "filled_metrics_csv"):
    register_artifact_contract(_name, _validate_metrics_csv)
register_artifact_contract("keyframes_json", _validate_keyframes_json)
for _name in ("interpolated_union_json", "filled_union_json"):
    register_artifact_contract(_name, _validate_union_json)
for _name in (
    "evaluation_summary",
    "validation_report",
    "combined_validation_report",
    "classwise_manifest",
):
    register_artifact_contract(_name, _validate_json_object)
