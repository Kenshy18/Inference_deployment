"""Export rich face observations as an immutable sidecar mask SQLite."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from collections import Counter
from pathlib import Path

from .face_privacy import derive_privacy_mask
from .sources import iter_face_frames


SCHEMA_NAME = "face-privacy-mask-sqlite"
SCHEMA_VERSION = "1"
ALGORITHM_VERSION = "face-privacy-geometry-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="overlay-export-face-masks",
        description=(
            "Derive face or eye privacy polygons from a schema-v3 inference "
            "SQLite without modifying the source."
        ),
    )
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=("face", "eyes"), required=True)
    parser.add_argument(
        "--eye-shape",
        choices=("ellipse", "rectangle"),
        default="ellipse",
    )
    parser.add_argument("--minimum-eye-confidence", type=float, default=0.35)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def export_face_masks(
    source: Path,
    output: Path,
    *,
    target: str,
    eye_shape: str = "ellipse",
    minimum_eye_confidence: float = 0.35,
    start_frame: int = 0,
    end_frame: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    resolved_source = Path(source).expanduser().resolve()
    resolved_output = Path(output).expanduser().resolve()
    if not resolved_source.is_file():
        raise FileNotFoundError(resolved_source)
    if resolved_source == resolved_output:
        raise ValueError("face mask output must differ from the inference SQLite")
    if target not in {"face", "eyes"}:
        raise ValueError("target must be face or eyes")
    if eye_shape not in {"ellipse", "rectangle"}:
        raise ValueError("eye_shape must be ellipse or rectangle")
    if not 0.0 <= minimum_eye_confidence <= 1.0:
        raise ValueError("minimum_eye_confidence must be between 0 and 1")
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if end_frame is not None and end_frame < start_frame:
        raise ValueError("end_frame must be >= start_frame")
    if resolved_output.exists() and not overwrite:
        raise FileExistsError(
            f"{resolved_output} already exists; pass --overwrite to replace it"
        )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_output.with_name(
        f".{resolved_output.name}.{uuid.uuid4().hex}.tmp"
    )
    counts: Counter[str] = Counter()
    first_frame: int | None = None
    last_frame: int | None = None
    try:
        with sqlite3.connect(temporary) as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=FULL;
                CREATE TABLE schema_info(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE masks(
                    frame INTEGER NOT NULL,
                    track_id TEXT NOT NULL,
                    polygons TEXT NOT NULL,
                    shape_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    source_observation_id INTEGER NOT NULL,
                    derivation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    PRIMARY KEY(frame, track_id),
                    CHECK(frame >= 0),
                    CHECK(shape_type IN ('ellipse', 'rectangle')),
                    CHECK(confidence >= 0 AND confidence <= 1)
                );
                CREATE INDEX idx_face_privacy_masks_frame
                    ON masks(frame);
                """
            )
            metadata = {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "algorithm_version": ALGORITHM_VERSION,
                "source_sqlite": str(resolved_source),
                "target": target,
                "eye_shape": eye_shape if target == "eyes" else "ellipse",
                "minimum_eye_confidence": repr(minimum_eye_confidence),
                "frame_semantics": "zero_based_original_video_frame",
                "coordinate_semantics": "original_video_pixel_coordinates",
            }
            connection.executemany(
                "INSERT INTO schema_info(key, value) VALUES (?, ?)",
                sorted(metadata.items()),
            )

            rows: list[tuple[object, ...]] = []
            for frame in iter_face_frames(
                resolved_source,
                include_ellipses=True,
                include_keypoints=target == "eyes",
                include_probability_masks=False,
                display_style="simple",
                require_privacy_geometry=True,
            ):
                if frame.frame_index < start_frame:
                    continue
                if end_frame is not None and frame.frame_index > end_frame:
                    break
                for item in frame.items:
                    privacy = derive_privacy_mask(
                        target,
                        item.ellipse,
                        item.keypoints,
                        eye_shape=eye_shape,
                        minimum_eye_confidence=minimum_eye_confidence,
                    )
                    if privacy is None:
                        counts["not_emitted"] += 1
                        continue
                    observation_id = int(item.identity.rsplit(":", 1)[1])
                    confidence = (
                        float(item.face_score or 0.0)
                        if privacy.derivation == "face-ellipse"
                        else privacy.confidence
                    )
                    polygons = json.dumps(
                        [
                            [
                                [float(point[0]), float(point[1])]
                                for point in privacy.polygon
                            ]
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    rows.append(
                        (
                            frame.frame_index,
                            f"face-observation:{observation_id}",
                            polygons,
                            privacy.shape,
                            "Face" if target == "face" else "Eyes",
                            observation_id,
                            privacy.derivation,
                            confidence,
                        )
                    )
                    counts[privacy.derivation] += 1
                    first_frame = (
                        frame.frame_index
                        if first_frame is None
                        else min(first_frame, frame.frame_index)
                    )
                    last_frame = (
                        frame.frame_index
                        if last_frame is None
                        else max(last_frame, frame.frame_index)
                    )
                    if len(rows) >= 1000:
                        connection.executemany(
                            """
                            INSERT INTO masks(
                                frame, track_id, polygons, shape_type, label,
                                source_observation_id, derivation, confidence
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            rows,
                        )
                        rows.clear()
            if rows:
                connection.executemany(
                    """
                    INSERT INTO masks(
                        frame, track_id, polygons, shape_type, label,
                        source_observation_id, derivation, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity != "ok":
                raise RuntimeError(f"face mask SQLite integrity check failed: {integrity}")
            connection.commit()
        os.replace(temporary, resolved_output)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "source": str(resolved_source),
        "output": str(resolved_output),
        "target": target,
        "shape": eye_shape if target == "eyes" else "ellipse",
        "rows": sum(
            count for name, count in counts.items() if name != "not_emitted"
        ),
        "first_frame": first_frame,
        "last_frame": last_frame,
        "derivations": dict(sorted(counts.items())),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = export_face_masks(
        args.sqlite,
        args.output,
        target=args.target,
        eye_shape=args.eye_shape,
        minimum_eye_confidence=args.minimum_eye_confidence,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
