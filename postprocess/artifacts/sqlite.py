"""Convert the public union JSON artifact to the output SQLite contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts.ellipses import ellipses_to_polygons_json
from contracts.mask_sqlite import MaskRow, read_mask_rows, write_mask_sqlite


def union2sqlite_parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-union-json", required=True)
    parser.add_argument("--output-sqlite", required=True)
    parser.add_argument("--reference-sqlite")
    return parser.parse_args(argv)


def union2sqlite_main(argv: list[str] | None = None) -> None:
    args = union2sqlite_parse_args(argv)
    rows = json.loads(Path(args.input_union_json).read_text(encoding="utf-8"))
    reference = None if not args.reference_sqlite else Path(args.reference_sqlite)
    labels_by_row: dict[tuple[int, str], str] = {}
    labels_by_track: dict[str, str] = {}
    if reference is not None:
        for source_row in read_mask_rows(reference):
            key = (source_row.frame, source_row.track_id)
            labels_by_row[key] = source_row.label
            if source_row.label:
                labels_by_track.setdefault(source_row.track_id, source_row.label)
    output: list[MaskRow] = []
    for row in rows:
        polygons_json: str | None = None
        if row.get("ellipse_params"):
            polygons_json = ellipses_to_polygons_json(row["ellipse_params"])
        elif row.get("polygon"):
            polygon = [[float(point[0]), float(point[1])] for point in row["polygon"]]
            polygons_json = json.dumps(
                [polygon], ensure_ascii=False, separators=(",", ":")
            )
        if polygons_json is not None:
            frame = int(row["frame"])
            track_id = str(row["track_id"])
            label = str(row.get("label") or "")
            if not label:
                label = labels_by_row.get(
                    (frame, track_id), labels_by_track.get(track_id, "")
                )
            output.append(
                MaskRow(
                    frame=frame,
                    track_id=track_id,
                    polygons=polygons_json,
                    label=label,
                    shape_type=("ellipse" if row.get("ellipse_params") else "polygon"),
                )
            )
    write_mask_sqlite(Path(args.output_sqlite), output, reference_sqlite=reference)
