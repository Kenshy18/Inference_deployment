"""Replaceable per-frame polygon approximation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from contracts.mask_sqlite import MaskRow, read_mask_rows, write_mask_sqlite


class PolygonApproximator(Protocol):
    """Contract for an algorithm that simplifies one frame's polygons."""

    name: str

    def approximate(self, polygons: list[list[list[float]]]) -> list[list[list[float]]]:
        """Return approximated polygons in source pixel coordinates."""


@dataclass(frozen=True)
class OpenCvRdpApproximator:
    """Ramer-Douglas-Peucker approximation backed by OpenCV."""

    name: str = "opencv_rdp"
    epsilon_ratio: float = 0.01
    minimum_epsilon_px: float = 0.5

    def approximate(self, polygons: list[list[list[float]]]) -> list[list[list[float]]]:
        output: list[list[list[float]]] = []
        for polygon in polygons:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) < 3:
                continue
            perimeter = float(cv2.arcLength(points.reshape(-1, 1, 2), True))
            epsilon = max(self.minimum_epsilon_px, perimeter * self.epsilon_ratio)
            simplified = cv2.approxPolyDP(
                points.reshape(-1, 1, 2), epsilon, True
            ).reshape(-1, 2)
            if len(simplified) < 3:
                simplified = points
            output.append([[float(point[0]), float(point[1])] for point in simplified])
        return output


def approximate_sqlite(
    input_sqlite: Path,
    output_sqlite: Path,
    *,
    approximator: PolygonApproximator | None = None,
) -> Path:
    implementation = approximator or OpenCvRdpApproximator()
    from common.live_preview import PreviewGeometry, active_postprocess_preview

    preview = active_postprocess_preview()
    output_rows: list[MaskRow] = []
    for row in read_mask_rows(input_sqlite):
        polygons = json.loads(row.polygons)
        approximated = implementation.approximate(polygons)
        if preview is not None and preview.should_sample("polygon_approximation"):
            preview.submit(
                PreviewGeometry(
                    row.frame,
                    "polygon_approximation",
                    "polygon approximation",
                    polygons=tuple(
                        tuple((float(point[0]), float(point[1])) for point in polygon)
                        for polygon in approximated
                    ),
                    track_id=row.track_id,
                    detail=f"RDP {sum(map(len, polygons))} -> {sum(map(len, approximated))} vertices",
                )
            )
        output_rows.append(
            MaskRow(
                frame=row.frame,
                track_id=row.track_id,
                polygons=json.dumps(
                    approximated, ensure_ascii=False, separators=(",", ":")
                ),
                label=row.label,
                shape_type="polygon",
            )
        )
    return write_mask_sqlite(output_sqlite, output_rows, reference_sqlite=input_sqlite)
