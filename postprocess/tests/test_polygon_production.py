from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from approximation.polygon.production import (
    DEFAULT_NUM_WORKERS,
    _resolve_cpp_compiler,
)
from approximation.polygon.preparation import (
    _expand_polygon,
    apply_border_expansion,
    apply_endpoint_extension,
)
from contracts.mask_sqlite import MaskRow, read_mask_rows, write_mask_sqlite


class PolygonProductionPreparationTests(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        rows = []
        for frame in range(10, 20):
            left = 2 + frame - 10
            rows.append(
                MaskRow(
                    frame,
                    "1",
                    json.dumps(
                        [[[left, 100], [left + 30, 100], [left + 30, 140], [left, 140]]]
                    ),
                    "sample",
                    "polygon",
                )
            )
        source = write_mask_sqlite(root / "source.sqlite", rows)
        with sqlite3.connect(source) as connection:
            connection.execute("CREATE TABLE cuts(frame INTEGER PRIMARY KEY)")
        return source

    def test_original_border_expansion_and_endpoint_extension_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            border, border_summary = apply_border_expansion(
                source, root / "border.sqlite"
            )
            self.assertEqual(9, border_summary["changed_rows"])
            first = read_mask_rows(border)[0]
            points = json.loads(first.polygons)[0]
            self.assertLess(min(point[0] for point in points), 0.0)

            endpoint, endpoint_summary = apply_endpoint_extension(
                border, root / "endpoint.sqlite"
            )
            self.assertEqual(10, endpoint_summary["inserted_rows"])
            rows = read_mask_rows(endpoint)
            self.assertEqual(20, len(rows))
            self.assertEqual(5, rows[0].frame)
            self.assertEqual(24, rows[-1].frame)

    def test_candidate_border_cap_and_corner_support(self) -> None:
        polygon = np.asarray(
            [[2.0, 24.0], [200.0, 2.0], [250.0, 100.0], [100.0, 250.0]],
            dtype=np.float32,
        )
        expanded, changed = _expand_polygon(
            polygon,
            width=400,
            height=400,
            trigger_px=10.0,
            expand_ratio=0.10,
            min_expand_px=6.0,
            max_expand_px=16.0,
            influence_px=16.0,
            corner_support=True,
        )
        self.assertTrue(changed)
        self.assertAlmostEqual(-14.0, float(expanded[0, 0]), places=5)
        self.assertAlmostEqual(8.0, float(expanded[0, 1]), places=5)
        self.assertLessEqual(float(polygon[0, 0] - expanded[0, 0]), 16.0)
        self.assertLessEqual(float(polygon[0, 1] - expanded[0, 1]), 16.0)

    def test_native_compiler_respects_explicit_environment(self) -> None:
        with mock.patch.dict(os.environ, {"CXX": "/test/compiler"}):
            self.assertEqual("/test/compiler", _resolve_cpp_compiler())

    def test_parallel_default_is_bounded_for_interactive_use(self) -> None:
        self.assertGreaterEqual(DEFAULT_NUM_WORKERS, 1)
        self.assertLessEqual(DEFAULT_NUM_WORKERS, 4)


if __name__ == "__main__":
    unittest.main()
