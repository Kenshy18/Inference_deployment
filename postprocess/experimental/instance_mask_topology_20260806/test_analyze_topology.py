#!/usr/bin/env python3
"""Synthetic unit tests for foreground-island versus hole classification."""

from __future__ import annotations

import unittest

import numpy as np

from analyze_topology import _polygon_relation


def _square(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.asarray([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=np.float32)


class PolygonRelationTests(unittest.TestCase):
    def test_disconnected_squares_are_two_foreground_components(self) -> None:
        depths, parents = _polygon_relation(
            [_square(0, 0, 10, 10), _square(20, 0, 30, 10)]
        )
        self.assertEqual(depths, [0, 0])
        self.assertEqual(parents, [None, None])

    def test_inner_ring_is_a_hole(self) -> None:
        depths, parents = _polygon_relation(
            [_square(0, 0, 30, 30), _square(5, 5, 25, 25)]
        )
        self.assertEqual(depths, [0, 1])
        self.assertEqual(parents, [None, 0])

    def test_island_inside_hole_returns_to_foreground(self) -> None:
        depths, parents = _polygon_relation(
            [
                _square(0, 0, 40, 40),
                _square(5, 5, 35, 35),
                _square(10, 10, 30, 30),
            ]
        )
        self.assertEqual(depths, [0, 1, 2])
        self.assertEqual(parents, [None, 0, 1])


if __name__ == "__main__":
    unittest.main()
