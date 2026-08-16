from __future__ import annotations

import argparse
import unittest

import numpy as np

from keyframes.ellipse.optimizer import kfbase_module as base
from keyframes.ellipse.trackk_dense_recall import kftrackk_parse_args


class EllipseBandedSolverTest(unittest.TestCase):
    def test_production_keyframe_defaults_preserve_fast_motion(self) -> None:
        args = kftrackk_parse_args(
            ["--input-metrics-csv", "input.csv", "--output-dir", "output"]
        )
        self.assertEqual(0.0, args.smooth_alpha)
        self.assertEqual(1, args.min_gap)
        self.assertEqual("raw", args.keyframe_value_source)

    def test_second_difference_solver_matches_dense_reference(self) -> None:
        rng = np.random.default_rng(1042)
        for length in (3, 4, 5, 31):
            weights = rng.uniform(0.18, 1.0, size=length)
            values = rng.normal(size=(length, 5))
            alpha = 1.25
            second = base.build_second_difference_matrix(length)
            system = np.diag(weights) + alpha * (second.T @ second)
            expected = np.linalg.solve(system, weights[:, None] * values)
            actual = base.solve_second_difference_system(weights, values, alpha)
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)

    def test_stream_smoothing_keeps_short_stream_behavior(self) -> None:
        states = np.asarray(
            [[10.0, 20.0, 5.0, 3.0, 0.0], [11.0, 21.0, 5.0, 3.0, 1.0]],
            dtype=np.float64,
        )
        stream = base.StreamSegment(
            stream_id="1:K1:run0:slot0",
            track_id="1",
            mode="K1",
            run_id=0,
            slot_id=0,
            frame_numbers=np.asarray([0, 1], dtype=np.int32),
            raw_states=states,
            confidence=np.ones(2, dtype=np.float64),
            weighted_error=np.zeros(2, dtype=np.float64),
        )
        args = argparse.Namespace(
            theta_weight_floor=0.2,
            min_segment_length=3,
            confidence_floor=0.18,
            smooth_alpha=1.0,
        )
        base.smooth_stream_segment(stream, args)
        np.testing.assert_array_equal(stream.smoothed_q, stream.raw_q)

    def test_global_refinement_matches_dense_reference(self) -> None:
        rng = np.random.default_rng(2048)
        for length, keyframes in (
            (3, [0, 2]),
            (17, [0, 3, 8, 12, 16]),
            (40, [0, 2, 7, 15, 21, 30, 39]),
        ):
            target = rng.normal(size=(length, 5))
            base_keys = target[keyframes] + rng.normal(
                scale=0.05, size=(len(keyframes), 5)
            )
            weights = rng.uniform(0.1, 2.0, size=length)
            ridge = 0.001
            interpolation = np.zeros((length, len(keyframes)), dtype=np.float64)
            for index, (left, right) in enumerate(zip(keyframes[:-1], keyframes[1:])):
                for position in range(left, right + 1):
                    alpha = (position - left) / float(right - left)
                    interpolation[position, index] = 1.0 - alpha
                    interpolation[position, index + 1] = alpha
            weighted = interpolation * np.sqrt(weights)[:, None]
            gram = weighted.T @ weighted + ridge * np.eye(len(keyframes))
            expected = np.empty_like(base_keys)
            for dimension in range(target.shape[1]):
                rhs = (
                    weighted.T @ (target[:, dimension] * np.sqrt(weights))
                    + ridge * base_keys[:, dimension]
                )
                expected[:, dimension] = np.linalg.solve(gram, rhs)
            actual = base.refine_keyframe_values_global_ls(
                target,
                base_keys,
                keyframes,
                weights,
                ridge,
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
