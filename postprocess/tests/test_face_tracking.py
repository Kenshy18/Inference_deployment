from __future__ import annotations

import unittest

from face_privacy.tracking import (
    FaceTrackObservation,
    FaceTrackingConfig,
    track_face_observations,
)


def observation(
    observation_id: int,
    frame: int,
    x: float,
    *,
    score: float = 0.95,
) -> FaceTrackObservation:
    return FaceTrackObservation(
        observation_id=observation_id,
        anchor_detection_id=observation_id,
        frame=frame,
        bbox=(x, 10.0, x + 40.0, 60.0),
        head_score=score,
        face_score=score,
    )


class FaceTrackingTests(unittest.TestCase):
    def test_assigns_one_id_across_a_short_detection_gap(self) -> None:
        assignments, summaries = track_face_observations(
            (
                observation(1, 0, 10.0),
                observation(2, 1, 12.0),
                observation(3, 3, 16.0),
            )
        )

        self.assertEqual(1, len({row.raw_track_id for row in assignments}))
        self.assertEqual(1, len(summaries))
        self.assertEqual(3, summaries[0].observed_frames)
        self.assertFalse(summaries[0].removed_by_short_track)

    def test_cut_between_observed_frames_always_splits_the_track(self) -> None:
        assignments, summaries = track_face_observations(
            (
                observation(1, 10, 20.0),
                observation(2, 12, 22.0),
            ),
            cuts={11},
        )

        self.assertEqual(2, len({row.raw_track_id for row in assignments}))
        self.assertEqual([0, 1], [row.scene_id for row in assignments])
        self.assertEqual(
            {"cut", "end_of_stream"}, {row.termination_reason for row in summaries}
        )

    def test_only_low_confidence_single_frame_track_is_removed(self) -> None:
        assignments, summaries = track_face_observations(
            (
                observation(1, 0, 10.0, score=0.40),
                observation(2, 10, 100.0, score=0.95),
            ),
            config=FaceTrackingConfig(
                max_gap_frames=2,
                short_track_max_hits=1,
                short_track_keep_score=0.85,
            ),
        )

        self.assertEqual(2, len(assignments))
        by_start = {row.start_frame: row for row in summaries}
        self.assertTrue(by_start[0].removed_by_short_track)
        self.assertIsNone(by_start[0].final_track_id)
        self.assertFalse(by_start[10].removed_by_short_track)
        self.assertIsNotNone(by_start[10].final_track_id)

    def test_default_removes_two_hit_low_confidence_track_but_keeps_strong_one(
        self,
    ) -> None:
        _assignments, summaries = track_face_observations(
            (
                observation(1, 0, 10.0, score=0.89),
                observation(2, 1, 11.0, score=0.88),
                observation(3, 10, 100.0, score=0.95),
                observation(4, 11, 101.0, score=0.89),
            )
        )

        by_start = {row.start_frame: row for row in summaries}
        self.assertTrue(by_start[0].removed_by_short_track)
        self.assertFalse(by_start[10].removed_by_short_track)


if __name__ == "__main__":
    unittest.main()
