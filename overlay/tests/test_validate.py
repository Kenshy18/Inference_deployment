from __future__ import annotations

import unittest

from overlay_renderer.validate import segment_boundaries


class ValidateTests(unittest.TestCase):
    def test_standard_renderer_summary_has_no_segment_boundaries(self) -> None:
        boundaries, records, errors = segment_boundaries(
            {
                "frames_written": 14400,
                "first_frame": 0,
                "last_frame": 14399,
            }
        )

        self.assertEqual([], boundaries)
        self.assertEqual([], records)
        self.assertEqual([], errors)

    def test_segmented_renderer_summary_validates_worker_ranges(self) -> None:
        boundaries, records, errors = segment_boundaries(
            {
                "start_frame": 0,
                "end_frame": 7,
                "frames": 8,
                "workers_detail": [
                    {
                        "start_frame": 0,
                        "end_frame": 3,
                        "renderer_summary": {"frames_written": 4},
                    },
                    {
                        "start_frame": 4,
                        "end_frame": 7,
                        "renderer_summary": {"frames_written": 4},
                    },
                ],
            }
        )

        self.assertEqual([4], boundaries)
        self.assertEqual(2, len(records))
        self.assertEqual([], errors)


if __name__ == "__main__":
    unittest.main()
