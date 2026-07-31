from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from common.progress import StageGraphProgress


class StageGraphProgressTests(unittest.TestCase):
    def test_in_stage_estimate_does_not_change_exact_completed_count(self) -> None:
        output = io.StringIO()
        progress = StageGraphProgress(10, interval_seconds=0.1)
        with (
            patch("common.progress.time.monotonic", return_value=100.0) as clock,
            redirect_stdout(output),
        ):
            progress.begin_stage(2, "tracking:running")
            clock.return_value = 108.0
            progress.activity("tracking:running")
            progress.finish_stage(3, "tracking:complete")

        events = [
            json.loads(line.split(" ", 1)[1])
            for line in output.getvalue().splitlines()
        ]
        self.assertEqual([2, 2, 3], [event["completed"] for event in events])
        self.assertEqual(0.2, events[0]["display_progress"])
        self.assertGreater(events[1]["display_progress"], 0.2)
        self.assertLess(events[1]["display_progress"], 0.3)
        self.assertTrue(events[1]["estimated"])
        self.assertEqual(0.3, events[2]["display_progress"])
        self.assertFalse(events[2]["estimated"])


if __name__ == "__main__":
    unittest.main()
