from __future__ import annotations

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from progress_protocol import (
    INTERVAL_ENVIRONMENT,
    PHASE_ENVIRONMENT,
    InferenceProgressReporter,
)


class ProgressProtocolTests(unittest.TestCase):
    def test_reporter_throttles_updates_but_never_drops_completion(self) -> None:
        output = io.StringIO()
        environment = {
            PHASE_ENVIRONMENT: "segmentation_inference",
            INTERVAL_ENVIRONMENT: "0.3",
        }
        with (
            patch.dict(os.environ, environment, clear=False),
            patch(
                "progress_protocol.time.monotonic",
                side_effect=(0.0, 0.1, 0.31, 0.31, 0.4),
            ),
            redirect_stdout(output),
        ):
            reporter = InferenceProgressReporter.from_environment(
                available_frames=100,
                max_frames=None,
            )
            assert reporter is not None
            reporter.start()
            reporter.update(10, fps=50.0)
            reporter.update(31, fps=52.0)
            reporter.complete(100, fps=55.0)

        events = [
            json.loads(line.split(" ", 1)[1])
            for line in output.getvalue().splitlines()
        ]
        self.assertEqual([0, 31, 100], [event["completed"] for event in events])
        self.assertEqual("complete", events[-1]["state"])
        self.assertEqual(100, events[-1]["total"])

    def test_max_frames_caps_the_reported_total(self) -> None:
        with patch.dict(
            os.environ,
            {PHASE_ENVIRONMENT: "face_inference"},
            clear=False,
        ):
            reporter = InferenceProgressReporter.from_environment(
                available_frames=5290,
                max_frames=600,
            )
        assert reporter is not None
        self.assertEqual(600, reporter.total)


if __name__ == "__main__":
    unittest.main()
