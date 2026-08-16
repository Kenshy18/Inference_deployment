from __future__ import annotations

import unittest

from deployment_tests.scripts.resource_sampler import select_pipeline_processes


class ResourceSamplerTests(unittest.TestCase):
    def test_marker_roots_include_all_descendants_but_not_neighbors(self) -> None:
        rows = [
            {"pid": 10, "ppid": 1, "command": "python -m orchestration"},
            {"pid": 11, "ppid": 10, "command": "python private_worker.py"},
            {"pid": 12, "ppid": 11, "command": "native_interval --batch"},
            {"pid": 20, "ppid": 1, "command": "python unrelated.py"},
            {"pid": 21, "ppid": 20, "command": "native_interval --unrelated"},
        ]
        selected = select_pipeline_processes(rows)
        self.assertEqual([10, 11, 12], [int(row["pid"]) for row in selected])

    def test_direct_postprocess_is_a_root(self) -> None:
        rows = [
            {
                "pid": 30,
                "ppid": 1,
                "command": "python /repo/postprocess/run_pipeline.py",
            },
            {"pid": 31, "ppid": 30, "command": "python opaque.py"},
        ]
        selected = select_pipeline_processes(rows)
        self.assertEqual([30, 31], [int(row["pid"]) for row in selected])


if __name__ == "__main__":
    unittest.main()
