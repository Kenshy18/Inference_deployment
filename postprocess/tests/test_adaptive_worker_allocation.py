from __future__ import annotations

import unittest

from experiments.runtime_floor_stability.adaptive_worker_allocation import (
    allocate_fixed_budget,
    allocation_imbalance,
    recommended_process_budget,
)


class AdaptiveWorkerAllocationTests(unittest.TestCase):
    def test_balanced_three_labels_receive_two_each(self) -> None:
        workloads = {"女性器": 9132, "男性器": 6499, "結合部分": 8872}
        self.assertEqual(
            allocate_fixed_budget(workloads, process_budget=6),
            {"女性器": 2, "男性器": 2, "結合部分": 2},
        )

    def test_dominant_label_receives_four_workers(self) -> None:
        workloads = {"女性器": 21916, "男性器": 2634, "結合部分": 4131}
        allocation = allocate_fixed_budget(workloads, process_budget=6)
        self.assertEqual(
            allocation,
            {"女性器": 4, "男性器": 1, "結合部分": 1},
        )
        self.assertLess(allocation_imbalance(workloads, allocation), 2.1)

    def test_single_active_label_receives_entire_budget(self) -> None:
        self.assertEqual(
            allocate_fixed_budget(
                {"女性器": 0, "男性器": 7591, "結合部分": 0},
                process_budget=6,
            ),
            {"男性器": 6},
        )

    def test_empty_workload_needs_no_process(self) -> None:
        self.assertEqual(
            allocate_fixed_budget({"女性器": 0}, process_budget=6),
            {},
        )

    def test_screened_24_core_budget(self) -> None:
        self.assertEqual(
            recommended_process_budget(cpu_count=24, active_label_count=3), 8
        )
        self.assertEqual(
            recommended_process_budget(cpu_count=24, active_label_count=2), 8
        )
        self.assertEqual(
            recommended_process_budget(cpu_count=24, active_label_count=1), 6
        )

    def test_budget_never_drops_below_active_labels(self) -> None:
        self.assertEqual(
            recommended_process_budget(cpu_count=4, active_label_count=3), 3
        )


if __name__ == "__main__":
    unittest.main()
