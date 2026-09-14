import sys
import unittest
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from analyze_g0_sparse_gate import acceptance_decision  # noqa: E402


def category_frame(zero, high):
    return pd.DataFrame([
        {"demand_category": "zero", "mae": zero},
        {"demand_category": "high_ge6", "mae": high},
    ])


class G0AcceptanceTest(unittest.TestCase):
    def test_all_preregistered_conditions_pass(self):
        result = acceptance_decision(
            {"mae": 4.0, "rmse": 6.0, "mae_net": 3.0},
            {"mae": 3.96, "rmse": 6.03, "mae_net": 3.015},
            category_frame(1.0, 8.0),
            category_frame(0.84, 8.04),
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["decision"], "ADVANCE_TO_B1")

    def test_joint_rmse_net_guard_uses_and_semantics(self):
        result = acceptance_decision(
            {"mae": 4.0, "rmse": 6.0, "mae_net": 3.0},
            {"mae": 3.9, "rmse": 6.12, "mae_net": 3.0},
            category_frame(1.0, 8.0),
            category_frame(0.8, 8.0),
        )
        self.assertTrue(result["accepted"])
        result = acceptance_decision(
            {"mae": 4.0, "rmse": 6.0, "mae_net": 3.0},
            {"mae": 3.9, "rmse": 6.12, "mae_net": 3.06},
            category_frame(1.0, 8.0),
            category_frame(0.8, 8.0),
        )
        self.assertFalse(result["accepted"])

    def test_failed_primary_condition_stops_g0(self):
        result = acceptance_decision(
            {"mae": 4.0, "rmse": 6.0, "mae_net": 3.0},
            {"mae": 3.99, "rmse": 6.0, "mae_net": 3.0},
            category_frame(1.0, 8.0),
            category_frame(0.9, 8.0),
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["decision"], "STOP_G0_DO_NOT_ADVANCE")


if __name__ == "__main__":
    unittest.main()
