import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from protocol import audit_anchor_coverage  # noqa: E402


class AnchorCoverageTest(unittest.TestCase):
    def test_default_range_requires_july_boundary_hour(self):
        coverage = audit_anchor_coverage(
            [],
            "2026-03-01",
            "2026-06-30",
            [0, 3, 6, 9, 12, 15, 18, 21],
            target_start_offset=1,
            pred_len=3,
        )
        self.assertEqual(coverage["expected_count"], 976)
        self.assertEqual(
            coverage["latest_requested_anchor_datetime"],
            "2026-06-30 21:00:00",
        )
        self.assertEqual(
            coverage["latest_required_target_datetime"],
            "2026-07-01 00:00:00",
        )

    def test_missing_final_anchor_is_reported_exactly(self):
        anchors = [0, 3, 6, 9, 12, 15, 18]
        sample_datetimes = ["2026-06-30 %02d:00:00" % hour for hour in anchors]
        coverage = audit_anchor_coverage(
            sample_datetimes,
            "2026-06-30",
            "2026-06-30",
            anchors + [21],
            target_start_offset=1,
            pred_len=3,
        )
        self.assertEqual(coverage["provided_count"], 7)
        self.assertEqual(
            coverage["missing_anchor_datetimes"],
            ["2026-06-30 21:00:00"],
        )

    def test_complete_range_has_no_missing_or_duplicate_anchors(self):
        sample_datetimes = np.asarray(
            [
                "2026-06-29 00:00:00",
                "2026-06-29 12:00:00",
                "2026-06-30 00:00:00",
                "2026-06-30 12:00:00",
            ],
            dtype=np.str_,
        )
        coverage = audit_anchor_coverage(
            sample_datetimes,
            "2026-06-29",
            "2026-06-30",
            [0, 12],
            target_start_offset=1,
            pred_len=3,
        )
        self.assertEqual(coverage["missing_anchor_datetimes"], [])
        self.assertEqual(coverage["duplicate_anchor_count"], 0)


if __name__ == "__main__":
    unittest.main()
