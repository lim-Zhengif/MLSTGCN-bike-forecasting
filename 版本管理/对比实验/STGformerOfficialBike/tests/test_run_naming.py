import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from protocol import (  # noqa: E402
    b0_holdout_names,
    b0_training_run_name,
    checkpoint_training_seed,
)


class RunNamingTest(unittest.TestCase):
    def test_training_name_tracks_seed_and_batch_size(self):
        self.assertEqual(
            b0_training_run_name(seed=2, batch_size=16),
            "b0_top150_hist168_pred3_seed2_bs16",
        )

    def test_holdout_names_track_seed_and_weather_regime(self):
        self.assertEqual(
            b0_holdout_names(1, "2026-03-01", "2026-06-30", 0),
            (
                "holdout_202603_202606_seed1_oracle",
                "stgformer_official_b0_seed1_202603_202606_oracle",
            ),
        )
        self.assertEqual(
            b0_holdout_names(2, "2026-03-01", "2026-06-30", 1),
            (
                "holdout_202603_202606_seed2_lag1",
                "stgformer_official_b0_seed2_202603_202606_lag1",
            ),
        )

    def test_checkpoint_seed_is_required(self):
        self.assertEqual(checkpoint_training_seed({"args": {"seed": 1}}), 1)
        with self.assertRaises(RuntimeError):
            checkpoint_training_seed({})


if __name__ == "__main__":
    unittest.main()
