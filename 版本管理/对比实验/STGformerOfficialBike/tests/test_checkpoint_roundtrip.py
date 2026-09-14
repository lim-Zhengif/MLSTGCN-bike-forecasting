import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models import STGformerOfficialBike, build_model_from_checkpoint  # noqa: E402
from protocol import MODEL_ID, UPSTREAM_COMMIT  # noqa: E402


def make_checkpoint():
    model = STGformerOfficialBike(
        num_nodes=5,
        in_steps=8,
        out_steps=3,
        input_dim=6,
        output_dim=2,
        feature_mean=np.zeros((1, 1, 1, 6), dtype=np.float64),
        feature_std=np.ones((1, 1, 1, 6), dtype=np.float64),
        hour_feature_index=3,
        dow_feature_index=4,
        input_embedding_dim=4,
        tod_embedding_dim=2,
        dow_embedding_dim=2,
        adaptive_embedding_dim=4,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        dropout_a=0.0,
    ).eval()
    checkpoint = {
        "model_id": MODEL_ID,
        "upstream_commit": UPSTREAM_COMMIT,
        "model_config": model.export_config(),
        "feature_mean": np.zeros((1, 1, 1, 6), dtype=np.float64),
        "feature_std": np.ones((1, 1, 1, 6), dtype=np.float64),
        "model_state_dict": model.state_dict(),
    }
    return model, checkpoint


class CheckpointRoundtripTest(unittest.TestCase):
    def test_strict_roundtrip_preserves_prediction(self):
        torch.manual_seed(2)
        model, checkpoint = make_checkpoint()
        x = torch.randn(2, 8, 5, 6)
        expected = model(x).detach()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            torch.save(checkpoint, path)
            restored_payload = torch.load(path, map_location="cpu")
        restored, audit = build_model_from_checkpoint(restored_payload, device=torch.device("cpu"))
        restored.eval()
        actual = restored(x).detach()
        self.assertEqual(audit, {"missing_keys": [], "unexpected_keys": []})
        torch.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)

    def test_mismatched_state_dict_is_rejected(self):
        _, checkpoint = make_checkpoint()
        checkpoint["model_state_dict"] = dict(checkpoint["model_state_dict"])
        checkpoint["model_state_dict"]["unexpected.weight"] = torch.ones(1)
        with self.assertRaises(RuntimeError):
            build_model_from_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
