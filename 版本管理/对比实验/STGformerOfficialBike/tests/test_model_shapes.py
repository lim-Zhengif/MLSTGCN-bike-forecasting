import sys
import unittest
from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models import STGformerOfficialBike  # noqa: E402


def make_model(in_steps, num_nodes, input_dim, out_steps=3, output_dim=2):
    return STGformerOfficialBike(
        num_nodes=num_nodes,
        in_steps=in_steps,
        out_steps=out_steps,
        input_dim=input_dim,
        output_dim=output_dim,
        feature_mean=torch.zeros(input_dim),
        feature_std=torch.ones(input_dim),
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
        kernel_size=1,
        order=2,
    )


class ModelShapeTest(unittest.TestCase):
    def test_exact_bike_contract_shape(self):
        torch.manual_seed(0)
        model = make_model(in_steps=168, num_nodes=150, input_dim=21).eval()
        x = torch.randn(1, 168, 150, 21)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(tuple(output.shape), (1, 3, 150, 2))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_forward_backward_is_finite(self):
        torch.manual_seed(1)
        model = make_model(in_steps=8, num_nodes=5, input_dim=6).train()
        x = torch.randn(2, 8, 5, 6)
        target = torch.randn(2, 3, 5, 2)
        output = model(x)
        loss = torch.nn.functional.smooth_l1_loss(output, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))

    def test_rejects_wrong_history_shape(self):
        model = make_model(in_steps=8, num_nodes=5, input_dim=6)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 7, 5, 6))


if __name__ == "__main__":
    unittest.main()
