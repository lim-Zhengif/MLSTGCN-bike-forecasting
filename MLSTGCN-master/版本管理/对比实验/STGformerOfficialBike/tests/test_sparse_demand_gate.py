import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models import (  # noqa: E402
    FrozenB0SparseGate,
    STGformerOfficialBike,
    SparseDemandGate,
    build_sparse_gate_from_checkpoint,
)


def make_wrapper():
    feature_mean = np.zeros((1, 1, 1, 6), dtype=np.float64)
    feature_std = np.ones((1, 1, 1, 6), dtype=np.float64)
    base = STGformerOfficialBike(
        num_nodes=5,
        in_steps=8,
        out_steps=3,
        input_dim=6,
        output_dim=2,
        feature_mean=feature_mean,
        feature_std=feature_std,
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
    )
    gate = SparseDemandGate(hidden_dim=8, gate_floor=0.1, initial_gate=0.98)
    wrapper = FrozenB0SparseGate(
        base_model=base,
        target_mean=np.zeros((1, 1, 1, 2), dtype=np.float64),
        target_std=np.ones((1, 1, 1, 2), dtype=np.float64),
        gate=gate,
        flow_feature_indices=(0, 1),
        hour_feature_index=3,
    )
    return wrapper, feature_mean, feature_std


class SparseDemandGateTest(unittest.TestCase):
    def test_initial_gate_is_near_identity_and_bounded(self):
        gate = SparseDemandGate(hidden_dim=8, gate_floor=0.1, initial_gate=0.98)
        features = torch.randn(4, 3, 5, 2, len(gate.FEATURE_NAMES))
        values = gate(features)
        torch.testing.assert_allclose(values, torch.full_like(values, 0.98), atol=1e-6, rtol=0)
        self.assertGreaterEqual(float(values.min()), 0.1)
        self.assertLessEqual(float(values.max()), 1.0)

    def test_wrapper_preserves_shape_and_external_softplus_contract(self):
        torch.manual_seed(7)
        wrapper, _, _ = make_wrapper()
        x = torch.randn(2, 8, 5, 6)
        x[..., 3] = torch.randint(0, 24, x[..., 3].shape).float()
        counts, gates, base = wrapper.forward_counts(x, return_diagnostics=True)
        pseudo = wrapper(x)
        recovered = F.softplus(
            pseudo * wrapper.target_std + wrapper.target_mean,
            beta=wrapper.softplus_beta,
        )
        self.assertEqual(tuple(counts.shape), (2, 3, 5, 2))
        self.assertEqual(tuple(gates.shape), tuple(counts.shape))
        torch.testing.assert_allclose(recovered, counts, atol=2e-6, rtol=1e-6)
        self.assertTrue(bool(torch.all(counts <= base)))

    def test_backbone_is_frozen_and_stays_in_eval_mode(self):
        wrapper, _, _ = make_wrapper()
        wrapper.train()
        self.assertFalse(wrapper.base_model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in wrapper.base_model.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in wrapper.gate.parameters()))

    def test_checkpoint_roundtrip_is_exact(self):
        torch.manual_seed(9)
        wrapper, feature_mean, feature_std = make_wrapper()
        checkpoint = {
            "model_config": wrapper.base_model.export_config(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": np.zeros((1, 1, 1, 2), dtype=np.float64),
            "target_std": np.ones((1, 1, 1, 2), dtype=np.float64),
            "base_model_state_dict": wrapper.base_model.state_dict(),
            "gate_config": wrapper.gate.export_config(),
            "gate_state_dict": wrapper.gate.state_dict(),
            "flow_feature_indices": [0, 1],
            "hour_feature_index": 3,
            "softplus_beta": 5.0,
        }
        restored, audit = build_sparse_gate_from_checkpoint(checkpoint, device=torch.device("cpu"))
        x = torch.randn(2, 8, 5, 6)
        expected = wrapper.eval()(x).detach()
        actual = restored.eval()(x).detach()
        self.assertEqual(audit, {"missing_keys": [], "unexpected_keys": []})
        torch.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
