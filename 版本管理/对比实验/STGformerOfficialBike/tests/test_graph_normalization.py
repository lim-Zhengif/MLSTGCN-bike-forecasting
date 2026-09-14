import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models import GraphPropagate, STGformerOfficialBike  # noqa: E402
from protocol import audit_graph_contract, row_normalize  # noqa: E402


class GraphContractTest(unittest.TestCase):
    def test_row_normalization(self):
        matrix = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=np.float32)
        normalized = row_normalize(matrix, add_self=True)
        np.testing.assert_allclose(normalized.sum(axis=1), np.ones(2), atol=1e-7)
        self.assertTrue(np.isfinite(normalized).all())

    def test_adaptive_graph_is_row_stochastic(self):
        model = STGformerOfficialBike(
            num_nodes=5,
            in_steps=8,
            out_steps=3,
            input_dim=6,
            output_dim=2,
            feature_mean=torch.zeros(6),
            feature_std=torch.ones(6),
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
        graph = model.build_adaptive_graph()
        self.assertEqual(tuple(graph.shape), (8, 5, 5))
        torch.testing.assert_allclose(graph.sum(dim=-1), torch.ones(8, 5), atol=1e-6, rtol=1e-6)

    def test_graph_propagation_identity(self):
        layer = GraphPropagate(Ks=2, dropout=0.0).eval()
        x = torch.randn(2, 3, 4, 5)
        graph = torch.eye(4).unsqueeze(0).expand(3, 4, 4)
        outputs = layer(x, graph)
        self.assertEqual(len(outputs), 2)
        torch.testing.assert_allclose(outputs[1], x)

    def test_repository_graph_matches_top150_contract(self):
        graph_dir = PROJECT_ROOT / "data" / "graph" / (
            "bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_graph_"
            "train2025_hist168_pred3_8anchors"
        )
        audit = audit_graph_contract(graph_dir, "dist", 150)
        self.assertEqual(audit["graph_shape"], [150, 150])
        self.assertEqual(audit["b0_graph_role"], "node_order_audit_only")
        self.assertEqual(len(audit["node_signature"]), 64)


if __name__ == "__main__":
    unittest.main()
