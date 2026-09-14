import sys
import unittest
from pathlib import Path
import torch
import numpy as np
import tempfile
import subprocess
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.stgformer_official_adapter import GraphPropagate
from models.multi_relation_graph_propagate import (
    MultiRelationGraphPropagate, B1STGformer, B1_MODEL_ID, build_b1_from_checkpoint)
from test_sparse_demand_gate import make_wrapper
from b1_graphs import load_relations
from train_stgformer_official_bike import DEFAULT_GRAPH_DIR


class B1Test(unittest.TestCase):
    def test_exact_separate_relation_powers(self):
        a = torch.tensor([[[0., 1.], [1., 0.]], [[1., 0.], [.5, .5]]])
        module = MultiRelationGraphPropagate(GraphPropagate(2, dropout=0), a)
        x = torch.randn(2, 3, 2, 4)
        explicit = sum(torch.einsum('nm,btmd->btnd', matrix, x)
                       for matrix in [a[0], a[0] @ a[0], a[1], a[1] @ a[1]]) / 4
        actual = module(x, torch.eye(2).repeat(3, 1, 1))[1]
        torch.testing.assert_allclose(actual, x + module.strength.tanh() * (explicit-x))
        actual.square().sum().backward()
        self.assertGreater(float(module.logits.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(module.strength.grad))

    def test_zero_injection_equivalence_gradient_and_roundtrip(self):
        wrapper, mean, std = make_wrapper()
        base = wrapper.base_model.eval()
        model = B1STGformer(supports=torch.eye(5).repeat(2, 1, 1),
                           relation_names=['a', 'b'], initial_strength=0.,
                           feature_mean=mean, feature_std=std, **base.export_config()).eval()
        missing = model.load_state_dict(base.state_dict(), strict=False).missing_keys
        self.assertTrue(all('.locals.' in name for name in missing))
        x = torch.randn(2, 8, 5, 6)
        torch.testing.assert_allclose(model(x), base(x), atol=0, rtol=0)
        model(x).sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        checkpoint = dict(model_id=B1_MODEL_ID, model_config=model.export_config(),
                          model_state_dict=model.state_dict(), feature_mean=mean, feature_std=std)
        restored, _ = build_b1_from_checkpoint(checkpoint)
        torch.testing.assert_allclose(model(x), restored.eval()(x), atol=0, rtol=0)
        checkpoint['model_state_dict'] = dict(checkpoint['model_state_dict'])
        del checkpoint['model_state_dict']['core.attn_layers_s.0.locals.powers']
        with self.assertRaises(RuntimeError):
            build_b1_from_checkpoint(checkpoint)

    def test_real_graph_contract(self):
        supports, names, audit = load_relations(DEFAULT_GRAPH_DIR, 150)
        self.assertEqual(supports.shape, (10, 150, 150))
        self.assertTrue(np.allclose(supports.sum(-1), 1, atol=1e-6))
        self.assertEqual(len(audit['relations']), len(names))

    def test_training_entrypoint_complete_mini_run(self):
        package = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(package), prefix='.tmp_b1_test_') as temp:
            folder = Path(temp)
            rng = np.random.RandomState(42)
            for split in ['train', 'val', 'test']:
                x = rng.rand(2, 168, 150, 6).astype(np.float32)
                x[..., 3] = 3
                x[..., 4] = 2
                y = rng.rand(2, 3, 150, 2).astype(np.float32)
                np.savez(folder / (split+'.npz'), x=x, y=y,
                         input_feature_cols=np.array(['out','in','f','hour','dow','g']),
                         target_cols=np.array(['out','in']))
            command = [sys.executable, '-B', str(package/'train_b1_stgformer.py'),
                       '--data_dir', str(folder), '--output_dir', str(folder/'result'),
                       '--device', 'cpu', '--epochs', '1', '--batch_size', '2',
                       '--input_embedding_dim', '4', '--tod_embedding_dim', '4',
                       '--dow_embedding_dim', '4', '--adaptive_embedding_dim', '4',
                       '--num_layers', '1', '--num_heads', '4']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((folder/'result/training_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['model'], B1_MODEL_ID)
            self.assertTrue((folder/'result/best_stgformer_b1.pt').exists())
            self.assertTrue(np.isfinite(summary['internal_test']['mae']))


if __name__ == '__main__':
    unittest.main()
