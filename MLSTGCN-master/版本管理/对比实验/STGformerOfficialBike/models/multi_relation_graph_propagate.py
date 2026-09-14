"""Static relation/order mixture injected into the official propagation path."""
import torch
from torch import nn
from .stgformer_official_adapter import STGformerOfficialBike

B1_MODEL_ID = "stgformer_official_static_multirelation_b1"


class MultiRelationGraphPropagate(nn.Module):
    def __init__(self, original, supports, max_order=2, initial_strength=0.01):
        super().__init__()
        if original.Ks < 2 or max_order < 1:
            raise ValueError("B1 requires official order >=2 and external order >=1")
        supports = torch.as_tensor(supports, dtype=torch.float32)
        if supports.ndim != 3 or supports.shape[1] != supports.shape[2]:
            raise ValueError("supports must be [R,N,N]")
        if not torch.isfinite(supports).all() or (supports < 0).any():
            raise ValueError("supports must be finite and nonnegative")
        if not torch.allclose(supports.sum(-1), torch.ones_like(supports.sum(-1)), atol=1e-5):
            raise ValueError("supports must be row stochastic")
        self.original = original
        powers = []
        current = supports
        for _ in range(max_order):
            powers.append(current)
            current = torch.matmul(current, supports)
        self.register_buffer("powers", torch.stack(powers, dim=1))
        self.logits = nn.Parameter(torch.zeros(len(supports), max_order))
        self.strength = nn.Parameter(torch.tensor(float(initial_strength)))

    def weights(self):
        return self.logits.flatten().softmax(0).reshape_as(self.logits)

    def forward(self, x, graph):
        paths = self.original(x, graph)
        # Sum_rk alpha_rk A_r^k X, never (sum_r alpha_r A_r)^k X.
        # Static weights permit this equivalent contraction without [B,T,N,R,K,D].
        support = torch.einsum("rk,rknm->nm", self.weights(), self.powers)
        external = torch.einsum("nm,btmd->btnd", support, x)
        strength = self.strength.tanh()
        return [paths[0]] + [p + strength * (external - x) for p in paths[1:]]


class B1STGformer(STGformerOfficialBike):
    def __init__(self, supports=None, relation_names=None, relation_audit=None,
                 external_order=2, initial_strength=0.01, **kwargs):
        super().__init__(**kwargs)
        self.relation_names = list(relation_names or [])
        if not self.relation_names or len(set(self.relation_names)) != len(self.relation_names):
            raise ValueError("unique relation_names are required")
        if supports is None:
            supports = torch.eye(self.num_nodes).repeat(len(self.relation_names), 1, 1)
        if len(supports) != len(self.relation_names):
            raise ValueError("relation/support count mismatch")
        self.relation_audit = relation_audit or {}
        self.external_order = int(external_order)
        self.initial_strength = float(initial_strength)
        for layer in self.core.attn_layers_s:
            layer.locals = MultiRelationGraphPropagate(
                layer.locals, supports, self.external_order, self.initial_strength)

    def export_config(self):
        config = super().export_config()
        config.update(relation_names=self.relation_names, relation_audit=self.relation_audit,
                      external_order=self.external_order, initial_strength=self.initial_strength)
        return config

    def diagnostics(self):
        rows = []
        for index, layer in enumerate(self.core.attn_layers_s):
            weights = layer.locals.weights().detach().cpu()
            for r, name in enumerate(self.relation_names):
                for k in range(self.external_order):
                    rows.append(dict(layer=index, relation=name, order=k+1,
                                     weight=float(weights[r, k]),
                                     strength=float(layer.locals.strength.detach().tanh().cpu())))
        return rows


def build_b1_from_checkpoint(checkpoint, device=None):
    if checkpoint.get("model_id") != B1_MODEL_ID:
        raise ValueError("Not a B1 checkpoint")
    model = B1STGformer(feature_mean=checkpoint["feature_mean"],
                       feature_std=checkpoint["feature_std"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if device is not None:
        model = model.to(device)
    return model, {"missing_keys": [], "unexpected_keys": []}
