"""History-only soft suppression gate on top of a frozen B0 checkpoint."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stgformer_official_adapter import STGformerOfficialBike


GATE_MODEL_ID = "stgformer_official_b0_sparse_demand_gate_g0"


def inverse_softplus(value, beta=5.0):
    scaled = torch.clamp(value * float(beta), min=1e-8)
    return torch.where(
        scaled > 20.0,
        value,
        torch.log(torch.expm1(scaled)) / float(beta),
    )


class SparseDemandGate(nn.Module):
    """Shared gate using only frozen predictions and observable history."""

    FEATURE_NAMES = (
        "base_log1p_prediction",
        "history_last_log1p",
        "history_mean_6h_log1p",
        "history_mean_24h_log1p",
        "history_zero_fraction_24h",
        "horizon_fraction",
        "target_hour_sin",
        "target_hour_cos",
    )

    def __init__(self, hidden_dim=16, gate_floor=0.1, initial_gate=0.98):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gate_floor = float(gate_floor)
        self.initial_gate = float(initial_gate)
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= self.gate_floor < self.initial_gate < 1.0:
            raise ValueError("Require 0 <= gate_floor < initial_gate < 1")
        self.network = nn.Sequential(
            nn.Linear(len(self.FEATURE_NAMES), self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        initial_probability = (
            (self.initial_gate - self.gate_floor) / (1.0 - self.gate_floor)
        )
        initial_logit = math.log(initial_probability / (1.0 - initial_probability))
        nn.init.constant_(self.network[-1].bias, initial_logit)

    def export_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "gate_floor": self.gate_floor,
            "initial_gate": self.initial_gate,
            "feature_names": list(self.FEATURE_NAMES),
        }

    def forward(self, features):
        probability = torch.sigmoid(self.network(features).squeeze(-1))
        return self.gate_floor + (1.0 - self.gate_floor) * probability


class FrozenB0SparseGate(nn.Module):
    """Frozen B0 plus a trainable history-conditioned multiplicative gate."""

    def __init__(
        self,
        base_model,
        target_mean,
        target_std,
        gate,
        flow_feature_indices=(0, 1),
        hour_feature_index=3,
        softplus_beta=5.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.gate = gate
        self.flow_feature_indices = tuple(int(value) for value in flow_feature_indices)
        self.hour_feature_index = int(hour_feature_index)
        self.softplus_beta = float(softplus_beta)
        if len(self.flow_feature_indices) != int(base_model.output_dim):
            raise ValueError("One history feature index is required per output channel")
        self.register_buffer(
            "target_mean", torch.as_tensor(target_mean, dtype=torch.float32).reshape(-1)
        )
        self.register_buffer(
            "target_std", torch.as_tensor(target_std, dtype=torch.float32).reshape(-1)
        )
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.base_model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self

    def gate_features(self, x_normalized, base_counts):
        mean = self.base_model.feature_mean
        std = self.base_model.feature_std
        flow_indices = torch.as_tensor(
            self.flow_feature_indices, dtype=torch.long, device=x_normalized.device
        )
        history = x_normalized.index_select(-1, flow_indices)
        history = history * std.index_select(0, flow_indices) + mean.index_select(0, flow_indices)
        last = history[:, -1]
        mean_6h = history[:, -min(6, history.shape[1]) :].mean(dim=1)
        recent_24h = history[:, -min(24, history.shape[1]) :]
        mean_24h = recent_24h.mean(dim=1)
        zero_fraction = (recent_24h <= 1e-6).to(history.dtype).mean(dim=1)

        batch_size, horizons, nodes, directions = base_counts.shape
        expand_shape = (batch_size, horizons, nodes, directions)
        last = last.unsqueeze(1).expand(expand_shape)
        mean_6h = mean_6h.unsqueeze(1).expand(expand_shape)
        mean_24h = mean_24h.unsqueeze(1).expand(expand_shape)
        zero_fraction = zero_fraction.unsqueeze(1).expand(expand_shape)

        horizon_ids = torch.arange(
            1, horizons + 1, dtype=base_counts.dtype, device=base_counts.device
        ).view(1, horizons, 1, 1)
        horizon_fraction = (horizon_ids / float(horizons)).expand(expand_shape)
        last_hour = (
            x_normalized[:, -1, :, self.hour_feature_index]
            * std[self.hour_feature_index]
            + mean[self.hour_feature_index]
        )
        target_hour = torch.remainder(
            torch.round(last_hour).unsqueeze(1).unsqueeze(-1) + horizon_ids,
            24.0,
        ).expand(expand_shape)
        angle = target_hour * (2.0 * math.pi / 24.0)
        return torch.stack(
            [
                torch.log1p(base_counts),
                last,
                mean_6h,
                mean_24h,
                zero_fraction,
                horizon_fraction,
                torch.sin(angle),
                torch.cos(angle),
            ],
            dim=-1,
        )

    def forward_counts(self, x_normalized, return_diagnostics=False):
        with torch.no_grad():
            base_raw = self.base_model(x_normalized)
            base_counts = F.softplus(
                base_raw * self.target_std + self.target_mean,
                beta=self.softplus_beta,
            )
        features = self.gate_features(x_normalized, base_counts)
        gate_values = self.gate(features)
        gated_counts = base_counts * gate_values
        if return_diagnostics:
            return gated_counts, gate_values, base_counts
        return gated_counts

    def forward(self, x_normalized):
        """Return pseudo-normalized values compatible with the B0 evaluator."""
        counts = self.forward_counts(x_normalized)
        pre_softplus = inverse_softplus(counts, beta=self.softplus_beta)
        return (pre_softplus - self.target_mean) / self.target_std


def build_sparse_gate_from_checkpoint(checkpoint, device=None):
    required = {
        "model_config",
        "feature_mean",
        "feature_std",
        "target_mean",
        "target_std",
        "base_model_state_dict",
        "gate_config",
        "gate_state_dict",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError("Sparse-gate checkpoint missing fields: %s" % missing)
    base_model = STGformerOfficialBike(
        feature_mean=checkpoint["feature_mean"],
        feature_std=checkpoint["feature_std"],
        **copy.deepcopy(checkpoint["model_config"]),
    )
    base_incompatible = base_model.load_state_dict(
        checkpoint["base_model_state_dict"], strict=False
    )
    gate_config = dict(checkpoint["gate_config"])
    gate_config.pop("feature_names", None)
    gate = SparseDemandGate(**gate_config)
    gate_incompatible = gate.load_state_dict(checkpoint["gate_state_dict"], strict=False)
    audit = {
        "missing_keys": list(base_incompatible.missing_keys) + [
            "gate." + key for key in gate_incompatible.missing_keys
        ],
        "unexpected_keys": list(base_incompatible.unexpected_keys) + [
            "gate." + key for key in gate_incompatible.unexpected_keys
        ],
    }
    if audit["missing_keys"] or audit["unexpected_keys"]:
        raise RuntimeError("Sparse-gate checkpoint/model mismatch: %s" % audit)
    model = FrozenB0SparseGate(
        base_model=base_model,
        target_mean=checkpoint["target_mean"],
        target_std=checkpoint["target_std"],
        gate=gate,
        flow_feature_indices=checkpoint.get("flow_feature_indices", (0, 1)),
        hour_feature_index=checkpoint.get("hour_feature_index", 3),
        softplus_beta=checkpoint.get("softplus_beta", 5.0),
    )
    if device is not None:
        model = model.to(device)
    return model, audit
