"""Faithful, executable adaptation of the pinned official STGformer core.

The class names and computation follow ``upstream/STGformer.py``.  The local
adapter makes time covariates explicit and replaces only the unavailable timm
MLP dependency.  B0 intentionally uses the official learned adaptive graph;
external relation graphs are reserved for B1 and later experiments.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """PyTorch-only equivalent of the timm MLP used by the upstream model."""

    def __init__(self, in_features, hidden_features, act_layer=nn.ReLU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(float(drop))
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(float(drop))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class FastAttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, qkv_bias=False, kernel=1):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.qkv = nn.Linear(self.model_dim, self.model_dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(
            2 * self.model_dim if kernel != 12 else self.model_dim,
            self.model_dim,
        )
        self.fast = 1

    def forward(self, x, edge_index=None, dim=0):
        del edge_index
        query, key, value = self.qkv(x).chunk(3, -1)
        qs = torch.stack(torch.split(query, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        ks = torch.stack(torch.split(key, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        vs = torch.stack(torch.split(value, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        out_s = self.fast_attention(x, qs, ks, vs, dim=dim)
        if x.size(1) > 1:
            query_t = query.transpose(1, 2)
            key_t = key.transpose(1, 2)
            value_t = value.transpose(1, 2)
            qs = torch.stack(torch.split(query_t, self.head_dim, dim=-1), dim=-2).flatten(
                start_dim=dim, end_dim=dim + 1
            )
            ks = torch.stack(torch.split(key_t, self.head_dim, dim=-1), dim=-2).flatten(
                start_dim=dim, end_dim=dim + 1
            )
            vs = torch.stack(torch.split(value_t, self.head_dim, dim=-1), dim=-2).flatten(
                start_dim=dim, end_dim=dim + 1
            )
            out_t = self.fast_attention(x.transpose(1, 2), qs, ks, vs, dim=dim).transpose(1, 2)
            return self.out_proj(torch.cat([out_s, out_t], dim=-1))
        return self.out_proj(out_s)

    @staticmethod
    def fast_attention(x, qs, ks, vs, dim=0):
        qs = F.normalize(qs, dim=-1)
        ks = F.normalize(ks, dim=-1)
        sequence_size = qs.shape[1]
        batch, length = x.shape[dim : dim + 2]
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
        numerator = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)
        numerator = numerator + sequence_size * vs
        ones = torch.ones(ks.shape[1], dtype=ks.dtype, device=ks.device)
        ks_sum = torch.einsum("blhm,l->bhm", ks, ones)
        denominator = torch.einsum("bnhm,bhm->bnh", qs, ks_sum).unsqueeze(-1)
        denominator = denominator + sequence_size
        out = numerator / denominator
        # torch.unflatten is not consistently available in the project's 1.9 builds.
        out = out.reshape(batch, length, *out.shape[1:]).flatten(start_dim=3)
        return out


class GraphPropagate(nn.Module):
    def __init__(self, Ks, gso=None, dropout=0.2):
        super().__init__()
        self.Ks = int(Ks)
        self.gso = gso
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x, graph):
        if self.Ks < 1:
            raise ValueError("Ks must be a positive integer, received %s" % self.Ks)
        x_k = x
        x_list = [x]
        for _ in range(1, self.Ks):
            x_k = torch.einsum("thi,btij->bthj", graph, x_k.clone())
            x_list.append(self.dropout(x_k))
        return x_list


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        mlp_ratio=2,
        num_heads=8,
        dropout=0.0,
        mask=False,
        kernel=1,
        supports=None,
        order=2,
    ):
        super().__init__()
        if order > 3:
            raise ValueError("The pinned upstream scale schedule supports order <= 3")
        support = supports[0] if supports else None
        self.locals = GraphPropagate(Ks=order, gso=support)
        self.attn = nn.ModuleList(
            [FastAttentionLayer(model_dim, num_heads, mask, kernel=kernel) for _ in range(order)]
        )
        self.pws = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in range(order)])
        for layer in self.pws:
            nn.init.constant_(layer.weight, 0.0)
            nn.init.constant_(layer.bias, 0.0)
        self.fc = Mlp(
            in_features=model_dim,
            hidden_features=int(model_dim * mlp_ratio),
            act_layer=nn.ReLU,
            drop=dropout,
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.scale = [1.0, 0.01, 0.001]

    def forward(self, x, graph):
        x_loc = self.locals(x, graph)
        # Preserve the upstream recurrence while avoiding mutation of the input tensor.
        residual = x
        x_glo = x
        context = x
        for index, propagated in enumerate(x_loc):
            att_outputs = self.attn[index](propagated)
            x_glo = x_glo + att_outputs * self.pws[index](context) * self.scale[index]
            context = att_outputs
        x = self.ln1(residual + self.dropout(x_glo))
        return self.ln2(x + self.dropout(self.fc(x)))


class STGformerCore(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps,
        out_steps,
        steps_per_day,
        input_dim,
        output_dim,
        input_embedding_dim=24,
        tod_embedding_dim=12,
        dow_embedding_dim=12,
        adaptive_embedding_dim=12,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        mlp_ratio=2.0,
        dropout_a=0.3,
        kernel_size=1,
        order=2,
    ):
        super().__init__()
        if adaptive_embedding_dim <= 0:
            raise ValueError("The official adaptive graph requires adaptive_embedding_dim > 0")
        if kernel_size < 1 or kernel_size > in_steps:
            raise ValueError("kernel_size must be in [1, in_steps]")
        self.num_nodes = int(num_nodes)
        self.in_steps = int(in_steps)
        self.out_steps = int(out_steps)
        self.steps_per_day = int(steps_per_day)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_embedding_dim = int(input_embedding_dim)
        self.tod_embedding_dim = int(tod_embedding_dim)
        self.dow_embedding_dim = int(dow_embedding_dim)
        self.adaptive_embedding_dim = int(adaptive_embedding_dim)
        self.model_dim = (
            self.input_embedding_dim
            + self.tod_embedding_dim
            + self.dow_embedding_dim
            + self.adaptive_embedding_dim
        )
        if self.model_dim % int(num_heads) != 0:
            raise ValueError("Combined model_dim must be divisible by num_heads")

        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
        self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(self.in_steps, self.num_nodes, self.adaptive_embedding_dim))
        )
        self.adaptive_dropout = nn.Dropout(float(dropout_a))
        self.pooling = nn.AvgPool2d(kernel_size=(1, int(kernel_size)), stride=1)
        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim,
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                    dropout=dropout,
                    kernel=kernel_size,
                    supports=[None],
                    order=order,
                )
            ]
        )
        encoded_steps = self.in_steps - int(kernel_size) + 1
        self.encoder_proj = nn.Linear(encoded_steps * self.model_dim, self.model_dim)
        self.encoder = nn.ModuleList(
            [
                Mlp(
                    in_features=self.model_dim,
                    hidden_features=int(self.model_dim * mlp_ratio),
                    act_layer=nn.ReLU,
                    drop=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.output_proj = nn.Linear(self.model_dim, self.out_steps * self.output_dim)
        self.temporal_proj = nn.Conv2d(
            self.model_dim,
            self.model_dim,
            kernel_size=(1, int(kernel_size)),
            stride=1,
            padding=0,
        )

    def build_adaptive_graph(self):
        graph = torch.matmul(self.adaptive_embedding, self.adaptive_embedding.transpose(1, 2))
        graph = self.pooling(graph.transpose(0, 2)).transpose(0, 2)
        return F.softmax(F.relu(graph), dim=-1)

    def forward(self, x, tod, dow):
        batch_size = x.shape[0]
        projected = self.input_proj(x)
        tod_ids = torch.remainder(torch.round(tod * self.steps_per_day).long(), self.steps_per_day)
        dow_ids = torch.remainder(torch.round(dow).long(), 7)
        adaptive = self.adaptive_embedding.unsqueeze(0).expand(
            batch_size, self.in_steps, self.num_nodes, self.adaptive_embedding_dim
        )
        features = [
            projected,
            self.tod_embedding(tod_ids),
            self.dow_embedding(dow_ids),
            self.adaptive_dropout(adaptive),
        ]
        hidden = torch.cat(features, dim=-1)
        hidden = self.temporal_proj(hidden.transpose(1, 3)).transpose(1, 3)
        graph = self.build_adaptive_graph()
        for layer in self.attn_layers_s:
            hidden = layer(hidden, graph)
        hidden = self.encoder_proj(hidden.transpose(1, 2).flatten(start_dim=-2))
        for layer in self.encoder:
            hidden = hidden + layer(hidden)
        out = self.output_proj(hidden).reshape(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        return out.transpose(1, 2)


class STGformerOfficialBike(nn.Module):
    """Local normalized-feature adapter around the official STGformer core."""

    CONFIG_KEYS = (
        "num_nodes",
        "in_steps",
        "out_steps",
        "input_dim",
        "output_dim",
        "steps_per_day",
        "hour_feature_index",
        "dow_feature_index",
        "input_embedding_dim",
        "tod_embedding_dim",
        "dow_embedding_dim",
        "adaptive_embedding_dim",
        "num_heads",
        "num_layers",
        "dropout",
        "mlp_ratio",
        "dropout_a",
        "kernel_size",
        "order",
    )

    def __init__(
        self,
        num_nodes,
        in_steps,
        out_steps,
        input_dim,
        output_dim,
        feature_mean,
        feature_std,
        steps_per_day=24,
        hour_feature_index=3,
        dow_feature_index=4,
        input_embedding_dim=24,
        tod_embedding_dim=12,
        dow_embedding_dim=12,
        adaptive_embedding_dim=12,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        mlp_ratio=2.0,
        dropout_a=0.3,
        kernel_size=1,
        order=2,
    ):
        super().__init__()
        constructor_values = locals().copy()
        self.num_nodes = int(num_nodes)
        self.in_steps = int(in_steps)
        self.out_steps = int(out_steps)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.steps_per_day = int(steps_per_day)
        self.hour_feature_index = int(hour_feature_index)
        self.dow_feature_index = int(dow_feature_index)
        for index, name in [
            (self.hour_feature_index, "hour_feature_index"),
            (self.dow_feature_index, "dow_feature_index"),
        ]:
            if not 0 <= index < self.input_dim:
                raise ValueError("%s is outside input_dim" % name)

        mean = torch.as_tensor(feature_mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(feature_std, dtype=torch.float32).reshape(-1)
        if mean.numel() != self.input_dim or std.numel() != self.input_dim:
            raise ValueError("feature_mean/std must contain exactly input_dim values")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)
        self.core = STGformerCore(
            num_nodes=self.num_nodes,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            adaptive_embedding_dim=adaptive_embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            dropout_a=dropout_a,
            kernel_size=kernel_size,
            order=order,
        )
        self._config = {key: copy.deepcopy(constructor_values[key]) for key in self.CONFIG_KEYS}

    def export_config(self):
        return copy.deepcopy(self._config)

    def build_adaptive_graph(self):
        return self.core.build_adaptive_graph()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("x must have shape [B,T,N,F]")
        expected = (self.in_steps, self.num_nodes, self.input_dim)
        if tuple(x.shape[1:]) != expected:
            raise ValueError("Expected x[:, %d, %d, %d], got %s" % (*expected, tuple(x.shape)))
        hour = (
            x[..., self.hour_feature_index] * self.feature_std[self.hour_feature_index]
            + self.feature_mean[self.hour_feature_index]
        )
        dow = (
            x[..., self.dow_feature_index] * self.feature_std[self.dow_feature_index]
            + self.feature_mean[self.dow_feature_index]
        )
        tod = torch.remainder(torch.round(hour), self.steps_per_day) / float(self.steps_per_day)
        return self.core(x, tod=tod, dow=dow)


def build_model_from_checkpoint(checkpoint, device=None):
    required = {"model_config", "feature_mean", "feature_std", "model_state_dict"}
    missing_fields = sorted(required.difference(checkpoint))
    if missing_fields:
        raise KeyError("Checkpoint missing fields: %s" % missing_fields)
    config = dict(checkpoint["model_config"])
    model = STGformerOfficialBike(
        feature_mean=checkpoint["feature_mean"],
        feature_std=checkpoint["feature_std"],
        **config,
    )
    if device is not None:
        model = model.to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    audit = {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    if audit["missing_keys"] or audit["unexpected_keys"]:
        raise RuntimeError("Checkpoint/model mismatch: %s" % audit)
    return model, audit
