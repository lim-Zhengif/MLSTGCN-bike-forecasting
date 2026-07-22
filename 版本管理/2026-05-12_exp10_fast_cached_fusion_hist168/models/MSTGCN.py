# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import eigs

def cheb_polynomial_torch(L_tilde, K):
    N = L_tilde.shape[0]

    cheb_polynomials = [torch.eye(N, device=L_tilde.device, dtype=L_tilde.dtype), L_tilde.clone()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


class CategoricalFeatureEmbedding(nn.Module):
    def __init__(self, index, num_embeddings, embedding_dim, mean, std):
        super(CategoricalFeatureEmbedding, self).__init__()
        self.index = index
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.register_buffer('mean', torch.tensor(float(mean), dtype=torch.float32))
        self.register_buffer('std', torch.tensor(float(std) if float(std) != 0 else 1.0, dtype=torch.float32))
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, feature_slice):
        raw_feature = feature_slice * self.std + self.mean
        feature_ids = raw_feature.round().clamp(0, self.num_embeddings - 1).long().squeeze(-1)
        return self.embedding(feature_ids)


class FeatureEmbeddingAdapter(nn.Module):
    def __init__(self, in_channels, categorical_feature_configs=None):
        super(FeatureEmbeddingAdapter, self).__init__()
        self.in_channels = int(in_channels)
        self.categorical_feature_configs = sorted(categorical_feature_configs or [], key=lambda item: item['index'])
        self.embedding_layers = nn.ModuleList([
            CategoricalFeatureEmbedding(
                index=item['index'],
                num_embeddings=item['num_embeddings'],
                embedding_dim=item['embedding_dim'],
                mean=item['mean'],
                std=item['std'],
            )
            for item in self.categorical_feature_configs
        ])
        self.effective_in_channels = self.in_channels - len(self.embedding_layers) + sum(
            layer.embedding_dim for layer in self.embedding_layers
        )

    def forward(self, x):
        if not self.embedding_layers:
            return x

        parts = []
        start_idx = 0
        for layer in self.embedding_layers:
            if start_idx < layer.index:
                parts.append(x[..., start_idx:layer.index])
            embedded = layer(x[..., layer.index:layer.index + 1])
            parts.append(embedded)
            start_idx = layer.index + 1

        if start_idx < x.shape[-1]:
            parts.append(x[..., start_idx:])

        return torch.cat(parts, dim=-1)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(ChannelAttention, self).__init__()
        channels = int(channels)
        hidden_channels = max(channels // int(reduction), 1)
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, channels)

    def forward(self, x):
        # x: (batch_size, N, F, T). Pool over stations and time, then reweight feature channels.
        weights = x.mean(dim=(1, 3))
        weights = F.relu(self.fc1(weights))
        weights = torch.sigmoid(self.fc2(weights)).unsqueeze(1).unsqueeze(-1)
        return x * weights


class TrendAlignmentDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_for_predict,
        out_dim,
        time_feature_index=-1,
        time_feature_mean=0.0,
        time_feature_std=1.0,
        time_cycle=24,
        time_embed_dim=16,
        attention_heads=4,
        dropout=0.1,
    ):
        super(TrendAlignmentDecoder, self).__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_for_predict = int(num_for_predict)
        self.out_dim = int(out_dim)
        self.time_feature_index = int(time_feature_index)
        self.time_cycle = max(int(time_cycle), 1)
        self.time_embed_dim = int(time_embed_dim)
        attention_heads = max(int(attention_heads), 1)
        self.register_buffer('time_feature_mean', torch.tensor(float(time_feature_mean), dtype=torch.float32))
        self.register_buffer(
            'time_feature_std',
            torch.tensor(float(time_feature_std) if float(time_feature_std) != 0 else 1.0, dtype=torch.float32),
        )

        self.time_embedding = nn.Embedding(self.time_cycle, self.time_embed_dim)
        attention_dim = self.hidden_dim + self.time_embed_dim
        while attention_dim % attention_heads != 0 and attention_heads > 1:
            attention_heads -= 1
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            dropout=float(dropout),
        )
        self.residual_proj = nn.Linear(self.hidden_dim, attention_dim)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        nn.init.xavier_uniform_(self.time_embedding.weight)

    def _history_time_ids(self, context_x, hist_len):
        # context_x is the raw scaled model input in (B, N, F, T).
        if self.time_feature_index >= 0 and self.time_feature_index < context_x.shape[2]:
            raw_time = (
                context_x[:, :, self.time_feature_index, :] * self.time_feature_std
                + self.time_feature_mean
            )
            return torch.remainder(raw_time.round().long(), self.time_cycle)
        positions = torch.arange(hist_len, device=context_x.device, dtype=torch.long)
        return positions.view(1, 1, hist_len).expand(context_x.shape[0], context_x.shape[1], hist_len)

    def forward(self, hidden_sequence, context_x):
        # hidden_sequence: (B, N, D, T), context_x: (B, N, F, T)
        batch_size, num_nodes, hidden_dim, hist_len = hidden_sequence.shape
        history_hidden = hidden_sequence.permute(0, 1, 3, 2)
        final_hidden = history_hidden[:, :, -1, :]

        history_time_ids = self._history_time_ids(context_x, hist_len)
        future_offsets = torch.arange(
            1,
            self.num_for_predict + 1,
            device=hidden_sequence.device,
            dtype=torch.long,
        ).view(1, 1, self.num_for_predict)
        future_time_ids = torch.remainder(history_time_ids[:, :, -1:].long() + future_offsets, self.time_cycle)

        history_time = self.time_embedding(history_time_ids)
        future_time = self.time_embedding(future_time_ids)
        keys = torch.cat([history_hidden, history_time], dim=-1)
        queries = torch.cat(
            [
                final_hidden.unsqueeze(2).expand(-1, -1, self.num_for_predict, -1),
                future_time,
            ],
            dim=-1,
        )

        flat_queries = queries.reshape(batch_size * num_nodes, self.num_for_predict, -1)
        flat_keys = keys.reshape(batch_size * num_nodes, hist_len, -1)
        aligned, _ = self.attention(
            flat_queries.transpose(0, 1),
            flat_keys.transpose(0, 1),
            flat_keys.transpose(0, 1),
            need_weights=False,
        )
        aligned = aligned.transpose(0, 1)
        residual = self.residual_proj(final_hidden).reshape(batch_size * num_nodes, 1, -1)
        output = self.output_proj(aligned + residual)
        output = output.view(batch_size, num_nodes, self.num_for_predict, self.out_dim)
        return output.permute(0, 2, 1, 3)


class CausalGatedTemporalBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, dilation=1, dropout=0.1):
        super(CausalGatedTemporalBlock, self).__init__()
        self.padding = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim * 2,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        gated = self.conv(F.pad(x, (self.padding, 0)))
        tanh_part, sigmoid_part = torch.chunk(gated, chunks=2, dim=1)
        out = torch.tanh(tanh_part) * torch.sigmoid(sigmoid_part)
        out = self.dropout(out)
        return self.norm(out + residual)


class GraphMaskedSpatialAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        heads=4,
        dropout=0.1,
        edge_bias=False,
        edge_bias_init=0.1,
        extra_edge_bias=False,
        extra_edge_bias_init=0.0,
        edge_bias_eps=1e-6,
    ):
        super(GraphMaskedSpatialAttention, self).__init__()
        hidden_dim = int(hidden_dim)
        heads = max(int(heads), 1)
        while hidden_dim % heads != 0 and heads > 1:
            heads -= 1
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.edge_bias = bool(edge_bias)
        self.edge_bias_eps = float(edge_bias_eps)
        if self.edge_bias:
            if self.edge_bias_eps <= 0:
                raise ValueError("edge_bias_eps must be > 0 when edge_bias is enabled.")
            self.edge_bias_scale = nn.Parameter(torch.tensor(float(edge_bias_init), dtype=torch.float32))
        self.extra_edge_bias = bool(extra_edge_bias)
        if self.extra_edge_bias:
            if self.edge_bias_eps <= 0:
                raise ValueError("edge_bias_eps must be > 0 when extra_edge_bias is enabled.")
            self.extra_edge_bias_scale = nn.Parameter(torch.tensor(float(extra_edge_bias_init), dtype=torch.float32))

    def _prepare_graph_mask(self, adj_for_run, batch_size, num_nodes, device):
        graph_mask = adj_for_run.detach() > 0
        eye_mask = torch.eye(num_nodes, device=device, dtype=torch.bool)
        if graph_mask.dim() == 2:
            graph_mask = graph_mask | eye_mask
            return graph_mask.view(1, 1, num_nodes, num_nodes)
        if graph_mask.dim() == 3:
            graph_mask = graph_mask | eye_mask.view(1, num_nodes, num_nodes)
            if graph_mask.shape[0] == 1 and batch_size > 1:
                graph_mask = graph_mask.expand(batch_size, -1, -1)
            if graph_mask.shape[0] != batch_size:
                raise ValueError("Batch graph mask size does not match node_state batch size.")
            return graph_mask.view(batch_size, 1, num_nodes, num_nodes)
        raise ValueError("adj_for_run must be a 2D or 3D adjacency tensor.")

    def _prepare_edge_bias(self, adj_for_run, batch_size, num_nodes, device, dtype):
        adj_weight = adj_for_run.detach().to(device=device, dtype=dtype)
        eye_weight = torch.eye(num_nodes, device=device, dtype=dtype)
        if adj_weight.dim() == 2:
            adj_weight = torch.maximum(adj_weight, eye_weight)
            row_sum = adj_weight.sum(dim=-1, keepdim=True).clamp_min(self.edge_bias_eps)
            adj_norm = (adj_weight / row_sum).clamp_min(self.edge_bias_eps)
            return torch.log(adj_norm).view(1, 1, num_nodes, num_nodes)
        if adj_weight.dim() == 3:
            adj_weight = torch.maximum(adj_weight, eye_weight.view(1, num_nodes, num_nodes))
            if adj_weight.shape[0] == 1 and batch_size > 1:
                adj_weight = adj_weight.expand(batch_size, -1, -1)
            if adj_weight.shape[0] != batch_size:
                raise ValueError("Batch edge-bias graph size does not match node_state batch size.")
            row_sum = adj_weight.sum(dim=-1, keepdim=True).clamp_min(self.edge_bias_eps)
            adj_norm = (adj_weight / row_sum).clamp_min(self.edge_bias_eps)
            return torch.log(adj_norm).view(batch_size, 1, num_nodes, num_nodes)
        raise ValueError("adj_for_run must be a 2D or 3D adjacency tensor.")

    def forward(self, node_state, adj_for_run, extra_edge_bias_adj=None):
        batch_size, num_nodes, hidden_dim = node_state.shape
        q = self.q_proj(node_state).view(batch_size, num_nodes, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(node_state).view(batch_size, num_nodes, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(node_state).view(batch_size, num_nodes, self.heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if adj_for_run is not None:
            graph_mask = self._prepare_graph_mask(adj_for_run, batch_size, num_nodes, node_state.device)
            if self.edge_bias:
                edge_bias = self._prepare_edge_bias(
                    adj_for_run,
                    batch_size,
                    num_nodes,
                    node_state.device,
                    scores.dtype,
                )
                scores = scores + self.edge_bias_scale.to(dtype=scores.dtype) * edge_bias
            if self.extra_edge_bias and extra_edge_bias_adj is not None:
                extra_edge_bias = self._prepare_edge_bias(
                    extra_edge_bias_adj,
                    batch_size,
                    num_nodes,
                    node_state.device,
                    scores.dtype,
                )
                scores = scores + self.extra_edge_bias_scale.to(dtype=scores.dtype) * extra_edge_bias
            scores = scores.masked_fill(~graph_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, hidden_dim)
        return self.out_proj(out)


class ASTTCNResidualBranch(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        num_for_predict,
        out_dim,
        layers=4,
        kernel_size=3,
        dilation_base=2,
        heads=4,
        dropout=0.1,
        residual_init=0.05,
        bounded_alpha=False,
        alpha_max=0.1,
        horizon_alpha=False,
        residual_horizon_mask=None,
        zero_init=False,
        residual_gate=False,
        residual_gate_hidden_dim=16,
        residual_gate_init=0.2,
        anchor_horizon_gate=False,
        anchor_embed_dim=8,
        horizon_embed_dim=4,
        anchor_horizon_gate_hidden_dim=16,
        anchor_horizon_gate_init=0.2,
        edge_bias=False,
        edge_bias_init=0.1,
        hgaurban_edge_bias=False,
        hgaurban_edge_bias_init=0.0,
        edge_bias_eps=1e-6,
    ):
        super(ASTTCNResidualBranch, self).__init__()
        self.num_for_predict = int(num_for_predict)
        self.out_dim = int(out_dim)
        self.bounded_alpha = bool(bounded_alpha)
        self.alpha_max = float(alpha_max)
        self.horizon_alpha = bool(horizon_alpha)
        self.use_residual_gate = bool(residual_gate)
        self.use_anchor_horizon_gate = bool(anchor_horizon_gate)
        if (
            residual_horizon_mask is None
            or residual_horizon_mask == ""
            or (
                not isinstance(residual_horizon_mask, str)
                and hasattr(residual_horizon_mask, "__len__")
                and len(residual_horizon_mask) == 0
            )
        ):
            mask = torch.ones(self.num_for_predict, dtype=torch.float32)
        elif isinstance(residual_horizon_mask, str):
            mask = torch.tensor(
                [float(item.strip()) for item in residual_horizon_mask.split(",") if item.strip()],
                dtype=torch.float32,
            )
        else:
            mask = torch.tensor(list(residual_horizon_mask), dtype=torch.float32)
        if mask.numel() != self.num_for_predict:
            raise ValueError(
                "residual_horizon_mask length must match num_for_predict: %d vs %d"
                % (mask.numel(), self.num_for_predict)
            )
        self.register_buffer("residual_horizon_mask", mask.view(1, self.num_for_predict, 1, 1))
        self.input_proj = nn.Linear(int(in_channels), int(hidden_dim))
        self.temporal_blocks = nn.ModuleList([
            CausalGatedTemporalBlock(
                hidden_dim=int(hidden_dim),
                kernel_size=int(kernel_size),
                dilation=int(dilation_base) ** layer_idx,
                dropout=float(dropout),
            )
            for layer_idx in range(int(layers))
        ])
        self.spatial_attention = GraphMaskedSpatialAttention(
            hidden_dim=int(hidden_dim),
            heads=int(heads),
            dropout=float(dropout),
            edge_bias=edge_bias,
            edge_bias_init=edge_bias_init,
            extra_edge_bias=hgaurban_edge_bias,
            extra_edge_bias_init=hgaurban_edge_bias_init,
            edge_bias_eps=edge_bias_eps,
        )
        self.stim_gate = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_for_predict * self.out_dim),
        )
        if self.use_residual_gate:
            gate_hidden_dim = int(residual_gate_hidden_dim)
            if gate_hidden_dim <= 0:
                gate_hidden_dim = max(int(hidden_dim) // 2, 1)
            self.residual_gate = nn.Sequential(
                nn.LayerNorm(int(hidden_dim) * 2),
                nn.Linear(int(hidden_dim) * 2, gate_hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(gate_hidden_dim, 1),
                nn.Sigmoid(),
            )
            gate_init = min(max(float(residual_gate_init), 1e-4), 1.0 - 1e-4)
            nn.init.zeros_(self.residual_gate[-2].weight)
            nn.init.constant_(self.residual_gate[-2].bias, math.log(gate_init / (1.0 - gate_init)))
        if self.use_anchor_horizon_gate:
            anchor_embed_dim = int(anchor_embed_dim)
            horizon_embed_dim = int(horizon_embed_dim)
            gate_hidden_dim = int(anchor_horizon_gate_hidden_dim)
            if anchor_embed_dim <= 0:
                anchor_embed_dim = 8
            if horizon_embed_dim <= 0:
                horizon_embed_dim = 4
            if gate_hidden_dim <= 0:
                gate_hidden_dim = max(int(hidden_dim) // 2, 1)
            self.anchor_embedding = nn.Embedding(24, anchor_embed_dim)
            self.horizon_embedding = nn.Embedding(self.num_for_predict, horizon_embed_dim)
            gate_input_dim = int(hidden_dim) + anchor_embed_dim + horizon_embed_dim
            self.anchor_horizon_gate = nn.Sequential(
                nn.LayerNorm(gate_input_dim),
                nn.Linear(gate_input_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(gate_hidden_dim, 1),
                nn.Sigmoid(),
            )
            gate_init = min(max(float(anchor_horizon_gate_init), 1e-4), 1.0 - 1e-4)
            nn.init.zeros_(self.anchor_horizon_gate[-2].weight)
            nn.init.constant_(self.anchor_horizon_gate[-2].bias, math.log(gate_init / (1.0 - gate_init)))
        if bool(zero_init):
            nn.init.zeros_(self.output_proj[-1].weight)
            nn.init.zeros_(self.output_proj[-1].bias)
        if self.bounded_alpha:
            if self.alpha_max <= 0:
                raise ValueError("alpha_max must be > 0 when bounded_alpha is enabled.")
            init_ratio = float(residual_init) / self.alpha_max
            init_ratio = min(max(init_ratio, 1e-4), 1.0 - 1e-4)
            raw_init = math.log(init_ratio / (1.0 - init_ratio))
            if self.horizon_alpha:
                init_value = torch.full((self.num_for_predict,), raw_init, dtype=torch.float32)
            else:
                init_value = torch.tensor(raw_init, dtype=torch.float32)
            self.residual_alpha = nn.Parameter(init_value)
        else:
            if self.horizon_alpha:
                init_value = torch.full((self.num_for_predict,), float(residual_init), dtype=torch.float32)
            else:
                init_value = torch.tensor(float(residual_init), dtype=torch.float32)
            self.residual_alpha = nn.Parameter(init_value)

    def residual_scale(self):
        if self.bounded_alpha:
            return self.alpha_max * torch.sigmoid(self.residual_alpha)
        return self.residual_alpha

    def residual_scale_for_output(self):
        scale = self.residual_scale()
        if self.horizon_alpha:
            return scale.view(1, self.num_for_predict, 1, 1)
        return scale

    def forward(self, x, adj_for_run=None, anchor_hours=None, hgaurban_adj=None):
        # x: (B, N, F, T). This branch predicts a small residual in scaled target space.
        batch_size, num_nodes, _, hist_len = x.shape
        hidden = self.input_proj(x.permute(0, 1, 3, 2))  # (B, N, T, H)
        hidden = hidden.reshape(batch_size * num_nodes, hist_len, -1).transpose(1, 2)
        for block in self.temporal_blocks:
            hidden = block(hidden)
        temporal_state = hidden[:, :, -1].view(batch_size, num_nodes, -1)

        spatial_state = self.spatial_attention(
            temporal_state,
            adj_for_run,
            extra_edge_bias_adj=hgaurban_adj,
        )
        gate = self.stim_gate(torch.cat([spatial_state, temporal_state], dim=-1))
        fused = gate * spatial_state + (1.0 - gate) * temporal_state
        residual = self.output_proj(fused).view(batch_size, num_nodes, self.num_for_predict, self.out_dim)
        residual = residual.permute(0, 2, 1, 3)
        if self.use_residual_gate:
            residual_gate = self.residual_gate(torch.cat([spatial_state, temporal_state], dim=-1))
            residual = residual * residual_gate.permute(0, 2, 1).unsqueeze(-1)
        if self.use_anchor_horizon_gate:
            if anchor_hours is None:
                anchor_ids = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            else:
                anchor_ids = anchor_hours.to(device=x.device, dtype=torch.long).view(-1).clamp(min=0, max=23)
                if anchor_ids.numel() != batch_size:
                    anchor_ids = anchor_ids[:batch_size]
                    if anchor_ids.numel() < batch_size:
                        pad = torch.zeros(batch_size - anchor_ids.numel(), dtype=torch.long, device=x.device)
                        anchor_ids = torch.cat([anchor_ids, pad], dim=0)
            pooled_state = fused.mean(dim=1)
            anchor_emb = self.anchor_embedding(anchor_ids)
            horizon_ids = torch.arange(self.num_for_predict, device=x.device, dtype=torch.long)
            horizon_emb = self.horizon_embedding(horizon_ids)
            gate_inputs = torch.cat(
                [
                    pooled_state.unsqueeze(1).expand(-1, self.num_for_predict, -1),
                    anchor_emb.unsqueeze(1).expand(-1, self.num_for_predict, -1),
                    horizon_emb.unsqueeze(0).expand(batch_size, -1, -1),
                ],
                dim=-1,
            )
            anchor_horizon_gate = self.anchor_horizon_gate(gate_inputs).view(
                batch_size,
                self.num_for_predict,
                1,
                1,
            )
            residual = residual * anchor_horizon_gate
        residual = residual * self.residual_horizon_mask.to(dtype=residual.dtype)
        return self.residual_scale_for_output() * residual


class STHybridMultiScaleResidualBranch(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        num_for_predict,
        out_dim,
        dilations=(1, 2, 4, 8),
        kernel_size=3,
        heads=4,
        dropout=0.1,
        residual_init=0.05,
        bounded_alpha=False,
        alpha_max=0.1,
        zero_init=False,
        edge_bias=False,
        edge_bias_init=0.1,
        edge_bias_eps=1e-6,
    ):
        super(STHybridMultiScaleResidualBranch, self).__init__()
        self.num_for_predict = int(num_for_predict)
        self.out_dim = int(out_dim)
        self.bounded_alpha = bool(bounded_alpha)
        self.alpha_max = float(alpha_max)
        dilation_values = [int(item) for item in dilations]
        if not dilation_values:
            raise ValueError("ST-Hybrid multi-scale residual requires at least one dilation.")
        self.input_proj = nn.Linear(int(in_channels), int(hidden_dim))
        self.scale_blocks = nn.ModuleList([
            CausalGatedTemporalBlock(
                hidden_dim=int(hidden_dim),
                kernel_size=int(kernel_size),
                dilation=dilation,
                dropout=float(dropout),
            )
            for dilation in dilation_values
        ])
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) * len(dilation_values)),
            nn.Linear(int(hidden_dim) * len(dilation_values), len(dilation_values)),
        )
        self.spatial_attention = GraphMaskedSpatialAttention(
            hidden_dim=int(hidden_dim),
            heads=int(heads),
            dropout=float(dropout),
            edge_bias=edge_bias,
            edge_bias_init=edge_bias_init,
            edge_bias_eps=edge_bias_eps,
        )
        self.st_fusion_gate = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_for_predict * self.out_dim),
        )
        if bool(zero_init):
            nn.init.zeros_(self.output_proj[-1].weight)
            nn.init.zeros_(self.output_proj[-1].bias)
        if self.bounded_alpha:
            if self.alpha_max <= 0:
                raise ValueError("alpha_max must be > 0 when bounded_alpha is enabled.")
            init_ratio = float(residual_init) / self.alpha_max
            init_ratio = min(max(init_ratio, 1e-4), 1.0 - 1e-4)
            self.residual_alpha = nn.Parameter(torch.tensor(math.log(init_ratio / (1.0 - init_ratio)), dtype=torch.float32))
        else:
            self.residual_alpha = nn.Parameter(torch.tensor(float(residual_init), dtype=torch.float32))

    def residual_scale(self):
        if self.bounded_alpha:
            return self.alpha_max * torch.sigmoid(self.residual_alpha)
        return self.residual_alpha

    def forward(self, x, adj_for_run=None):
        # x: (B, N, F, T). The branch returns a small residual in scaled target space.
        batch_size, num_nodes, _, hist_len = x.shape
        hidden = self.input_proj(x.permute(0, 1, 3, 2))
        hidden = hidden.reshape(batch_size * num_nodes, hist_len, -1).transpose(1, 2)

        scale_states = []
        for block in self.scale_blocks:
            scale_states.append(block(hidden)[:, :, -1].view(batch_size, num_nodes, -1))
        stacked = torch.stack(scale_states, dim=2)
        gate_input = torch.cat(scale_states, dim=-1)
        scale_weights = torch.softmax(self.scale_gate(gate_input), dim=-1)
        temporal_state = torch.sum(stacked * scale_weights.unsqueeze(-1), dim=2)

        spatial_state = self.spatial_attention(temporal_state, adj_for_run)
        gate = self.st_fusion_gate(torch.cat([spatial_state, temporal_state], dim=-1))
        fused = gate * spatial_state + (1.0 - gate) * temporal_state
        residual = self.output_proj(fused).view(batch_size, num_nodes, self.num_for_predict, self.out_dim)
        residual = residual.permute(0, 2, 1, 3)
        return self.residual_scale() * residual


class LocalGlobalTemporalResidualBranch(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        num_for_predict,
        out_dim,
        heads=4,
        local_kernel_size=5,
        landmark_count=24,
        dropout=0.1,
        residual_init=0.01,
        bounded_alpha=True,
        alpha_max=0.1,
        zero_init=True,
        gate_init=0.2,
        spatial_transformer=False,
        spatial_heads=None,
        spatial_edge_bias=True,
        spatial_edge_bias_init=0.05,
        spatial_edge_bias_eps=1e-6,
    ):
        super(LocalGlobalTemporalResidualBranch, self).__init__()
        self.num_for_predict = int(num_for_predict)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.landmark_count = max(int(landmark_count), 1)
        self.bounded_alpha = bool(bounded_alpha)
        self.alpha_max = float(alpha_max)
        self.use_spatial_transformer = bool(spatial_transformer)

        heads = max(int(heads), 1)
        while self.hidden_dim % heads != 0 and heads > 1:
            heads -= 1
        local_kernel_size = int(local_kernel_size)
        if local_kernel_size <= 0:
            raise ValueError("local_kernel_size must be > 0.")
        self.input_proj = nn.Linear(int(in_channels), self.hidden_dim)
        self.local_conv = nn.Conv1d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=local_kernel_size,
            padding=local_kernel_size // 2,
            groups=self.hidden_dim,
        )
        self.global_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=heads,
            dropout=float(dropout),
        )
        self.norm_local = nn.LayerNorm(self.hidden_dim)
        self.norm_global = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.norm_ffn = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        if self.use_spatial_transformer:
            spatial_heads = heads if spatial_heads is None else spatial_heads
            self.spatial_attention = GraphMaskedSpatialAttention(
                hidden_dim=self.hidden_dim,
                heads=spatial_heads,
                dropout=dropout,
                edge_bias=spatial_edge_bias,
                edge_bias_init=spatial_edge_bias_init,
                edge_bias_eps=spatial_edge_bias_eps,
            )
            self.norm_spatial = nn.LayerNorm(self.hidden_dim)
            self.spatial_fusion_gate = nn.Sequential(
                nn.LayerNorm(self.hidden_dim * 2),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Sigmoid(),
            )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_for_predict * self.out_dim),
        )
        self.gate_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.num_for_predict * self.out_dim),
            nn.Sigmoid(),
        )
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.gate_proj[1].weight)
        nn.init.constant_(self.gate_proj[1].bias, math.log(gate_init / (1.0 - gate_init)))
        if bool(zero_init):
            nn.init.zeros_(self.output_proj[-1].weight)
            nn.init.zeros_(self.output_proj[-1].bias)
        if self.bounded_alpha:
            if self.alpha_max <= 0:
                raise ValueError("alpha_max must be > 0 when bounded_alpha is enabled.")
            init_ratio = float(residual_init) / self.alpha_max
            init_ratio = min(max(init_ratio, 1e-4), 1.0 - 1e-4)
            self.residual_alpha = nn.Parameter(torch.tensor(math.log(init_ratio / (1.0 - init_ratio)), dtype=torch.float32))
        else:
            self.residual_alpha = nn.Parameter(torch.tensor(float(residual_init), dtype=torch.float32))

    def residual_scale(self):
        if self.bounded_alpha:
            return self.alpha_max * torch.sigmoid(self.residual_alpha)
        return self.residual_alpha

    def forward(self, x, adj_for_run=None):
        # x: (B, N, F, T). Optionally applies graph-aware spatial attention after local-global temporal encoding.
        batch_size, num_nodes, _, hist_len = x.shape
        hidden = self.input_proj(x.permute(0, 1, 3, 2))
        hidden = hidden.reshape(batch_size * num_nodes, hist_len, self.hidden_dim)

        local = self.local_conv(hidden.transpose(1, 2)).transpose(1, 2)
        if local.shape[1] > hist_len:
            local = local[:, :hist_len]
        elif local.shape[1] < hist_len:
            local = F.pad(local, (0, 0, 0, hist_len - local.shape[1]))
        hidden = self.norm_local(hidden + self.dropout(local))
        if hist_len > self.landmark_count:
            pooled = F.adaptive_avg_pool1d(hidden.transpose(1, 2), self.landmark_count).transpose(1, 2)
        else:
            pooled = hidden
        global_out, _ = self.global_attention(
            hidden.transpose(0, 1),
            pooled.transpose(0, 1),
            pooled.transpose(0, 1),
            need_weights=False,
        )
        global_out = global_out.transpose(0, 1)
        hidden = self.norm_global(hidden + self.dropout(global_out))
        hidden = self.norm_ffn(hidden + self.dropout(self.ffn(hidden)))

        state = hidden[:, -1].view(batch_size, num_nodes, self.hidden_dim)
        if self.use_spatial_transformer and adj_for_run is not None:
            spatial_state = self.spatial_attention(state, adj_for_run)
            spatial_state = self.norm_spatial(state + self.dropout(spatial_state))
            fusion_gate = self.spatial_fusion_gate(torch.cat([state, spatial_state], dim=-1))
            state = fusion_gate * spatial_state + (1.0 - fusion_gate) * state
        residual = self.output_proj(state).view(batch_size, num_nodes, self.num_for_predict, self.out_dim)
        residual = residual.permute(0, 2, 1, 3)
        gate = self.gate_proj(state).view(batch_size, num_nodes, self.num_for_predict, self.out_dim)
        gate = gate.permute(0, 2, 1, 3)
        return self.residual_scale() * gate * residual


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(
        self,
        K,
        fusiongraph,
        in_channels,
        out_channels,
        device,
        adaptive_support_gate=False,
        support_gate_hidden_dim=32,
        support_gate_temperature=1.0,
        dynamic_graph=False,
        dynamic_graph_hidden_dim=32,
        dynamic_graph_topk=20,
        dynamic_graph_init_lambda=0.8,
        dynamic_graph_residual=False,
        dynamic_graph_residual_init=0.01,
    ):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.fusiongraph = fusiongraph
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = device
        self.use_adaptive_support_gate = bool(adaptive_support_gate)
        self.support_gate_temperature = float(support_gate_temperature)
        self.use_dynamic_graph = bool(dynamic_graph)
        self.use_dynamic_graph_residual = bool(dynamic_graph_residual)
        self.dynamic_graph_topk = int(dynamic_graph_topk)
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
        if self.use_adaptive_support_gate:
            support_gate_hidden_dim = max(int(support_gate_hidden_dim), 1)
            self.support_gate = nn.Sequential(
                nn.LayerNorm(int(in_channels)),
                nn.Linear(int(in_channels), support_gate_hidden_dim),
                nn.GELU(),
                nn.Linear(support_gate_hidden_dim, int(K)),
            )
            nn.init.zeros_(self.support_gate[-1].weight)
            nn.init.zeros_(self.support_gate[-1].bias)
        else:
            self.support_gate = None
        if self.use_dynamic_graph or self.use_dynamic_graph_residual:
            dynamic_graph_hidden_dim = max(int(dynamic_graph_hidden_dim), 1)
            self.dynamic_node_norm = nn.LayerNorm(int(in_channels))
            self.dynamic_query = nn.Linear(int(in_channels), dynamic_graph_hidden_dim)
            self.dynamic_key = nn.Linear(int(in_channels), dynamic_graph_hidden_dim)
            if self.use_dynamic_graph:
                self.dynamic_lambda_gate = nn.Sequential(
                    nn.LayerNorm(int(in_channels)),
                    nn.Linear(int(in_channels), dynamic_graph_hidden_dim),
                    nn.GELU(),
                    nn.Linear(dynamic_graph_hidden_dim, 1),
                )
                init_lambda = min(max(float(dynamic_graph_init_lambda), 1e-4), 1.0 - 1e-4)
                nn.init.zeros_(self.dynamic_lambda_gate[-1].weight)
                nn.init.constant_(self.dynamic_lambda_gate[-1].bias, math.log(init_lambda / (1.0 - init_lambda)))
            else:
                self.dynamic_lambda_gate = None
            if self.use_dynamic_graph_residual:
                init_beta = max(float(dynamic_graph_residual_init), 0.0)
                self.dynamic_residual_beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32))
            else:
                self.dynamic_residual_beta = None
        else:
            self.dynamic_node_norm = None
            self.dynamic_query = None
            self.dynamic_key = None
            self.dynamic_lambda_gate = None
            self.dynamic_residual_beta = None

    def build_cheb_polynomials(self, adj_for_run):
        degree = adj_for_run.sum(dim=1)
        L_tilde = torch.diag(degree) - adj_for_run
        return cheb_polynomial_torch(L_tilde, self.K)

    def _prepare_static_adj(self, adj_for_run, batch_size, device, dtype):
        if adj_for_run is None:
            adj_for_run = self.fusiongraph()
        static_adj = adj_for_run.to(device=device, dtype=dtype)
        if static_adj.dim() == 2:
            static_adj = static_adj.unsqueeze(0).expand(batch_size, -1, -1)
        elif static_adj.dim() == 3:
            if static_adj.size(0) != batch_size:
                if static_adj.size(0) == 1:
                    static_adj = static_adj.expand(batch_size, -1, -1)
                else:
                    static_adj = static_adj[:batch_size]
        else:
            raise ValueError("adj_for_run must be a 2D or 3D adjacency tensor.")
        static_adj = torch.clamp(static_adj, min=0.0)
        return static_adj / static_adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    def _build_dynamic_adj(self, x):
        node_context = x.mean(dim=3)
        node_context = self.dynamic_node_norm(node_context)
        query = self.dynamic_query(node_context)
        key = self.dynamic_key(node_context)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(max(query.size(-1), 1))
        dynamic_adj = torch.softmax(scores, dim=-1)
        if 0 < self.dynamic_graph_topk < dynamic_adj.size(-1):
            topk_values, topk_indices = torch.topk(dynamic_adj, k=self.dynamic_graph_topk, dim=-1)
            sparse_adj = torch.zeros_like(dynamic_adj)
            dynamic_adj = sparse_adj.scatter(-1, topk_indices, topk_values)
            dynamic_adj = dynamic_adj / dynamic_adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return dynamic_adj

    def build_dynamic_cheb_polynomials(self, x, adj_for_run=None):
        batch_size, num_nodes = x.size(0), x.size(1)
        static_adj = self._prepare_static_adj(adj_for_run, batch_size, x.device, x.dtype)
        dynamic_adj = self._build_dynamic_adj(x)
        context_summary = x.mean(dim=(1, 3))
        static_lambda = torch.sigmoid(self.dynamic_lambda_gate(context_summary)).view(batch_size, 1, 1)
        mixed_adj = static_lambda * static_adj + (1.0 - static_lambda) * dynamic_adj
        degree = mixed_adj.sum(dim=-1)
        eye = torch.eye(num_nodes, device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        laplacian = torch.diag_embed(degree) - mixed_adj
        supports = [eye, laplacian]
        for order in range(2, self.K):
            supports.append(2 * torch.matmul(laplacian, supports[order - 1]) - supports[order - 2])
        return supports[: self.K]

    def build_dynamic_residual_cheb_polynomials(self, x):
        batch_size, num_nodes = x.size(0), x.size(1)
        dynamic_adj = self._build_dynamic_adj(x)
        degree = dynamic_adj.sum(dim=-1)
        eye = torch.eye(num_nodes, device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        laplacian = torch.diag_embed(degree) - dynamic_adj
        supports = [eye, laplacian]
        for order in range(2, self.K):
            supports.append(2 * torch.matmul(laplacian, supports[order - 1]) - supports[order - 2])
        return supports[: self.K]

    def forward(self, x, cheb_polynomials=None, adj_for_run=None):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        if self.use_dynamic_graph:
            cheb_polynomials = self.build_dynamic_cheb_polynomials(x, adj_for_run=adj_for_run)
        elif cheb_polynomials is None:
            cheb_polynomials = self.build_cheb_polynomials(self.fusiongraph())

        theta = torch.stack(list(self.Theta), dim=0).to(device=x.device, dtype=x.dtype)
        supports = torch.stack(
            [poly.to(device=x.device, dtype=x.dtype) for poly in cheb_polynomials[: self.K]],
            dim=1 if self.use_dynamic_graph else 0,
        )
        # Original implementation looped over every history step. This einsum
        # keeps the same right-multiplied graph convention while processing the
        # whole temporal axis in one batched operation.
        if self.use_adaptive_support_gate:
            context_summary = x.mean(dim=(1, 3))
            support_logits = self.support_gate(context_summary)
            support_weights = torch.softmax(support_logits / max(self.support_gate_temperature, 1e-6), dim=-1)
            if self.use_dynamic_graph:
                output = torch.einsum('bmct,bkmn,kco,bk->bnot', x, supports, theta, support_weights)
            else:
                output = torch.einsum('bmct,kmn,kco,bk->bnot', x, supports, theta, support_weights)
        else:
            if self.use_dynamic_graph:
                output = torch.einsum('bmct,bkmn,kco->bnot', x, supports, theta)
            else:
                output = torch.einsum('bmct,kmn,kco->bnot', x, supports, theta)
        if self.use_dynamic_graph_residual:
            dynamic_supports = torch.stack(
                [
                    poly.to(device=x.device, dtype=x.dtype)
                    for poly in self.build_dynamic_residual_cheb_polynomials(x)
                ],
                dim=1,
            )
            dynamic_output = torch.einsum('bmct,bkmn,kco->bnot', x, dynamic_supports, theta)
            beta = torch.clamp(self.dynamic_residual_beta.to(device=x.device, dtype=x.dtype), min=0.0)
            output = output + beta * dynamic_output
        return F.relu(output)


class CausalMultiScaleTemporalMixer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_strides,
        kernel_size=3,
        dilations=(1, 2, 4),
        dropout=0.1,
        gate_hidden_dim=32,
    ):
        super(CausalMultiScaleTemporalMixer, self).__init__()
        self.kernel_size = int(kernel_size)
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be > 0.")
        self.dilations = tuple(int(item) for item in dilations if int(item) > 0)
        if not self.dilations:
            raise ValueError("dilations must include at least one positive value.")
        self.branches = nn.ModuleList([
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=(1, self.kernel_size),
                stride=(1, int(time_strides)),
                padding=0,
                dilation=(1, int(dilation)),
            )
            for dilation in self.dilations
        ])
        gate_hidden_dim = max(int(gate_hidden_dim), 1)
        self.branch_gate = nn.Sequential(
            nn.LayerNorm(int(in_channels)),
            nn.Linear(int(in_channels), gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, len(self.dilations)),
        )
        nn.init.zeros_(self.branch_gate[-1].weight)
        nn.init.zeros_(self.branch_gate[-1].bias)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x, context_summary=None):
        if context_summary is None:
            context_summary = x.mean(dim=(2, 3))
        branch_weights = torch.softmax(self.branch_gate(context_summary), dim=-1)
        branch_outputs = []
        for dilation, branch in zip(self.dilations, self.branches):
            left_padding = int(dilation) * (self.kernel_size - 1)
            branch_input = F.pad(x, (left_padding, 0, 0, 0))
            branch_outputs.append(branch(branch_input))
        stacked = torch.stack(branch_outputs, dim=1)
        fused = torch.sum(stacked * branch_weights[:, :, None, None, None], dim=1)
        return self.dropout(fused)


class HorizonAwarePredictionHead(nn.Module):
    def __init__(
        self,
        input_time_steps,
        hidden_dim,
        num_for_predict,
        out_dim=1,
        head_hidden_dim=64,
        horizon_embed_dim=8,
        dropout=0.1,
    ):
        super(HorizonAwarePredictionHead, self).__init__()
        self.num_for_predict = int(num_for_predict)
        self.out_dim = int(out_dim)
        head_hidden_dim = max(int(head_hidden_dim), 1)
        horizon_embed_dim = max(int(horizon_embed_dim), 1)
        self.shared_state_proj = nn.Conv2d(
            int(input_time_steps),
            head_hidden_dim,
            kernel_size=(1, int(hidden_dim)),
        )
        self.horizon_embedding = nn.Embedding(self.num_for_predict, horizon_embed_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(head_hidden_dim + horizon_embed_dim),
            nn.Linear(head_hidden_dim + horizon_embed_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(head_hidden_dim, self.out_dim),
        )

    def forward(self, x):
        # x: (B, N, F, T). The shared projection keeps the temporal summary
        # common, while horizon embeddings specialize the final mapping.
        decoder_x = x.permute(0, 3, 1, 2)
        state = self.shared_state_proj(decoder_x).squeeze(-1).permute(0, 2, 1)
        batch_size, num_nodes, _ = state.shape
        horizon_ids = torch.arange(self.num_for_predict, device=x.device, dtype=torch.long)
        horizon_emb = self.horizon_embedding(horizon_ids)
        state = state.unsqueeze(1).expand(-1, self.num_for_predict, -1, -1)
        horizon_emb = horizon_emb.view(1, self.num_for_predict, 1, -1).expand(
            batch_size,
            -1,
            num_nodes,
            -1,
        )
        output = self.decoder(torch.cat([state, horizon_emb], dim=-1))
        return output


class HorizonAwareResidualCorrectionHead(nn.Module):
    """Predict a small horizon-aware residual delta on top of a base forecast."""

    def __init__(
        self,
        hidden_dim,
        num_for_predict,
        out_dim=1,
        head_hidden_dim=64,
        horizon_embed_dim=8,
        anchor_embed_dim=8,
        dropout=0.1,
        init_residual=0.05,
        use_anchor=True,
    ):
        super(HorizonAwareResidualCorrectionHead, self).__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_for_predict = int(num_for_predict)
        self.out_dim = int(out_dim)
        self.use_anchor = bool(use_anchor)
        head_hidden_dim = max(int(head_hidden_dim), 1)
        horizon_embed_dim = max(int(horizon_embed_dim), 1)
        anchor_embed_dim = max(int(anchor_embed_dim), 1)

        self.correction_alpha = nn.Parameter(torch.tensor(float(init_residual), dtype=torch.float32))
        if self.use_anchor:
            self.anchor_embedding = nn.Embedding(24, anchor_embed_dim)
            anchor_dim = anchor_embed_dim
        else:
            self.anchor_embedding = None
            anchor_dim = 0
        self.horizon_embedding = nn.Embedding(self.num_for_predict, horizon_embed_dim)

        correction_input_dim = self.hidden_dim + self.out_dim + anchor_dim + horizon_embed_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(correction_input_dim),
            nn.Linear(correction_input_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(head_hidden_dim, self.out_dim),
        )
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, hidden_sequence, base_output, anchor_hours=None):
        # hidden_sequence: (B, N, D, T), base_output: (B, H, N, O)
        batch_size, num_nodes, _, _ = hidden_sequence.shape
        hidden_state = hidden_sequence[:, :, :, -1]  # (B, N, D)
        hidden_state = hidden_state.unsqueeze(1).expand(-1, self.num_for_predict, -1, -1)

        pieces = [hidden_state]

        if self.use_anchor:
            if anchor_hours is None:
                anchor_ids = torch.zeros(batch_size, dtype=torch.long, device=hidden_sequence.device)
            else:
                anchor_ids = anchor_hours.to(device=hidden_sequence.device, dtype=torch.long).view(-1).clamp(min=0, max=23)
            anchor_emb = self.anchor_embedding(anchor_ids)
            anchor_emb = anchor_emb.view(batch_size, 1, 1, -1).expand(
                batch_size,
                self.num_for_predict,
                num_nodes,
                -1,
            )
            pieces.append(anchor_emb)

        horizon_ids = torch.arange(self.num_for_predict, device=hidden_sequence.device, dtype=torch.long)
        horizon_emb = self.horizon_embedding(horizon_ids).view(1, self.num_for_predict, 1, -1)
        horizon_emb = horizon_emb.expand(batch_size, -1, num_nodes, -1)
        pieces.append(horizon_emb)

        pieces.append(base_output)
        correction_input = torch.cat(pieces, dim=-1)
        delta = self.output_proj(correction_input.reshape(-1, correction_input.size(-1)))
        delta = delta.view(batch_size, self.num_for_predict, num_nodes, self.out_dim)
        correction_alpha = torch.clamp(self.correction_alpha, min=0.0)
        return base_output + correction_alpha * delta


class MSTGCN_block(nn.Module):

    def __init__(
        self,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        fusiongraph,
        device,
        time_kernel_size=3,
        channel_attention=False,
        channel_attention_reduction=4,
        backbone_adaptive_graph=False,
        backbone_support_gate_hidden_dim=32,
        backbone_support_gate_temperature=1.0,
        backbone_multiscale_temporal=False,
        backbone_temporal_dilations=None,
        backbone_temporal_gate_hidden_dim=32,
        backbone_branch_gate=False,
        backbone_branch_gate_hidden_dim=32,
        backbone_dynamic_graph=False,
        backbone_dynamic_graph_hidden_dim=32,
        backbone_dynamic_graph_topk=20,
        backbone_dynamic_graph_init_lambda=0.8,
        backbone_dynamic_graph_residual=False,
        backbone_dynamic_graph_residual_init=0.01,
    ):
        super(MSTGCN_block, self).__init__()
        time_kernel_size = int(time_kernel_size)
        time_padding = time_kernel_size // 2
        self.use_backbone_adaptive_graph = bool(backbone_adaptive_graph)
        self.use_backbone_multiscale_temporal = bool(backbone_multiscale_temporal)
        self.use_backbone_branch_gate = bool(backbone_branch_gate)
        self.use_backbone_dynamic_graph = bool(backbone_dynamic_graph)
        self.cheb_conv = cheb_conv(
            K,
            fusiongraph,
            in_channels,
            nb_chev_filter,
            device,
            adaptive_support_gate=self.use_backbone_adaptive_graph,
            support_gate_hidden_dim=backbone_support_gate_hidden_dim,
            support_gate_temperature=backbone_support_gate_temperature,
            dynamic_graph=self.use_backbone_dynamic_graph,
            dynamic_graph_hidden_dim=backbone_dynamic_graph_hidden_dim,
            dynamic_graph_topk=backbone_dynamic_graph_topk,
            dynamic_graph_init_lambda=backbone_dynamic_graph_init_lambda,
            dynamic_graph_residual=backbone_dynamic_graph_residual,
            dynamic_graph_residual_init=backbone_dynamic_graph_residual_init,
        )
        self.graph_proj = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 1), stride=(1, 1))
        if self.use_backbone_multiscale_temporal:
            if backbone_temporal_dilations is None:
                backbone_temporal_dilations = (1, 2, 4)
            self.time_mixer = CausalMultiScaleTemporalMixer(
                nb_chev_filter,
                nb_time_filter,
                time_strides,
                kernel_size=time_kernel_size,
                dilations=backbone_temporal_dilations,
                dropout=0.1,
                gate_hidden_dim=backbone_temporal_gate_hidden_dim,
            )
        else:
            self.time_conv = nn.Conv2d(
                nb_chev_filter,
                nb_time_filter,
                kernel_size=(1, time_kernel_size),
                stride=(1, time_strides),
                padding=(0, time_padding),
            )
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)
        if self.use_backbone_branch_gate:
            self.backbone_branch_gate = nn.Sequential(
                nn.LayerNorm(nb_chev_filter),
                nn.Linear(nb_chev_filter, max(int(backbone_branch_gate_hidden_dim), 1)),
                nn.GELU(),
                nn.Linear(max(int(backbone_branch_gate_hidden_dim), 1), 3),
            )
            nn.init.zeros_(self.backbone_branch_gate[-1].weight)
            nn.init.zeros_(self.backbone_branch_gate[-1].bias)
        else:
            self.backbone_branch_gate = None
        if channel_attention:
            self.channel_attention = ChannelAttention(nb_time_filter, reduction=channel_attention_reduction)
        else:
            self.channel_attention = nn.Identity()

    def forward(self, x, cheb_polynomials=None, adj_for_run=None):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        # cheb gcn
        spatial_gcn = self.cheb_conv(
            x,
            cheb_polynomials=cheb_polynomials,
            adj_for_run=adj_for_run,
        )  # (b,N,F,T)

        spatial_gcn_c = spatial_gcn.permute(0, 2, 1, 3)  # (b,F,N,T)
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,F,N,T)
        graph_branch = self.graph_proj(spatial_gcn_c)
        if self.use_backbone_multiscale_temporal:
            temporal_branch = self.time_mixer(spatial_gcn_c, context_summary=spatial_gcn.mean(dim=(1, 3)))
        else:
            temporal_branch = self.time_conv(spatial_gcn_c)
        if self.use_backbone_branch_gate:
            branch_weights = torch.softmax(self.backbone_branch_gate(spatial_gcn.mean(dim=(1, 3))), dim=-1)
            fused = (
                branch_weights[:, 0].view(-1, 1, 1, 1) * graph_branch
                + branch_weights[:, 1].view(-1, 1, 1, 1) * temporal_branch
                + branch_weights[:, 2].view(-1, 1, 1, 1) * x_residual
            )
        else:
            fused = x_residual + temporal_branch
        x_residual = self.ln(F.relu(fused).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)  # (b,N,F,T)
        x_residual = self.channel_attention(x_residual)

        return x_residual


class BranchFusionHead(nn.Module):
    """Context-aware fusion over full prediction candidates."""

    def __init__(
        self,
        context_channels,
        num_branches,
        hidden_dim=32,
        dropout=0.0,
        init_main_bias=2.0,
    ):
        super().__init__()
        self.num_branches = int(num_branches)
        hidden_dim = max(int(hidden_dim), 1)
        self.gate = nn.Sequential(
            nn.LayerNorm(int(context_channels)),
            nn.Linear(int(context_channels), hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.num_branches),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        with torch.no_grad():
            self.gate[-1].bias[0] = float(init_main_bias)

    def forward(self, context_x, candidates):
        if len(candidates) != self.num_branches:
            raise ValueError(
                "BranchFusionHead expected %d candidates, got %d"
                % (self.num_branches, len(candidates))
            )
        weights = self.branch_weights(context_x)
        stacked = torch.stack(candidates, dim=1)
        return (weights.view(weights.size(0), self.num_branches, 1, 1, 1) * stacked).sum(dim=1)

    def branch_weights(self, context_x):
        context = context_x.mean(dim=(1, 3))
        return torch.softmax(self.gate(context), dim=-1)


class ResidualBranchGateHead(nn.Module):
    """Anchor/horizon-aware gates for residual correction branches."""

    def __init__(
        self,
        context_channels,
        num_branches,
        num_for_predict,
        hidden_dim=32,
        dropout=0.0,
        init_gate=0.95,
        anchor_embed_dim=8,
        horizon_embed_dim=4,
        branch_embed_dim=4,
        use_anchor=True,
    ):
        super().__init__()
        self.num_branches = int(num_branches)
        self.num_for_predict = int(num_for_predict)
        self.use_anchor = bool(use_anchor)
        hidden_dim = max(int(hidden_dim), 1)
        anchor_embed_dim = max(int(anchor_embed_dim), 1)
        horizon_embed_dim = max(int(horizon_embed_dim), 1)
        branch_embed_dim = max(int(branch_embed_dim), 1)

        if self.use_anchor:
            self.anchor_embedding = nn.Embedding(24, anchor_embed_dim)
            anchor_dim = anchor_embed_dim
        else:
            self.anchor_embedding = None
            anchor_dim = 0
        self.horizon_embedding = nn.Embedding(self.num_for_predict, horizon_embed_dim)
        self.branch_embedding = nn.Embedding(self.num_branches, branch_embed_dim)

        gate_input_dim = int(context_channels) + anchor_dim + horizon_embed_dim + branch_embed_dim
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        init_gate = min(max(float(init_gate), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(init_gate / (1.0 - init_gate)))

    def forward(self, context_x, anchor_hours=None):
        batch_size = context_x.size(0)
        context = context_x.mean(dim=(1, 3))
        context = context.view(batch_size, 1, 1, -1).expand(
            batch_size,
            self.num_branches,
            self.num_for_predict,
            -1,
        )

        pieces = [context]
        if self.use_anchor:
            if anchor_hours is None:
                anchor_ids = torch.zeros(batch_size, dtype=torch.long, device=context_x.device)
            else:
                anchor_ids = anchor_hours.to(device=context_x.device, dtype=torch.long).view(-1).clamp(min=0, max=23)
            anchor_emb = self.anchor_embedding(anchor_ids)
            anchor_emb = anchor_emb.view(batch_size, 1, 1, -1).expand(
                batch_size,
                self.num_branches,
                self.num_for_predict,
                -1,
            )
            pieces.append(anchor_emb)

        horizon_ids = torch.arange(self.num_for_predict, device=context_x.device, dtype=torch.long)
        horizon_emb = self.horizon_embedding(horizon_ids).view(1, 1, self.num_for_predict, -1)
        horizon_emb = horizon_emb.expand(batch_size, self.num_branches, self.num_for_predict, -1)
        pieces.append(horizon_emb)

        branch_ids = torch.arange(self.num_branches, device=context_x.device, dtype=torch.long)
        branch_emb = self.branch_embedding(branch_ids).view(1, self.num_branches, 1, -1)
        branch_emb = branch_emb.expand(batch_size, self.num_branches, self.num_for_predict, -1)
        pieces.append(branch_emb)

        gate_input = torch.cat(pieces, dim=-1)
        gate = self.gate(gate_input.reshape(-1, gate_input.size(-1)))
        return torch.sigmoid(gate).view(batch_size, self.num_branches, self.num_for_predict)


class AnchorResidualScale(nn.Module):
    """Conservative anchor-hour scale for a residual branch."""

    def __init__(self, init_scale=1.0, max_delta=0.2, anchor_hours=None):
        super().__init__()
        self.max_delta = float(max_delta)
        if self.max_delta <= 0:
            raise ValueError("max_delta must be > 0.")
        anchor_mask = torch.zeros(24, dtype=torch.float32)
        if anchor_hours:
            for hour in anchor_hours:
                hour = int(hour)
                if hour < 0 or hour > 23:
                    raise ValueError("anchor hour values must be within [0, 23].")
                anchor_mask[hour] = 1.0
        else:
            anchor_mask.fill_(1.0)
        self.register_buffer("anchor_mask", anchor_mask)

        bounded = (float(init_scale) - 1.0) / self.max_delta
        bounded = min(max(bounded, -0.999), 0.999)
        initial_logit = 0.5 * math.log((1.0 + bounded) / (1.0 - bounded))
        self.scale_logit = nn.Parameter(torch.full((24,), float(initial_logit), dtype=torch.float32))

    def forward(self, residual, anchor_hours=None):
        batch_size = residual.size(0)
        if anchor_hours is None:
            anchor_ids = torch.zeros(batch_size, dtype=torch.long, device=residual.device)
        else:
            anchor_ids = anchor_hours.to(device=residual.device, dtype=torch.long).view(-1).clamp(min=0, max=23)
        scale = 1.0 + self.anchor_mask.to(residual.device, residual.dtype) * self.max_delta * torch.tanh(
            self.scale_logit.to(device=residual.device, dtype=residual.dtype)
        )
        scale = scale.index_select(0, anchor_ids).view(batch_size, 1, 1, 1)
        return residual * scale


class MSTGCN_submodule(nn.Module):

    def __init__(
        self,
        device,
        fusiongraph,
        in_channels,
        len_input,
        num_for_predict,
        out_dim=1,
        categorical_feature_configs=None,
        cheb_k=3,
        nb_block=2,
        nb_chev_filter=64,
        nb_time_filter=64,
        time_kernel_size=3,
        channel_attention=False,
        channel_attention_reduction=4,
        trend_alignment_decoder=False,
        trend_time_feature_index=-1,
        trend_time_feature_mean=0.0,
        trend_time_feature_std=1.0,
        trend_time_cycle=24,
        trend_time_embed_dim=16,
        trend_attention_heads=4,
        trend_dropout=0.1,
        horizon_specific_prediction_head=False,
        horizon_aware_prediction_head=False,
        horizon_head_hidden_dim=64,
        horizon_head_embed_dim=8,
        horizon_head_dropout=0.1,
        horizon_graph_fusion_decoder=False,
        horizon_graph_decoder_residual=0.2,
        ast_tcn_residual=False,
        ast_tcn_hidden_dim=32,
        ast_tcn_layers=4,
        ast_tcn_kernel_size=3,
        ast_tcn_dilation_base=2,
        ast_tcn_heads=4,
        ast_tcn_dropout=0.1,
        ast_tcn_residual_init=0.05,
        ast_tcn_bounded_alpha=False,
        ast_tcn_alpha_max=0.1,
        ast_tcn_horizon_alpha=False,
        ast_tcn_residual_horizon_mask=None,
        ast_tcn_zero_init=False,
        ast_tcn_residual_gate=False,
        ast_tcn_residual_gate_hidden_dim=16,
        ast_tcn_residual_gate_init=0.2,
        ast_tcn_anchor_horizon_gate=False,
        ast_tcn_anchor_embed_dim=8,
        ast_tcn_horizon_embed_dim=4,
        ast_tcn_anchor_horizon_gate_hidden_dim=16,
        ast_tcn_anchor_horizon_gate_init=0.2,
        ast_tcn_edge_bias=False,
        ast_tcn_edge_bias_init=0.1,
        ast_tcn_hgaurban_edge_bias=False,
        ast_tcn_hgaurban_edge_bias_init=0.0,
        hgaurban_edge_bias_matrix=None,
        ast_tcn_edge_bias_eps=1e-6,
        sthybrid_ms_residual=False,
        sthybrid_ms_hidden_dim=32,
        sthybrid_ms_dilations=None,
        sthybrid_ms_kernel_size=3,
        sthybrid_ms_heads=4,
        sthybrid_ms_dropout=0.1,
        sthybrid_ms_residual_init=0.01,
        sthybrid_ms_bounded_alpha=True,
        sthybrid_ms_alpha_max=0.1,
        sthybrid_ms_zero_init=True,
        sthybrid_ms_edge_bias=True,
        sthybrid_ms_edge_bias_init=0.05,
        sthybrid_ms_edge_bias_eps=1e-6,
        stgformer_temporal_residual=False,
        stgformer_temporal_hidden_dim=32,
        stgformer_temporal_heads=4,
        stgformer_temporal_local_kernel_size=5,
        stgformer_temporal_landmark_count=24,
        stgformer_temporal_dropout=0.1,
        stgformer_temporal_residual_init=0.01,
        stgformer_temporal_bounded_alpha=True,
        stgformer_temporal_alpha_max=0.1,
        stgformer_temporal_zero_init=True,
        stgformer_temporal_gate_init=0.2,
        stgformer_spatial_transformer=False,
        stgformer_spatial_heads=4,
        stgformer_spatial_edge_bias=True,
        stgformer_spatial_edge_bias_init=0.05,
        stgformer_spatial_edge_bias_eps=1e-6,
        stgformer_anchor_residual_scale=False,
        stgformer_anchor_residual_init=1.0,
        stgformer_anchor_residual_max_delta=0.2,
        stgformer_anchor_residual_hours=None,
        backbone_adaptive_graph=False,
        backbone_support_gate_hidden_dim=32,
        backbone_support_gate_temperature=1.0,
        backbone_multiscale_temporal=False,
        backbone_temporal_dilations=None,
        backbone_temporal_gate_hidden_dim=32,
        backbone_branch_gate=False,
        backbone_branch_gate_hidden_dim=32,
        backbone_dynamic_graph=False,
        backbone_dynamic_graph_hidden_dim=32,
        backbone_dynamic_graph_topk=20,
        backbone_dynamic_graph_init_lambda=0.8,
        backbone_dynamic_graph_residual=False,
        backbone_dynamic_graph_residual_init=0.01,
        branch_fusion=False,
        branch_fusion_hidden_dim=32,
        branch_fusion_dropout=0.0,
        branch_fusion_init_main_bias=2.0,
        residual_gate_fusion=False,
        residual_gate_hidden_dim=32,
        residual_gate_dropout=0.0,
        residual_gate_init=0.95,
        residual_gate_anchor_embed_dim=8,
        residual_gate_horizon_embed_dim=4,
        residual_gate_branch_embed_dim=4,
        residual_gate_use_anchor=True,
        residual_correction_head=False,
        residual_correction_hidden_dim=64,
        residual_correction_horizon_embed_dim=8,
        residual_correction_anchor_embed_dim=8,
        residual_correction_dropout=0.1,
        residual_correction_init=0.05,
        residual_correction_use_anchor=True,
    ):


    # def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        # Fusion Graph
        DEVICE = torch.device(device)

        # -----------------

        # Parameters
        K = int(cheb_k)
        nb_block = int(nb_block)
        nb_chev_filter = int(nb_chev_filter)
        nb_time_filter = int(nb_time_filter)
        time_kernel_size = int(time_kernel_size)
        channel_attention_reduction = int(channel_attention_reduction)
        time_strides = 1


        super(MSTGCN_submodule, self).__init__()
        self.fusiongraph = fusiongraph
        self.input_adapter = FeatureEmbeddingAdapter(in_channels, categorical_feature_configs)
        effective_in_channels = self.input_adapter.effective_in_channels
        if channel_attention:
            self.input_channel_attention = ChannelAttention(
                effective_in_channels,
                reduction=channel_attention_reduction,
            )
        else:
            self.input_channel_attention = nn.Identity()

        self.BlockList = nn.ModuleList([
            MSTGCN_block(
                effective_in_channels,
                K,
                nb_chev_filter,
                nb_time_filter,
                time_strides,
                fusiongraph,
                DEVICE,
                time_kernel_size=time_kernel_size,
                channel_attention=channel_attention,
                channel_attention_reduction=channel_attention_reduction,
                backbone_adaptive_graph=backbone_adaptive_graph,
                backbone_support_gate_hidden_dim=backbone_support_gate_hidden_dim,
                backbone_support_gate_temperature=backbone_support_gate_temperature,
                backbone_multiscale_temporal=backbone_multiscale_temporal,
                backbone_temporal_dilations=backbone_temporal_dilations,
                backbone_temporal_gate_hidden_dim=backbone_temporal_gate_hidden_dim,
                backbone_branch_gate=backbone_branch_gate,
                backbone_branch_gate_hidden_dim=backbone_branch_gate_hidden_dim,
                backbone_dynamic_graph=backbone_dynamic_graph,
                backbone_dynamic_graph_hidden_dim=backbone_dynamic_graph_hidden_dim,
                backbone_dynamic_graph_topk=backbone_dynamic_graph_topk,
                backbone_dynamic_graph_init_lambda=backbone_dynamic_graph_init_lambda,
                backbone_dynamic_graph_residual=backbone_dynamic_graph_residual,
                backbone_dynamic_graph_residual_init=backbone_dynamic_graph_residual_init,
            )
        ])

        self.BlockList.extend([
            MSTGCN_block(
                nb_time_filter,
                K,
                nb_chev_filter,
                nb_time_filter,
                1,
                fusiongraph,
                DEVICE,
                time_kernel_size=time_kernel_size,
                channel_attention=channel_attention,
                channel_attention_reduction=channel_attention_reduction,
                backbone_adaptive_graph=backbone_adaptive_graph,
                backbone_support_gate_hidden_dim=backbone_support_gate_hidden_dim,
                backbone_support_gate_temperature=backbone_support_gate_temperature,
                backbone_multiscale_temporal=backbone_multiscale_temporal,
                backbone_temporal_dilations=backbone_temporal_dilations,
                backbone_temporal_gate_hidden_dim=backbone_temporal_gate_hidden_dim,
                backbone_branch_gate=backbone_branch_gate,
                backbone_branch_gate_hidden_dim=backbone_branch_gate_hidden_dim,
                backbone_dynamic_graph=backbone_dynamic_graph,
                backbone_dynamic_graph_hidden_dim=backbone_dynamic_graph_hidden_dim,
                backbone_dynamic_graph_topk=backbone_dynamic_graph_topk,
                backbone_dynamic_graph_init_lambda=backbone_dynamic_graph_init_lambda,
                backbone_dynamic_graph_residual=backbone_dynamic_graph_residual,
                backbone_dynamic_graph_residual_init=backbone_dynamic_graph_residual_init,
            )
            for _ in range(nb_block - 1)
        ])

        self.use_trend_alignment_decoder = bool(trend_alignment_decoder)
        self.use_horizon_specific_prediction_head = bool(horizon_specific_prediction_head)
        self.use_horizon_aware_prediction_head = bool(horizon_aware_prediction_head)
        if self.use_trend_alignment_decoder:
            self.trend_decoder = TrendAlignmentDecoder(
                hidden_dim=nb_time_filter,
                num_for_predict=num_for_predict,
                out_dim=out_dim,
                time_feature_index=trend_time_feature_index,
                time_feature_mean=trend_time_feature_mean,
                time_feature_std=trend_time_feature_std,
                time_cycle=trend_time_cycle,
                time_embed_dim=trend_time_embed_dim,
                attention_heads=trend_attention_heads,
                dropout=trend_dropout,
            )
        elif self.use_horizon_aware_prediction_head:
            self.horizon_aware_head = HorizonAwarePredictionHead(
                input_time_steps=int(len_input / time_strides),
                hidden_dim=nb_time_filter,
                num_for_predict=num_for_predict,
                out_dim=out_dim,
                head_hidden_dim=horizon_head_hidden_dim,
                horizon_embed_dim=horizon_head_embed_dim,
                dropout=horizon_head_dropout,
            )
        elif self.use_horizon_specific_prediction_head:
            self.horizon_final_convs = nn.ModuleList([
                nn.Conv2d(
                    int(len_input / time_strides),
                    out_dim,
                    kernel_size=(1, nb_time_filter),
                )
                for _ in range(num_for_predict)
            ])
        else:
            self.final_conv = nn.Conv2d(
                int(len_input / time_strides),
                num_for_predict * out_dim,
                kernel_size=(1, nb_time_filter),
            )

        self.use_ast_tcn_residual = bool(ast_tcn_residual)
        self.use_ast_tcn_hgaurban_edge_bias = bool(ast_tcn_hgaurban_edge_bias)
        if self.use_ast_tcn_hgaurban_edge_bias:
            if hgaurban_edge_bias_matrix is None:
                raise ValueError("ast_tcn_hgaurban_edge_bias requires hgaurban_edge_bias_matrix.")
            hgaurban_tensor = torch.as_tensor(hgaurban_edge_bias_matrix, dtype=torch.float32)
            self.register_buffer("hgaurban_edge_bias_matrix", hgaurban_tensor)
        else:
            self.hgaurban_edge_bias_matrix = None
        if self.use_ast_tcn_residual:
            self.ast_tcn_branch = ASTTCNResidualBranch(
                in_channels=effective_in_channels,
                hidden_dim=ast_tcn_hidden_dim,
                num_for_predict=num_for_predict,
                out_dim=out_dim,
                layers=ast_tcn_layers,
                kernel_size=ast_tcn_kernel_size,
                dilation_base=ast_tcn_dilation_base,
                heads=ast_tcn_heads,
                dropout=ast_tcn_dropout,
                residual_init=ast_tcn_residual_init,
                bounded_alpha=ast_tcn_bounded_alpha,
                alpha_max=ast_tcn_alpha_max,
                horizon_alpha=ast_tcn_horizon_alpha,
                residual_horizon_mask=ast_tcn_residual_horizon_mask,
                zero_init=ast_tcn_zero_init,
                residual_gate=ast_tcn_residual_gate,
                residual_gate_hidden_dim=ast_tcn_residual_gate_hidden_dim,
                residual_gate_init=ast_tcn_residual_gate_init,
                anchor_horizon_gate=ast_tcn_anchor_horizon_gate,
                anchor_embed_dim=ast_tcn_anchor_embed_dim,
                horizon_embed_dim=ast_tcn_horizon_embed_dim,
                anchor_horizon_gate_hidden_dim=ast_tcn_anchor_horizon_gate_hidden_dim,
                anchor_horizon_gate_init=ast_tcn_anchor_horizon_gate_init,
                edge_bias=ast_tcn_edge_bias,
                edge_bias_init=ast_tcn_edge_bias_init,
                hgaurban_edge_bias=ast_tcn_hgaurban_edge_bias,
                hgaurban_edge_bias_init=ast_tcn_hgaurban_edge_bias_init,
                edge_bias_eps=ast_tcn_edge_bias_eps,
            )

        self.use_sthybrid_ms_residual = bool(sthybrid_ms_residual)
        if self.use_sthybrid_ms_residual:
            if sthybrid_ms_dilations is None:
                sthybrid_ms_dilations = (1, 2, 4, 8)
            self.sthybrid_ms_branch = STHybridMultiScaleResidualBranch(
                in_channels=effective_in_channels,
                hidden_dim=sthybrid_ms_hidden_dim,
                num_for_predict=num_for_predict,
                out_dim=out_dim,
                dilations=sthybrid_ms_dilations,
                kernel_size=sthybrid_ms_kernel_size,
                heads=sthybrid_ms_heads,
                dropout=sthybrid_ms_dropout,
                residual_init=sthybrid_ms_residual_init,
                bounded_alpha=sthybrid_ms_bounded_alpha,
                alpha_max=sthybrid_ms_alpha_max,
                zero_init=sthybrid_ms_zero_init,
                edge_bias=sthybrid_ms_edge_bias,
                edge_bias_init=sthybrid_ms_edge_bias_init,
                edge_bias_eps=sthybrid_ms_edge_bias_eps,
            )

        self.use_stgformer_temporal_residual = bool(stgformer_temporal_residual)
        if self.use_stgformer_temporal_residual:
            self.stgformer_temporal_branch = LocalGlobalTemporalResidualBranch(
                in_channels=effective_in_channels,
                hidden_dim=stgformer_temporal_hidden_dim,
                num_for_predict=num_for_predict,
                out_dim=out_dim,
                heads=stgformer_temporal_heads,
                local_kernel_size=stgformer_temporal_local_kernel_size,
                landmark_count=stgformer_temporal_landmark_count,
                dropout=stgformer_temporal_dropout,
                residual_init=stgformer_temporal_residual_init,
                bounded_alpha=stgformer_temporal_bounded_alpha,
                alpha_max=stgformer_temporal_alpha_max,
                zero_init=stgformer_temporal_zero_init,
                gate_init=stgformer_temporal_gate_init,
                spatial_transformer=stgformer_spatial_transformer,
                spatial_heads=stgformer_spatial_heads,
                spatial_edge_bias=stgformer_spatial_edge_bias,
                spatial_edge_bias_init=stgformer_spatial_edge_bias_init,
                spatial_edge_bias_eps=stgformer_spatial_edge_bias_eps,
            )
        self.use_stgformer_anchor_residual_scale = bool(stgformer_anchor_residual_scale)
        if self.use_stgformer_anchor_residual_scale:
            if not self.use_stgformer_temporal_residual:
                raise ValueError("stgformer_anchor_residual_scale requires stgformer_temporal_residual.")
            self.stgformer_anchor_residual_scale = AnchorResidualScale(
                init_scale=stgformer_anchor_residual_init,
                max_delta=stgformer_anchor_residual_max_delta,
                anchor_hours=stgformer_anchor_residual_hours,
            )
        else:
            self.stgformer_anchor_residual_scale = None

        self.use_branch_fusion = bool(branch_fusion)
        self.branch_fusion_branch_names = ["graph"]
        if self.use_ast_tcn_residual:
            self.branch_fusion_branch_names.append("ast_tcn")
        if self.use_sthybrid_ms_residual:
            self.branch_fusion_branch_names.append("sthybrid")
        if self.use_stgformer_temporal_residual:
            self.branch_fusion_branch_names.append("stgformer")
        if self.use_branch_fusion:
            if len(self.branch_fusion_branch_names) <= 1:
                raise ValueError("branch_fusion requires at least one residual branch.")
            self.branch_fusion_head = BranchFusionHead(
                context_channels=effective_in_channels,
                num_branches=len(self.branch_fusion_branch_names),
                hidden_dim=branch_fusion_hidden_dim,
                dropout=branch_fusion_dropout,
                init_main_bias=branch_fusion_init_main_bias,
            )
        else:
            self.branch_fusion_head = None

        self.use_residual_gate_fusion = bool(residual_gate_fusion)
        self.residual_gate_branch_names = []
        if self.use_ast_tcn_residual:
            self.residual_gate_branch_names.append("ast_tcn")
        if self.use_sthybrid_ms_residual:
            self.residual_gate_branch_names.append("sthybrid")
        if self.use_stgformer_temporal_residual:
            self.residual_gate_branch_names.append("stgformer")
        if self.use_residual_gate_fusion:
            if self.use_branch_fusion:
                raise ValueError("residual_gate_fusion cannot be combined with branch_fusion.")
            if len(self.residual_gate_branch_names) <= 0:
                raise ValueError("residual_gate_fusion requires at least one residual branch.")
            self.residual_gate_head = ResidualBranchGateHead(
                context_channels=effective_in_channels,
                num_branches=len(self.residual_gate_branch_names),
                num_for_predict=num_for_predict,
                hidden_dim=residual_gate_hidden_dim,
                dropout=residual_gate_dropout,
                init_gate=residual_gate_init,
                anchor_embed_dim=residual_gate_anchor_embed_dim,
                horizon_embed_dim=residual_gate_horizon_embed_dim,
                branch_embed_dim=residual_gate_branch_embed_dim,
                use_anchor=residual_gate_use_anchor,
            )
        else:
            self.residual_gate_head = None

        self.use_residual_correction_head = bool(residual_correction_head)
        if self.use_residual_correction_head:
            self.residual_correction_head = HorizonAwareResidualCorrectionHead(
                hidden_dim=nb_time_filter,
                num_for_predict=num_for_predict,
                out_dim=out_dim,
                head_hidden_dim=residual_correction_hidden_dim,
                horizon_embed_dim=residual_correction_horizon_embed_dim,
                anchor_embed_dim=residual_correction_anchor_embed_dim,
                dropout=residual_correction_dropout,
                init_residual=residual_correction_init,
                use_anchor=residual_correction_use_anchor,
            )
        else:
            self.residual_correction_head = None

        self.DEVICE = DEVICE
        self.num_for_predict = num_for_predict
        self.out_dim = out_dim
        self.use_horizon_graph_fusion_decoder = bool(horizon_graph_fusion_decoder)
        self.horizon_graph_decoder_residual = float(horizon_graph_decoder_residual)
        self.to(self.DEVICE)

    def forward(self, x, anchor_hours=None):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''

        context_x = x.permute((0, 2, 3, 1))
        x = self.input_adapter(x)
        x = x.permute((0, 2, 3, 1))
        x = self.input_channel_attention(x)
        branch_x = x

        adj_for_run = self.fusiongraph(context_x, anchor_hours=anchor_hours)
        if any(getattr(block, "use_backbone_dynamic_graph", False) for block in self.BlockList):
            cheb_polynomials = None
        else:
            cheb_polynomials = self.BlockList[0].cheb_conv.build_cheb_polynomials(adj_for_run)

        for block in self.BlockList:
            x = block(x, cheb_polynomials=cheb_polynomials, adj_for_run=adj_for_run)

        if self.use_trend_alignment_decoder:
            output = self.trend_decoder(x, context_x)
        elif self.use_horizon_graph_fusion_decoder:
            horizon_graphs = self.fusiongraph.horizon_graphs_for_run(
                context_x,
                anchor_hours=anchor_hours,
                num_for_predict=self.num_for_predict,
            )
            if horizon_graphs is None:
                raise RuntimeError('horizon_graph_fusion_decoder requires FusionGraphModel.horizon_graph_fusion_gate.')
            horizon_outputs = []
            for horizon_idx in range(self.num_for_predict):
                adj_h = horizon_graphs[horizon_idx].to(x.device, dtype=x.dtype)
                adj_h = adj_h / adj_h.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                graph_x = torch.einsum('ij,bjft->bift', adj_h, x)
                decoder_x = (
                    (1.0 - self.horizon_graph_decoder_residual) * x
                    + self.horizon_graph_decoder_residual * graph_x
                )
                horizon_all = self.final_conv(decoder_x.permute(0, 3, 1, 2))[:, :, :, -1]
                batch_size, _, num_nodes = horizon_all.shape
                horizon_all = horizon_all.view(batch_size, self.num_for_predict, self.out_dim, num_nodes)
                horizon_output = horizon_all[:, horizon_idx].permute(0, 2, 1).unsqueeze(1)
                horizon_outputs.append(horizon_output)
            output = torch.cat(horizon_outputs, dim=1)
        elif self.use_horizon_aware_prediction_head:
            output = self.horizon_aware_head(x)
        elif self.use_horizon_specific_prediction_head:
            decoder_x = x.permute(0, 3, 1, 2)
            horizon_outputs = []
            for horizon_head in self.horizon_final_convs:
                horizon_output = horizon_head(decoder_x)[:, :, :, -1]
                horizon_output = horizon_output.permute(0, 2, 1).unsqueeze(1)
                horizon_outputs.append(horizon_output)
            output = torch.cat(horizon_outputs, dim=1)
        else:
            output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1]
            batch_size, _, num_nodes = output.shape
            output = output.view(batch_size, self.num_for_predict, self.out_dim, num_nodes)
            output = output.permute(0, 1, 3, 2)

        branch_weights = None
        residual_branch_index = 1
        if self.use_branch_fusion:
            branch_weights = self.branch_fusion_head.branch_weights(branch_x)
        residual_gates = None
        residual_gate_index = 0
        if self.use_residual_gate_fusion:
            residual_gates = self.residual_gate_head(branch_x, anchor_hours=anchor_hours)
        if self.use_ast_tcn_residual:
            hgaurban_adj = (
                self.hgaurban_edge_bias_matrix.to(device=branch_x.device, dtype=branch_x.dtype)
                if self.use_ast_tcn_hgaurban_edge_bias
                else None
            )
            ast_residual = self.ast_tcn_branch(
                branch_x,
                adj_for_run=adj_for_run,
                anchor_hours=anchor_hours,
                hgaurban_adj=hgaurban_adj,
            )
            if self.use_branch_fusion:
                output = output + branch_weights[:, residual_branch_index].view(-1, 1, 1, 1) * ast_residual
                residual_branch_index += 1
            elif self.use_residual_gate_fusion:
                gate = residual_gates[:, residual_gate_index].view(-1, self.num_for_predict, 1, 1)
                output = output + gate * ast_residual
                residual_gate_index += 1
            else:
                output = output + ast_residual
        if self.use_sthybrid_ms_residual:
            sthybrid_residual = self.sthybrid_ms_branch(branch_x, adj_for_run=adj_for_run)
            if self.use_branch_fusion:
                output = output + branch_weights[:, residual_branch_index].view(-1, 1, 1, 1) * sthybrid_residual
                residual_branch_index += 1
            elif self.use_residual_gate_fusion:
                gate = residual_gates[:, residual_gate_index].view(-1, self.num_for_predict, 1, 1)
                output = output + gate * sthybrid_residual
                residual_gate_index += 1
            else:
                output = output + sthybrid_residual
        if self.use_stgformer_temporal_residual:
            stgformer_residual = self.stgformer_temporal_branch(branch_x, adj_for_run=adj_for_run)
            if self.use_stgformer_anchor_residual_scale:
                stgformer_residual = self.stgformer_anchor_residual_scale(
                    stgformer_residual,
                    anchor_hours=anchor_hours,
                )
            if self.use_branch_fusion:
                output = output + branch_weights[:, residual_branch_index].view(-1, 1, 1, 1) * stgformer_residual
                residual_branch_index += 1
            elif self.use_residual_gate_fusion:
                gate = residual_gates[:, residual_gate_index].view(-1, self.num_for_predict, 1, 1)
                output = output + gate * stgformer_residual
                residual_gate_index += 1
            else:
                output = output + stgformer_residual

        if self.use_residual_correction_head:
            output = self.residual_correction_head(x, output, anchor_hours=anchor_hours)

        return output
