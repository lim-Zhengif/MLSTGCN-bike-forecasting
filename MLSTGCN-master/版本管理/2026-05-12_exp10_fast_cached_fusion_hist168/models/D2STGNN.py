# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MSTGCN import FeatureEmbeddingAdapter


def row_normalize(adj):
    adj = torch.relu(adj)
    adj = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    rowsum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return adj / rowsum


class DiffusionGraphConv(nn.Module):
    def __init__(self, channels, support_len, order=2, dropout=0.1):
        super(DiffusionGraphConv, self).__init__()
        self.order = int(order)
        self.dropout = float(dropout)
        input_channels = channels * (support_len * self.order + 1)
        self.proj = nn.Conv2d(input_channels, channels, kernel_size=(1, 1))

    def forward(self, x, supports):
        # x: [B, C, N, T], supports: list of [N, N]
        outputs = [x]
        for support in supports:
            x1 = torch.einsum('bcnt,nm->bcmt', x, support)
            outputs.append(x1)
            for _ in range(2, self.order + 1):
                x1 = torch.einsum('bcnt,nm->bcmt', x1, support)
                outputs.append(x1)
        h = torch.cat(outputs, dim=1)
        h = self.proj(h)
        return F.dropout(h, p=self.dropout, training=self.training)


class D2STGNNBlock(nn.Module):
    def __init__(self, hidden_dim, support_len, gcn_order=2, dropout=0.1, kernel_size=2, dilation=1):
        super(D2STGNNBlock, self).__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.filter_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=(1, self.kernel_size),
            dilation=(1, self.dilation),
        )
        self.gate_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=(1, self.kernel_size),
            dilation=(1, self.dilation),
        )
        self.graph_conv = DiffusionGraphConv(hidden_dim, support_len, order=gcn_order, dropout=dropout)
        self.residual_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.dropout = float(dropout)

    def forward(self, x, supports):
        residual = x
        pad = (self.kernel_size - 1) * self.dilation
        x_padded = F.pad(x, (pad, 0, 0, 0))
        temporal = torch.tanh(self.filter_conv(x_padded)) * torch.sigmoid(self.gate_conv(x_padded))
        spatial = self.graph_conv(temporal, supports)
        spatial = self.residual_conv(spatial)
        out = spatial + residual[..., -spatial.shape[-1]:]
        return self.norm(out)


class D2STGNNFusionBackbone(nn.Module):
    """D2STGNN-style backbone that keeps FusionGraphModel as the graph prior.

    It consumes the same shape as MSTGCN_submodule:
    input  [B, T, N, F]
    output [B, pred_len, N, out_dim]
    """

    def __init__(
        self,
        device,
        fusiongraph,
        in_channels,
        len_input,
        num_for_predict,
        out_dim=1,
        categorical_feature_configs=None,
        hidden_dim=64,
        num_layers=4,
        dropout=0.1,
        dilation_cycle=2,
        kernel_size=2,
        gcn_order=2,
        node_embed_dim=16,
        adaptive_adj=True,
        use_reverse=True,
        fusion_init=1.0,
    ):
        super(D2STGNNFusionBackbone, self).__init__()
        self.DEVICE = torch.device(device)
        self.fusiongraph = fusiongraph
        self.num_for_predict = int(num_for_predict)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.adaptive_adj = bool(adaptive_adj)
        self.use_reverse = bool(use_reverse)
        self.input_adapter = FeatureEmbeddingAdapter(in_channels, categorical_feature_configs)
        effective_in_channels = self.input_adapter.effective_in_channels

        self.start_conv = nn.Conv2d(effective_in_channels, self.hidden_dim, kernel_size=(1, 1))
        self.node_embed_dim = int(node_embed_dim)
        self.fusion_alpha = nn.Parameter(torch.tensor(float(fusion_init), dtype=torch.float32))
        num_nodes = int(getattr(getattr(fusiongraph, 'graph', None), 'node_num', 0))
        if self.adaptive_adj and num_nodes <= 0:
            raise ValueError('D2STGNN adaptive adjacency requires fusiongraph.graph.node_num.')
        if self.adaptive_adj:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, self.node_embed_dim, device=self.DEVICE))
            self.nodevec2 = nn.Parameter(torch.randn(self.node_embed_dim, num_nodes, device=self.DEVICE))
            nn.init.xavier_uniform_(self.nodevec1)
            nn.init.xavier_uniform_(self.nodevec2)
        else:
            self.register_parameter('nodevec1', None)
            self.register_parameter('nodevec2', None)

        support_len = 1
        if self.use_reverse:
            support_len += 1

        self.blocks = nn.ModuleList()
        for layer_idx in range(int(num_layers)):
            dilation = 2 ** (layer_idx % max(int(dilation_cycle), 1))
            self.blocks.append(
                D2STGNNBlock(
                    hidden_dim=self.hidden_dim,
                    support_len=support_len,
                    gcn_order=gcn_order,
                    dropout=dropout,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_for_predict * self.out_dim),
        )
        self.to(self.DEVICE)

    def _build_supports(self, x_context):
        fused_adj = row_normalize(self.fusiongraph(x_context))
        if self.adaptive_adj:
            adaptive_adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            blend = torch.sigmoid(self.fusion_alpha)
            adj = blend * fused_adj + (1.0 - blend) * adaptive_adj
            adj = row_normalize(adj)
        else:
            adj = fused_adj

        supports = [adj]
        if self.use_reverse:
            supports.append(row_normalize(adj.transpose(0, 1)))
        return supports

    def forward(self, x):
        # Raw context is kept for FusionGraphModel's optional context gate.
        x_context = x.permute(0, 2, 3, 1)
        x = self.input_adapter(x)
        x = x.permute(0, 3, 2, 1)
        supports = self._build_supports(x_context)
        h = self.start_conv(x)
        for block in self.blocks:
            h = block(h, supports)
        node_state = h[..., -1].permute(0, 2, 1)
        output = self.decoder(node_state)
        batch_size, num_nodes, _ = output.shape
        output = output.view(batch_size, num_nodes, self.num_for_predict, self.out_dim)
        return output.permute(0, 2, 1, 3)
