import os
import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def detect_project_root(start_dir):
    current = os.path.abspath(start_dir)
    while True:
        if (
            os.path.isdir(os.path.join(current, 'data'))
            and os.path.isdir(os.path.join(current, 'models'))
            and os.path.isdir(os.path.join(current, 'datasets'))
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.abspath(start_dir)
        current = parent


PROJECT_ROOT = detect_project_root(SCRIPT_DIR)


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
    def forward(self, x):
        return self.mlp(x)

class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = x.to(self.conv.weight.device)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class SGEmbedding(nn.Module):
    """
    multi-graph spatial embedding
    SE:     [num_vertices, D]
    GE:     [num_vertices, num_graphs, 1]
    D:      output dims = M * d
    retrun: [num_vertices, num_graphs, num_vertices, D]
    """
    def __init__(self, D, bn_decay):
        super(SGEmbedding, self).__init__()
        self.FC_se = FC(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)

        self.FC_ge = FC(
            input_dims=[5, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)  # input_dims = graph_nums

    def forward(self, SE, GE):
        # spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE)
        # multi-graph embedding
        GE = F.one_hot(GE[..., 0].to(torch.int64) % 5, 5).to(SE.device).float()
        GE = GE.unsqueeze(dim=2)
        GE = self.FC_ge(GE)
        return SE + GE


class spatialAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [num_vertices, num_graphs, num_vertices, D]
    SGE:    [num_vertices, num_graphs, num_vertices, D]
    M:      number of attention heads
    d:      dimension of each attention outputs
    return: [num_vertices, num_graphs, num_vertices, D]
    '''
    def __init__(self, M, d, bn_decay):
        super(spatialAttention, self).__init__()
        self.d = d
        self.M = M
        D = self.M * self.d
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, SGE):
        num_vertex = X.shape[0]
        X = torch.cat((X, SGE), dim=-1)  
        # [num_vertices, num_graphs, num_vertices, 2 * D]

        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)                # [M * num_vertices, num_graphs, num_vertices, d]

        query = torch.cat(torch.split(query, self.M, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.M, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.M, dim=-1), dim=0)

        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)

        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, num_vertex, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class graphAttention(nn.Module):
    '''
    multi-graph attention mechanism
    X:      [num_vertices, num_graphs, num_vertices, D]
    SGE:    [num_vertices, num_graphs, num_vertices, D]
    M:      number of attention heads
    d:      dimension of each attention outputs
    return: [num_vertices, num_graphs, num_vertices, D]
    '''
    def __init__(self, M, d, bn_decay, mask=True):
        super(graphAttention, self).__init__()
        self.d = d
        self.M = M
        D = self.M * self.d
        self.mask = mask
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, SGE):
        num_vertex_ = X.shape[0]
        X = torch.cat((X, SGE), dim=-1)
        # [num_vertices, num_graphs, num_vertices, 2 * D]

        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)

        query = torch.cat(torch.split(query, self.M, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.M, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.M, dim=-1), dim=0)    
        # [M * num_vertices, num_graphs, num_vertices, d]

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)

        if self.mask:
            num_vertex = X.shape[0]
            num_step = X.shape[1]
            mask = torch.ones(num_step, num_step, device=attention.device)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.M * num_vertex, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + 1)

        attention = F.softmax(attention, dim=-1)

        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, num_vertex_, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [num_vertices, num_graphs, num_vertices, D]
    HG:     [num_vertices, num_graphs, num_vertices, D]
    D:      output dims = M * d
    return: [num_vertices, num_graphs, num_vertices, D]
    '''

    def __init__(self, D, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, HS, HG):
        XS = self.FC_xs(HS)
        XG = self.FC_xt(HG)
        z = torch.sigmoid(torch.add(XS, XG))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HG))
        H = self.FC_h(H)
        del XS, XG, z
        return H


class STAttBlock(nn.Module):
    def __init__(self, M, d, bn_decay, mask=False):
        super(STAttBlock, self).__init__()
        self.spatialAttention = spatialAttention(M, d, bn_decay)
        self.graphAttention = graphAttention(M, d, bn_decay, mask=mask)
        self.gatedFusion = gatedFusion(M * d, bn_decay)

    def forward(self, X, SGE):
        HS = self.spatialAttention(X, SGE)
        HT = self.graphAttention(X, SGE)
        H = self.gatedFusion(HS, HT)
        del HS, HT
        return torch.add(X, H)

class FusionGraphModel(nn.Module):
    def __init__(self, graph, device, conf_graph, conf_data, M, d, bn_decay):
        super(FusionGraphModel, self).__init__()
        self.M = M
        self.d = d
        self.bn_decay = bn_decay
        self.device = torch.device(device)
        D = self.M * self.d
        self.SG_ATT = STAttBlock(M, d, bn_decay)
        self.SGEmbedding = SGEmbedding(D, bn_decay)

        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=self.bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=self.bn_decay)

        self.graph = graph
        self.matrix_w = conf_graph['matrix_weight']
        # matrix_weight: if True, turn the weight matrices trainable.         
        self.attention = conf_graph['attention']
        # attention: if True, the SG-ATT is used.
        self.sparsify_mode = conf_graph.get('sparsify_mode', 'none')
        self.sparsify_topk = int(conf_graph.get('sparsify_topk', 0))
        self.sparsify_symmetric = bool(conf_graph.get('sparsify_symmetric', True))
        self.sparsify_keep_self = bool(conf_graph.get('sparsify_keep_self', True))
        self.task = conf_data['type']
        self.se_path = self._resolve_se_path()

        if self.graph.graph_num == 1:
            self.fusion_graph = False
            self.A_single = self.graph.get_graph(graph.use_graph[0])
        else:
            self.fusion_graph = True
            self.softmax = nn.Softmax(dim=1)

            if self.matrix_w:
                adj_w = nn.Parameter(torch.randn(self.graph.graph_num, self.graph.node_num, self.graph.node_num))
                adj_w_bias = nn.Parameter(torch.randn(self.graph.node_num, self.graph.node_num))
                self.adj_w_bias = nn.Parameter(adj_w_bias.to(self.device), requires_grad=True)
                self.linear = linear(5, 1)

            else:
                adj_w = nn.Parameter(torch.randn(1, self.graph.graph_num))

            self.adj_w = nn.Parameter(adj_w.to(self.device), requires_grad=True)
            self.used_graphs = self.graph.get_used_graphs()
            assert len(self.used_graphs) == self.graph.graph_num

        self.register_buffer('SE', self._load_spatial_embedding())

    def _resolve_se_path(self):
        mapping = {
            'pm25': os.path.join(PROJECT_ROOT, 'data', 'SE', 'se_pm25.csv'),
            'parking': os.path.join(PROJECT_ROOT, 'data', 'SE', 'se_parking.csv'),
            'bike': os.path.join(PROJECT_ROOT, 'data', 'SE', 'se_bike.csv'),
        }
        if self.task not in mapping:
            raise NotImplementedError('Unsupported task for spatial embedding: %s' % self.task)
        return mapping[self.task]

    def _load_spatial_embedding(self):
        if not os.path.exists(self.se_path):
            raise FileNotFoundError('Missing spatial embedding file: %s' % self.se_path)
        return torch.from_numpy(
            np.float32(pd.read_csv(self.se_path, header=None).values)
        ).to(self.device)

    def _apply_graph_sparsification(self, adj_matrix):
        if self.sparsify_mode == 'none':
            return adj_matrix

        num_nodes = adj_matrix.shape[0]
        keep_mask = torch.zeros_like(adj_matrix, dtype=torch.bool)

        if self.sparsify_keep_self:
            eye_mask = torch.eye(num_nodes, device=adj_matrix.device, dtype=torch.bool)
            keep_mask = keep_mask | eye_mask

        offdiag_mask = ~torch.eye(num_nodes, device=adj_matrix.device, dtype=torch.bool)

        if self.sparsify_mode in {'topk', 'topk_or_row_mean'} and self.sparsify_topk > 0:
            topk = min(self.sparsify_topk, max(num_nodes - 1, 1))
            offdiag_values = adj_matrix.masked_fill(~offdiag_mask, float('-inf'))
            _, topk_indices = torch.topk(offdiag_values, k=topk, dim=-1)
            row_indices = torch.arange(num_nodes, device=adj_matrix.device).unsqueeze(-1).expand_as(topk_indices)
            keep_mask[row_indices, topk_indices] = True

        if self.sparsify_mode in {'row_mean', 'topk_or_row_mean'}:
            valid_counts = offdiag_mask.sum(dim=-1).clamp(min=1)
            row_mean = (adj_matrix * offdiag_mask.float()).sum(dim=-1) / valid_counts
            keep_mask = keep_mask | (adj_matrix >= row_mean.unsqueeze(-1))
            keep_mask = keep_mask & (adj_matrix > 0)

        sparsified = adj_matrix * keep_mask.float()

        if self.sparsify_symmetric:
            sparsified = torch.maximum(sparsified, sparsified.T)

        return sparsified

    def forward(self):

        if self.graph.fix_weight:
            return self.graph.get_fix_weight()

        if self.fusion_graph:
            if not self.matrix_w:
                self.A_w = self.softmax(self.adj_w)[0]
                adj_list = [self.used_graphs[i] * self.A_w[i] for i in range(self.graph.graph_num)]
                self.adj_for_run = torch.sum(torch.stack(adj_list), dim=0)      
                # create a graph stack

            else:
                if self.attention:
                    W = torch.stack((self.used_graphs))
                    GE = W[:,:,0].permute(1, 0).unsqueeze(dim=2)
                    # generate graph embbeding

                    SGE = self.SGEmbedding(self.SE, GE)
                    W = self.FC_1(torch.unsqueeze(W.permute(1, 0, 2), -1))
                    W = self.SG_ATT(W, SGE)     
                    # multi-graph spatial attention

                    W = self.FC_2(W).squeeze(dim=-1)
                    W = torch.sum(self.adj_w * W.permute(1, 0, 2), dim=0)

                else:
                    W= torch.sum(self.adj_w * torch.stack(self.used_graphs), dim=0)
                act = nn.ReLU()
                W = act(W)
                W = self._apply_graph_sparsification(W)
                self.adj_for_run = W

        else:
            self.adj_for_run = self.A_single

        return self.adj_for_run
