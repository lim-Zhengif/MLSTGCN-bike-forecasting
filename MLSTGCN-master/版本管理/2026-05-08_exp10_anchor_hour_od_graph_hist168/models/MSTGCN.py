# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import eigs

from torch_geometric.utils import dense_to_sparse, get_laplacian, to_dense_adj


def cheb_polynomial_torch(L_tilde, K):
    N = L_tilde.shape[0]

    cheb_polynomials = [torch.eye(N).to(L_tilde.device), L_tilde.clone()]

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


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, fusiongraph, in_channels, out_channels, device):
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
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        adj_for_run = self.fusiongraph()

        edge_idx, edge_attr = dense_to_sparse(adj_for_run)
        # edge_idx_l, edge_attr_l = get_laplacian(edge_idx, edge_attr, 'sym')
        edge_idx_l, edge_attr_l = get_laplacian(edge_idx, edge_attr)

        L_tilde = to_dense_adj(edge_idx_l, edge_attr=edge_attr_l)[0]
        cheb_polynomials = cheb_polynomial_torch(L_tilde, self.K)

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


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
    ):
        super(MSTGCN_block, self).__init__()
        time_kernel_size = int(time_kernel_size)
        time_padding = time_kernel_size // 2
        self.cheb_conv = cheb_conv(K, fusiongraph, in_channels, nb_chev_filter, device)
        self.time_conv = nn.Conv2d(
            nb_chev_filter,
            nb_time_filter,
            kernel_size=(1, time_kernel_size),
            stride=(1, time_strides),
            padding=(0, time_padding),
        )
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)
        if channel_attention:
            self.channel_attention = ChannelAttention(nb_time_filter, reduction=channel_attention_reduction)
        else:
            self.channel_attention = nn.Identity()

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        # cheb gcn
        spatial_gcn = self.cheb_conv(x)  # (b,N,F,T)

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)  # (b,N,F,T)
        x_residual = self.channel_attention(x_residual)

        return x_residual


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
            )
            for _ in range(nb_block - 1)
        ])

        self.final_conv = nn.Conv2d(
            int(len_input / time_strides),
            num_for_predict * out_dim,
            kernel_size=(1, nb_time_filter),
        )

        self.DEVICE = DEVICE
        self.num_for_predict = num_for_predict
        self.out_dim = out_dim
        self.to(self.DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''

        x = self.input_adapter(x)
        x = x.permute((0, 2, 3, 1))
        x = self.input_channel_attention(x)

        for block in self.BlockList:
            x = block(x)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1]
        batch_size, _, num_nodes = output.shape
        output = output.view(batch_size, self.num_for_predict, self.out_dim, num_nodes)
        output = output.permute(0, 1, 3, 2)

        return output
