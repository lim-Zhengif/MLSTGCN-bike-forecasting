import torch
import torch.nn as nn
import torch.nn.functional as F


class AVWGCN(nn.Module):
    """Adaptive vertex-wise graph convolution used by AGCRN."""

    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.cheb_k = int(cheb_k)
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, self.cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, dim_out))
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.xavier_uniform_(self.bias_pool)

    def forward(self, x, node_embeddings):
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num, device=x.device), supports]
        for _ in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set[: self.cheb_k], dim=0)

        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        return torch.einsum("bnki,nkio->bno", x_g, weights) + bias


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.node_num = int(node_num)
        self.hidden_dim = int(dim_out)
        self.gate = AVWGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        return r * state + (1.0 - r) * hc

    def init_hidden_state(self, batch_size, device):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, device=device)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super().__init__()
        self.node_num = int(node_num)
        self.input_dim = int(dim_in)
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        cells = [AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim)]
        for _ in range(1, self.num_layers):
            cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))
        self.dcrnn_cells = nn.ModuleList(cells)

    def forward(self, x, node_embeddings):
        # x: [B, T, N, F]
        if x.shape[2] != self.node_num or x.shape[3] != self.input_dim:
            raise ValueError("Unexpected AGCRN input shape: %s" % (tuple(x.shape),))

        current_inputs = x
        output_hidden = []
        batch_size = x.shape[0]
        for layer_idx, cell in enumerate(self.dcrnn_cells):
            state = cell.init_hidden_state(batch_size, x.device)
            inner_states = []
            for step in range(current_inputs.shape[1]):
                state = cell(current_inputs[:, step, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden


class AGCRNBaseline(nn.Module):
    """AGCRN adapted to `[B, hist_len, N, in_dim] -> [B, pred_len, N, out_dim]`."""

    def __init__(
        self,
        num_nodes,
        input_dim,
        rnn_units,
        output_dim,
        horizon,
        num_layers=1,
        cheb_k=2,
        embed_dim=10,
        dropout=0.0,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(rnn_units)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)
        self.num_layers = int(num_layers)
        self.dropout = nn.Dropout(float(dropout))

        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, int(embed_dim)))
        self.encoder = AVWDCRNN(
            self.num_nodes,
            self.input_dim,
            self.hidden_dim,
            int(cheb_k),
            int(embed_dim),
            self.num_layers,
        )
        self.end_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.horizon * self.output_dim,
            kernel_size=(1, self.hidden_dim),
            bias=True,
        )

    def forward(self, x):
        output, _ = self.encoder(x, self.node_embeddings)
        output = self.dropout(output[:, -1:, :, :])
        output = self.end_conv(output)
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_nodes)
        return output.permute(0, 1, 3, 2)
