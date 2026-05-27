# For relative import
import os
import sys

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_DIR)

import numpy as np
import torch
from torch.utils import data

from util import StandardScaler


class BikeGraph:
    def __init__(self, graph_dir, config_graph, device):
        self.device = torch.device(device)

        use_graph = config_graph['use']
        fix_weight = config_graph['fix_weight']
        tempp_diag_zero = config_graph['tempp_diag_zero']

        self.A_dist = torch.from_numpy(
            np.float32(np.load(os.path.join(graph_dir, 'dist.npy')))
        ).to(self.device)
        self.A_neighb = torch.from_numpy(
            np.float32(np.load(os.path.join(graph_dir, 'neigh.npy')))
        ).to(self.device)
        self.A_func = torch.from_numpy(
            np.float32(np.load(os.path.join(graph_dir, 'func.npy')))
        ).to(self.device)
        self.A_distri = torch.from_numpy(
            np.float32(np.load(os.path.join(graph_dir, 'bike_heuristic.npy')))
        ).to(self.device)
        self.A_tempp = torch.from_numpy(
            np.float32(np.load(os.path.join(graph_dir, 'tempp_bike.npy')))
        ).to(self.device)

        self.node_num = self.A_dist.shape[0]

        if tempp_diag_zero:
            self.A_tempp.fill_diagonal_(0)

        self.use_graph = use_graph
        self.fix_weight = fix_weight
        self.graph_num = len(use_graph)

    def get_used_graphs(self):
        return [self.get_graph(name) for name in self.use_graph]

    def get_fix_weight(self):
        return (
            self.A_dist * 0.2
            + self.A_neighb * 0.2
            + self.A_distri * 0.2
            + self.A_tempp * 0.2
            + self.A_func * 0.2
        )

    def get_graph(self, name):
        if name == 'dist':
            return self.A_dist
        if name == 'neighb':
            return self.A_neighb
        if name == 'distri':
            return self.A_distri
        if name == 'tempp':
            return self.A_tempp
        if name == 'func':
            return self.A_func
        raise NotImplementedError


class Bike(data.Dataset):
    def __init__(self, data_dir, data_type):
        assert data_type in ['train', 'val', 'test']
        self.data_type = data_type
        self._load_data(data_dir)

    def _load_data(self, data_dir):
        self.data = {}
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(data_dir, category + '.npz'), allow_pickle=True)
            self.data['x_' + category] = cat_data['x'].astype(np.float32)
            self.data['y_' + category] = cat_data['y'].astype(np.float32)

        feature_mean = self.data['x_train'].mean(axis=(0, 1, 2), keepdims=True)
        feature_std = self.data['x_train'].std(axis=(0, 1, 2), keepdims=True)
        feature_std = np.where(feature_std == 0, 1.0, feature_std)

        target_mean = self.data['y_train'].mean(axis=(0, 1, 2), keepdims=True)
        target_std = self.data['y_train'].std(axis=(0, 1, 2), keepdims=True)
        target_std = np.where(target_std == 0, 1.0, target_std)

        self.scaler = StandardScaler(
            mean=target_mean,
            std=target_std,
        )
        for category in ['train', 'val', 'test']:
            self.data['x_' + category] = (
                self.data['x_' + category] - feature_mean
            ) / feature_std
        self.x = self.data['x_%s' % self.data_type]
        self.y = self.data['y_%s' % self.data_type]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
