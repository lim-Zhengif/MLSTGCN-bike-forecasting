# For relative import
import os
import sys

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_DIR)


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


PROJECT_ROOT = detect_project_root(PROJ_DIR)

import numpy as np
import torch
from torch.utils import data

from util import StandardScaler


class BikeGraph:
    def __init__(self, graph_dir, config_graph, device):
        self.device = torch.device(device)
        self.graph_dir = graph_dir

        use_graph = config_graph['use']
        fix_weight = config_graph['fix_weight']
        tempp_diag_zero = config_graph['tempp_diag_zero']
        hgaurban_graph_prior_path = config_graph.get('hgaurban_graph_prior_path') or ''
        cssg_rw_graph_prior_path = config_graph.get('cssg_rw_graph_prior_path') or ''

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
        self.extra_graphs = {}
        for graph_file in os.listdir(graph_dir):
            graph_name, ext = os.path.splitext(graph_file)
            if ext != '.npy' or not graph_name.startswith('od'):
                continue
            self.extra_graphs[graph_name] = torch.from_numpy(
                np.float32(np.load(os.path.join(graph_dir, graph_file)))
            ).to(self.device)
        if 'hgaurban' in use_graph:
            if not hgaurban_graph_prior_path:
                hgaurban_graph_prior_path = os.path.join(graph_dir, 'hgaurban_graph_prior.npy')
            if not os.path.isabs(hgaurban_graph_prior_path):
                candidate_paths = [
                    os.path.join(graph_dir, hgaurban_graph_prior_path),
                    os.path.join(PROJECT_ROOT, hgaurban_graph_prior_path),
                    os.path.join(PROJ_DIR, hgaurban_graph_prior_path),
                ]
                hgaurban_graph_prior_path = next(
                    (path for path in candidate_paths if os.path.exists(path)),
                    candidate_paths[0],
                )
            if not os.path.exists(hgaurban_graph_prior_path):
                raise FileNotFoundError(
                    'Missing HGAurban graph prior for graph_use=hgaurban: %s' % hgaurban_graph_prior_path
                )
            self.extra_graphs['hgaurban'] = torch.from_numpy(
                np.float32(np.load(hgaurban_graph_prior_path))
            ).to(self.device)
        if 'cssg_rw' in use_graph:
            if not cssg_rw_graph_prior_path:
                cssg_rw_graph_prior_path = os.path.join(graph_dir, 'cssg_rw_graph_prior.npy')
            if not os.path.isabs(cssg_rw_graph_prior_path):
                candidate_paths = [
                    os.path.join(graph_dir, cssg_rw_graph_prior_path),
                    os.path.join(PROJECT_ROOT, cssg_rw_graph_prior_path),
                    os.path.join(PROJ_DIR, cssg_rw_graph_prior_path),
                ]
                cssg_rw_graph_prior_path = next(
                    (path for path in candidate_paths if os.path.exists(path)),
                    candidate_paths[0],
                )
            if not os.path.exists(cssg_rw_graph_prior_path):
                raise FileNotFoundError(
                    'Missing CSSG RW graph prior for graph_use=cssg_rw: %s' % cssg_rw_graph_prior_path
                )
            self.extra_graphs['cssg_rw'] = torch.from_numpy(
                np.float32(np.load(cssg_rw_graph_prior_path))
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
        if name in self.extra_graphs:
            return self.extra_graphs[name]
        raise NotImplementedError


class Bike(data.Dataset):
    def __init__(self, data_dir, data_type, return_anchor=False):
        assert data_type in ['train', 'val', 'test']
        self.data_type = data_type
        self.return_anchor = bool(return_anchor)
        self._load_data(data_dir)

    def _load_data(self, data_dir):
        self.data = {}
        self.input_feature_cols = []
        self.history_feature_cols = []
        self.known_future_feature_cols = []
        self.log1p_feature_cols = []
        self.target_cols = []
        self.anchor_hours_by_split = {}
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(data_dir, category + '.npz'), allow_pickle=True)
            self.data['x_' + category] = cat_data['x']
            self.data['y_' + category] = cat_data['y']
            if 'anchor_hours' in cat_data:
                self.anchor_hours_by_split[category] = cat_data['anchor_hours'].astype(np.int64)
            else:
                self.anchor_hours_by_split[category] = np.full(len(self.data['x_' + category]), -1, dtype=np.int64)
            if category == 'train':
                if 'input_feature_cols' in cat_data:
                    self.input_feature_cols = [str(item) for item in cat_data['input_feature_cols'].tolist()]
                if 'history_feature_cols' in cat_data:
                    self.history_feature_cols = [str(item) for item in cat_data['history_feature_cols'].tolist()]
                if 'known_future_feature_cols' in cat_data:
                    self.known_future_feature_cols = [str(item) for item in cat_data['known_future_feature_cols'].tolist()]
                if 'log1p_feature_cols' in cat_data:
                    self.log1p_feature_cols = [str(item) for item in cat_data['log1p_feature_cols'].tolist()]
                if 'target_cols' in cat_data:
                    self.target_cols = [str(item) for item in cat_data['target_cols'].tolist()]

        x_train_for_stats = self.data['x_train'].astype(np.float32)
        y_train_for_stats = self.data['y_train'].astype(np.float32)
        feature_mean = x_train_for_stats.mean(axis=(0, 1, 2), keepdims=True)
        feature_std = x_train_for_stats.std(axis=(0, 1, 2), keepdims=True)
        feature_std = np.where(feature_std == 0, 1.0, feature_std)

        target_mean = y_train_for_stats.mean(axis=(0, 1, 2), keepdims=True)
        target_std = y_train_for_stats.std(axis=(0, 1, 2), keepdims=True)
        target_std = np.where(target_std == 0, 1.0, target_std)

        self.scaler = StandardScaler(
            mean=target_mean,
            std=target_std,
        )
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = target_std.astype(np.float32)
        for category in ['train', 'val', 'test']:
            self.data['x_' + category] = (
                self.data['x_' + category] - feature_mean
            ).astype(np.float32) / feature_std.astype(np.float32)
            self.data['y_' + category] = self.data['y_' + category].astype(np.float32)
        self.x = self.data['x_%s' % self.data_type]
        self.y = self.data['y_%s' % self.data_type]
        self.anchor_hours = self.anchor_hours_by_split[self.data_type]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.return_anchor:
            return self.x[index], self.y[index], self.anchor_hours[index]
        return self.x[index], self.y[index]
