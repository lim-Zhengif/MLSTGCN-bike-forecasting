# For relative import
import os
import sys

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)
import argparse

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
try:
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    WandbLogger = None
try:
    import wandb
except ImportError:
    wandb = None

from models.MSTGCN import MSTGCN_submodule
from models.fusiongraph import FusionGraphModel

from datasets.air import *
from datasets.bike import Bike, BikeGraph
from util import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_type', default='pm25', choices=['pm25', 'bike'])
parser.add_argument('--device', default='auto', help="Use 'auto', 'cpu', or a cuda device such as 'cuda:0'.")
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--hist_len', type=int, default=None)
parser.add_argument('--pred_len', type=int, default=None)
args = parser.parse_args()

os.environ['WANDB_MODE'] = 'offline'                        # select one from ['online','offline']

hyperparameter_defaults = dict(
    server=dict(
        gpu_id=0,
    ),
    graph=dict(
        use=['dist', 'neighb', 'distri','tempp', 'func'],   # select no more than five graphs from ['dist', 'neighb', 'distri', 'tempp', 'func'].
        fix_weight=False,                                   # if True, the weight of each graph is fixed.
        tempp_diag_zero=True,                               # if Ture, the values of temporal pattern similarity weight matrix turn to 0.
        matrix_weight=True,                                 # if True, turn the weight matrices trainable.
        distri_type='exp',                                  # select one from ['kl', 'ws', 'exp']: 'kl' is for Kullback-Leibler divergence, 'ws' is for Wasserstein, and 'exp' is for expotential fitting
        func_type='ours',                                   # select one from ['ours', 'others'], 'others' is for the functionality graph proposed by "Spatiotemporal Multi-Graph Convolution Network for Ride-hailing Demand Forecasting"
        attention=True,                                    # if True, the SG-ATT is used.
    ),
    model=dict(
        # TODO: check batch_size
        use='MSTGCN'                                        
    ),
    data=dict(
        in_dim=1,
        out_dim=1,
        hist_len=24,
        pred_len=24,
        type=args.data_type,

    ),
    STMGCN=dict(
        use_fusion_graph=True,
    ),
    train=dict(
        seed=0,
        epoch=40,
        batch_size=32,
        lr=1e-4,
        weight_decay=1e-4,
        M=24,                                                
        d=6,                                                
        bn_decay=0.1,
    )
)

if args.epochs is not None:
    hyperparameter_defaults['train']['epoch'] = args.epochs
if args.batch_size is not None:
    hyperparameter_defaults['train']['batch_size'] = args.batch_size
if args.hist_len is not None:
    hyperparameter_defaults['data']['hist_len'] = args.hist_len
if args.pred_len is not None:
    hyperparameter_defaults['data']['pred_len'] = args.pred_len


def resolve_device(device_arg, gpu_id):
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda:%d' % gpu_id)
        return torch.device('cpu')
    return torch.device(device_arg)


def create_logger(config):
    project_name = config['data']['type']
    if wandb is not None and WandbLogger is not None:
        wandb.init(config=hyperparameter_defaults, project=project_name)
        return wandb.config, WandbLogger(project=project_name)
    return hyperparameter_defaults, CSVLogger(save_dir='logs', name=project_name)


config, lightning_logger = create_logger(hyperparameter_defaults)

pl.utilities.seed.seed_everything(config['train']['seed'])

gpu_id = config['server']['gpu_id']
device = resolve_device(args.device, gpu_id)

root_dir = 'data'
if config['data']['type'] == 'pm25':
    data_dir = os.path.join(root_dir, 'temporal_data', 'pm25')
    graph_dir = os.path.join(root_dir, 'graph', 'pm25')
    graph = AirGraph(graph_dir, config['graph'], device)
    train_set = Air(data_dir, 'train')
    val_set = Air(data_dir, 'val')
    test_set = Air(data_dir, 'test')
elif config['data']['type'] == 'bike':
    data_dir = os.path.join(root_dir, 'temporal_data', 'bike')
    graph_dir = os.path.join(root_dir, 'graph', 'bike')
    graph = BikeGraph(graph_dir, config['graph'], device)
    train_set = Bike(data_dir, 'train')
    val_set = Bike(data_dir, 'val')
    test_set = Bike(data_dir, 'test')
else:
    raise NotImplementedError

scaler = train_set.scaler

# Auto-sync with prepared temporal files (supports multi-feature bike input).
config['data']['in_dim'] = int(train_set.x.shape[-1])
config['data']['out_dim'] = int(train_set.y.shape[-1])
config['data']['hist_len'] = int(train_set.x.shape[1])
config['data']['pred_len'] = int(train_set.y.shape[1])

class LightningData(LightningDataModule):
    def __init__(self, train_set, val_set, test_set):
        super().__init__()
        self.batch_size = config['train']['batch_size']
        self.pin_memory = device.type == 'cuda'
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                   pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=0,
                                 pin_memory=self.pin_memory, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0,
                                  pin_memory=self.pin_memory, drop_last=False)

class LightningModel(LightningModule):
    def __init__(self, scaler, fusiongraph):
        super().__init__()

        self.scaler = scaler
        self.fusiongraph = fusiongraph

        self.metric_lightning = LightningMetric()

        self.loss = nn.L1Loss(reduction='mean')

        if config['model']['use'] == 'ASTGCN':
            raise NotImplementedError('ASTGCN_submodule is not included in this repository.')
        elif config['model']['use'] == 'MSTGCN':
            self.model = MSTGCN_submodule(
                device,
                fusiongraph,
                config['data']['in_dim'],
                config['data']['hist_len'],
                config['data']['pred_len'],
                config['data']['out_dim'],
            )
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)
        else:
            raise NotImplementedError

        self.log_dict(config)

    def forward(self, x):
        return self.model(x)

    def _run_model(self, batch):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = self(x)

        y_hat = self.scaler.inverse_transform(y_hat)

        loss = masked_mae(y_hat, y, 0.0)

        return y_hat, y, loss

    def training_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        self.metric_lightning(y_hat.cpu(), y.cpu())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):
        test_metric_dict = self.metric_lightning.compute()
        self.log_dict(test_metric_dict)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])


def main():
    fusiongraph = FusionGraphModel(graph, device, config['graph'], config['data'], config['train']['M'], config['train']['d'], config['train']['bn_decay'])

    lightning_data = LightningData(train_set, val_set, test_set)

    lightning_model = LightningModel(scaler, fusiongraph)

    trainer_kwargs = dict(
        logger=lightning_logger,
        max_epochs=config['train']['epoch'],
    )
    if int(pl.__version__.split('.')[0]) >= 2:
        trainer_kwargs.update({
            'accelerator': 'gpu' if device.type == 'cuda' else 'cpu',
            'devices': 1,
        })
    else:
        trainer_kwargs.update({
            'gpus': [gpu_id] if device.type == 'cuda' else 0,
        })
    trainer = Trainer(**trainer_kwargs)

    trainer.fit(lightning_model, lightning_data)
    trainer.test(lightning_model, datamodule=lightning_data)
    print('Graph USE', config['graph']['use'])
    print('Data', config['data'])
    print('Device', str(device))


if __name__ == '__main__':
    main()
