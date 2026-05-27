# For relative import
import os
import sys

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)


def detect_project_root(start_dir):
    current = os.path.abspath(start_dir)
    while True:
        if (
            os.path.isdir(os.path.join(current, '二月份数据处理'))
            and os.path.isdir(os.path.join(current, 'models'))
            and os.path.isdir(os.path.join(current, 'datasets'))
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.abspath(start_dir)
        current = parent


PROJECT_ROOT = detect_project_root(PROJ_DIR)

import argparse

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

try:
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    WandbLogger = None

try:
    import wandb
except ImportError:
    wandb = None

from datasets.bike import Bike, BikeGraph
from models.MSTGCN import MSTGCN_submodule
from models.fusiongraph import FusionGraphModel
from util import LightningMetric, masked_huber, masked_mae


parser = argparse.ArgumentParser()
parser.add_argument('--device', default='auto', help="Use 'auto', 'cpu', or a cuda device such as 'cuda:0'.")
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hist_len', type=int, default=7)
parser.add_argument('--pred_len', type=int, default=1)
parser.add_argument('--data_dir', default=os.path.join('data', 'temporal_data', 'bike'))
parser.add_argument('--graph_dir', default=os.path.join('data', 'graph', 'bike'))
parser.add_argument('--project', default='bike')
parser.add_argument('--logger', choices=['auto', 'csv', 'wandb'], default='auto')
parser.add_argument('--early_stop_patience', type=int, default=10)
parser.add_argument('--lr_patience', type=int, default=4)
parser.add_argument('--lr_factor', type=float, default=0.5)
parser.add_argument('--min_lr', type=float, default=1e-6)
parser.add_argument('--loss', choices=['mae', 'huber'], default='huber')
parser.add_argument('--huber_delta', type=float, default=3.0)
parser.add_argument('--mape_epsilon', type=float, default=1.0)
parser.add_argument('--graph_sparsify_mode', choices=['none', 'topk', 'row_mean', 'topk_or_row_mean'], default='topk')
parser.add_argument('--graph_topk', type=int, default=15)
parser.add_argument('--weekday_embed_dim', type=int, default=8)
args = parser.parse_args()

os.environ['WANDB_MODE'] = 'offline'

hyperparameter_defaults = dict(
    server=dict(
        gpu_id=0,
    ),
    graph=dict(
        use=['dist', 'neighb', 'distri', 'tempp', 'func'],
        fix_weight=False,
        tempp_diag_zero=True,
        matrix_weight=True,
        distri_type='exp',
        func_type='ours',
        attention=True,
        sparsify_mode=args.graph_sparsify_mode,
        sparsify_topk=args.graph_topk,
        sparsify_symmetric=True,
        sparsify_keep_self=True,
    ),
    model=dict(
        use='MSTGCN'
    ),
    data=dict(
        in_dim=1,
        out_dim=1,
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        type='bike',
    ),
    train=dict(
        seed=0,
        epoch=args.epochs,
        batch_size=args.batch_size,
        lr=1e-4,
        weight_decay=1e-4,
        M=24,
        d=6,
        bn_decay=0.1,
        loss=args.loss,
        huber_delta=args.huber_delta,
        mape_epsilon=args.mape_epsilon,
        weekday_embed_dim=args.weekday_embed_dim,
    )
)


def resolve_device(device_arg, gpu_id):
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda:%d' % gpu_id)
        return torch.device('cpu')
    return torch.device(device_arg)


def resolve_project_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(base_dir, path_value)


def create_logger(config):
    if args.logger == 'csv':
        return hyperparameter_defaults, CSVLogger(save_dir=os.path.join(PROJECT_ROOT, 'logs'), name=args.project)

    if wandb is not None and WandbLogger is not None:
        try:
            wandb.init(config=hyperparameter_defaults, project=args.project)
            return wandb.config, WandbLogger(project=args.project)
        except Exception as exc:
            if args.logger == 'wandb':
                raise
            print('Wandb init failed, falling back to CSVLogger:', exc)

    if args.logger == 'wandb':
        raise RuntimeError('Wandb logger requested, but wandb is unavailable in the current environment.')

    return hyperparameter_defaults, CSVLogger(save_dir=os.path.join(PROJECT_ROOT, 'logs'), name=args.project)


config, lightning_logger = create_logger(hyperparameter_defaults)
pl.utilities.seed.seed_everything(config['train']['seed'])

gpu_id = config['server']['gpu_id']
device = resolve_device(args.device, gpu_id)
args.data_dir = resolve_project_path(PROJECT_ROOT, args.data_dir)
args.graph_dir = resolve_project_path(PROJECT_ROOT, args.graph_dir)

if not os.path.exists(args.data_dir):
    raise FileNotFoundError(
        'Missing bike temporal data directory: %s. Run prepare_bike_training_data.py first.' % args.data_dir
    )
if not os.path.exists(args.graph_dir):
    raise FileNotFoundError(
        'Missing bike graph directory: %s. Run prepare_bike_training_data.py first.' % args.graph_dir
    )

graph = BikeGraph(args.graph_dir, config['graph'], device)
train_set = Bike(args.data_dir, 'train')
val_set = Bike(args.data_dir, 'val')
test_set = Bike(args.data_dir, 'test')
scaler = train_set.scaler

# Auto-sync model input/output dimensions with generated temporal files.
config['data']['in_dim'] = int(train_set.x.shape[-1])
config['data']['out_dim'] = int(train_set.y.shape[-1])
config['data']['hist_len'] = int(train_set.x.shape[1])
config['data']['pred_len'] = int(train_set.y.shape[1])


def build_categorical_feature_configs(dataset, weekday_embed_dim):
    if weekday_embed_dim <= 0:
        return []
    configs = []
    feature_cols = getattr(dataset, 'input_feature_cols', [])
    if not feature_cols:
        return configs

    feature_mean = dataset.feature_mean.reshape(-1)
    feature_std = dataset.feature_std.reshape(-1)
    categorical_map = {
        '星期几': 7,
        'future_星期几': 7,
    }
    for feature_idx, feature_name in enumerate(feature_cols):
        if feature_name not in categorical_map:
            continue
        configs.append(
            {
                'index': feature_idx,
                'num_embeddings': categorical_map[feature_name],
                'embedding_dim': weekday_embed_dim,
                'mean': float(feature_mean[feature_idx]),
                'std': float(feature_std[feature_idx]),
                'name': feature_name,
            }
        )
    return configs


categorical_feature_configs = build_categorical_feature_configs(
    train_set,
    weekday_embed_dim=config['train']['weekday_embed_dim'],
)


class LightningData(LightningDataModule):
    def __init__(self, train_set, val_set, test_set):
        super().__init__()
        self.batch_size = config['train']['batch_size']
        self.pin_memory = device.type == 'cuda'
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


class LightningModel(LightningModule):
    def __init__(self, scaler, fusiongraph, categorical_feature_configs):
        super().__init__()
        self.scaler = scaler
        self.fusiongraph = fusiongraph
        self.metric_lightning = LightningMetric(mape_eps=args.mape_epsilon)
        self.loss_name = args.loss
        self.huber_delta = args.huber_delta
        if self.loss_name == 'huber':
            self.loss = nn.SmoothL1Loss(beta=self.huber_delta, reduction='mean')
        else:
            self.loss = nn.L1Loss(reduction='mean')

        self.model = MSTGCN_submodule(
            device,
            fusiongraph,
            config['data']['in_dim'],
            config['data']['hist_len'],
            config['data']['pred_len'],
            config['data']['out_dim'],
            categorical_feature_configs=categorical_feature_configs,
        )
        for param in self.model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param)

        self.log_dict(config)

    def forward(self, x):
        return self.model(x)

    def _apply_output_constraint(self, y_hat):
        # Bike demand targets are counts, so we enforce non-negative predictions in raw space.
        return F.softplus(y_hat, beta=5.0)

    def _compute_loss(self, y_hat, y):
        if self.loss_name == 'huber':
            return masked_huber(y_hat, y, delta=self.huber_delta)
        return masked_mae(y_hat, y)

    def _run_model(self, batch):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = self(x)
        y_hat = self.scaler.inverse_transform(y_hat)
        y_hat = self._apply_output_constraint(y_hat)
        loss = self._compute_loss(y_hat, y)
        mae_loss = masked_mae(y_hat, y)
        return y_hat, y, loss, mae_loss

    def training_step(self, batch, batch_idx):
        y_hat, y, loss, mae_loss = self._run_model(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', mae_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, loss, mae_loss = self._run_model(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', mae_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        y_hat, y, loss, mae_loss = self._run_model(batch)
        self.metric_lightning(y_hat.cpu(), y.cpu())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mae', mae_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def test_epoch_end(self, outputs):
        self.log_dict(self.metric_lightning.compute())

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_mae',
                'interval': 'epoch',
                'frequency': 1,
            },
        }


def main():
    fusiongraph = FusionGraphModel(
        graph,
        device,
        config['graph'],
        config['data'],
        config['train']['M'],
        config['train']['d'],
        config['train']['bn_decay'],
    )
    lightning_data = LightningData(train_set, val_set, test_set)
    lightning_model = LightningModel(scaler, fusiongraph, categorical_feature_configs)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mae',
        mode='min',
        save_top_k=1,
        save_last=True,
        filename='best-{epoch:02d}-{val_mae:.4f}',
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_mae',
        mode='min',
        patience=args.early_stop_patience,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer_kwargs = dict(
        logger=lightning_logger,
        max_epochs=config['train']['epoch'],
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
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
    best_model_path = checkpoint_callback.best_model_path
    try:
        trainer.test(datamodule=lightning_data, ckpt_path='best')
    except TypeError:
        if best_model_path:
            trainer.test(ckpt_path=best_model_path, datamodule=lightning_data)
        else:
            trainer.test(lightning_model, datamodule=lightning_data)
    print('Bike graph use:', config['graph']['use'])
    print('Bike data:', config['data'])
    print('Project root:', PROJECT_ROOT)
    print('Data dir:', args.data_dir)
    print('Graph dir:', args.graph_dir)
    print('Applied log1p input feature cols:', getattr(train_set, 'log1p_feature_cols', []))
    print('Categorical feature configs:', categorical_feature_configs)
    print('Device:', str(device))
    print('Best checkpoint:', best_model_path if best_model_path else 'None')


if __name__ == '__main__':
    main()
