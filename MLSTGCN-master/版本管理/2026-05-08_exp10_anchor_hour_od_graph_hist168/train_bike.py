# For relative import
import os
import sys

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_DIR)


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
import json
import subprocess

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
from analysis_result_utils import build_analysis_task_dir, build_and_save_analysis_registry, ensure_analysis_dir


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
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--lr_patience', type=int, default=4)
parser.add_argument('--lr_factor', type=float, default=0.5)
parser.add_argument('--min_lr', type=float, default=1e-6)

# Training/runtime controls exposed for fast CLI experimentation.
parser.add_argument('--seed', type=int, default=0, help='Random seed for training reproducibility.')
parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker count. Increase if data loading becomes a bottleneck.')
parser.add_argument('--precision', default='32', help="Lightning precision mode, e.g. '32', '16-mixed', 'bf16-mixed'.")
parser.add_argument('--grad_clip_val', type=float, default=0.0, help='Gradient clipping threshold. Set > 0 to stabilize harder runs.')
parser.add_argument(
    '--monitor_metric',
    choices=['val_mae_epoch', 'val_loss_epoch'],
    default='val_mae_epoch',
    help='Metric used for best-checkpoint selection, early stopping, and LR scheduler monitoring.',
)
parser.add_argument(
    '--monitor_mode',
    choices=['min', 'max'],
    default='min',
    help="Optimization direction for --monitor_metric. Forecast errors should usually use 'min'.",
)
parser.add_argument('--loss', choices=['mae', 'huber'], default='huber')
parser.add_argument('--huber_delta', type=float, default=3.0)
parser.add_argument('--mape_epsilon', type=float, default=1.0)
parser.add_argument('--graph_sparsify_mode', choices=['none', 'topk', 'row_mean', 'topk_or_row_mean'], default='topk')
parser.add_argument('--graph_topk', type=int, default=15)

# Graph sparsification knobs. These are especially useful when comparing top-k style graph pruning variants.
parser.add_argument(
    '--graph_sparsify_symmetric',
    default='true',
    help="Use 'true' or 'false' to force the sparsified graph to be symmetric.",
)
parser.add_argument(
    '--graph_sparsify_keep_self',
    default='true',
    help="Use 'true' or 'false' to keep self-loops after graph sparsification.",
)
parser.add_argument('--weekday_embed_dim', type=int, default=8)

# Output non-negativity control for bike-count prediction heads.
parser.add_argument(
    '--output_constraint',
    choices=['softplus', 'relu', 'none'],
    default='softplus',
    help="Post-process model outputs with 'softplus', 'relu', or no constraint.",
)
parser.add_argument(
    '--output_softplus_beta',
    type=float,
    default=5.0,
    help='Softplus sharpness used when --output_constraint=softplus. Larger means closer to ReLU.',
)
parser.add_argument('--graph_use', default='dist,neighb,distri,tempp,func')
parser.add_argument('--graph_attention', default='true', help="Use 'true' or 'false' to enable fusion-graph attention.")
parser.add_argument('--matrix_weight', default='true', help="Use 'true' or 'false' to enable trainable graph-specific matrices.")
parser.add_argument('--graph_fix_weight', default='false', help="Use 'true' or 'false' to freeze graph weights from prebuilt graphs.")
parser.add_argument('--tempp_diag_zero', default='true', help="Use 'true' or 'false' to zero the diagonal of temporal proximity graph.")
parser.add_argument('--graph_distri_type', default='exp')
parser.add_argument('--graph_func_type', default='ours')
parser.add_argument('--fusion_heads', type=int, default=24)
parser.add_argument('--fusion_head_dim', type=int, default=6)
parser.add_argument('--bn_decay', type=float, default=0.1)
parser.add_argument('--cheb_k', type=int, default=3)
parser.add_argument('--nb_block', type=int, default=2)
parser.add_argument('--nb_chev_filter', type=int, default=64)
parser.add_argument('--nb_time_filter', type=int, default=64)
parser.add_argument(
    '--time_kernel_size',
    type=int,
    default=3,
    help='Temporal convolution kernel size inside MSTGCN blocks. Must be an odd number such as 3, 5, or 7.',
)
parser.add_argument('--channel_attention', default='false', help="Use 'true' or 'false' to enable channel attention in MSTGCN.")
parser.add_argument('--channel_attention_reduction', type=int, default=4)
args = parser.parse_args()


def parse_bool_arg(value, flag_name):
    normalized = str(value).strip().lower()
    mapping = {
        '1': True,
        'true': True,
        'yes': True,
        'y': True,
        '0': False,
        'false': False,
        'no': False,
        'n': False,
    }
    if normalized not in mapping:
        parser.error('%s only accepts true/false/1/0/yes/no, got: %s' % (flag_name, value))
    return mapping[normalized]


def parse_graph_use_arg(value):
    graph_use = [item.strip() for item in str(value).split(',') if item.strip()]
    if not graph_use:
        parser.error('--graph_use must include at least one graph name.')
    allowed_graphs = {'dist', 'neighb', 'distri', 'tempp', 'func', 'od00', 'od06', 'od12', 'od16', 'od20'}
    unknown_graphs = [item for item in graph_use if item not in allowed_graphs]
    if unknown_graphs:
        parser.error('Unsupported graph names in --graph_use: %s' % ', '.join(unknown_graphs))
    return graph_use


args.graph_attention = parse_bool_arg(args.graph_attention, '--graph_attention')
args.matrix_weight = parse_bool_arg(args.matrix_weight, '--matrix_weight')
args.graph_fix_weight = parse_bool_arg(args.graph_fix_weight, '--graph_fix_weight')
args.tempp_diag_zero = parse_bool_arg(args.tempp_diag_zero, '--tempp_diag_zero')
args.graph_sparsify_symmetric = parse_bool_arg(args.graph_sparsify_symmetric, '--graph_sparsify_symmetric')
args.graph_sparsify_keep_self = parse_bool_arg(args.graph_sparsify_keep_self, '--graph_sparsify_keep_self')
args.channel_attention = parse_bool_arg(args.channel_attention, '--channel_attention')
args.graph_use = parse_graph_use_arg(args.graph_use)

if args.graph_topk < 0:
    parser.error('--graph_topk must be >= 0.')
if args.num_workers < 0:
    parser.error('--num_workers must be >= 0.')
if args.weekday_embed_dim < 0:
    parser.error('--weekday_embed_dim must be >= 0.')
if args.fusion_heads <= 0:
    parser.error('--fusion_heads must be > 0.')
if args.fusion_head_dim <= 0:
    parser.error('--fusion_head_dim must be > 0.')
if args.bn_decay <= 0:
    parser.error('--bn_decay must be > 0.')
if args.cheb_k <= 0:
    parser.error('--cheb_k must be > 0.')
if args.nb_block <= 0:
    parser.error('--nb_block must be > 0.')
if args.nb_chev_filter <= 0:
    parser.error('--nb_chev_filter must be > 0.')
if args.nb_time_filter <= 0:
    parser.error('--nb_time_filter must be > 0.')
if args.time_kernel_size <= 0:
    parser.error('--time_kernel_size must be > 0.')
if args.time_kernel_size % 2 == 0:
    parser.error('--time_kernel_size must be an odd number so time padding stays centered.')
if args.channel_attention_reduction <= 0:
    parser.error('--channel_attention_reduction must be > 0.')
if args.output_softplus_beta <= 0:
    parser.error('--output_softplus_beta must be > 0.')
if args.grad_clip_val < 0:
    parser.error('--grad_clip_val must be >= 0.')


def parse_precision_arg(value):
    normalized = str(value).strip().lower()
    precision_map = {
        '16': 16,
        '32': 32,
        '64': 64,
        '16-mixed': '16-mixed',
        'bf16-mixed': 'bf16-mixed',
        'bf16': 'bf16',
        '32-true': '32-true',
        '64-true': '64-true',
    }
    if normalized not in precision_map:
        parser.error("--precision must be one of: 16, 32, 64, 16-mixed, bf16-mixed, bf16, 32-true, 64-true")
    return precision_map[normalized]


args.precision = parse_precision_arg(args.precision)

os.environ['WANDB_MODE'] = 'offline'

hyperparameter_defaults = dict(
    server=dict(
        gpu_id=0,
    ),
    graph=dict(
        use=args.graph_use,
        fix_weight=args.graph_fix_weight,
        tempp_diag_zero=args.tempp_diag_zero,
        matrix_weight=args.matrix_weight,
        distri_type=args.graph_distri_type,
        func_type=args.graph_func_type,
        attention=args.graph_attention,
        sparsify_mode=args.graph_sparsify_mode,
        sparsify_topk=args.graph_topk,
        sparsify_symmetric=args.graph_sparsify_symmetric,
        sparsify_keep_self=args.graph_sparsify_keep_self,
    ),
    model=dict(
        use='MSTGCN',
        cheb_k=args.cheb_k,
        nb_block=args.nb_block,
        nb_chev_filter=args.nb_chev_filter,
        nb_time_filter=args.nb_time_filter,
        time_kernel_size=args.time_kernel_size,
        channel_attention=args.channel_attention,
        channel_attention_reduction=args.channel_attention_reduction,
    ),
    data=dict(
        in_dim=1,
        out_dim=1,
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        type='bike',
    ),
    train=dict(
        seed=args.seed,
        epoch=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        M=args.fusion_heads,
        d=args.fusion_head_dim,
        bn_decay=args.bn_decay,
        loss=args.loss,
        huber_delta=args.huber_delta,
        mape_epsilon=args.mape_epsilon,
        weekday_embed_dim=args.weekday_embed_dim,
        output_constraint=args.output_constraint,
        output_softplus_beta=args.output_softplus_beta,
        precision=args.precision,
        grad_clip_val=args.grad_clip_val,
        monitor_metric=args.monitor_metric,
        monitor_mode=args.monitor_mode,
    )
)


LATEST_TRAINING_RECORD_NAME = 'latest_hourly_training_checkpoint.json'
TRAINING_SUMMARY_NAME = 'training_summary.json'
TRAIN_ENTRY_SCRIPT_ENV = 'MLSTGCN_TRAIN_ENTRY_SCRIPT'
TRAIN_ENTRY_ARGS_ENV = 'MLSTGCN_TRAIN_ENTRY_ARGS_JSON'
TRAIN_RECORD_DIR_ENV = 'MLSTGCN_TRAIN_RECORD_DIR'
TRAIN_VERSION_TAG_ENV = 'MLSTGCN_VERSION_TAG_OVERRIDE'


def get_version_tag():
    return os.environ.get(TRAIN_VERSION_TAG_ENV) or os.path.basename(PROJ_DIR)


def get_training_record_dir():
    override_dir = os.environ.get(TRAIN_RECORD_DIR_ENV)
    if override_dir:
        os.makedirs(override_dir, exist_ok=True)
        return override_dir
    return PROJ_DIR


def get_latest_training_record_path():
    return os.path.join(get_training_record_dir(), LATEST_TRAINING_RECORD_NAME)


def get_training_summary_dir():
    return build_analysis_task_dir(
        project_root=PROJECT_ROOT,
        version_tag=get_version_tag(),
        task_type='training_result',
        extra_tag=args.project,
    )


def _to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except TypeError:
        return float(value.item())


def _load_json_list_from_env(env_name):
    raw_value = os.environ.get(env_name)
    if not raw_value:
        return None
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, list) else None


def collect_command_metadata():
    entry_script = os.environ.get(TRAIN_ENTRY_SCRIPT_ENV) or sys.argv[0]
    entry_args = _load_json_list_from_env(TRAIN_ENTRY_ARGS_ENV) or list(sys.argv[1:])
    resolved_argv = list(sys.argv)
    python_exec = os.path.basename(sys.executable) or 'python'
    return {
        'python_executable': python_exec,
        'entry_script': entry_script,
        'entry_args': entry_args,
        'entry_command': subprocess.list2cmdline([python_exec, entry_script] + entry_args),
        'resolved_train_argv': resolved_argv,
        'resolved_train_command': subprocess.list2cmdline([python_exec] + resolved_argv),
    }


COMMAND_METADATA = collect_command_metadata()


def write_latest_training_record(best_checkpoint, last_checkpoint, logger_obj):
    preferred_checkpoint = best_checkpoint or last_checkpoint
    if not preferred_checkpoint:
        return None

    record = {
        'project': args.project,
        'project_root': PROJECT_ROOT,
        'version_tag': get_version_tag(),
        'preferred_checkpoint': preferred_checkpoint,
        'best_checkpoint': best_checkpoint or None,
        'last_checkpoint': last_checkpoint or None,
        'logger_dir': getattr(logger_obj, 'log_dir', None),
        'data_dir': args.data_dir,
        'graph_dir': args.graph_dir,
        'python_executable': COMMAND_METADATA.get('python_executable'),
        'entry_script': COMMAND_METADATA.get('entry_script'),
        'entry_args': COMMAND_METADATA.get('entry_args'),
        'entry_command': COMMAND_METADATA.get('entry_command'),
        'resolved_train_argv': COMMAND_METADATA.get('resolved_train_argv'),
        'resolved_train_command': COMMAND_METADATA.get('resolved_train_command'),
    }
    record_path = get_latest_training_record_path()
    with open(record_path, 'w', encoding='utf-8') as fp:
        json.dump(record, fp, ensure_ascii=False, indent=2)
    return record_path


def write_training_summary(best_checkpoint, last_checkpoint, logger_obj, best_val_mae_epoch, test_results):
    summary_dir = ensure_analysis_dir(get_training_summary_dir())
    summary_path = os.path.join(summary_dir, TRAINING_SUMMARY_NAME)
    test_result = test_results[0] if test_results else {}
    summary = {
        'project': args.project,
        'version_tag': get_version_tag(),
        'log_dir': getattr(logger_obj, 'log_dir', None),
        'data_dir': args.data_dir,
        'graph_dir': args.graph_dir,
        'python_executable': COMMAND_METADATA.get('python_executable'),
        'entry_script': COMMAND_METADATA.get('entry_script'),
        'entry_args': COMMAND_METADATA.get('entry_args'),
        'entry_command': COMMAND_METADATA.get('entry_command'),
        'resolved_train_argv': COMMAND_METADATA.get('resolved_train_argv'),
        'resolved_train_command': COMMAND_METADATA.get('resolved_train_command'),
        'best_checkpoint': best_checkpoint or None,
        'last_checkpoint': last_checkpoint or None,
        'best_val_mae_epoch': _to_float_or_none(best_val_mae_epoch),
        'test_mae': _to_float_or_none(test_result.get('test_mae')),
        'test_loss': _to_float_or_none(test_result.get('test_loss')),
    }
    with open(summary_path, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    return summary_path


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
        self.num_workers = config['train']['num_workers']
        self.pin_memory = device.type == 'cuda'
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
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
            cheb_k=config['model']['cheb_k'],
            nb_block=config['model']['nb_block'],
            nb_chev_filter=config['model']['nb_chev_filter'],
            nb_time_filter=config['model']['nb_time_filter'],
            time_kernel_size=config['model']['time_kernel_size'],
            channel_attention=config['model']['channel_attention'],
            channel_attention_reduction=config['model']['channel_attention_reduction'],
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
        # Bike demand targets are counts, so we optionally enforce non-negative predictions in raw space.
        output_constraint = config['train']['output_constraint']
        if output_constraint == 'none':
            return y_hat
        if output_constraint == 'relu':
            return F.relu(y_hat)
        return F.softplus(y_hat, beta=config['train']['output_softplus_beta'])

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
        self.log('val_loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('val_mae_step', mae_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('val_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae_epoch', mae_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

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
                'monitor': config['train']['monitor_metric'],
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
    checkpoint_filename = 'best-{epoch:02d}-' + args.monitor_metric + '={' + args.monitor_metric + ':.4f}'
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor_metric,
        mode=args.monitor_mode,
        save_top_k=1,
        save_last=True,
        filename=checkpoint_filename,
    )
    early_stopping_callback = EarlyStopping(
        monitor=args.monitor_metric,
        mode=args.monitor_mode,
        patience=args.early_stop_patience,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer_kwargs = dict(
        logger=lightning_logger,
        max_epochs=config['train']['epoch'],
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        precision=config['train']['precision'],
        gradient_clip_val=config['train']['grad_clip_val'],
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
    last_model_path = checkpoint_callback.last_model_path
    test_results = None
    try:
        test_results = trainer.test(datamodule=lightning_data, ckpt_path='best')
    except TypeError:
        if best_model_path:
            test_results = trainer.test(ckpt_path=best_model_path, datamodule=lightning_data)
        else:
            test_results = trainer.test(lightning_model, datamodule=lightning_data)
    training_record_path = write_latest_training_record(
        best_checkpoint=best_model_path,
        last_checkpoint=last_model_path,
        logger_obj=lightning_logger,
    )
    training_summary_path = write_training_summary(
        best_checkpoint=best_model_path,
        last_checkpoint=last_model_path,
        logger_obj=lightning_logger,
        best_val_mae_epoch=checkpoint_callback.best_model_score,
        test_results=test_results,
    )
    try:
        build_and_save_analysis_registry(PROJECT_ROOT)
    except ModuleNotFoundError as exc:
        if exc.name != 'openpyxl':
            raise
        print('Skip analysis registry Excel export because openpyxl is not installed.')
    print('Bike graph use:', config['graph']['use'])
    print('Bike data:', config['data'])
    print('Project root:', PROJECT_ROOT)
    print('Data dir:', args.data_dir)
    print('Graph dir:', args.graph_dir)
    print('Applied log1p input feature cols:', getattr(train_set, 'log1p_feature_cols', []))
    print('Categorical feature configs:', categorical_feature_configs)
    print('Device:', str(device))
    print('Best checkpoint:', best_model_path if best_model_path else 'None')
    print('Last checkpoint:', last_model_path if last_model_path else 'None')
    print('Latest training record:', training_record_path if training_record_path else 'None')
    print('Training summary:', training_summary_path if training_summary_path else 'None')


if __name__ == '__main__':
    main()
