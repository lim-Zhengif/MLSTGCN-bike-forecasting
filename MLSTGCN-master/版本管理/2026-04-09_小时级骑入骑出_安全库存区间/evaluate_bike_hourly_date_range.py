import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)

from datasets.bike import BikeGraph
from hourly_pipeline_utils import (
    aggregate_hourly_trip_counts,
    apply_log1p_transform,
    build_hourly_feature_frame,
    build_hourly_samples,
    detect_project_root,
    load_daily_feature_table,
    resolve_project_path,
    resolve_trip_files,
)
from models.MSTGCN import MSTGCN_submodule
from models.fusiongraph import FusionGraphModel


PROJECT_ROOT = detect_project_root(PROJ_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from analysis_result_utils import (
    build_analysis_task_dir,
    build_and_save_analysis_registry,
    ensure_analysis_dir,
    resolve_version_tag,
)


LATEST_TRAINING_RECORD_NAME = 'latest_hourly_training_checkpoint.json'
TRAINING_SUMMARY_NAME = 'training_summary.json'

EVAL_CONFIG_DEFAULTS = {
    'graph_use': 'dist,neighb,distri,tempp,func',
    'graph_attention': 'true',
    'matrix_weight': 'true',
    'graph_fix_weight': 'false',
    'tempp_diag_zero': 'true',
    'graph_distri_type': 'exp',
    'graph_func_type': 'ours',
    'graph_sparsify_mode': 'topk',
    'graph_topk': 15,
    'graph_sparsify_symmetric': 'true',
    'graph_sparsify_keep_self': 'true',
    'fusion_heads': 24,
    'fusion_head_dim': 6,
    'bn_decay': 0.1,
    'cheb_k': 3,
    'nb_block': 2,
    'nb_chev_filter': 64,
    'nb_time_filter': 64,
    'time_kernel_size': 3,
    'weekday_embed_dim': 8,
    'output_constraint': 'softplus',
    'output_softplus_beta': 5.0,
}

RESTORABLE_CONFIG_KEYS = tuple(EVAL_CONFIG_DEFAULTS.keys())


def resolve_device(device_arg, gpu_id=0):
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda:%d' % gpu_id)
        return torch.device('cpu')
    return torch.device(device_arg)


def find_latest_checkpoint(project_root, project_name, extra_search_roots=None):
    search_roots = [project_root]
    for root in extra_search_roots or []:
        if root and root not in search_roots:
            search_roots.append(root)

    patterns = []
    for root in search_roots:
        patterns.extend(
            [
                os.path.join(root, 'logs', project_name, '**', '*.ckpt'),
                os.path.join(root, 'wandb', '**', project_name, '**', '*.ckpt'),
                os.path.join(root, 'wandb', '**', 'checkpoints', '*.ckpt'),
            ]
        )
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))
    if not candidates:
        raise FileNotFoundError('No checkpoint was found for project %s.' % project_name)
    best_candidates = [item for item in candidates if 'best-' in os.path.basename(item)]
    selected = best_candidates if best_candidates else candidates
    selected = sorted(selected, key=os.path.getmtime, reverse=True)
    return selected[0]


def find_latest_checkpoint_any_project(project_root, extra_search_roots=None):
    search_roots = [project_root]
    for root in extra_search_roots or []:
        if root and root not in search_roots:
            search_roots.append(root)

    patterns = []
    for root in search_roots:
        patterns.extend(
            [
                os.path.join(root, 'logs', '**', '*.ckpt'),
                os.path.join(root, 'wandb', '**', '*.ckpt'),
            ]
        )

    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))
    candidates = [item for item in candidates if os.path.isfile(item)]
    if not candidates:
        raise FileNotFoundError('No checkpoint was found under logs/ or wandb/.')
    best_candidates = [item for item in candidates if 'best-' in os.path.basename(item)]
    selected = best_candidates if best_candidates else candidates
    selected = sorted(selected, key=os.path.getmtime, reverse=True)
    return selected[0]


def load_latest_training_record(script_dir):
    record_path = os.path.join(script_dir, LATEST_TRAINING_RECORD_NAME)
    if not os.path.exists(record_path):
        return None, None
    try:
        with open(record_path, 'r', encoding='utf-8') as fp:
            return json.load(fp), record_path
    except Exception as exc:
        print('Ignore invalid latest training record %s: %s' % (record_path, exc))
        return None, record_path


def resolve_checkpoint_path(args, cli_argv):
    if args.checkpoint:
        return resolve_project_path(PROJECT_ROOT, args.checkpoint), 'explicit --checkpoint'

    project_was_explicit = any(item == '--project' or str(item).startswith('--project=') for item in cli_argv)
    if not project_was_explicit:
        training_record, training_record_path = load_latest_training_record(PROJ_DIR)
        if training_record:
            for key in ('preferred_checkpoint', 'best_checkpoint', 'last_checkpoint'):
                checkpoint_path = training_record.get(key)
                if checkpoint_path and os.path.exists(checkpoint_path):
                    return checkpoint_path, '%s from %s' % (key, training_record_path)
            print(
                'Latest training record exists but checkpoint files are missing, fallback to project search: %s'
                % training_record_path
            )
        try:
            checkpoint_path = find_latest_checkpoint_any_project(
                PROJECT_ROOT,
                extra_search_roots=[PROJ_DIR],
            )
            return checkpoint_path, 'latest checkpoint across all projects'
        except FileNotFoundError:
            pass

    checkpoint_path = find_latest_checkpoint(
        PROJECT_ROOT,
        args.project,
        extra_search_roots=[PROJ_DIR],
    )
    return checkpoint_path, 'latest checkpoint under project %s' % args.project


def infer_project_name_from_checkpoint(checkpoint_path):
    if not checkpoint_path:
        return None
    normalized = os.path.normpath(checkpoint_path)
    parts = normalized.split(os.sep)
    if 'logs' in parts:
        idx = parts.index('logs')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if 'files' in parts:
        idx = parts.index('files')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def sanitize_experiment_tag(tag_value):
    if not tag_value:
        return None
    safe_chars = []
    for char in str(tag_value):
        if char.isalnum() or char in ('-', '_'):
            safe_chars.append(char)
        elif char in ('.', '='):
            safe_chars.append('-')
    safe_value = ''.join(safe_chars).strip('-_')
    return safe_value or None


def normalize_path_for_compare(path_value):
    if not path_value:
        return None
    return os.path.normcase(os.path.abspath(os.path.normpath(path_value)))


def parse_bool_value(value, name):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if normalized in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise ValueError('%s must be true or false, got %r.' % (name, value))


def parse_graph_use_value(value):
    if isinstance(value, (list, tuple)):
        parsed = [str(item).strip() for item in value if str(item).strip()]
    else:
        parsed = [item.strip() for item in str(value).split(',') if item.strip()]
    if not parsed:
        raise ValueError('--graph_use cannot be empty.')
    return parsed


def parse_option_values(argv):
    options = {}
    idx = 0
    while idx < len(argv):
        token = str(argv[idx])
        if not token.startswith('--'):
            idx += 1
            continue
        if '=' in token:
            flag, value = token[2:].split('=', 1)
            options[flag.replace('-', '_')] = value
            idx += 1
            continue
        key = token[2:].replace('-', '_')
        if idx + 1 < len(argv) and not str(argv[idx + 1]).startswith('--'):
            options[key] = argv[idx + 1]
            idx += 2
        else:
            options[key] = 'true'
            idx += 1
    return options


def find_training_summary_by_project(project_name, checkpoint_path=None):
    if not project_name:
        return None, None
    pattern = os.path.join(
        PROJECT_ROOT,
        '分析结果',
        '**',
        'training_result_%s' % project_name,
        TRAINING_SUMMARY_NAME,
    )
    candidates = [item for item in glob.glob(pattern, recursive=True) if os.path.isfile(item)]
    if not candidates:
        return None, None

    checkpoint_norm = normalize_path_for_compare(checkpoint_path)
    matched = []
    for summary_path in candidates:
        try:
            with open(summary_path, 'r', encoding='utf-8') as fp:
                summary = json.load(fp)
        except Exception:
            continue
        summary_checkpoints = [
            normalize_path_for_compare(summary.get('best_checkpoint')),
            normalize_path_for_compare(summary.get('last_checkpoint')),
        ]
        if checkpoint_norm and checkpoint_norm in summary_checkpoints:
            matched.append((summary_path, summary))
    if matched:
        return sorted(matched, key=lambda item: os.path.getmtime(item[0]), reverse=True)[0]

    candidates = sorted(candidates, key=os.path.getmtime, reverse=True)
    with open(candidates[0], 'r', encoding='utf-8') as fp:
        return candidates[0], json.load(fp)


def find_training_summary_by_checkpoint(checkpoint_path):
    checkpoint_norm = normalize_path_for_compare(checkpoint_path)
    if not checkpoint_norm:
        return None, None
    pattern = os.path.join(PROJECT_ROOT, '分析结果', '**', TRAINING_SUMMARY_NAME)
    for summary_path in glob.glob(pattern, recursive=True):
        try:
            with open(summary_path, 'r', encoding='utf-8') as fp:
                summary = json.load(fp)
        except Exception:
            continue
        summary_checkpoints = [
            normalize_path_for_compare(summary.get('best_checkpoint')),
            normalize_path_for_compare(summary.get('last_checkpoint')),
        ]
        if checkpoint_norm in summary_checkpoints:
            return summary_path, summary
    return None, None


def load_training_config_source(checkpoint_path, checkpoint_project):
    summary_path, summary = find_training_summary_by_project(checkpoint_project, checkpoint_path)
    if summary:
        return summary, summary_path

    summary_path, summary = find_training_summary_by_checkpoint(checkpoint_path)
    if summary:
        return summary, summary_path

    record, record_path = load_latest_training_record(PROJ_DIR)
    if not record:
        return None, None

    checkpoint_norm = normalize_path_for_compare(checkpoint_path)
    record_checkpoints = [
        normalize_path_for_compare(record.get('preferred_checkpoint')),
        normalize_path_for_compare(record.get('best_checkpoint')),
        normalize_path_for_compare(record.get('last_checkpoint')),
    ]
    if checkpoint_norm in record_checkpoints or record.get('project') == checkpoint_project:
        return record, record_path
    return None, None


def build_restored_eval_config(args, cli_argv, checkpoint_path, checkpoint_project):
    training_source, training_source_path = load_training_config_source(checkpoint_path, checkpoint_project)
    restored_options = {}
    if training_source:
        restored_argv = training_source.get('resolved_train_argv') or []
        if not restored_argv:
            restored_argv = training_source.get('entry_args') or []
        restored_options = parse_option_values(restored_argv)

    cli_options = parse_option_values(cli_argv)
    raw_config = dict(EVAL_CONFIG_DEFAULTS)
    for key in RESTORABLE_CONFIG_KEYS:
        if key in restored_options:
            raw_config[key] = restored_options[key]
        if key in cli_options:
            raw_config[key] = cli_options[key]

    graph_config = {
        'use': parse_graph_use_value(raw_config['graph_use']),
        'fix_weight': parse_bool_value(raw_config['graph_fix_weight'], 'graph_fix_weight'),
        'tempp_diag_zero': parse_bool_value(raw_config['tempp_diag_zero'], 'tempp_diag_zero'),
        'matrix_weight': parse_bool_value(raw_config['matrix_weight'], 'matrix_weight'),
        'attention': parse_bool_value(raw_config['graph_attention'], 'graph_attention'),
        'distri_type': str(raw_config['graph_distri_type']),
        'func_type': str(raw_config['graph_func_type']),
        'sparsify_mode': str(raw_config['graph_sparsify_mode']),
        'sparsify_topk': int(raw_config['graph_topk']),
        'sparsify_symmetric': parse_bool_value(raw_config['graph_sparsify_symmetric'], 'graph_sparsify_symmetric'),
        'sparsify_keep_self': parse_bool_value(raw_config['graph_sparsify_keep_self'], 'graph_sparsify_keep_self'),
    }
    fusion_config = {
        'M': int(raw_config['fusion_heads']),
        'd': int(raw_config['fusion_head_dim']),
        'bn_decay': float(raw_config['bn_decay']),
    }
    model_config = {
        'cheb_k': int(raw_config['cheb_k']),
        'nb_block': int(raw_config['nb_block']),
        'nb_chev_filter': int(raw_config['nb_chev_filter']),
        'nb_time_filter': int(raw_config['nb_time_filter']),
        'time_kernel_size': int(raw_config['time_kernel_size']),
    }
    prediction_config = {
        'weekday_embed_dim': int(raw_config['weekday_embed_dim']),
        'output_constraint': str(raw_config['output_constraint']),
        'output_softplus_beta': float(raw_config['output_softplus_beta']),
    }

    return {
        'training_config_source': training_source_path,
        'raw_options': raw_config,
        'graph': graph_config,
        'fusion': fusion_config,
        'model': model_config,
        'prediction': prediction_config,
    }


def load_train_metadata(data_dir):
    train_npz = np.load(os.path.join(data_dir, 'train.npz'), allow_pickle=True)
    x_train = train_npz['x'].astype(np.float32)
    y_train = train_npz['y'].astype(np.float32)
    return {
        'feature_cols': [str(item) for item in train_npz['input_feature_cols'].tolist()],
        'history_feature_cols': [str(item) for item in train_npz['history_feature_cols'].tolist()],
        'known_future_feature_cols': [str(item) for item in train_npz['known_future_feature_cols'].tolist()],
        'log1p_feature_cols': [str(item) for item in train_npz['log1p_feature_cols'].tolist()],
        'target_cols': [str(item) for item in train_npz['target_cols'].tolist()],
        'hist_len': int(x_train.shape[1]),
        'pred_len': int(y_train.shape[1]),
        'in_dim': int(x_train.shape[-1]),
        'out_dim': int(y_train.shape[-1]),
        'feature_mean': x_train.mean(axis=(0, 1, 2), keepdims=True),
        'feature_std': np.where(x_train.std(axis=(0, 1, 2), keepdims=True) == 0, 1.0, x_train.std(axis=(0, 1, 2), keepdims=True)),
        'target_mean': y_train.mean(axis=(0, 1, 2), keepdims=True),
        'target_std': np.where(y_train.std(axis=(0, 1, 2), keepdims=True) == 0, 1.0, y_train.std(axis=(0, 1, 2), keepdims=True)),
    }


def build_categorical_feature_configs(metadata, weekday_embed_dim):
    if weekday_embed_dim <= 0:
        return []
    configs = []
    categorical_map = {'星期几': 7, 'future_星期几': 7}
    feature_mean = metadata['feature_mean'].reshape(-1)
    feature_std = metadata['feature_std'].reshape(-1)
    for feature_idx, feature_name in enumerate(metadata['feature_cols']):
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


class InferenceModel(torch.nn.Module):
    def __init__(self, graph, device, metadata, restored_config, categorical_feature_configs):
        super().__init__()
        config_graph = restored_config['graph']
        config_model = restored_config['model']
        config_fusion = restored_config['fusion']
        config_data = dict(
            in_dim=metadata['in_dim'],
            out_dim=metadata['out_dim'],
            hist_len=metadata['hist_len'],
            pred_len=metadata['pred_len'],
            type='bike',
        )
        self.fusiongraph = FusionGraphModel(
            graph,
            device,
            config_graph,
            config_data,
            M=config_fusion['M'],
            d=config_fusion['d'],
            bn_decay=config_fusion['bn_decay'],
        )
        self.model = MSTGCN_submodule(
            device,
            self.fusiongraph,
            metadata['in_dim'],
            metadata['hist_len'],
            metadata['pred_len'],
            metadata['out_dim'],
            categorical_feature_configs=categorical_feature_configs,
            cheb_k=config_model['cheb_k'],
            nb_block=config_model['nb_block'],
            nb_chev_filter=config_model['nb_chev_filter'],
            nb_time_filter=config_model['nb_time_filter'],
            time_kernel_size=config_model['time_kernel_size'],
        )

    def forward(self, x):
        return self.model(x)


def load_model(checkpoint_path, graph_dir, device, metadata, restored_config):
    graph_config = restored_config['graph']
    graph = BikeGraph(graph_dir, graph_config, device)
    model = InferenceModel(
        graph=graph,
        device=device,
        metadata=metadata,
        restored_config=restored_config,
        categorical_feature_configs=build_categorical_feature_configs(
            metadata,
            restored_config['prediction']['weekday_embed_dim'],
        ),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ignored_missing = [item for item in missing if item.startswith('fusiongraph.SE')]
    real_missing = [item for item in missing if item not in ignored_missing]
    if real_missing:
        raise RuntimeError('Missing keys when loading checkpoint: %s' % real_missing)
    unexpected = [item for item in unexpected if not item.startswith('metric_lightning')]
    unexpected = [item for item in unexpected if not item.startswith('loss')]
    if unexpected:
        print('Ignoring unexpected checkpoint keys:', unexpected)
    model.eval()
    return model


def apply_output_constraint(y_hat, prediction_config):
    output_constraint = prediction_config['output_constraint']
    if output_constraint == 'none':
        return y_hat
    if output_constraint == 'relu':
        return F.relu(y_hat)
    return F.softplus(y_hat, beta=prediction_config['output_softplus_beta'])


def masked_mape_np(pred, true):
    denominator = np.clip(np.abs(true), a_min=1.0, a_max=None)
    return float(np.mean(np.abs(pred - true) / denominator))


def compute_metrics(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return {
        'mae': float(np.mean(np.abs(pred - true))),
        'rmse': float(np.sqrt(np.mean((pred - true) ** 2))),
        'mape': masked_mape_np(pred, true),
    }


def evaluate_dates(model, device, feature_df, mapping_df, metadata, start_date, end_date, prediction_config):
    sample_bundle = build_hourly_samples(
        feature_df=feature_df,
        mapping_df=mapping_df,
        history_feature_cols=metadata['history_feature_cols'],
        target_cols=metadata['target_cols'],
        known_future_feature_cols=metadata['known_future_feature_cols'],
        hist_len=metadata['hist_len'],
        pred_len=metadata['pred_len'],
        min_known_future_coverage=0.0,
    )
    sample_date_to_idx = {date_value: idx for idx, date_value in enumerate(sample_bundle['sample_dates'])}
    requested_dates = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d').tolist()
    missing_dates = [date_value for date_value in requested_dates if date_value not in sample_date_to_idx]
    if missing_dates:
        raise ValueError('Requested dates are missing from hourly samples: %s' % missing_dates)

    feature_mean = metadata['feature_mean']
    feature_std = metadata['feature_std']
    target_mean = torch.as_tensor(metadata['target_mean'], dtype=torch.float32)
    target_std = torch.as_tensor(metadata['target_std'], dtype=torch.float32)
    station_names = mapping_df['站点名称'].tolist()

    station_hour_rows = []
    daily_rows = []
    with torch.no_grad():
        for date_value in requested_dates:
            sample_idx = sample_date_to_idx[date_value]
            x_window = sample_bundle['x'][sample_idx]
            y_true = sample_bundle['y'][sample_idx]
            x_window, _ = apply_log1p_transform(
                x_window,
                metadata['feature_cols'],
                metadata['log1p_feature_cols'],
            )
            x_scaled = (x_window - feature_mean[0]) / feature_std[0]
            x_tensor = torch.from_numpy(x_scaled[np.newaxis, ...].astype(np.float32)).to(device)
            pred_scaled = model(x_tensor).cpu()
            pred = (pred_scaled * target_std) + target_mean
            pred = apply_output_constraint(pred, prediction_config).numpy()[0]

            overall_metrics = compute_metrics(pred.reshape(-1), y_true.reshape(-1))
            out_metrics = compute_metrics(pred[..., 0].reshape(-1), y_true[..., 0].reshape(-1))
            in_metrics = compute_metrics(pred[..., 1].reshape(-1), y_true[..., 1].reshape(-1))
            net_pred = pred[..., 1] - pred[..., 0]
            net_true = y_true[..., 1] - y_true[..., 0]
            net_metrics = compute_metrics(net_pred.reshape(-1), net_true.reshape(-1))
            daily_rows.append(
                {
                    '日期': date_value,
                    'overall_mae': overall_metrics['mae'],
                    'overall_rmse': overall_metrics['rmse'],
                    'overall_mape': overall_metrics['mape'],
                    '小时骑出量_mae': out_metrics['mae'],
                    '小时骑入量_mae': in_metrics['mae'],
                    '小时净流量_mae': net_metrics['mae'],
                }
            )

            for hour_idx in range(metadata['pred_len']):
                for station_idx, station_name in enumerate(station_names):
                    pred_out = float(pred[hour_idx, station_idx, 0])
                    pred_in = float(pred[hour_idx, station_idx, 1])
                    true_out = float(y_true[hour_idx, station_idx, 0])
                    true_in = float(y_true[hour_idx, station_idx, 1])
                    station_hour_rows.append(
                        {
                            '日期': date_value,
                            '小时': hour_idx,
                            '站点名称': station_name,
                            'pred_小时骑出量': pred_out,
                            'pred_小时骑入量': pred_in,
                            'pred_小时净流量': pred_in - pred_out,
                            'true_小时骑出量': true_out,
                            'true_小时骑入量': true_in,
                            'true_小时净流量': true_in - true_out,
                            'abs_error_小时骑出量': abs(pred_out - true_out),
                            'abs_error_小时骑入量': abs(pred_in - true_in),
                            'abs_error_小时净流量': abs((pred_in - pred_out) - (true_in - true_out)),
                        }
                    )

    station_hour_df = pd.DataFrame(station_hour_rows)
    daily_df = pd.DataFrame(daily_rows)
    hour_df = (
        station_hour_df.groupby('小时', as_index=False)[
            ['abs_error_小时骑出量', 'abs_error_小时骑入量', 'abs_error_小时净流量']
        ]
        .mean()
        .rename(
            columns={
                'abs_error_小时骑出量': 'mean_abs_error_小时骑出量',
                'abs_error_小时骑入量': 'mean_abs_error_小时骑入量',
                'abs_error_小时净流量': 'mean_abs_error_小时净流量',
            }
        )
    )
    summary = {
        'date_start': requested_dates[0],
        'date_end': requested_dates[-1],
        'num_days': len(requested_dates),
        'num_stations': len(station_names),
        'overall_mae_mean': float(daily_df['overall_mae'].mean()),
        'overall_rmse_mean': float(daily_df['overall_rmse'].mean()),
        'overall_mape_mean': float(daily_df['overall_mape'].mean()),
        'hourly_out_mae_mean': float(daily_df['小时骑出量_mae'].mean()),
        'hourly_in_mae_mean': float(daily_df['小时骑入量_mae'].mean()),
        'hourly_net_mae_mean': float(daily_df['小时净流量_mae'].mean()),
    }
    return station_hour_df, daily_df, hour_df, summary


def main():
    cli_argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--project', default='bike_hourly_safe_inventory')
    parser.add_argument('--data_dir', default=os.path.join('data', 'temporal_data', 'bike_hourly_safe_inventory'))
    parser.add_argument('--graph_dir', default=os.path.join('data', 'graph', 'bike_hourly_safe_inventory'))
    parser.add_argument('--mapping_file', default=os.path.join('data', 'graph', 'bike_hourly_safe_inventory', 'selected_node_mapping.csv'))
    parser.add_argument('--source_dir', default='二月份数据处理')
    parser.add_argument('--trip_files', default=None)
    parser.add_argument('--daily_feature_file', default='ST_Master_Feature_Table.csv')
    parser.add_argument('--weather_file', default=None)
    parser.add_argument('--aux_temporal_files', default='FINAL_JC_BaseTable_with_POI.csv,CLEAN_JC_BaseTable_2026-02-01_to_2026-02-28.csv')
    parser.add_argument('--start_date', default='2026-02-01')
    parser.add_argument('--end_date', default='2026-02-10')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--version_tag', default=None)
    parser.add_argument('--task_name', default='小时级滚动预测')
    parser.add_argument('--extra_tag', default=None)
    parser.add_argument('--graph_use', default=None)
    parser.add_argument('--graph_attention', default=None)
    parser.add_argument('--matrix_weight', default=None)
    parser.add_argument('--graph_fix_weight', default=None)
    parser.add_argument('--tempp_diag_zero', default=None)
    parser.add_argument('--graph_distri_type', default=None)
    parser.add_argument('--graph_func_type', default=None)
    parser.add_argument('--graph_sparsify_mode', choices=['none', 'topk', 'row_mean', 'topk_or_row_mean'], default=None)
    parser.add_argument('--graph_topk', type=int, default=None)
    parser.add_argument('--graph_sparsify_symmetric', default=None)
    parser.add_argument('--graph_sparsify_keep_self', default=None)
    parser.add_argument('--fusion_heads', type=int, default=None)
    parser.add_argument('--fusion_head_dim', type=int, default=None)
    parser.add_argument('--bn_decay', type=float, default=None)
    parser.add_argument('--cheb_k', type=int, default=None)
    parser.add_argument('--nb_block', type=int, default=None)
    parser.add_argument('--nb_chev_filter', type=int, default=None)
    parser.add_argument('--nb_time_filter', type=int, default=None)
    parser.add_argument('--time_kernel_size', type=int, default=None)
    parser.add_argument('--weekday_embed_dim', type=int, default=None)
    parser.add_argument('--output_constraint', choices=['softplus', 'relu', 'none'], default=None)
    parser.add_argument('--output_softplus_beta', type=float, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    args.data_dir = resolve_project_path(PROJECT_ROOT, args.data_dir)
    args.graph_dir = resolve_project_path(PROJECT_ROOT, args.graph_dir)
    args.mapping_file = resolve_project_path(PROJECT_ROOT, args.mapping_file)
    args.source_dir = resolve_project_path(PROJECT_ROOT, args.source_dir)
    checkpoint_path, checkpoint_source = resolve_checkpoint_path(args, cli_argv)
    checkpoint_project = infer_project_name_from_checkpoint(checkpoint_path)
    restored_config = build_restored_eval_config(args, cli_argv, checkpoint_path, checkpoint_project)
    if not args.extra_tag:
        args.extra_tag = sanitize_experiment_tag(checkpoint_project)
    version_tag = resolve_version_tag(PROJECT_ROOT, args.version_tag, [checkpoint_path, PROJ_DIR])
    if args.output_dir:
        args.output_dir = resolve_project_path(PROJECT_ROOT, args.output_dir)
    else:
        args.output_dir = build_analysis_task_dir(
            project_root=PROJECT_ROOT,
            version_tag=version_tag,
            task_type=args.task_name,
            start_date=args.start_date,
            end_date=args.end_date,
            extra_tag=args.extra_tag,
        )
    metadata = load_train_metadata(args.data_dir)
    mapping_df = pd.read_csv(args.mapping_file).sort_values('Node_ID').reset_index(drop=True)
    station_names = mapping_df['站点名称'].astype(str).tolist()

    trip_files = resolve_trip_files(args.source_dir, args.trip_files)
    hourly_df, _ = aggregate_hourly_trip_counts(trip_files, station_names)
    all_dates = sorted(hourly_df['日期'].dropna().unique())
    daily_feature_df, daily_feature_path, aux_paths, weather_path, aux_merge_summary = load_daily_feature_table(
        source_dir=args.source_dir,
        station_names=station_names,
        all_dates=all_dates,
        daily_feature_file=args.daily_feature_file,
        aux_temporal_files=args.aux_temporal_files,
        weather_file=args.weather_file,
    )
    feature_df = build_hourly_feature_frame(hourly_df, daily_feature_df)

    model = load_model(
        checkpoint_path,
        args.graph_dir,
        device,
        metadata,
        restored_config=restored_config,
    )
    station_hour_df, daily_df, hour_df, summary = evaluate_dates(
        model=model,
        device=device,
        feature_df=feature_df,
        mapping_df=mapping_df,
        metadata=metadata,
        start_date=args.start_date,
        end_date=args.end_date,
        prediction_config=restored_config['prediction'],
    )

    ensure_analysis_dir(args.output_dir)
    station_hour_path = os.path.join(args.output_dir, 'station_hourly_predictions.csv')
    daily_path = os.path.join(args.output_dir, 'daily_metrics.csv')
    hour_path = os.path.join(args.output_dir, 'hourly_metrics.csv')
    summary_path = os.path.join(args.output_dir, 'summary.json')
    station_hour_df.to_csv(station_hour_path, index=False, encoding='utf-8-sig')
    daily_df.to_csv(daily_path, index=False, encoding='utf-8-sig')
    hour_df.to_csv(hour_path, index=False, encoding='utf-8-sig')

    summary.update(
        {
            'checkpoint': checkpoint_path,
            'checkpoint_source': checkpoint_source,
            'checkpoint_project': checkpoint_project,
            'training_config_source': restored_config['training_config_source'],
            'restored_graph_config': restored_config['graph'],
            'restored_fusion_config': restored_config['fusion'],
            'restored_model_config': restored_config['model'],
            'restored_prediction_config': restored_config['prediction'],
            'experiment_tag': args.extra_tag,
            'trip_files': trip_files,
            'daily_feature_file': daily_feature_path,
            'weather_file': weather_path,
            'aux_temporal_files': aux_paths,
            'aux_merge_summary': aux_merge_summary,
            'version_tag': version_tag,
            'task_name': args.task_name,
        }
    )
    with open(summary_path, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    try:
        build_and_save_analysis_registry(PROJECT_ROOT)
    except ModuleNotFoundError as exc:
        if exc.name != 'openpyxl':
            raise
        print('Skip analysis registry Excel export because openpyxl is not installed.')
    print('Checkpoint:', checkpoint_path)
    print('Checkpoint source:', checkpoint_source)
    print('Checkpoint project:', checkpoint_project if checkpoint_project else 'None')
    print('Training config source:', restored_config['training_config_source'] if restored_config['training_config_source'] else 'defaults only')
    print('Restored graph config:', json.dumps(restored_config['graph'], ensure_ascii=False))
    print('Restored model config:', json.dumps(restored_config['model'], ensure_ascii=False))
    print('Experiment tag:', args.extra_tag if args.extra_tag else 'None')
    print('Evaluated dates:', args.start_date, '->', args.end_date)
    print('Saved station hourly predictions to:', station_hour_path)
    print('Saved daily metrics to:', daily_path)
    print('Saved hourly metrics to:', hour_path)
    print('Saved summary to:', summary_path)
    print('Summary:', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
