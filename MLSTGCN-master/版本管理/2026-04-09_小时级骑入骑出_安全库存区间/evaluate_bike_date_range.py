import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

os.environ.setdefault('WANDB_MODE', 'disabled')

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

from datasets.bike import BikeGraph
from models.MSTGCN import MSTGCN_submodule
from models.fusiongraph import FusionGraphModel
from prepare_bike_data import (
    DEFAULT_AUX_TEMPORAL_FILES,
    DEFAULT_WEATHER_FILES,
    add_calendar_features,
    merge_prefer_existing,
    merge_auxiliary_feature_tables,
    parse_csv_list,
    resolve_optional_input_files,
)


def normalize_date_series(series, label):
    parsed = pd.to_datetime(series, errors='coerce')
    if parsed.isna().all():
        raise ValueError('Failed to parse any values in %s as dates' % label)
    return parsed.dt.strftime('%Y-%m-%d')


def resolve_device(device_arg, gpu_id=0):
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda:%d' % gpu_id)
        return torch.device('cpu')
    return torch.device(device_arg)


def resolve_project_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(base_dir, path_value)


def find_latest_bike_checkpoint(project_root):
    patterns = [
        os.path.join(project_root, 'logs', 'bike', '**', '*.ckpt'),
        os.path.join(project_root, 'wandb', '**', 'bike', '**', '*.ckpt'),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))
    if not candidates:
        raise FileNotFoundError('No bike checkpoint was found under logs/ or wandb/.')
    best_candidates = [item for item in candidates if 'best-' in os.path.basename(item)]
    selected = best_candidates if best_candidates else candidates
    selected = sorted(selected, key=os.path.getmtime, reverse=True)
    return selected[0]


def load_train_metadata(data_dir):
    train_npz = np.load(os.path.join(data_dir, 'train.npz'), allow_pickle=True)
    x_train = train_npz['x'].astype(np.float32)
    y_train = train_npz['y'].astype(np.float32)
    feature_cols = [str(item) for item in train_npz['input_feature_cols'].tolist()]
    history_feature_cols = [str(item) for item in train_npz['history_feature_cols'].tolist()] if 'history_feature_cols' in train_npz else list(feature_cols)
    known_future_feature_cols = [str(item) for item in train_npz['known_future_feature_cols'].tolist()] if 'known_future_feature_cols' in train_npz else []
    log1p_feature_cols = [str(item) for item in train_npz['log1p_feature_cols'].tolist()] if 'log1p_feature_cols' in train_npz else []
    target_cols = [str(item) for item in train_npz['target_cols'].tolist()]

    feature_mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    feature_std = x_train.std(axis=(0, 1, 2), keepdims=True)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)

    target_mean = y_train.mean(axis=(0, 1, 2), keepdims=True)
    target_std = y_train.std(axis=(0, 1, 2), keepdims=True)
    target_std = np.where(target_std == 0, 1.0, target_std)

    return {
        'x_train': x_train,
        'y_train': y_train,
        'feature_cols': feature_cols,
        'history_feature_cols': history_feature_cols,
        'known_future_feature_cols': known_future_feature_cols,
        'log1p_feature_cols': log1p_feature_cols,
        'target_cols': target_cols,
        'hist_len': int(x_train.shape[1]),
        'pred_len': int(y_train.shape[1]),
        'in_dim': int(x_train.shape[-1]),
        'out_dim': int(y_train.shape[-1]),
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'target_mean': target_mean,
        'target_std': target_std,
    }


def load_mapping(mapping_file):
    mapping_df = pd.read_csv(mapping_file)
    if 'Node_ID' not in mapping_df.columns or '站点名称' not in mapping_df.columns:
        raise KeyError('Mapping file must contain Node_ID and 站点名称: %s' % mapping_file)
    mapping_df = mapping_df.sort_values('Node_ID').reset_index(drop=True)
    return mapping_df


def load_temporal_table(source_dir, temporal_files, weather_file, aux_temporal_files):
    frames = []
    for relative_path in temporal_files:
        full_path = relative_path if os.path.isabs(relative_path) else os.path.join(source_dir, relative_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError('Missing temporal file: %s' % full_path)
        df = pd.read_csv(full_path)
        if '日期' not in df.columns or '站点名称' not in df.columns:
            raise KeyError('Temporal file must contain 日期 and 站点名称: %s' % full_path)
        df['日期'] = normalize_date_series(df['日期'], '日期')
        frames.append(df)

    temporal_df = pd.concat(frames, ignore_index=True, sort=False)
    temporal_df = temporal_df.drop_duplicates(subset=['日期', '站点名称'], keep='last').copy()
    full_dates = sorted(temporal_df['日期'].dropna().unique())
    full_station_names = sorted(temporal_df['站点名称'].dropna().astype(str).unique())
    full_index = pd.MultiIndex.from_product([full_dates, full_station_names], names=['日期', '站点名称']).to_frame(index=False)
    temporal_df['站点名称'] = temporal_df['站点名称'].astype(str)
    temporal_df = full_index.merge(temporal_df, on=['日期', '站点名称'], how='left')
    temporal_df, aux_merge_summary = merge_auxiliary_feature_tables(temporal_df, aux_temporal_files, '日期', '站点名称')
    temporal_df = add_calendar_features(temporal_df, '日期')

    weather_candidates = resolve_optional_input_files(source_dir, weather_file, DEFAULT_WEATHER_FILES)
    weather_path = weather_candidates[0] if weather_candidates else None
    if weather_path is not None:
        weather_df = pd.read_csv(weather_path)
        if '日期' not in weather_df.columns:
            raise KeyError('Weather file must contain 日期: %s' % weather_path)
        weather_df['日期'] = normalize_date_series(weather_df['日期'], 'weather 日期')
        weather_df = weather_df.drop_duplicates(subset=['日期'], keep='last').copy()
        weather_cols = [col for col in weather_df.columns if col != '日期']
        temporal_df = merge_prefer_existing(temporal_df, weather_df, ['日期'], weather_cols)

    return temporal_df, weather_path, aux_merge_summary


def build_feature_target_arrays(feature_df, mapping_df, history_feature_cols, known_future_feature_cols, target_cols):
    station_names = mapping_df['站点名称'].tolist()
    all_dates = sorted(feature_df['日期'].dropna().unique())

    history_matrices = []
    for feature_name in history_feature_cols:
        if feature_name not in feature_df.columns:
            raise KeyError('Missing feature column in temporal table: %s' % feature_name)
        pivot = feature_df.pivot_table(
            index='日期',
            columns='站点名称',
            values=feature_name,
            aggfunc='mean',
        )
        pivot = pivot.reindex(index=all_dates, columns=station_names).fillna(0.0)
        history_matrices.append(pivot.values.astype(np.float32))

    y_matrices = []
    for target_name in target_cols:
        if target_name not in feature_df.columns:
            raise KeyError('Missing target column in temporal table: %s' % target_name)
        pivot = feature_df.pivot_table(
            index='日期',
            columns='站点名称',
            values=target_name,
            aggfunc='mean',
        )
        pivot = pivot.reindex(index=all_dates, columns=station_names).fillna(0.0)
        y_matrices.append(pivot.values.astype(np.float32))

    x_values = np.stack(history_matrices, axis=-1)
    known_future_values = None
    if known_future_feature_cols:
        known_future_matrices = []
        for feature_name in known_future_feature_cols:
            if feature_name not in feature_df.columns:
                raise KeyError('Missing known future column in temporal table: %s' % feature_name)
            pivot = feature_df.pivot_table(
                index='日期',
                columns='站点名称',
                values=feature_name,
                aggfunc='mean',
            )
            pivot = pivot.reindex(index=all_dates, columns=station_names).fillna(0.0)
            known_future_matrices.append(pivot.values.astype(np.float32))
        known_future_values = np.stack(known_future_matrices, axis=-1)
    y_values = np.stack(y_matrices, axis=-1)
    return all_dates, x_values, known_future_values, y_values


class InferenceModel(torch.nn.Module):
    def __init__(
        self,
        graph,
        device,
        in_dim,
        hist_len,
        pred_len,
        out_dim,
        graph_sparsify_mode='topk',
        graph_topk=15,
        categorical_feature_configs=None,
    ):
        super().__init__()
        self.config_graph = dict(
            use=['dist', 'neighb', 'distri', 'tempp', 'func'],
            fix_weight=False,
            tempp_diag_zero=True,
            matrix_weight=True,
            distri_type='exp',
            func_type='ours',
            attention=True,
            sparsify_mode=graph_sparsify_mode,
            sparsify_topk=graph_topk,
            sparsify_symmetric=True,
            sparsify_keep_self=True,
        )
        self.config_data = dict(
            in_dim=in_dim,
            out_dim=out_dim,
            hist_len=hist_len,
            pred_len=pred_len,
            type='bike',
        )
        self.fusiongraph = FusionGraphModel(
            graph,
            device,
            self.config_graph,
            self.config_data,
            M=24,
            d=6,
            bn_decay=0.1,
        )
        self.model = MSTGCN_submodule(
            device,
            self.fusiongraph,
            in_dim,
            hist_len,
            pred_len,
            out_dim,
            categorical_feature_configs=categorical_feature_configs,
        )

    def forward(self, x):
        return self.model(x)


def build_categorical_feature_configs(metadata, weekday_embed_dim):
    if weekday_embed_dim <= 0:
        return []
    configs = []
    categorical_map = {
        '星期几': 7,
        'future_星期几': 7,
    }
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


def load_model(checkpoint_path, graph_dir, device, metadata, graph_sparsify_mode='topk', graph_topk=15, weekday_embed_dim=8):
    graph_config = dict(
        use=['dist', 'neighb', 'distri', 'tempp', 'func'],
        fix_weight=False,
        tempp_diag_zero=True,
        matrix_weight=True,
        attention=True,
        distri_type='exp',
        func_type='ours',
        sparsify_mode=graph_sparsify_mode,
        sparsify_topk=graph_topk,
        sparsify_symmetric=True,
        sparsify_keep_self=True,
    )
    graph = BikeGraph(graph_dir, graph_config, device)
    categorical_feature_configs = build_categorical_feature_configs(metadata, weekday_embed_dim)
    model = InferenceModel(
        graph=graph,
        device=device,
        in_dim=metadata['in_dim'],
        hist_len=metadata['hist_len'],
        pred_len=metadata['pred_len'],
        out_dim=metadata['out_dim'],
        graph_sparsify_mode=graph_sparsify_mode,
        graph_topk=graph_topk,
        categorical_feature_configs=categorical_feature_configs,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ignored_missing = [item for item in missing if item.startswith('fusiongraph.SE')]
    real_missing = [item for item in missing if item not in ignored_missing]
    if real_missing:
        raise RuntimeError('Missing keys when loading checkpoint: %s' % real_missing)
    if unexpected:
        unexpected = [item for item in unexpected if not item.startswith('metric_lightning')]
        unexpected = [item for item in unexpected if not item.startswith('loss')]
        if unexpected:
            print('Ignoring unexpected checkpoint keys:', unexpected)
    model.eval()
    return model


def masked_mape_np(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
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


def apply_output_constraint(y_hat):
    return F.softplus(y_hat, beta=5.0)


def apply_log1p_to_input_window(x_window, feature_cols, selected_cols):
    if not selected_cols:
        return x_window

    transformed = np.array(x_window, copy=True)
    feature_index = {name: idx for idx, name in enumerate(feature_cols)}
    for feature_name in selected_cols:
        feature_idx = feature_index.get(feature_name)
        if feature_idx is None:
            continue
        feature_slice = transformed[..., feature_idx]
        min_value = np.nanmin(feature_slice)
        if min_value < 0:
            continue
        transformed[..., feature_idx] = np.log1p(feature_slice)
    return transformed


def evaluate_dates(model, device, all_dates, x_values, known_future_values, y_values, mapping_df, metadata, start_date, end_date):
    if metadata['pred_len'] != 1:
        raise NotImplementedError('This evaluation script currently supports pred_len=1 only.')

    date_to_idx = {date_value: idx for idx, date_value in enumerate(all_dates)}
    requested_dates = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d').tolist()

    missing_dates = [date_value for date_value in requested_dates if date_value not in date_to_idx]
    if missing_dates:
        raise ValueError('Requested dates are missing from temporal table: %s' % missing_dates)

    prediction_rows = []
    daily_rows = []
    station_names = mapping_df['站点名称'].tolist()
    feature_mean = metadata['feature_mean']
    feature_std = metadata['feature_std']
    target_mean = torch.as_tensor(metadata['target_mean'], dtype=torch.float32)
    target_std = torch.as_tensor(metadata['target_std'], dtype=torch.float32)

    with torch.no_grad():
        for target_date in requested_dates:
            target_idx = date_to_idx[target_date]
            hist_start = target_idx - metadata['hist_len']
            if hist_start < 0:
                raise ValueError('Not enough history for target date %s with hist_len=%s' % (target_date, metadata['hist_len']))

            x_window = x_values[hist_start:target_idx]
            if known_future_values is not None:
                future_features = known_future_values[target_idx]
                future_features = np.repeat(future_features[np.newaxis, ...], metadata['hist_len'], axis=0)
                x_window = np.concatenate([x_window, future_features], axis=-1)
            y_true = y_values[target_idx:target_idx + 1]
            x_window = apply_log1p_to_input_window(
                x_window,
                metadata['feature_cols'],
                metadata['log1p_feature_cols'],
            )

            x_scaled = (x_window - feature_mean[0]) / feature_std[0]
            x_tensor = torch.from_numpy(x_scaled[np.newaxis, ...].astype(np.float32)).to(device)
            pred_scaled = model(x_tensor).cpu()
            pred = (pred_scaled * target_std) + target_mean
            pred = apply_output_constraint(pred).numpy()[0, 0]
            true = y_true[0]

            overall_metrics = compute_metrics(pred.reshape(-1), true.reshape(-1))
            daily_row = {
                '日期': target_date,
                'overall_mae': overall_metrics['mae'],
                'overall_rmse': overall_metrics['rmse'],
                'overall_mape': overall_metrics['mape'],
            }

            for target_idx_dim, target_name in enumerate(metadata['target_cols']):
                target_metrics = compute_metrics(pred[:, target_idx_dim], true[:, target_idx_dim])
                daily_row['%s_mae' % target_name] = target_metrics['mae']
                daily_row['%s_rmse' % target_name] = target_metrics['rmse']
                daily_row['%s_mape' % target_name] = target_metrics['mape']

            daily_rows.append(daily_row)

            for station_idx, station_name in enumerate(station_names):
                row = {
                    '日期': target_date,
                    '站点名称': station_name,
                }
                for target_idx_dim, target_name in enumerate(metadata['target_cols']):
                    pred_value = float(pred[station_idx, target_idx_dim])
                    true_value = float(true[station_idx, target_idx_dim])
                    row['pred_%s' % target_name] = pred_value
                    row['true_%s' % target_name] = true_value
                    row['abs_error_%s' % target_name] = abs(pred_value - true_value)
                prediction_rows.append(row)

    pred_df = pd.DataFrame(prediction_rows)
    daily_df = pd.DataFrame(daily_rows)
    summary = {
        'date_start': requested_dates[0],
        'date_end': requested_dates[-1],
        'num_days': len(requested_dates),
        'num_stations': len(station_names),
        'checkpoint': None,
        'overall_mae_mean': float(daily_df['overall_mae'].mean()),
        'overall_rmse_mean': float(daily_df['overall_rmse'].mean()),
        'overall_mape_mean': float(daily_df['overall_mape'].dropna().mean()) if daily_df['overall_mape'].notna().any() else None,
    }
    for target_name in metadata['target_cols']:
        summary['%s_mae_mean' % target_name] = float(daily_df['%s_mae' % target_name].mean())
        summary['%s_rmse_mean' % target_name] = float(daily_df['%s_rmse' % target_name].mean())
        target_mape_col = '%s_mape' % target_name
        summary['%s_mape_mean' % target_name] = (
            float(daily_df[target_mape_col].dropna().mean()) if daily_df[target_mape_col].notna().any() else None
        )

    return pred_df, daily_df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--data_dir', default=os.path.join('data', 'temporal_data', 'bike'))
    parser.add_argument('--graph_dir', default=os.path.join('data', 'graph', 'bike'))
    parser.add_argument('--mapping_file', default=os.path.join('data', 'graph', 'bike', 'selected_node_mapping.csv'))
    parser.add_argument('--source_dir', default='二月份数据处理')
    parser.add_argument(
        '--temporal_files',
        default='2Years_Daily_NetFlow.csv,CLEAN_JC_BaseTable_2026-02-01_to_2026-02-28.csv',
        help='Comma-separated temporal csv files under source_dir.',
    )
    parser.add_argument(
        '--weather_file',
        default=None,
    )
    parser.add_argument(
        '--aux_temporal_files',
        default=None,
        help='Optional comma-separated auxiliary temporal tables under source_dir.',
    )
    parser.add_argument('--start_date', default='2026-02-01')
    parser.add_argument('--end_date', default='2026-02-10')
    parser.add_argument('--output_dir', default=os.path.join('outputs', 'bike_eval_2026-02-01_to_2026-02-10'))
    parser.add_argument('--graph_sparsify_mode', choices=['none', 'topk', 'row_mean', 'topk_or_row_mean'], default='topk')
    parser.add_argument('--graph_topk', type=int, default=15)
    parser.add_argument('--weekday_embed_dim', type=int, default=8)
    args = parser.parse_args()

    device = resolve_device(args.device)
    args.data_dir = resolve_project_path(PROJECT_ROOT, args.data_dir)
    args.graph_dir = resolve_project_path(PROJECT_ROOT, args.graph_dir)
    args.mapping_file = resolve_project_path(PROJECT_ROOT, args.mapping_file)
    args.source_dir = resolve_project_path(PROJECT_ROOT, args.source_dir)
    args.output_dir = resolve_project_path(PROJECT_ROOT, args.output_dir)
    checkpoint_path = args.checkpoint or find_latest_bike_checkpoint(PROJECT_ROOT)
    metadata = load_train_metadata(args.data_dir)
    mapping_df = load_mapping(args.mapping_file)
    temporal_files = parse_csv_list(args.temporal_files)
    aux_temporal_files = resolve_optional_input_files(args.source_dir, args.aux_temporal_files, DEFAULT_AUX_TEMPORAL_FILES)
    feature_df, weather_path, aux_merge_summary = load_temporal_table(
        args.source_dir,
        temporal_files,
        args.weather_file,
        aux_temporal_files,
    )
    all_dates, x_values, known_future_values, y_values = build_feature_target_arrays(
        feature_df,
        mapping_df,
        metadata['history_feature_cols'],
        metadata['known_future_feature_cols'],
        metadata['target_cols'],
    )
    model = load_model(
        checkpoint_path,
        args.graph_dir,
        device,
        metadata,
        graph_sparsify_mode=args.graph_sparsify_mode,
        graph_topk=args.graph_topk,
        weekday_embed_dim=args.weekday_embed_dim,
    )

    pred_df, daily_df, summary = evaluate_dates(
        model=model,
        device=device,
        all_dates=all_dates,
        x_values=x_values,
        known_future_values=known_future_values,
        y_values=y_values,
        mapping_df=mapping_df,
        metadata=metadata,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, 'station_predictions.csv')
    daily_path = os.path.join(args.output_dir, 'daily_metrics.csv')
    summary_path = os.path.join(args.output_dir, 'summary.json')

    pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
    daily_df.to_csv(daily_path, index=False, encoding='utf-8-sig')
    summary['checkpoint'] = checkpoint_path
    summary['weather_file'] = weather_path
    summary['temporal_files'] = temporal_files
    summary['aux_temporal_files'] = aux_temporal_files
    summary['aux_merge_summary'] = aux_merge_summary
    with open(summary_path, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print('Checkpoint:', checkpoint_path)
    print('Weather file:', weather_path)
    print('Evaluated dates:', args.start_date, '->', args.end_date)
    print('Saved station predictions to:', pred_path)
    print('Saved daily metrics to:', daily_path)
    print('Saved summary to:', summary_path)
    print('Summary:', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
