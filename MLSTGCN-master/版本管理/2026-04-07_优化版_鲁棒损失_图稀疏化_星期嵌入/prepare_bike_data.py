import argparse
import os

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


PROJECT_ROOT = detect_project_root(SCRIPT_DIR)


SOURCE_GRAPH_FILES = {
    'dist.npy': 'graph_spatial_distance.npy',
    'neigh.npy': 'graph_od_transition.npy',
    'func.npy': 'graph_poi_semantic.npy',
    'bike_heuristic.npy': 'graph_heuristic.npy',
    'tempp_bike.npy': 'graph_demand_correlation.npy',
}

SYMMETRIC_OUTPUTS = {'neigh.npy', 'bike_heuristic.npy'}

DEFAULT_TEMPORAL_FILES = ['2Years_Daily_NetFlow.csv', 'ST_Master_Feature_Table.csv']
DEFAULT_AUX_TEMPORAL_FILES = [
    'FINAL_JC_BaseTable_with_POI.csv',
    'ST_Master_Feature_Table.csv',
    'CLEAN_JC_BaseTable_2026-02-01_to_2026-02-28.csv',
]
DEFAULT_WEATHER_FILES = [
    os.path.join('weather-get', 'NYC_Weather_2024-01-01_to_2026-02-31.csv'),
    os.path.join('weather-get', 'NYC_Weather_2024-01-01_to_2026-01-31.csv'),
]
DATE_COLUMN_CANDIDATES = ['\u65e5\u671f', 'date', 'Date', 'datetime', '\u65f6\u95f4']
STATION_COLUMN_CANDIDATES = ['\u7ad9\u70b9\u540d\u79f0', 'station_name', 'Station', 'station', '\u7ad9\u70b9']
WEATHER_FEATURE_CANDIDATES = [
    '\u6700\u9ad8\u6c14\u6e29(\u00b0C)',
    '\u6700\u4f4e\u6c14\u6e29(\u00b0C)',
    '\u603b\u964d\u6c34\u91cf(mm)',
    '\u603b\u964d\u96ea\u91cf(cm)',
    '\u6700\u5927\u98ce\u901f(km/h)',
]
DEMAND_FEATURE_CANDIDATES = [
    '\u65e5\u95f4\u9a91\u51fa\u91cf',
    '\u65e5\u95f4\u9a91\u5165\u91cf',
    '\u65e5\u95f4\u7eaf\u7528\u6237\u51c0\u6d41\u91cf',
    '\u65e9\u6668\u521d\u59cb\u8f66\u8f86\u6570',
    '\u665a\u95f4\u5b9e\u9645\u8f66\u8f86\u6570',
    '\u5b98\u65b9\u767d\u5929\u5e72\u9884\u91cf',
]
HISTORY_DEMAND_FEATURE_CANDIDATES = [
    '\u65e5\u95f4\u9a91\u51fa\u91cf',
    '\u65e5\u95f4\u9a91\u5165\u91cf',
    '\u65e5\u95f4\u7eaf\u7528\u6237\u51c0\u6d41\u91cf',
]
KNOWN_FUTURE_FEATURE_CANDIDATES = [
    '\u65e9\u6668\u521d\u59cb\u8f66\u8f86\u6570',
    '\u603b\u6869\u6570',
    '\u661f\u671f\u51e0',
    '\u662f\u5426\u5468\u672b',
    '\u662f\u5426\u8282\u5047\u65e5',
] + WEATHER_FEATURE_CANDIDATES
LOG1P_FEATURE_CANDIDATES = [
    '\u65e5\u95f4\u9a91\u51fa\u91cf',
    '\u65e5\u95f4\u9a91\u5165\u91cf',
    '\u65e5\u95f4\u7eaf\u7528\u6237\u51c0\u6d41\u91cf',
    '\u65e9\u6668\u521d\u59cb\u8f66\u8f86\u6570',
    '\u665a\u95f4\u5b9e\u9645\u8f66\u8f86\u6570',
    '\u603b\u6869\u6570',
]


def resolve_input_file(source_dir, user_value, fallback_candidates, label):
    if user_value:
        candidate = user_value if os.path.isabs(user_value) else os.path.join(source_dir, user_value)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError('Missing %s file: %s' % (label, candidate))

    for name in fallback_candidates:
        candidate = os.path.join(source_dir, name)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        'Missing %s file under %s. Tried: %s'
        % (label, source_dir, fallback_candidates)
    )


def resolve_project_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(base_dir, path_value)


def resolve_column(df, preferred, candidates, fallback_idx, label):
    if preferred:
        if preferred in df.columns:
            return preferred
        raise KeyError('Specified %s not found: %s' % (label, preferred))

    for name in candidates:
        if name in df.columns:
            return name

    if len(df.columns) > fallback_idx:
        return df.columns[fallback_idx]

    raise KeyError('Cannot resolve %s from columns: %s' % (label, list(df.columns)))


def parse_feature_columns(raw_value):
    if raw_value is None:
        return None
    feature_cols = [item.strip() for item in raw_value.split(',') if item.strip()]
    return feature_cols if feature_cols else None


def parse_csv_list(raw_value):
    if raw_value is None:
        return []
    return [item.strip() for item in raw_value.split(',') if item.strip()]


def parse_target_columns(raw_value):
    target_cols = [item.strip() for item in raw_value.split(',') if item.strip()]
    if not target_cols:
        raise ValueError('target_cols must contain at least one column')
    return target_cols


def normalize_date_series(series, label):
    parsed = pd.to_datetime(series, errors='coerce')
    if parsed.isna().all():
        raise ValueError('Failed to parse any values in %s as dates' % label)
    return parsed.dt.strftime('%Y-%m-%d')


def resolve_optional_input_files(source_dir, user_value, fallback_candidates):
    candidates = parse_csv_list(user_value) if user_value else list(fallback_candidates)
    resolved = []
    for name in candidates:
        candidate = name if os.path.isabs(name) else os.path.join(source_dir, name)
        if os.path.exists(candidate):
            resolved.append(candidate)
    return resolved


def merge_prefer_existing(left_df, right_df, on, value_cols):
    if right_df.empty or not value_cols:
        return left_df

    merged = left_df.merge(right_df, on=on, how='left', suffixes=('', '__aux'))
    for col in value_cols:
        aux_col = col + '__aux'
        if aux_col not in merged.columns:
            continue
        if col in merged.columns:
            merged[col] = merged[col].combine_first(merged[aux_col])
        else:
            merged[col] = merged[aux_col]
        merged.drop(columns=[aux_col], inplace=True)
    return merged


def add_calendar_features(feature_df, date_col):
    parsed_dates = pd.to_datetime(feature_df[date_col], errors='coerce')
    if parsed_dates.isna().all():
        raise ValueError('Failed to derive calendar features from %s' % date_col)

    feature_df['星期几'] = parsed_dates.dt.dayofweek.astype(np.float32)
    feature_df['是否周末'] = (parsed_dates.dt.dayofweek >= 5).astype(np.float32)

    calendar = USFederalHolidayCalendar()
    holiday_dates = calendar.holidays(start=parsed_dates.min(), end=parsed_dates.max())
    holiday_set = set(pd.to_datetime(holiday_dates).strftime('%Y-%m-%d'))
    feature_df['是否节假日'] = feature_df[date_col].isin(holiday_set).astype(np.float32)
    return feature_df


def merge_auxiliary_feature_tables(feature_df, aux_paths, date_col, station_col):
    merge_summary = []
    for aux_path in aux_paths:
        aux_df = pd.read_csv(aux_path)
        if aux_df.empty:
            continue

        aux_station_col = resolve_column(
            aux_df,
            station_col if station_col in aux_df.columns else None,
            [station_col] + STATION_COLUMN_CANDIDATES,
            1,
            'auxiliary station column',
        )

        aux_date_col = None
        for candidate in [date_col] + DATE_COLUMN_CANDIDATES:
            if candidate in aux_df.columns:
                aux_date_col = candidate
                break

        rename_map = {}
        if aux_station_col != station_col:
            rename_map[aux_station_col] = station_col
        if aux_date_col and aux_date_col != date_col:
            rename_map[aux_date_col] = date_col
        if rename_map:
            aux_df = aux_df.rename(columns=rename_map)

        if date_col in aux_df.columns:
            aux_df[date_col] = normalize_date_series(aux_df[date_col], 'auxiliary date column')
        aux_df[station_col] = aux_df[station_col].astype(str)

        value_cols = [col for col in aux_df.columns if col not in {date_col, station_col}]
        if not value_cols:
            continue

        static_cols = []
        dynamic_cols = []
        if date_col in aux_df.columns:
            nunique_by_station = aux_df.groupby(station_col)[value_cols].nunique(dropna=True)
            for col in value_cols:
                if (nunique_by_station[col] <= 1).all():
                    static_cols.append(col)
                else:
                    dynamic_cols.append(col)
        else:
            static_cols = list(value_cols)

        if static_cols:
            static_df = aux_df[[station_col] + static_cols].drop_duplicates(subset=[station_col], keep='last')
            feature_df = merge_prefer_existing(feature_df, static_df, [station_col], static_cols)
        if dynamic_cols:
            dynamic_df = aux_df[[date_col, station_col] + dynamic_cols].drop_duplicates(
                subset=[date_col, station_col],
                keep='last',
            )
            feature_df = merge_prefer_existing(feature_df, dynamic_df, [date_col, station_col], dynamic_cols)

        merge_summary.append(
            {
                'file': aux_path,
                'static_cols': static_cols,
                'dynamic_cols': dynamic_cols,
            }
        )
    return feature_df, merge_summary


def generate_graph_seq2seq_io_data(x_values, y_values, x_offsets, y_offsets, known_future_values=None):
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(x_values.shape[0] - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_sample = x_values[t + x_offsets, ...]
        if known_future_values is not None:
            future_sample = known_future_values[t + y_offsets[0], ...]
            future_sample = np.repeat(future_sample[np.newaxis, ...], len(x_offsets), axis=0)
            x_sample = np.concatenate([x_sample, future_sample], axis=-1)
        x.append(x_sample)
        y.append(y_values[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def apply_log1p_transform(feature_values, feature_cols, selected_cols):
    if feature_values is None or not selected_cols:
        return feature_values, []

    transformed = feature_values.copy()
    applied_cols = []
    for feature_name in selected_cols:
        if feature_name not in feature_cols:
            continue
        feature_idx = feature_cols.index(feature_name)
        feature_slice = transformed[..., feature_idx]
        min_value = np.nanmin(feature_slice)
        if min_value < 0:
            continue
        transformed[..., feature_idx] = np.log1p(feature_slice)
        applied_cols.append(feature_name)
    return transformed, applied_cols


def split_and_save(
    x_values,
    y_values,
    output_dir,
    hist_len,
    pred_len,
    history_feature_cols,
    target_cols,
    known_future_values=None,
    known_future_feature_cols=None,
    log1p_feature_cols=None,
):
    x_offsets = np.sort(np.arange(-(hist_len - 1), 1, 1))
    y_offsets = np.sort(np.arange(1, pred_len + 1, 1))
    x, y = generate_graph_seq2seq_io_data(x_values, y_values, x_offsets, y_offsets, known_future_values)
    if len(x) == 0:
        raise ValueError(
            'No training samples were created. Reduce hist_len/pred_len or provide a longer time series.'
        )

    known_future_feature_cols = known_future_feature_cols or []
    log1p_feature_cols = log1p_feature_cols or []
    input_feature_cols = list(history_feature_cols) + ['future_' + col for col in known_future_feature_cols]
    input_log1p_feature_cols = (
        [col for col in history_feature_cols if col in log1p_feature_cols]
        + ['future_' + col for col in known_future_feature_cols if col in log1p_feature_cols]
    )

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    split_map = {
        'train': (x[:num_train], y[:num_train]),
        'val': (x[num_train:num_train + num_val], y[num_train:num_train + num_val]),
        'test': (x[-num_test:], y[-num_test:]),
    }

    os.makedirs(output_dir, exist_ok=True)
    for split_name, (split_x, split_y) in split_map.items():
        np.savez_compressed(
            os.path.join(output_dir, '%s.npz' % split_name),
            x=split_x,
            y=split_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
            input_feature_cols=np.array(input_feature_cols),
            history_feature_cols=np.array(history_feature_cols),
            known_future_feature_cols=np.array(known_future_feature_cols),
            log1p_feature_cols=np.array(input_log1p_feature_cols),
            target_cols=np.array(target_cols),
        )

    return {
        'num_samples': num_samples,
        'train': split_map['train'][0].shape,
        'val': split_map['val'][0].shape,
        'test': split_map['test'][0].shape,
    }


def generate_fallback_embedding(graphs, embed_dim, seed):
    features = np.concatenate(graphs, axis=1).astype(np.float32)
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((features.shape[1], embed_dim)).astype(np.float32)
    projection /= np.sqrt(features.shape[1])
    return features @ projection


def maybe_write_spatial_embedding(se_output_path, graphs, embed_dim, seed, se_strategy):
    if se_strategy == 'skip':
        return 'skipped'

    if se_strategy == 'preserve' and os.path.exists(se_output_path):
        return 'kept_existing'

    embedding = generate_fallback_embedding(graphs, embed_dim, seed)
    pd.DataFrame(embedding).to_csv(se_output_path, header=False, index=False)
    return 'saved_fallback'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default='\u4e8c\u6708\u4efd\u6570\u636e\u5904\u7406')
    parser.add_argument('--mapping_file', default='GNN_Node_Mapping.csv')
    parser.add_argument(
        '--temporal_file',
        default=None,
        help='Temporal table file under source_dir. If omitted, try 2Years_Daily_NetFlow.csv then ST_Master_Feature_Table.csv',
    )
    parser.add_argument('--mapping_station_col', default='\u7ad9\u70b9\u540d\u79f0')
    parser.add_argument('--date_col', default=None)
    parser.add_argument('--station_col', default=None)
    parser.add_argument(
        '--weather_file',
        default=None,
        help='Optional weather file under source_dir to merge by date',
    )
    parser.add_argument(
        '--aux_temporal_files',
        default=None,
        help='Optional comma-separated auxiliary temporal tables under source_dir for known-future/static features.',
    )
    parser.add_argument('--weather_date_col', default=None)
    parser.add_argument(
        '--input_feature_cols',
        default=None,
        help='Comma-separated historical feature columns for x. If omitted, auto uses demand history + historical weather.',
    )
    parser.add_argument(
        '--known_future_cols',
        default=None,
        help='Comma-separated target-day known feature columns. If omitted, auto uses morning inventory, capacity, calendar, and weather.',
    )
    parser.add_argument(
        '--log1p_feature_cols',
        default=None,
        help='Optional comma-separated non-negative input feature columns to transform with log1p before scaling.',
    )
    parser.add_argument(
        '--min_known_future_coverage',
        type=float,
        default=0.05,
        help='Skip target-day known feature columns whose date-station coverage is below this ratio.',
    )
    parser.add_argument('--output_name', default='bike')
    parser.add_argument(
        '--target_cols',
        default='\u65e5\u95f4\u9a91\u51fa\u91cf,\u65e5\u95f4\u9a91\u5165\u91cf',
        help='Comma-separated target columns for y',
    )
    parser.add_argument('--hist_len', type=int, default=7)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--keep_strategy', choices=['drop_incomplete', 'fill_zero'], default='fill_zero')
    parser.add_argument('--embedding_dim', type=int, default=144)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--keep_directed', action='store_true')
    parser.add_argument(
        '--se_strategy',
        choices=['preserve', 'fallback', 'skip'],
        default='preserve',
        help='How to handle data/SE/se_<output_name>.csv when regenerating data.',
    )
    args = parser.parse_args()

    source_dir = resolve_project_path(PROJECT_ROOT, args.source_dir)
    mapping_path = resolve_input_file(source_dir, args.mapping_file, [args.mapping_file], 'mapping')
    temporal_path = resolve_input_file(source_dir, args.temporal_file, DEFAULT_TEMPORAL_FILES, 'temporal')
    aux_paths = resolve_optional_input_files(source_dir, args.aux_temporal_files, DEFAULT_AUX_TEMPORAL_FILES)
    weather_candidates = resolve_optional_input_files(source_dir, args.weather_file, DEFAULT_WEATHER_FILES)
    weather_path = weather_candidates[0] if weather_candidates else None

    mapping_df = pd.read_csv(mapping_path).sort_values('Node_ID').reset_index(drop=True)
    if args.mapping_station_col not in mapping_df.columns:
        raise KeyError('Mapping station column not found: %s' % args.mapping_station_col)
    if 'Node_ID' not in mapping_df.columns:
        raise KeyError('Node_ID column not found in mapping file: %s' % mapping_path)

    feature_df = pd.read_csv(temporal_path)
    if feature_df.empty:
        raise ValueError('Temporal table is empty: %s' % temporal_path)

    date_col = resolve_column(feature_df, args.date_col, DATE_COLUMN_CANDIDATES, 0, 'date column')
    station_col = resolve_column(
        feature_df,
        args.station_col,
        [args.mapping_station_col] + STATION_COLUMN_CANDIDATES,
        1,
        'station column',
    )

    feature_df[station_col] = feature_df[station_col].astype(str)
    feature_df[date_col] = normalize_date_series(feature_df[date_col], 'temporal date column')
    full_dates = sorted(feature_df[date_col].dropna().unique())
    full_station_names = mapping_df[args.mapping_station_col].astype(str).tolist()
    full_index = pd.MultiIndex.from_product(
        [full_dates, full_station_names],
        names=[date_col, station_col],
    ).to_frame(index=False)
    feature_df = full_index.merge(feature_df, on=[date_col, station_col], how='left')
    feature_df, aux_merge_summary = merge_auxiliary_feature_tables(feature_df, aux_paths, date_col, station_col)
    feature_df = add_calendar_features(feature_df, date_col)

    resolved_weather_feature_cols = []
    missing_weather_dates = []
    if weather_path is not None:
        weather_df = pd.read_csv(weather_path)
        if weather_df.empty:
            raise ValueError('Weather table is empty: %s' % weather_path)
        weather_date_col = resolve_column(
            weather_df,
            args.weather_date_col,
            DATE_COLUMN_CANDIDATES,
            0,
            'weather date column',
        )
        resolved_weather_feature_cols = [col for col in weather_df.columns if col != weather_date_col]
        if not resolved_weather_feature_cols:
            raise ValueError('No weather feature columns found in %s' % weather_path)
        weather_df[weather_date_col] = normalize_date_series(weather_df[weather_date_col], 'weather date column')
        weather_df = weather_df.drop_duplicates(subset=[weather_date_col]).copy()
        if weather_date_col != date_col:
            weather_df = weather_df.rename(columns={weather_date_col: date_col})
        feature_dates = set(feature_df[date_col].dropna().unique())
        weather_dates = set(weather_df[date_col].dropna().unique())
        missing_weather_dates = sorted(feature_dates - weather_dates)
        feature_df = merge_prefer_existing(feature_df, weather_df, [date_col], resolved_weather_feature_cols)

    target_cols = parse_target_columns(args.target_cols)
    missing_targets = [col for col in target_cols if col not in feature_df.columns]
    if missing_targets:
        raise KeyError('Target columns not found: %s' % missing_targets)

    history_feature_cols = parse_feature_columns(args.input_feature_cols)
    if history_feature_cols is None:
        history_feature_cols = list(target_cols)
        for col in HISTORY_DEMAND_FEATURE_CANDIDATES:
            if col in feature_df.columns and col not in history_feature_cols:
                history_feature_cols.append(col)
        candidate_weather_cols = resolved_weather_feature_cols or WEATHER_FEATURE_CANDIDATES
        for col in candidate_weather_cols:
            if col in feature_df.columns and col not in history_feature_cols:
                history_feature_cols.append(col)
    for target_col in reversed(target_cols):
        if target_col not in history_feature_cols:
            history_feature_cols = [target_col] + history_feature_cols
    history_feature_cols = list(dict.fromkeys(history_feature_cols))
    missing_features = [col for col in history_feature_cols if col not in feature_df.columns]
    if missing_features:
        raise KeyError('Input feature columns not found: %s' % missing_features)

    known_future_feature_cols = parse_feature_columns(args.known_future_cols)
    if known_future_feature_cols is None:
        known_future_feature_cols = []
        candidate_weather_cols = resolved_weather_feature_cols or WEATHER_FEATURE_CANDIDATES
        for col in KNOWN_FUTURE_FEATURE_CANDIDATES:
            resolved_col = col
            if col in WEATHER_FEATURE_CANDIDATES and col not in candidate_weather_cols:
                continue
            if resolved_col in feature_df.columns and resolved_col not in known_future_feature_cols:
                known_future_feature_cols.append(resolved_col)
    known_future_feature_cols = [col for col in known_future_feature_cols if col in feature_df.columns]
    known_future_feature_cols = [col for col in known_future_feature_cols if col not in target_cols]
    known_future_feature_cols = list(dict.fromkeys(known_future_feature_cols))

    all_dates = sorted(feature_df[date_col].dropna().unique())
    ordered_stations = mapping_df[args.mapping_station_col].tolist()

    target_feature_pivots = []
    for target_col in target_cols:
        target_pivot = feature_df.pivot_table(
            index=date_col,
            columns=station_col,
            values=target_col,
            aggfunc='sum',
        )
        target_pivot = target_pivot.reindex(index=all_dates, columns=ordered_stations)
        target_feature_pivots.append(target_pivot)

    if args.keep_strategy == 'drop_incomplete':
        complete_mask = np.ones(len(ordered_stations), dtype=bool)
        for target_pivot in target_feature_pivots:
            complete_mask &= ~target_pivot.isna().any(axis=0).to_numpy()
        kept_stations = target_feature_pivots[0].columns[complete_mask]
        selected_mapping = mapping_df[mapping_df[args.mapping_station_col].isin(kept_stations)].copy()
    else:
        selected_mapping = mapping_df.copy()

    if selected_mapping.empty:
        raise ValueError('No stations left after keep_strategy=%s' % args.keep_strategy)

    selected_mapping = selected_mapping.sort_values('Node_ID').reset_index(drop=True)
    selected_station_names = selected_mapping[args.mapping_station_col].tolist()

    y_feature_matrices = []
    for target_pivot in target_feature_pivots:
        target_pivot = target_pivot.reindex(columns=selected_station_names).fillna(0.0)
        y_feature_matrices.append(target_pivot.values.astype(np.float32))
    y_values = np.stack(y_feature_matrices, axis=-1)

    x_feature_matrices = []
    for feature_name in history_feature_cols:
        feature_pivot = feature_df.pivot_table(
            index=date_col,
            columns=station_col,
            values=feature_name,
            aggfunc='mean',
        )
        feature_pivot = feature_pivot.reindex(index=all_dates, columns=selected_station_names).fillna(0.0)
        x_feature_matrices.append(feature_pivot.values.astype(np.float32))
    x_values = np.stack(x_feature_matrices, axis=-1)

    known_future_values = None
    known_future_coverage = {}
    skipped_known_future_cols = {}
    if known_future_feature_cols:
        known_future_matrices = []
        filtered_known_future_cols = []
        for feature_name in known_future_feature_cols:
            feature_pivot = feature_df.pivot_table(
                index=date_col,
                columns=station_col,
                values=feature_name,
                aggfunc='mean',
            )
            feature_pivot = feature_pivot.reindex(index=all_dates, columns=selected_station_names)
            coverage = float(feature_pivot.notna().mean().mean())
            known_future_coverage[feature_name] = coverage
            if coverage < args.min_known_future_coverage:
                skipped_known_future_cols[feature_name] = coverage
                continue
            feature_pivot = feature_pivot.fillna(0.0)
            known_future_matrices.append(feature_pivot.values.astype(np.float32))
            filtered_known_future_cols.append(feature_name)
        known_future_feature_cols = filtered_known_future_cols
        if known_future_matrices:
            known_future_values = np.stack(known_future_matrices, axis=-1)

    requested_log1p_cols = parse_feature_columns(args.log1p_feature_cols)
    if requested_log1p_cols is None:
        requested_log1p_cols = []
        for col in LOG1P_FEATURE_CANDIDATES:
            if col in history_feature_cols or col in known_future_feature_cols:
                requested_log1p_cols.append(col)
    requested_log1p_cols = list(dict.fromkeys(requested_log1p_cols))

    x_values, applied_history_log1p_cols = apply_log1p_transform(
        x_values,
        history_feature_cols,
        requested_log1p_cols,
    )
    known_future_values, applied_known_future_log1p_cols = apply_log1p_transform(
        known_future_values,
        known_future_feature_cols,
        requested_log1p_cols,
    )
    applied_log1p_feature_cols = [
        col for col in requested_log1p_cols
        if col in applied_history_log1p_cols or col in applied_known_future_log1p_cols
    ]

    if x_values.shape[0] != y_values.shape[0] or x_values.shape[1] != y_values.shape[1]:
        raise ValueError('x/y alignment mismatch: x=%s, y=%s' % (x_values.shape, y_values.shape))

    temporal_output_dir = os.path.join(PROJECT_ROOT, 'data', 'temporal_data', args.output_name)
    graph_output_dir = os.path.join(PROJECT_ROOT, 'data', 'graph', args.output_name)
    se_output_path = os.path.join(PROJECT_ROOT, 'data', 'SE', 'se_%s.csv' % args.output_name)

    os.makedirs(graph_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(se_output_path), exist_ok=True)

    node_indices = selected_mapping['Node_ID'].to_numpy(dtype=np.int64)
    graph_list = []
    for output_name, source_name in SOURCE_GRAPH_FILES.items():
        graph_path = os.path.join(source_dir, source_name)
        if not os.path.exists(graph_path):
            raise FileNotFoundError('Missing source graph file: %s' % graph_path)
        graph = np.load(graph_path).astype(np.float32)
        if graph.shape != (len(mapping_df), len(mapping_df)):
            raise ValueError('Unexpected graph shape for %s: %s' % (source_name, graph.shape))
        graph = graph[np.ix_(node_indices, node_indices)]
        if output_name in SYMMETRIC_OUTPUTS and not args.keep_directed:
            graph = 0.5 * (graph + graph.T)
            np.fill_diagonal(graph, 1.0)
        np.save(os.path.join(graph_output_dir, output_name), graph)
        graph_list.append(graph)

    se_status = maybe_write_spatial_embedding(
        se_output_path,
        graph_list,
        args.embedding_dim,
        args.seed,
        args.se_strategy,
    )
    selected_mapping.to_csv(os.path.join(graph_output_dir, 'selected_node_mapping.csv'), index=False)

    split_summary = split_and_save(
        x_values,
        y_values,
        temporal_output_dir,
        args.hist_len,
        args.pred_len,
        history_feature_cols,
        target_cols,
        known_future_values=known_future_values,
        known_future_feature_cols=known_future_feature_cols,
        log1p_feature_cols=applied_log1p_feature_cols,
    )

    print('Prepared dataset:', args.output_name)
    print('Project root:', PROJECT_ROOT)
    print('Temporal source:', temporal_path)
    print('Auxiliary sources:', aux_paths)
    print('Weather source:', weather_path if weather_path is not None else 'None')
    print('Weather feature columns:', resolved_weather_feature_cols)
    print('Missing weather dates:', len(missing_weather_dates))
    print('Calendar features:', ['星期几', '是否周末', '是否节假日'])
    print('Auxiliary merge summary:', aux_merge_summary)
    print('Temporal columns:', {'date_col': date_col, 'station_col': station_col, 'target_cols': target_cols})
    print('Historical input feature columns:', history_feature_cols)
    print('Known future feature columns:', known_future_feature_cols)
    print('Applied log1p feature columns:', applied_log1p_feature_cols)
    print('Known future coverage:', known_future_coverage)
    print('Skipped known future columns:', skipped_known_future_cols)
    print('Target columns:', target_cols)
    print('Dates:', len(all_dates))
    if all_dates:
        print('Date range:', all_dates[0], '->', all_dates[-1])
    print('Original nodes:', len(mapping_df))
    print('Selected nodes:', len(selected_mapping))
    print('Temporal x shape:', x_values.shape)
    print('Temporal y shape:', y_values.shape)
    print('Saved temporal data to:', temporal_output_dir)
    print('Saved graph data to:', graph_output_dir)
    print('Spatial embedding status:', se_status)
    print('Spatial embedding path:', se_output_path)
    print('Split summary:', split_summary)


if __name__ == '__main__':
    main()
