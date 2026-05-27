import glob
import os

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


DATE_COLUMN_CANDIDATES = ['日期', 'date', 'Date', 'datetime', '时间']
STATION_COLUMN_CANDIDATES = ['站点名称', 'station_name', 'Station', 'station', '站点']
WEATHER_FEATURE_CANDIDATES = [
    '最高气温(°C)',
    '最低气温(°C)',
    '总降水量(mm)',
    '总降雪量(cm)',
    '最大风速(km/h)',
]
KNOWN_FUTURE_FEATURE_CANDIDATES = [
    '早晨初始车辆数',
    '总桩数',
    '星期几',
    '是否周末',
    '是否节假日',
] + WEATHER_FEATURE_CANDIDATES
HOURLY_HISTORY_FEATURE_CANDIDATES = [
    '小时骑出量',
    '小时骑入量',
    '小时净流量',
    '小时',
    '星期几',
    '是否周末',
    '是否节假日',
] + WEATHER_FEATURE_CANDIDATES
LOG1P_FEATURE_CANDIDATES = [
    '小时骑出量',
    '小时骑入量',
    '小时净流量',
    '早晨初始车辆数',
    '总桩数',
]
DEFAULT_TRIP_PATTERNS = [
    os.path.join('纽约单车订单数据', 'JC-*-citibike-tripdata.csv'),
    'JC-202602-citibike-tripdata.csv',
]
DEFAULT_AUX_TEMPORAL_FILES = [
    'FINAL_JC_BaseTable_with_POI.csv',
    'CLEAN_JC_BaseTable_2026-02-01_to_2026-02-28.csv',
]
DEFAULT_WEATHER_FILES = [
    os.path.join('weather-get', 'NYC_Weather_2024-01-01_to_2026-02-31.csv'),
    os.path.join('weather-get', 'NYC_Weather_2024-01-01_to_2026-01-31.csv'),
]
SOURCE_GRAPH_FILES = {
    'dist.npy': 'graph_spatial_distance.npy',
    'neigh.npy': 'graph_od_transition.npy',
    'func.npy': 'graph_poi_semantic.npy',
    'bike_heuristic.npy': 'graph_heuristic.npy',
    'tempp_bike.npy': 'graph_demand_correlation.npy',
}
SYMMETRIC_OUTPUTS = {'neigh.npy', 'bike_heuristic.npy'}


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


def resolve_project_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(base_dir, path_value)


def parse_csv_list(raw_value):
    if raw_value is None:
        return []
    return [item.strip() for item in raw_value.split(',') if item.strip()]


def parse_feature_columns(raw_value):
    if raw_value is None:
        return None
    items = [item.strip() for item in raw_value.split(',') if item.strip()]
    return items if items else None


def resolve_optional_input_files(source_dir, user_value, fallback_candidates):
    candidates = parse_csv_list(user_value) if user_value else list(fallback_candidates)
    resolved = []
    for name in candidates:
        candidate = name if os.path.isabs(name) else os.path.join(source_dir, name)
        if os.path.exists(candidate):
            resolved.append(candidate)
    return resolved


def resolve_input_file(source_dir, user_value, fallback_candidates, label):
    if user_value:
        candidate = user_value if os.path.isabs(user_value) else os.path.join(source_dir, user_value)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError('Missing %s file: %s' % (label, candidate))

    for name in fallback_candidates:
        candidate = name if os.path.isabs(name) else os.path.join(source_dir, name)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError('Missing %s file under %s' % (label, source_dir))


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


def normalize_date_series(series, label):
    parsed = pd.to_datetime(series, errors='coerce')
    if parsed.isna().all():
        raise ValueError('Failed to parse any values in %s as dates' % label)
    return parsed.dt.strftime('%Y-%m-%d')


def normalize_datetime_series(series, label):
    parsed = pd.to_datetime(series, errors='coerce')
    if parsed.isna().all():
        raise ValueError('Failed to parse any values in %s as datetimes' % label)
    return parsed.dt.floor('h')


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

        aux_df[station_col] = aux_df[station_col].astype(str)
        if date_col in aux_df.columns:
            aux_df[date_col] = normalize_date_series(aux_df[date_col], 'auxiliary date column')

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

        merge_summary.append({'file': aux_path, 'static_cols': static_cols, 'dynamic_cols': dynamic_cols})
    return feature_df, merge_summary


def load_mapping(source_dir, mapping_file, mapping_station_col='站点名称'):
    mapping_path = resolve_input_file(source_dir, mapping_file, [mapping_file], 'mapping')
    mapping_df = pd.read_csv(mapping_path)
    if mapping_station_col not in mapping_df.columns:
        raise KeyError('Mapping station column not found: %s' % mapping_station_col)
    if 'Node_ID' not in mapping_df.columns:
        raise KeyError('Node_ID column not found in mapping file: %s' % mapping_path)
    mapping_df = mapping_df.sort_values('Node_ID').reset_index(drop=True)
    mapping_df[mapping_station_col] = mapping_df[mapping_station_col].astype(str)
    return mapping_path, mapping_df


def resolve_trip_files(source_dir, trip_files):
    patterns = parse_csv_list(trip_files) if trip_files else list(DEFAULT_TRIP_PATTERNS)
    resolved = []
    for pattern in patterns:
        candidate = pattern if os.path.isabs(pattern) else os.path.join(source_dir, pattern)
        if os.path.isfile(candidate):
            resolved.append(candidate)
            continue
        resolved.extend(sorted(glob.glob(candidate)))
    resolved = sorted(set(resolved))
    if not resolved:
        raise FileNotFoundError('No trip files matched under %s' % source_dir)
    return resolved


def aggregate_hourly_trip_counts(trip_files, station_names):
    station_set = set(station_names)
    hourly_frames = []
    use_cols = ['started_at', 'ended_at', 'start_station_name', 'end_station_name']
    for trip_path in trip_files:
        trip_df = pd.read_csv(trip_path, usecols=lambda col: col in use_cols)
        if trip_df.empty:
            continue

        out_df = trip_df[['started_at', 'start_station_name']].copy()
        out_df = out_df.rename(columns={'started_at': 'datetime', 'start_station_name': '站点名称'})
        out_df['datetime'] = normalize_datetime_series(out_df['datetime'], 'started_at')
        out_df['站点名称'] = out_df['站点名称'].astype(str)
        out_df = out_df[out_df['站点名称'].isin(station_set)]
        out_df['小时骑出量'] = 1.0
        out_df = out_df.groupby(['datetime', '站点名称'], as_index=False)['小时骑出量'].sum()

        in_df = trip_df[['ended_at', 'end_station_name']].copy()
        in_df = in_df.rename(columns={'ended_at': 'datetime', 'end_station_name': '站点名称'})
        in_df['datetime'] = normalize_datetime_series(in_df['datetime'], 'ended_at')
        in_df['站点名称'] = in_df['站点名称'].astype(str)
        in_df = in_df[in_df['站点名称'].isin(station_set)]
        in_df['小时骑入量'] = 1.0
        in_df = in_df.groupby(['datetime', '站点名称'], as_index=False)['小时骑入量'].sum()

        merged = out_df.merge(in_df, on=['datetime', '站点名称'], how='outer').fillna(0.0)
        hourly_frames.append(merged)

    if not hourly_frames:
        raise ValueError('No hourly trip rows were generated from trip files.')

    hourly_df = pd.concat(hourly_frames, ignore_index=True)
    hourly_df = hourly_df.groupby(['datetime', '站点名称'], as_index=False)[['小时骑出量', '小时骑入量']].sum()
    start_dt = hourly_df['datetime'].min().floor('h')
    end_dt = hourly_df['datetime'].max().floor('h')
    full_hours = pd.date_range(start=start_dt, end=end_dt, freq='h')
    full_index = pd.MultiIndex.from_product([full_hours, station_names], names=['datetime', '站点名称']).to_frame(index=False)
    hourly_df = full_index.merge(hourly_df, on=['datetime', '站点名称'], how='left')
    hourly_df['小时骑出量'] = hourly_df['小时骑出量'].fillna(0.0).astype(np.float32)
    hourly_df['小时骑入量'] = hourly_df['小时骑入量'].fillna(0.0).astype(np.float32)
    hourly_df['小时净流量'] = (hourly_df['小时骑入量'] - hourly_df['小时骑出量']).astype(np.float32)
    hourly_df['日期'] = hourly_df['datetime'].dt.strftime('%Y-%m-%d')
    hourly_df['小时'] = hourly_df['datetime'].dt.hour.astype(np.float32)
    return hourly_df, full_hours


def load_daily_feature_table(
    source_dir,
    station_names,
    all_dates,
    daily_feature_file=None,
    aux_temporal_files=None,
    weather_file=None,
):
    base_df = pd.MultiIndex.from_product(
        [all_dates, station_names],
        names=['日期', '站点名称'],
    ).to_frame(index=False)

    daily_candidates = [daily_feature_file] if daily_feature_file else ['ST_Master_Feature_Table.csv']
    resolved_daily_path = None
    for name in daily_candidates:
        if not name:
            continue
        candidate = name if os.path.isabs(name) else os.path.join(source_dir, name)
        if os.path.exists(candidate):
            resolved_daily_path = candidate
            break

    merge_summary = []
    feature_df = base_df.copy()
    if resolved_daily_path is not None:
        daily_df = pd.read_csv(resolved_daily_path)
        if not daily_df.empty:
            date_col = resolve_column(daily_df, None, DATE_COLUMN_CANDIDATES, 0, 'daily feature date column')
            station_col = resolve_column(daily_df, None, STATION_COLUMN_CANDIDATES, 1, 'daily feature station column')
            daily_df = daily_df.rename(columns={date_col: '日期', station_col: '站点名称'})
            daily_df['日期'] = normalize_date_series(daily_df['日期'], 'daily feature date column')
            daily_df['站点名称'] = daily_df['站点名称'].astype(str)
            value_cols = [col for col in daily_df.columns if col not in {'日期', '站点名称'}]
            feature_df = merge_prefer_existing(
                feature_df,
                daily_df[['日期', '站点名称'] + value_cols].drop_duplicates(subset=['日期', '站点名称'], keep='last'),
                ['日期', '站点名称'],
                value_cols,
            )

    aux_paths = resolve_optional_input_files(source_dir, aux_temporal_files, DEFAULT_AUX_TEMPORAL_FILES)
    feature_df, aux_summary = merge_auxiliary_feature_tables(feature_df, aux_paths, '日期', '站点名称')
    merge_summary.extend(aux_summary)
    feature_df = add_calendar_features(feature_df, '日期')

    weather_path = None
    weather_candidates = resolve_optional_input_files(source_dir, weather_file, DEFAULT_WEATHER_FILES)
    if weather_candidates:
        weather_path = weather_candidates[0]
        weather_df = pd.read_csv(weather_path)
        weather_date_col = resolve_column(weather_df, None, DATE_COLUMN_CANDIDATES, 0, 'weather date column')
        weather_df = weather_df.rename(columns={weather_date_col: '日期'})
        weather_df['日期'] = normalize_date_series(weather_df['日期'], 'weather date column')
        weather_cols = [col for col in weather_df.columns if col != '日期']
        feature_df = merge_prefer_existing(
            feature_df,
            weather_df[['日期'] + weather_cols].drop_duplicates(subset=['日期'], keep='last'),
            ['日期'],
            weather_cols,
        )

    return feature_df, resolved_daily_path, aux_paths, weather_path, merge_summary


def build_hourly_feature_frame(hourly_df, daily_feature_df):
    return hourly_df.merge(daily_feature_df, on=['日期', '站点名称'], how='left')


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


def copy_graphs_and_embeddings(
    source_dir,
    graph_output_dir,
    se_output_path,
    mapping_df,
    keep_directed,
    embedding_dim,
    seed,
    se_strategy,
):
    os.makedirs(graph_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(se_output_path), exist_ok=True)

    node_indices = mapping_df['Node_ID'].to_numpy(dtype=np.int64)
    graph_list = []
    for output_name, source_name in SOURCE_GRAPH_FILES.items():
        graph_path = os.path.join(source_dir, source_name)
        if not os.path.exists(graph_path):
            raise FileNotFoundError('Missing source graph file: %s' % graph_path)
        graph = np.load(graph_path).astype(np.float32)
        graph = graph[np.ix_(node_indices, node_indices)]
        if output_name in SYMMETRIC_OUTPUTS and not keep_directed:
            graph = 0.5 * (graph + graph.T)
            np.fill_diagonal(graph, 1.0)
        np.save(os.path.join(graph_output_dir, output_name), graph)
        graph_list.append(graph)

    se_status = maybe_write_spatial_embedding(
        se_output_path,
        graph_list,
        embedding_dim,
        seed,
        se_strategy,
    )
    mapping_df.to_csv(os.path.join(graph_output_dir, 'selected_node_mapping.csv'), index=False, encoding='utf-8-sig')
    return se_status


def build_hourly_samples(
    feature_df,
    mapping_df,
    history_feature_cols,
    target_cols,
    known_future_feature_cols,
    hist_len,
    pred_len,
    min_known_future_coverage,
):
    station_names = mapping_df['站点名称'].tolist()
    datetimes = pd.Index(sorted(feature_df['datetime'].dropna().unique()))
    dates = sorted(feature_df['日期'].dropna().unique())

    history_matrices = []
    for feature_name in history_feature_cols:
        pivot = feature_df.pivot_table(
            index='datetime',
            columns='站点名称',
            values=feature_name,
            aggfunc='mean',
        )
        pivot = pivot.reindex(index=datetimes, columns=station_names).fillna(0.0)
        history_matrices.append(pivot.to_numpy(dtype=np.float32))
    history_values = np.stack(history_matrices, axis=-1)

    target_matrices = []
    for target_name in target_cols:
        pivot = feature_df.pivot_table(
            index='datetime',
            columns='站点名称',
            values=target_name,
            aggfunc='mean',
        )
        pivot = pivot.reindex(index=datetimes, columns=station_names).fillna(0.0)
        target_matrices.append(pivot.to_numpy(dtype=np.float32))
    target_values = np.stack(target_matrices, axis=-1)

    known_future_values = None
    known_future_coverage = {}
    skipped_known_future_cols = {}
    filtered_known_future_cols = []
    if known_future_feature_cols:
        daily_feature_df = feature_df[['日期', '站点名称'] + known_future_feature_cols].drop_duplicates(
            subset=['日期', '站点名称'],
            keep='last',
        )
        future_matrices = []
        for feature_name in known_future_feature_cols:
            pivot = daily_feature_df.pivot_table(
                index='日期',
                columns='站点名称',
                values=feature_name,
                aggfunc='mean',
            )
            pivot = pivot.reindex(index=dates, columns=station_names)
            coverage = float(pivot.notna().mean().mean())
            known_future_coverage[feature_name] = coverage
            if coverage < min_known_future_coverage:
                skipped_known_future_cols[feature_name] = coverage
                continue
            future_matrices.append(pivot.fillna(0.0).to_numpy(dtype=np.float32))
            filtered_known_future_cols.append(feature_name)
        known_future_feature_cols = filtered_known_future_cols
        if future_matrices:
            known_future_values = np.stack(future_matrices, axis=-1)

    datetime_to_idx = {dt: idx for idx, dt in enumerate(datetimes)}
    date_to_idx = {date_value: idx for idx, date_value in enumerate(dates)}

    x_samples = []
    y_samples = []
    sample_dates = []
    for date_value in dates:
        target_start = pd.Timestamp('%s 00:00:00' % date_value)
        if target_start not in datetime_to_idx:
            continue
        target_start_idx = datetime_to_idx[target_start]
        hist_start_idx = target_start_idx - hist_len
        target_end_idx = target_start_idx + pred_len
        if hist_start_idx < 0 or target_end_idx > len(datetimes):
            continue

        x_window = history_values[hist_start_idx:target_start_idx]
        y_window = target_values[target_start_idx:target_end_idx]
        if known_future_values is not None:
            future_date_idx = date_to_idx[date_value]
            future_window = np.repeat(
                known_future_values[future_date_idx][np.newaxis, ...],
                hist_len,
                axis=0,
            )
            x_window = np.concatenate([x_window, future_window], axis=-1)

        x_samples.append(x_window.astype(np.float32))
        y_samples.append(y_window.astype(np.float32))
        sample_dates.append(date_value)

    if not x_samples:
        raise ValueError('No hourly samples were created. Check hist_len/pred_len and source data coverage.')

    return {
        'x': np.stack(x_samples, axis=0),
        'y': np.stack(y_samples, axis=0),
        'sample_dates': sample_dates,
        'known_future_feature_cols': known_future_feature_cols,
        'known_future_coverage': known_future_coverage,
        'skipped_known_future_cols': skipped_known_future_cols,
    }


def split_and_save_hourly_dataset(
    x_values,
    y_values,
    sample_dates,
    output_dir,
    history_feature_cols,
    target_cols,
    known_future_feature_cols,
    log1p_feature_cols,
):
    num_samples = x_values.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    split_map = {
        'train': (x_values[:num_train], y_values[:num_train], sample_dates[:num_train]),
        'val': (
            x_values[num_train:num_train + num_val],
            y_values[num_train:num_train + num_val],
            sample_dates[num_train:num_train + num_val],
        ),
        'test': (x_values[-num_test:], y_values[-num_test:], sample_dates[-num_test:]),
    }

    input_feature_cols = list(history_feature_cols) + ['future_' + col for col in known_future_feature_cols]
    input_log1p_feature_cols = (
        [col for col in history_feature_cols if col in log1p_feature_cols]
        + ['future_' + col for col in known_future_feature_cols if col in log1p_feature_cols]
    )

    os.makedirs(output_dir, exist_ok=True)
    for split_name, (split_x, split_y, split_dates) in split_map.items():
        np.savez_compressed(
            os.path.join(output_dir, '%s.npz' % split_name),
            x=split_x,
            y=split_y,
            input_feature_cols=np.array(input_feature_cols),
            history_feature_cols=np.array(history_feature_cols),
            known_future_feature_cols=np.array(known_future_feature_cols),
            log1p_feature_cols=np.array(input_log1p_feature_cols),
            target_cols=np.array(target_cols),
            sample_dates=np.array(split_dates),
        )

    return {
        'num_samples': num_samples,
        'train': split_map['train'][0].shape,
        'val': split_map['val'][0].shape,
        'test': split_map['test'][0].shape,
    }


def build_safe_inventory_bounds(net_flow, capacity):
    net_flow = np.asarray(net_flow, dtype=np.float64)
    cumulative = np.cumsum(net_flow)
    max_drawdown = float(np.max(np.concatenate([[0.0], -cumulative])))
    s_min = int(np.ceil(max(0.0, max_drawdown)))
    if capacity is None or np.isnan(capacity):
        s_max = np.nan
    else:
        max_increase = float(np.max(np.concatenate([[0.0], cumulative])))
        s_max = int(np.floor(float(capacity) - max_increase))
    feasible = bool(np.isnan(s_max) or s_min <= s_max)
    return s_min, s_max, cumulative, feasible


def compute_adjustment_to_interval(current_inventory, s_min, s_max):
    if current_inventory is None or pd.isna(current_inventory):
        return np.nan, 'unknown'
    current_inventory = float(current_inventory)
    if current_inventory < s_min:
        return float(s_min - current_inventory), 'dispatch_in'
    if not pd.isna(s_max) and current_inventory > s_max:
        return float(current_inventory - s_max), 'dispatch_out'
    return 0.0, 'already_safe'
