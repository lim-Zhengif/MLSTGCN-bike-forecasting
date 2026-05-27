import argparse
import os

import numpy as np

from hourly_pipeline_utils import (
    DEFAULT_TRIP_PATTERNS,
    HOURLY_HISTORY_FEATURE_CANDIDATES,
    KNOWN_FUTURE_FEATURE_CANDIDATES,
    LOG1P_FEATURE_CANDIDATES,
    aggregate_hourly_trip_counts,
    apply_log1p_transform,
    build_hourly_feature_frame,
    build_hourly_samples,
    copy_graphs_and_embeddings,
    detect_project_root,
    load_daily_feature_table,
    load_mapping,
    parse_feature_columns,
    resolve_project_path,
    resolve_trip_files,
    split_and_save_hourly_dataset,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = detect_project_root(SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default='二月份数据处理')
    parser.add_argument('--mapping_file', default='GNN_Node_Mapping.csv')
    parser.add_argument('--mapping_station_col', default='站点名称')
    parser.add_argument(
        '--trip_files',
        default=','.join(DEFAULT_TRIP_PATTERNS),
        help='Comma-separated trip csv paths or glob patterns under source_dir.',
    )
    parser.add_argument('--daily_feature_file', default='ST_Master_Feature_Table.csv')
    parser.add_argument(
        '--aux_temporal_files',
        default='FINAL_JC_BaseTable_with_POI.csv,CLEAN_JC_BaseTable_2026-02-01_to_2026-02-28.csv',
    )
    parser.add_argument('--weather_file', default=None)
    parser.add_argument('--history_feature_cols', default=None)
    parser.add_argument('--known_future_cols', default=None)
    parser.add_argument('--log1p_feature_cols', default=None)
    parser.add_argument('--hist_len', type=int, default=24 * 7)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--min_known_future_coverage', type=float, default=0.05)
    parser.add_argument('--output_name', default='bike_hourly_safe_inventory')
    parser.add_argument('--embedding_dim', type=int, default=144)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--keep_directed', action='store_true')
    parser.add_argument('--se_strategy', choices=['preserve', 'fallback', 'skip'], default='preserve')
    args = parser.parse_args()

    source_dir = resolve_project_path(PROJECT_ROOT, args.source_dir)
    _, mapping_df = load_mapping(source_dir, args.mapping_file, args.mapping_station_col)
    station_names = mapping_df[args.mapping_station_col].tolist()

    trip_files = resolve_trip_files(source_dir, args.trip_files)
    hourly_df, all_hours = aggregate_hourly_trip_counts(trip_files, station_names)
    all_dates = sorted(hourly_df['日期'].dropna().unique())

    daily_feature_df, daily_feature_path, aux_paths, weather_path, aux_merge_summary = load_daily_feature_table(
        source_dir=source_dir,
        station_names=station_names,
        all_dates=all_dates,
        daily_feature_file=args.daily_feature_file,
        aux_temporal_files=args.aux_temporal_files,
        weather_file=args.weather_file,
    )
    feature_df = build_hourly_feature_frame(hourly_df, daily_feature_df)

    history_feature_cols = parse_feature_columns(args.history_feature_cols) or [
        col for col in HOURLY_HISTORY_FEATURE_CANDIDATES if col in feature_df.columns
    ]
    target_cols = ['小时骑出量', '小时骑入量']
    for target_col in reversed(target_cols):
        if target_col not in history_feature_cols:
            history_feature_cols = [target_col] + history_feature_cols
    history_feature_cols = list(dict.fromkeys(history_feature_cols))

    known_future_feature_cols = parse_feature_columns(args.known_future_cols) or [
        col for col in KNOWN_FUTURE_FEATURE_CANDIDATES if col in feature_df.columns
    ]
    known_future_feature_cols = [col for col in known_future_feature_cols if col not in target_cols]
    known_future_feature_cols = list(dict.fromkeys(known_future_feature_cols))

    sample_bundle = build_hourly_samples(
        feature_df=feature_df,
        mapping_df=mapping_df.rename(columns={args.mapping_station_col: '站点名称'}),
        history_feature_cols=history_feature_cols,
        target_cols=target_cols,
        known_future_feature_cols=known_future_feature_cols,
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        min_known_future_coverage=args.min_known_future_coverage,
    )

    requested_log1p_cols = parse_feature_columns(args.log1p_feature_cols) or [
        col for col in LOG1P_FEATURE_CANDIDATES
        if col in history_feature_cols or col in sample_bundle['known_future_feature_cols']
    ]
    requested_log1p_cols = list(dict.fromkeys(requested_log1p_cols))

    x_values = sample_bundle['x']
    history_x, applied_history_cols = apply_log1p_transform(
        x_values[..., :len(history_feature_cols)],
        history_feature_cols,
        requested_log1p_cols,
    )
    if sample_bundle['known_future_feature_cols']:
        future_x, applied_future_cols = apply_log1p_transform(
            x_values[..., len(history_feature_cols):],
            sample_bundle['known_future_feature_cols'],
            requested_log1p_cols,
        )
        x_values = np.concatenate([history_x, future_x], axis=-1)
    else:
        applied_future_cols = []
        x_values = history_x

    applied_log1p_feature_cols = [
        col for col in requested_log1p_cols
        if col in applied_history_cols or col in applied_future_cols
    ]

    temporal_output_dir = os.path.join(PROJECT_ROOT, 'data', 'temporal_data', args.output_name)
    graph_output_dir = os.path.join(PROJECT_ROOT, 'data', 'graph', args.output_name)
    se_output_path = os.path.join(PROJECT_ROOT, 'data', 'SE', 'se_%s.csv' % args.output_name)

    split_summary = split_and_save_hourly_dataset(
        x_values=x_values,
        y_values=sample_bundle['y'],
        sample_dates=sample_bundle['sample_dates'],
        output_dir=temporal_output_dir,
        history_feature_cols=history_feature_cols,
        target_cols=target_cols,
        known_future_feature_cols=sample_bundle['known_future_feature_cols'],
        log1p_feature_cols=applied_log1p_feature_cols,
    )

    se_status = copy_graphs_and_embeddings(
        source_dir=source_dir,
        graph_output_dir=graph_output_dir,
        se_output_path=se_output_path,
        mapping_df=mapping_df,
        keep_directed=args.keep_directed,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
        se_strategy=args.se_strategy,
    )

    print('Prepared hourly dataset:', args.output_name)
    print('Project root:', PROJECT_ROOT)
    print('Trip files:', len(trip_files))
    print('Hourly datetime range:', str(all_hours.min()), '->', str(all_hours.max()))
    print('Daily feature file:', daily_feature_path)
    print('Auxiliary feature files:', aux_paths)
    print('Weather file:', weather_path)
    print('Auxiliary merge summary:', aux_merge_summary)
    print('History feature columns:', history_feature_cols)
    print('Known future feature columns:', sample_bundle['known_future_feature_cols'])
    print('Known future coverage:', sample_bundle['known_future_coverage'])
    print('Skipped known future columns:', sample_bundle['skipped_known_future_cols'])
    print('Applied log1p feature columns:', applied_log1p_feature_cols)
    print('Temporal x shape:', x_values.shape)
    print('Temporal y shape:', sample_bundle['y'].shape)
    print('Saved temporal data to:', temporal_output_dir)
    print('Saved graph data to:', graph_output_dir)
    print('Spatial embedding status:', se_status)
    print('Spatial embedding path:', se_output_path)
    print('Split summary:', split_summary)


if __name__ == '__main__':
    main()
