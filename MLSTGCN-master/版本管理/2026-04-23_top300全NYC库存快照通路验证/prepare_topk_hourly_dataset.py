import argparse
import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOURLY_UTILS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "2026-04-09_小时级骑入骑出_安全库存区间")
if HOURLY_UTILS_DIR not in sys.path:
    sys.path.insert(0, HOURLY_UTILS_DIR)

from hourly_pipeline_utils import (  # noqa: E402
    HOURLY_HISTORY_FEATURE_CANDIDATES,
    KNOWN_FUTURE_FEATURE_CANDIDATES,
    LOG1P_FEATURE_CANDIDATES,
    WEATHER_FEATURE_CANDIDATES,
    add_calendar_features,
    aggregate_hourly_trip_counts,
    apply_log1p_transform,
    build_hourly_feature_frame,
    build_hourly_samples,
    copy_graphs_and_embeddings,
    detect_project_root,
    load_daily_feature_table,
    load_mapping,
    resolve_trip_files,
    split_and_save_hourly_dataset,
)


PROJECT_ROOT = detect_project_root(SCRIPT_DIR)


def select_history_and_target_cols(hourly_df, feature_df):
    target_cols = [hourly_df.columns[2], hourly_df.columns[3]]
    history_cols = target_cols + [hourly_df.columns[4], hourly_df.columns[6]]
    for col in HOURLY_HISTORY_FEATURE_CANDIDATES:
        if col in feature_df.columns and col not in history_cols:
            history_cols.append(col)
    history_cols = list(dict.fromkeys(history_cols))
    return history_cols, target_cols


def select_known_future_cols(feature_df):
    preferred = ["capacity", "morning_bikes", "morning_docks", "evening_bikes", "evening_docks"]
    cols = [col for col in preferred if col in feature_df.columns]
    for col in KNOWN_FUTURE_FEATURE_CANDIDATES:
        if col in feature_df.columns and col not in cols:
            cols.append(col)
    return cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_dir", default=os.path.join("二月份数据处理", "nyc_top300_inventory_validation"))
    parser.add_argument("--trip_subdir", default=os.path.join("二月份数据处理", "纽约单车订单数据"))
    parser.add_argument("--trip_pattern", default="20*-citibike-tripdata*.csv")
    parser.add_argument("--output_name", default="bike_hourly_safe_inventory_top300_nyc")
    parser.add_argument("--hist_len", type=int, default=24 * 7)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--min_known_future_coverage", type=float, default=0.2)
    parser.add_argument("--keep_directed", action="store_true")
    parser.add_argument("--embedding_dim", type=int, default=144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--se_strategy", choices=["preserve", "fallback", "skip"], default="preserve")
    args = parser.parse_args()

    asset_dir = os.path.abspath(args.asset_dir)
    trip_dir = os.path.abspath(args.trip_subdir)
    mapping_path, mapping_df = load_mapping(asset_dir, "GNN_Node_Mapping_topk.csv", mapping_station_col="station_name")
    station_names = mapping_df["station_name"].tolist()

    trip_files = resolve_trip_files(trip_dir, args.trip_pattern)
    hourly_df, all_hours = aggregate_hourly_trip_counts(trip_files, station_names)
    all_dates = sorted(hourly_df["日期" if "日期" in hourly_df.columns else hourly_df.columns[5]].dropna().unique())

    daily_feature_df, daily_feature_path, aux_paths, weather_path, aux_merge_summary = load_daily_feature_table(
        source_dir=asset_dir,
        station_names=station_names,
        all_dates=all_dates,
        daily_feature_file="snapshot_daily_features_topk.csv",
        aux_temporal_files="station_static_features_topk.csv",
        weather_file=os.path.join(PROJECT_ROOT, "二月份数据处理", "weather-get", "NYC_Weather_2024-01-01_to_2026-02-31.csv"),
    )
    feature_df = build_hourly_feature_frame(hourly_df, daily_feature_df)
    history_feature_cols, target_cols = select_history_and_target_cols(hourly_df, feature_df)
    known_future_feature_cols = select_known_future_cols(feature_df)

    sample_bundle = build_hourly_samples(
        feature_df=feature_df,
        mapping_df=mapping_df.rename(columns={"station_name": "站点名称"}),
        history_feature_cols=history_feature_cols,
        target_cols=target_cols,
        known_future_feature_cols=known_future_feature_cols,
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        min_known_future_coverage=args.min_known_future_coverage,
    )

    requested_log1p_cols = [
        col for col in LOG1P_FEATURE_CANDIDATES
        if col in history_feature_cols or col in sample_bundle["known_future_feature_cols"]
    ]
    requested_log1p_cols = list(dict.fromkeys(requested_log1p_cols))

    x_values = sample_bundle["x"]
    history_x, applied_history_cols = apply_log1p_transform(
        x_values[..., :len(history_feature_cols)],
        history_feature_cols,
        requested_log1p_cols,
    )
    if sample_bundle["known_future_feature_cols"]:
        future_x, applied_future_cols = apply_log1p_transform(
            x_values[..., len(history_feature_cols):],
            sample_bundle["known_future_feature_cols"],
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

    temporal_output_dir = os.path.join(PROJECT_ROOT, "data", "temporal_data", args.output_name)
    graph_output_dir = os.path.join(PROJECT_ROOT, "data", "graph", args.output_name)
    se_output_path = os.path.join(PROJECT_ROOT, "data", "SE", "se_%s.csv" % args.output_name)

    split_summary = split_and_save_hourly_dataset(
        x_values=x_values,
        y_values=sample_bundle["y"],
        sample_dates=sample_bundle["sample_dates"],
        output_dir=temporal_output_dir,
        history_feature_cols=history_feature_cols,
        target_cols=target_cols,
        known_future_feature_cols=sample_bundle["known_future_feature_cols"],
        log1p_feature_cols=applied_log1p_feature_cols,
    )

    se_status = copy_graphs_and_embeddings(
        source_dir=asset_dir,
        graph_output_dir=graph_output_dir,
        se_output_path=se_output_path,
        mapping_df=mapping_df.rename(columns={"station_name": "站点名称"}),
        keep_directed=args.keep_directed,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
        se_strategy=args.se_strategy,
    )

    print("Prepared top-k hourly dataset:", args.output_name)
    print("Mapping path:", mapping_path)
    print("Trip files:", len(trip_files))
    print("Hourly datetime range:", str(all_hours.min()), "->", str(all_hours.max()))
    print("Daily feature file:", daily_feature_path)
    print("Auxiliary feature files:", aux_paths)
    print("Weather file:", weather_path)
    print("Auxiliary merge summary:", aux_merge_summary)
    print("History feature columns:", history_feature_cols)
    print("Known future feature columns:", sample_bundle["known_future_feature_cols"])
    print("Known future coverage:", sample_bundle["known_future_coverage"])
    print("Skipped known future columns:", sample_bundle["skipped_known_future_cols"])
    print("Applied log1p feature columns:", applied_log1p_feature_cols)
    print("Temporal x shape:", x_values.shape)
    print("Temporal y shape:", sample_bundle["y"].shape)
    print("Saved temporal data to:", temporal_output_dir)
    print("Saved graph data to:", graph_output_dir)
    print("Spatial embedding status:", se_status)
    print("Spatial embedding path:", se_output_path)
    print("Split summary:", split_summary)


if __name__ == "__main__":
    main()
