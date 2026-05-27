import argparse
import os
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION_ROOT = os.path.dirname(SCRIPT_DIR)
HOURLY_UTILS_DIR = next(
    (
        os.path.join(VERSION_ROOT, name)
        for name in os.listdir(VERSION_ROOT)
        if name.startswith("2026-04-09")
    ),
    os.path.join(VERSION_ROOT, "2026-04-09_å°æ—¶çº§éª‘å…¥éª‘å‡º_å®‰å…¨åº“å­˜åŒºé—´"),
)
if HOURLY_UTILS_DIR not in sys.path:
    sys.path.insert(0, HOURLY_UTILS_DIR)

from hourly_pipeline_utils import (  # noqa: E402
    HOURLY_HISTORY_FEATURE_CANDIDATES,
    KNOWN_FUTURE_FEATURE_CANDIDATES,
    LOG1P_FEATURE_CANDIDATES,
    aggregate_hourly_trip_counts,
    apply_log1p_transform,
    build_hourly_feature_frame,
    copy_graphs_and_embeddings,
    detect_project_root,
    load_daily_feature_table,
    load_mapping,
    resolve_trip_files,
)


PROJECT_ROOT = detect_project_root(SCRIPT_DIR)
DATE_COL = None
STATION_COL = None


def select_history_and_target_cols(hourly_df, feature_df):
    target_cols = [hourly_df.columns[2], hourly_df.columns[3]]
    history_cols = target_cols + [hourly_df.columns[4], hourly_df.columns[6]]
    for col in HOURLY_HISTORY_FEATURE_CANDIDATES:
        if col in feature_df.columns and col not in history_cols:
            history_cols.append(col)
    return list(dict.fromkeys(history_cols)), target_cols


def select_known_future_cols(feature_df):
    preferred = ["capacity", "morning_bikes", "morning_docks", "evening_bikes", "evening_docks"]
    cols = [col for col in preferred if col in feature_df.columns]
    for col in KNOWN_FUTURE_FEATURE_CANDIDATES:
        if col in feature_df.columns and col not in cols:
            cols.append(col)
    return cols


def parse_anchor_hours(value):
    anchors = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        hour = int(item)
        if hour < 0 or hour > 23:
            raise ValueError("anchor hour must be between 0 and 23: %s" % item)
        anchors.append(hour)
    anchors = sorted(set(anchors))
    if not anchors:
        raise ValueError("No anchor hours were provided.")
    return anchors


def build_hourly_samples_for_anchors(
    feature_df,
    mapping_df,
    history_feature_cols,
    target_cols,
    known_future_feature_cols,
    hist_len,
    pred_len,
    anchor_hours,
    min_known_future_coverage,
    target_start_offset,
):
    station_col = STATION_COL if STATION_COL in mapping_df.columns else mapping_df.columns[1]
    date_col = DATE_COL if DATE_COL in feature_df.columns else feature_df.columns[5]
    station_names = mapping_df[station_col].tolist()
    datetimes = pd.Index(sorted(feature_df["datetime"].dropna().unique()))
    dates = sorted(feature_df[date_col].dropna().unique())

    history_matrices = []
    for feature_name in history_feature_cols:
        pivot = feature_df.pivot_table(
            index="datetime",
            columns=station_col,
            values=feature_name,
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=datetimes, columns=station_names).fillna(0.0)
        history_matrices.append(pivot.to_numpy(dtype=np.float32))
    history_values = np.stack(history_matrices, axis=-1)

    target_matrices = []
    for target_name in target_cols:
        pivot = feature_df.pivot_table(
            index="datetime",
            columns=station_col,
            values=target_name,
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=datetimes, columns=station_names).fillna(0.0)
        target_matrices.append(pivot.to_numpy(dtype=np.float32))
    target_values = np.stack(target_matrices, axis=-1)

    known_future_values = None
    known_future_coverage = {}
    skipped_known_future_cols = {}
    filtered_known_future_cols = []
    if known_future_feature_cols:
        daily_feature_df = feature_df[[date_col, station_col] + known_future_feature_cols].drop_duplicates(
            subset=[date_col, station_col],
            keep="last",
        )
        future_matrices = []
        for feature_name in known_future_feature_cols:
            pivot = daily_feature_df.pivot_table(
                index=date_col,
                columns=station_col,
                values=feature_name,
                aggfunc="mean",
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
    sample_datetimes = []
    anchor_hour_values = []
    target_start_datetimes = []
    for date_value in dates:
        for anchor_hour in anchor_hours:
            anchor_dt = pd.Timestamp("%s %02d:00:00" % (date_value, anchor_hour))
            # Decision at t predicts t+1..t+pred_len by default.
            target_start = anchor_dt + pd.Timedelta(hours=target_start_offset)
            if anchor_dt not in datetime_to_idx or target_start not in datetime_to_idx:
                continue
            anchor_idx = datetime_to_idx[anchor_dt]
            target_start_idx = datetime_to_idx[target_start]
            hist_start_idx = anchor_idx - hist_len
            target_end_idx = target_start_idx + pred_len
            if hist_start_idx < 0 or target_end_idx > len(datetimes):
                continue

            x_window = history_values[hist_start_idx:anchor_idx]
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
            sample_datetimes.append(anchor_dt.strftime("%Y-%m-%d %H:%M:%S"))
            anchor_hour_values.append(anchor_hour)
            target_start_datetimes.append(target_start.strftime("%Y-%m-%d %H:%M:%S"))

    if not x_samples:
        raise ValueError("No rolling-anchor hourly samples were created.")

    return {
        "x": np.stack(x_samples, axis=0),
        "y": np.stack(y_samples, axis=0),
        "sample_dates": sample_dates,
        "sample_datetimes": sample_datetimes,
        "anchor_hours": anchor_hour_values,
        "target_start_datetimes": target_start_datetimes,
        "known_future_feature_cols": known_future_feature_cols,
        "known_future_coverage": known_future_coverage,
        "skipped_known_future_cols": skipped_known_future_cols,
    }


def split_and_save_rolling_dataset(
    x_values,
    y_values,
    sample_dates,
    sample_datetimes,
    anchor_hours,
    target_start_datetimes,
    output_dir,
    history_feature_cols,
    target_cols,
    known_future_feature_cols,
    log1p_feature_cols,
    save_dtype=np.float32,
):
    num_samples = x_values.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    slices = {
        "train": slice(0, num_train),
        "val": slice(num_train, num_train + num_val),
        "test": slice(num_samples - num_test, num_samples),
    }
    input_feature_cols = list(history_feature_cols) + ["future_" + col for col in known_future_feature_cols]
    input_log1p_feature_cols = (
        [col for col in history_feature_cols if col in log1p_feature_cols]
        + ["future_" + col for col in known_future_feature_cols if col in log1p_feature_cols]
    )

    os.makedirs(output_dir, exist_ok=True)
    split_summary = {"num_samples": num_samples}
    sample_dates = np.array(sample_dates)
    sample_datetimes = np.array(sample_datetimes)
    anchor_hours = np.array(anchor_hours, dtype=np.int16)
    target_start_datetimes = np.array(target_start_datetimes)
    for split_name, split_slice in slices.items():
        np.savez_compressed(
            os.path.join(output_dir, "%s.npz" % split_name),
            x=x_values[split_slice].astype(save_dtype),
            y=y_values[split_slice].astype(save_dtype),
            input_feature_cols=np.array(input_feature_cols),
            history_feature_cols=np.array(history_feature_cols),
            known_future_feature_cols=np.array(known_future_feature_cols),
            log1p_feature_cols=np.array(input_log1p_feature_cols),
            target_cols=np.array(target_cols),
            sample_dates=sample_dates[split_slice],
            sample_datetimes=sample_datetimes[split_slice],
            anchor_hours=anchor_hours[split_slice],
            target_start_datetimes=target_start_datetimes[split_slice],
        )
        split_summary[split_name] = x_values[split_slice].shape
    return split_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_dir", required=True)
    parser.add_argument("--trip_subdir", required=True)
    parser.add_argument("--trip_pattern", default="20*-citibike-tripdata*.csv")
    parser.add_argument("--output_name", default="bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20")
    parser.add_argument("--hist_len", type=int, default=24 * 7)
    parser.add_argument("--pred_len", type=int, default=6)
    parser.add_argument("--anchor_hours", default="0,6,12,16,20")
    parser.add_argument("--target_start_offset", type=int, default=1)
    parser.add_argument("--min_known_future_coverage", type=float, default=0.2)
    parser.add_argument("--save_dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--keep_directed", action="store_true")
    parser.add_argument("--embedding_dim", type=int, default=144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--se_strategy", choices=["preserve", "fallback", "skip"], default="preserve")
    args = parser.parse_args()

    asset_dir = os.path.abspath(args.asset_dir)
    trip_dir = os.path.abspath(args.trip_subdir)
    anchor_hours = parse_anchor_hours(args.anchor_hours)

    mapping_path, mapping_df = load_mapping(asset_dir, "GNN_Node_Mapping_topk.csv", mapping_station_col="station_name")
    station_names = mapping_df["station_name"].tolist()

    trip_files = resolve_trip_files(trip_dir, args.trip_pattern)
    hourly_df, all_hours = aggregate_hourly_trip_counts(trip_files, station_names)
    global DATE_COL, STATION_COL
    STATION_COL = hourly_df.columns[1]
    DATE_COL = hourly_df.columns[5]
    pipeline_mapping = mapping_df.rename(columns={"station_name": STATION_COL})
    all_dates = sorted(hourly_df[DATE_COL].dropna().unique())

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

    sample_bundle = build_hourly_samples_for_anchors(
        feature_df=feature_df,
        mapping_df=pipeline_mapping,
        history_feature_cols=history_feature_cols,
        target_cols=target_cols,
        known_future_feature_cols=known_future_feature_cols,
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        anchor_hours=anchor_hours,
        min_known_future_coverage=args.min_known_future_coverage,
        target_start_offset=args.target_start_offset,
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

    split_summary = split_and_save_rolling_dataset(
        x_values=x_values,
        y_values=sample_bundle["y"],
        sample_dates=sample_bundle["sample_dates"],
        sample_datetimes=sample_bundle["sample_datetimes"],
        anchor_hours=sample_bundle["anchor_hours"],
        target_start_datetimes=sample_bundle["target_start_datetimes"],
        output_dir=temporal_output_dir,
        history_feature_cols=history_feature_cols,
        target_cols=target_cols,
        known_future_feature_cols=sample_bundle["known_future_feature_cols"],
        log1p_feature_cols=applied_log1p_feature_cols,
        save_dtype=np.float16 if args.save_dtype == "float16" else np.float32,
    )

    se_status = copy_graphs_and_embeddings(
        source_dir=asset_dir,
        graph_output_dir=graph_output_dir,
        se_output_path=se_output_path,
        mapping_df=pipeline_mapping,
        keep_directed=args.keep_directed,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
        se_strategy=args.se_strategy,
    )

    anchor_counts = pd.Series(sample_bundle["anchor_hours"]).value_counts().sort_index().to_dict()
    print("Prepared rolling-anchor top-k hourly dataset:", args.output_name)
    print("Mapping path:", mapping_path)
    print("Trip files:", len(trip_files))
    print("Hourly datetime range:", str(all_hours.min()), "->", str(all_hours.max()))
    print("Anchor hours:", anchor_hours)
    print("Target start offset:", args.target_start_offset)
    print("Anchor sample counts:", anchor_counts)
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
