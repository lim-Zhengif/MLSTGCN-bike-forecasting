import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hourly_pipeline_utils import (  # noqa: E402
    HOURLY_HISTORY_FEATURE_CANDIDATES,
    KNOWN_FUTURE_FEATURE_CANDIDATES,
    LOG1P_FEATURE_CANDIDATES,
    add_calendar_features,
    apply_log1p_transform,
    copy_graphs_and_embeddings,
    detect_project_root,
    load_daily_feature_table,
    load_mapping,
    resolve_trip_files,
)


PROJECT_ROOT = detect_project_root(SCRIPT_DIR)
DEFAULT_ASSET_DIR = os.path.join("二月份数据处理", "nyc_top150_inventory_validation")
DEFAULT_TRIP_SUBDIR = os.path.join("二月份数据处理", "纽约单车订单数据")
DEFAULT_WEATHER_FILE = os.path.join(
    PROJECT_ROOT,
    "二月份数据处理",
    "weather-get",
    "NYC_Weather_2024-01-01_to_2026-02-31.csv",
)


def normalize_trip_datetime(series, label, freq):
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().all():
        raise ValueError("Failed to parse any values in %s as datetimes" % label)
    return parsed.dt.floor(freq)


def aggregate_trip_counts_by_frequency(trip_files, station_names, freq="30min"):
    station_set = set(station_names)
    frames = []
    use_cols = ["started_at", "ended_at", "start_station_name", "end_station_name"]
    for trip_path in trip_files:
        trip_df = pd.read_csv(trip_path, usecols=lambda col: col in use_cols)
        if trip_df.empty:
            continue

        out_df = trip_df[["started_at", "start_station_name"]].copy()
        out_df = out_df.rename(columns={"started_at": "datetime", "start_station_name": "站点名称"})
        out_df["datetime"] = normalize_trip_datetime(out_df["datetime"], "started_at", freq)
        out_df["站点名称"] = out_df["站点名称"].astype(str)
        out_df = out_df[out_df["站点名称"].isin(station_set)]
        out_df["半小时骑出量"] = 1.0
        out_df = out_df.groupby(["datetime", "站点名称"], as_index=False)["半小时骑出量"].sum()

        in_df = trip_df[["ended_at", "end_station_name"]].copy()
        in_df = in_df.rename(columns={"ended_at": "datetime", "end_station_name": "站点名称"})
        in_df["datetime"] = normalize_trip_datetime(in_df["datetime"], "ended_at", freq)
        in_df["站点名称"] = in_df["站点名称"].astype(str)
        in_df = in_df[in_df["站点名称"].isin(station_set)]
        in_df["半小时骑入量"] = 1.0
        in_df = in_df.groupby(["datetime", "站点名称"], as_index=False)["半小时骑入量"].sum()

        frames.append(out_df.merge(in_df, on=["datetime", "站点名称"], how="outer").fillna(0.0))

    if not frames:
        raise ValueError("No trip rows were generated from trip files.")

    interval_df = pd.concat(frames, ignore_index=True)
    interval_df = interval_df.groupby(["datetime", "站点名称"], as_index=False)[["半小时骑出量", "半小时骑入量"]].sum()
    start_dt = interval_df["datetime"].min().floor(freq)
    end_dt = interval_df["datetime"].max().floor(freq)
    full_times = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    full_index = pd.MultiIndex.from_product(
        [full_times, station_names],
        names=["datetime", "站点名称"],
    ).to_frame(index=False)
    interval_df = full_index.merge(interval_df, on=["datetime", "站点名称"], how="left")
    interval_df["半小时骑出量"] = interval_df["半小时骑出量"].fillna(0.0).astype(np.float32)
    interval_df["半小时骑入量"] = interval_df["半小时骑入量"].fillna(0.0).astype(np.float32)
    interval_df["半小时净流量"] = (interval_df["半小时骑入量"] - interval_df["半小时骑出量"]).astype(np.float32)
    interval_df["日期"] = interval_df["datetime"].dt.strftime("%Y-%m-%d")
    interval_df["小时"] = interval_df["datetime"].dt.hour.astype(np.float32)
    interval_df["分钟"] = interval_df["datetime"].dt.minute.astype(np.float32)
    interval_df["日内半小时序号"] = (interval_df["datetime"].dt.hour * 2 + interval_df["datetime"].dt.minute // 30).astype(np.float32)
    return interval_df, full_times


def build_feature_frame(interval_df, daily_feature_df):
    return interval_df.merge(daily_feature_df, on=["日期", "站点名称"], how="left")


def select_history_and_target_cols(interval_df, feature_df):
    target_cols = ["半小时骑出量", "半小时骑入量"]
    history_cols = target_cols + ["半小时净流量", "小时", "分钟", "日内半小时序号"]
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


def build_continuous_samples(
    feature_df,
    mapping_df,
    history_feature_cols,
    target_cols,
    known_future_feature_cols,
    hist_len,
    pred_len,
    stride,
    min_known_future_coverage,
):
    station_names = mapping_df["站点名称"].tolist()
    datetimes = pd.Index(sorted(feature_df["datetime"].dropna().unique()))
    dates = sorted(feature_df["日期"].dropna().unique())

    history_values = []
    for feature_name in history_feature_cols:
        pivot = feature_df.pivot_table(index="datetime", columns="站点名称", values=feature_name, aggfunc="mean")
        pivot = pivot.reindex(index=datetimes, columns=station_names).fillna(0.0)
        history_values.append(pivot.to_numpy(dtype=np.float32))
    history_values = np.stack(history_values, axis=-1)

    target_values = []
    for target_name in target_cols:
        pivot = feature_df.pivot_table(index="datetime", columns="站点名称", values=target_name, aggfunc="mean")
        pivot = pivot.reindex(index=datetimes, columns=station_names).fillna(0.0)
        target_values.append(pivot.to_numpy(dtype=np.float32))
    target_values = np.stack(target_values, axis=-1)

    known_future_values = None
    known_future_coverage = {}
    skipped_known_future_cols = {}
    filtered_known_future_cols = []
    if known_future_feature_cols:
        daily_feature_df = feature_df[["日期", "站点名称"] + known_future_feature_cols].drop_duplicates(
            subset=["日期", "站点名称"],
            keep="last",
        )
        future_matrices = []
        for feature_name in known_future_feature_cols:
            pivot = daily_feature_df.pivot_table(index="日期", columns="站点名称", values=feature_name, aggfunc="mean")
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

    date_to_idx = {date_value: idx for idx, date_value in enumerate(dates)}
    x_samples = []
    y_samples = []
    sample_datetimes = []
    target_start_datetimes = []
    sample_dates = []
    for anchor_idx in range(hist_len, len(datetimes) - pred_len + 1, stride):
        target_start_idx = anchor_idx
        target_end_idx = target_start_idx + pred_len
        anchor_dt = pd.Timestamp(datetimes[anchor_idx - 1])
        target_start_dt = pd.Timestamp(datetimes[target_start_idx])
        date_value = target_start_dt.strftime("%Y-%m-%d")

        x_window = history_values[anchor_idx - hist_len:anchor_idx]
        y_window = target_values[target_start_idx:target_end_idx]
        if known_future_values is not None:
            future_date_idx = date_to_idx[date_value]
            future_window = np.repeat(known_future_values[future_date_idx][np.newaxis, ...], hist_len, axis=0)
            x_window = np.concatenate([x_window, future_window], axis=-1)

        x_samples.append(x_window.astype(np.float32))
        y_samples.append(y_window.astype(np.float32))
        sample_datetimes.append(anchor_dt.strftime("%Y-%m-%d %H:%M:%S"))
        target_start_datetimes.append(target_start_dt.strftime("%Y-%m-%d %H:%M:%S"))
        sample_dates.append(date_value)

    if not x_samples:
        raise ValueError("No 30min continuous samples were created.")

    return {
        "x": np.stack(x_samples, axis=0),
        "y": np.stack(y_samples, axis=0),
        "sample_dates": sample_dates,
        "sample_datetimes": sample_datetimes,
        "target_start_datetimes": target_start_datetimes,
        "known_future_feature_cols": known_future_feature_cols,
        "known_future_coverage": known_future_coverage,
        "skipped_known_future_cols": skipped_known_future_cols,
    }


def split_and_save_dataset(
    x_values,
    y_values,
    sample_dates,
    sample_datetimes,
    target_start_datetimes,
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
    sample_dates = np.asarray(sample_dates)
    sample_datetimes = np.asarray(sample_datetimes)
    target_start_datetimes = np.asarray(target_start_datetimes)
    summary = {"num_samples": int(num_samples)}
    for split_name, split_slice in slices.items():
        np.savez_compressed(
            os.path.join(output_dir, "%s.npz" % split_name),
            x=x_values[split_slice],
            y=y_values[split_slice],
            input_feature_cols=np.array(input_feature_cols),
            history_feature_cols=np.array(history_feature_cols),
            known_future_feature_cols=np.array(known_future_feature_cols),
            log1p_feature_cols=np.array(input_log1p_feature_cols),
            target_cols=np.array(target_cols),
            sample_dates=sample_dates[split_slice],
            sample_datetimes=sample_datetimes[split_slice],
            target_start_datetimes=target_start_datetimes[split_slice],
        )
        summary[split_name] = tuple(x_values[split_slice].shape)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_dir", default=DEFAULT_ASSET_DIR)
    parser.add_argument("--trip_subdir", default=DEFAULT_TRIP_SUBDIR)
    parser.add_argument("--trip_pattern", default="20*-citibike-tripdata*.csv")
    parser.add_argument("--output_name", default="bike_halfhour_top150_paper_aligned_hist12_pred12_continuous")
    parser.add_argument("--freq", default="30min")
    parser.add_argument("--hist_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
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

    interval_df, all_times = aggregate_trip_counts_by_frequency(trip_files, station_names, freq=args.freq)
    all_dates = sorted(interval_df["日期"].dropna().unique())
    daily_feature_df, daily_feature_path, aux_paths, weather_path, aux_merge_summary = load_daily_feature_table(
        source_dir=asset_dir,
        station_names=station_names,
        all_dates=all_dates,
        daily_feature_file="snapshot_daily_features_topk.csv",
        aux_temporal_files="station_static_features_topk.csv",
        weather_file=DEFAULT_WEATHER_FILE,
    )
    daily_feature_df = add_calendar_features(daily_feature_df, "日期") if "星期几" not in daily_feature_df.columns else daily_feature_df
    feature_df = build_feature_frame(interval_df, daily_feature_df)
    history_feature_cols, target_cols = select_history_and_target_cols(interval_df, feature_df)
    known_future_feature_cols = select_known_future_cols(feature_df)

    sample_bundle = build_continuous_samples(
        feature_df=feature_df,
        mapping_df=mapping_df.rename(columns={"station_name": "站点名称"}),
        history_feature_cols=history_feature_cols,
        target_cols=target_cols,
        known_future_feature_cols=known_future_feature_cols,
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        stride=args.stride,
        min_known_future_coverage=args.min_known_future_coverage,
    )

    requested_log1p_cols = [
        col
        for col in LOG1P_FEATURE_CANDIDATES + ["半小时骑出量", "半小时骑入量", "半小时净流量"]
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
        col for col in requested_log1p_cols if col in applied_history_cols or col in applied_future_cols
    ]

    temporal_output_dir = os.path.join(PROJECT_ROOT, "data", "temporal_data", args.output_name)
    graph_output_dir = os.path.join(PROJECT_ROOT, "data", "graph", args.output_name)
    se_output_path = os.path.join(PROJECT_ROOT, "data", "SE", "se_%s.csv" % args.output_name)
    split_summary = split_and_save_dataset(
        x_values=x_values,
        y_values=sample_bundle["y"],
        sample_dates=sample_bundle["sample_dates"],
        sample_datetimes=sample_bundle["sample_datetimes"],
        target_start_datetimes=sample_bundle["target_start_datetimes"],
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
    metadata = {
        "freq": args.freq,
        "hist_len": args.hist_len,
        "pred_len": args.pred_len,
        "stride": args.stride,
        "task_note": "paper-aligned 30min continuous multi-step forecasting; 12 steps = 6 hours",
        "mapping_path": mapping_path,
        "trip_files": len(trip_files),
        "time_range": [str(all_times.min()), str(all_times.max())],
        "daily_feature_path": daily_feature_path,
        "auxiliary_feature_files": aux_paths,
        "weather_file": weather_path,
        "auxiliary_merge_summary": aux_merge_summary,
        "history_feature_cols": history_feature_cols,
        "known_future_feature_cols": sample_bundle["known_future_feature_cols"],
        "known_future_coverage": sample_bundle["known_future_coverage"],
        "skipped_known_future_cols": sample_bundle["skipped_known_future_cols"],
        "target_cols": target_cols,
        "applied_log1p_feature_cols": applied_log1p_feature_cols,
        "x_shape": tuple(x_values.shape),
        "y_shape": tuple(sample_bundle["y"].shape),
        "split_summary": split_summary,
        "graph_output_dir": graph_output_dir,
        "se_status": se_status,
        "se_output_path": se_output_path,
    }
    os.makedirs(temporal_output_dir, exist_ok=True)
    with open(os.path.join(temporal_output_dir, "paper_aligned_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Prepared 30min paper-aligned dataset:", args.output_name)
    print("Temporal x shape:", x_values.shape)
    print("Temporal y shape:", sample_bundle["y"].shape)
    print("Saved temporal data to:", temporal_output_dir)
    print("Saved graph data to:", graph_output_dir)
    print("Spatial embedding status:", se_status)
    print("Split summary:", split_summary)


if __name__ == "__main__":
    main()
