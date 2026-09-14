# -*- coding: utf-8 -*-
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
HOURLY_PIPELINE_DIR = PROJECT_ROOT / "版本管理" / "2026-04-09_小时级骑入骑出_安全库存区间"
ROLLING_ANCHOR_PIPELINE_DIR = PROJECT_ROOT / "版本管理" / "2026-04-23_top300全NYC库存快照通路验证"
for source_dir in [SCRIPT_DIR, SCRIPT_DIR.parent, HOURLY_PIPELINE_DIR, ROLLING_ANCHOR_PIPELINE_DIR]:
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from protocol import (  # noqa: E402
    MODEL_ID,
    UPSTREAM_COMMIT,
    anchor_metrics,
    apply_log1p_transform_inplace,
    audit_anchor_coverage,
    audit_graph_contract,
    compute_metrics,
    horizon_metrics,
    load_npz_metadata,
    resolve_device,
    save_json,
    truth_key_signature,
)
from hourly_pipeline_utils import (  # noqa: E402
    aggregate_hourly_trip_counts,
    build_hourly_feature_frame,
    load_daily_feature_table,
)
from models import build_model_from_checkpoint  # noqa: E402
from prepare_topk_hourly_dataset_rolling_anchors import (  # noqa: E402
    build_hourly_samples_for_anchors,
    parse_anchor_hours,
)


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "temporal_data" / (
    "bike_hourly_safe_inventory_top150_nyc_full_predlen3_"
    "anchors_00_03_06_09_12_15_18_21_train2025_hist168"
)
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "data" / "graph" / (
    "bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_graph_"
    "train2025_hist168_pred3_8anchors"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "分析结果" / "对比实验" / "STGformerOfficialBike" / (
    "b0_top150_hist168_pred3_seed0_bs16"
) / "best_stgformer_official_b0.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "分析结果" / "对比实验" / "STGformerOfficialBike" / "holdout_202603_202606"
DEFAULT_ORDER_DIR = PROJECT_ROOT / "二月份数据处理" / "纽约单车订单数据"
DEFAULT_ASSET_DIR = PROJECT_ROOT / "二月份数据处理" / "nyc_top300_inventory_validation"
DEFAULT_WEATHER_FILE = PROJECT_ROOT / "二月份数据处理" / "weather-get" / (
    "NYC_Weather_2024-06-01_to_2026-06-30.csv"
)
STATION_COL = "站点名称"
WEATHER_FUTURE_FEATURE_NAMES = {
    "最高气温(°C)",
    "最低气温(°C)",
    "总降水量(mm)",
    "总降雪量(cm)",
    "最大风速(km/h)",
}


def absolute_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def assert_checkpoint_contract(checkpoint, metadata, graph_audit):
    errors = []
    if checkpoint.get("model_id") != MODEL_ID:
        errors.append("model_id mismatch")
    if checkpoint.get("upstream_commit") != UPSTREAM_COMMIT:
        errors.append("upstream_commit mismatch")
    config = checkpoint.get("model_config", {})
    expected_shape = {
        "in_steps": metadata["hist_len"],
        "out_steps": metadata["pred_len"],
        "num_nodes": metadata["num_nodes"],
        "input_dim": metadata["in_dim"],
        "output_dim": metadata["out_dim"],
    }
    for key, expected in expected_shape.items():
        if int(config.get(key, -1)) != int(expected):
            errors.append("%s mismatch" % key)
    for key in ["feature_mean", "feature_std", "target_mean", "target_std"]:
        if key not in checkpoint or not np.allclose(
            np.asarray(checkpoint[key], dtype=np.float64),
            np.asarray(metadata[key], dtype=np.float64),
            rtol=1e-7,
            atol=1e-8,
        ):
            errors.append("%s mismatch" % key)
    if checkpoint.get("input_feature_cols") != metadata["input_feature_cols"]:
        errors.append("input_feature_cols mismatch")
    if checkpoint.get("target_cols") != metadata["target_cols"]:
        errors.append("target_cols mismatch")
    checkpoint_graph = checkpoint.get("graph_audit", {})
    if checkpoint_graph.get("node_signature") != graph_audit["node_signature"]:
        errors.append("node_signature mismatch")
    if errors:
        raise RuntimeError("Checkpoint/data contract failed: %s" % ", ".join(errors))


def grouped_month_metrics(frame):
    rows = []
    for month, group in frame.groupby("month", sort=True):
        pred = group[["pred_out", "pred_in"]].to_numpy(dtype=np.float64)
        true = group[["true_out", "true_in"]].to_numpy(dtype=np.float64)
        metrics = compute_metrics(pred, true)
        rows.append({"month": str(month), "rows": len(group), **metrics})
    return pd.DataFrame(rows)


def apply_future_weather_lag(sample_bundle, feature_df, weather_file, history_feature_cols, lag_days):
    """Replace future weather only with observations available ``lag_days`` earlier."""
    lag_days = int(lag_days)
    if lag_days <= 0:
        return sample_bundle, []
    future_cols = list(sample_bundle.get("known_future_feature_cols", []))
    weather_cols = [name for name in future_cols if name in WEATHER_FUTURE_FEATURE_NAMES]
    if not weather_cols:
        return sample_bundle, []
    # This array is freshly built for this evaluation, so weather replacement
    # can safely happen in place instead of duplicating several GiB.
    x_values = np.asarray(sample_bundle["x"], dtype=np.float32)
    feature_daily = {}
    for name in weather_cols:
        if name in feature_df.columns:
            values = pd.to_numeric(feature_df[name], errors="coerce")
            feature_daily[name] = values.groupby(feature_df["日期"].astype(str).str[:10]).mean().to_dict()
    weather_daily = {}
    if Path(weather_file).exists():
        weather_df = pd.read_csv(weather_file)
        date_col = next(
            (name for name in ["日期", "date", "Date", "datetime", "时间"] if name in weather_df.columns),
            weather_df.columns[0],
        )
        parsed = pd.to_datetime(weather_df[date_col], errors="coerce")
        weather_df = weather_df.loc[parsed.notna()].copy()
        weather_df["__date"] = parsed.loc[parsed.notna()].dt.strftime("%Y-%m-%d")
        for name in weather_cols:
            if name in weather_df.columns:
                weather_daily[name] = (
                    pd.to_numeric(weather_df[name], errors="coerce")
                    .groupby(weather_df["__date"])
                    .mean()
                    .to_dict()
                )
    history_count = len(history_feature_cols)
    missing_dates = []
    for row_index, date_value in enumerate(sample_bundle["sample_dates"]):
        source_date = (pd.Timestamp(str(date_value)[:10]) - pd.Timedelta(days=lag_days)).strftime("%Y-%m-%d")
        replaced = False
        for name in weather_cols:
            value = weather_daily.get(name, {}).get(source_date)
            if value is None or pd.isna(value):
                value = feature_daily.get(name, {}).get(source_date)
            if value is None or pd.isna(value):
                continue
            feature_index = history_count + future_cols.index(name)
            x_values[row_index, :, :, feature_index] = np.float32(value)
            replaced = True
        if not replaced:
            missing_dates.append(str(date_value)[:10])
    updated = dict(sample_bundle)
    updated["x"] = x_values
    return updated, sorted(set(missing_dates))


def main():
    parser = argparse.ArgumentParser(description="Evaluate B0 official-core STGformer on an external holdout")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--graph_dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--graph_name", default="dist")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start_date", default="2026-03-01")
    parser.add_argument("--end_date", default="2026-06-30")
    parser.add_argument(
        "--trip_glob",
        default="20260[2-7]-citibike-tripdata*.csv",
        help=(
            "Trip-file glob. The default includes July because the 2026-06-30 21:00 "
            "anchor predicts through 2026-07-01 00:00."
        ),
    )
    parser.add_argument("--order_dir", default=str(DEFAULT_ORDER_DIR))
    parser.add_argument("--asset_dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--weather_file", default=str(DEFAULT_WEATHER_FILE))
    parser.add_argument(
        "--future_weather_lag_days",
        type=int,
        default=0,
        help="0 uses same-day observed weather (oracle); positive values use only earlier observations.",
    )
    parser.add_argument("--eval_tag", default="stgformer_official_b0_202603_202606")
    parser.add_argument("--anchor_hours", default="0,3,6,9,12,15,18,21")
    parser.add_argument("--target_start_offset", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--allow_missing_dates", action="store_true")
    args = parser.parse_args()
    if args.future_weather_lag_days < 0:
        raise ValueError("future_weather_lag_days must be non-negative")

    data_dir = absolute_path(args.data_dir)
    graph_dir = absolute_path(args.graph_dir)
    checkpoint_path = absolute_path(args.checkpoint)
    output_dir = absolute_path(args.output_dir)
    order_dir = absolute_path(args.order_dir)
    asset_dir = absolute_path(args.asset_dir)
    weather_file = absolute_path(args.weather_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_npz_metadata(data_dir)
    graph_audit = audit_graph_contract(graph_dir, args.graph_name, metadata["num_nodes"])
    mapping = pd.read_csv(graph_dir / "selected_node_mapping.csv").sort_values("Node_ID").reset_index(drop=True)
    if STATION_COL not in mapping.columns:
        raise KeyError("Missing station-name column in selected_node_mapping.csv")
    station_names = mapping[STATION_COL].astype(str).tolist()

    trip_files = sorted(glob.glob(str(order_dir / args.trip_glob)))
    if not trip_files:
        raise FileNotFoundError("No trip files matched: %s" % (order_dir / args.trip_glob))
    print("Aggregating hourly trip counts from %d files..." % len(trip_files))
    hourly_df, full_hours = aggregate_hourly_trip_counts(trip_files, station_names)
    requested_coverage = audit_anchor_coverage(
        [],
        args.start_date,
        args.end_date,
        parse_anchor_hours(args.anchor_hours),
        args.target_start_offset,
        metadata["pred_len"],
    )
    loaded_hourly_end = pd.Timestamp(full_hours.max())
    required_target_end = pd.Timestamp(requested_coverage["latest_required_target_datetime"])
    if loaded_hourly_end < required_target_end and not args.allow_missing_dates:
        raise RuntimeError(
            "The requested holdout needs target data through %s, but the loaded trips end at %s. "
            "The final anchor %s uses t+%d..t+%d targets. Expand --trip_glob to include the "
            "boundary month (July 2026 for the default range), or shorten --end_date."
            % (
                required_target_end.strftime("%Y-%m-%d %H:%M:%S"),
                loaded_hourly_end.strftime("%Y-%m-%d %H:%M:%S"),
                requested_coverage["latest_requested_anchor_datetime"],
                args.target_start_offset,
                args.target_start_offset + metadata["pred_len"] - 1,
            )
        )
    anchor_hours = parse_anchor_hours(args.anchor_hours)
    required_history_start = (
        pd.Timestamp(args.start_date).normalize()
        + pd.Timedelta(hours=min(anchor_hours) - metadata["hist_len"])
    )
    loaded_hourly_start = pd.Timestamp(full_hours.min())
    if loaded_hourly_start > required_history_start and not args.allow_missing_dates:
        raise RuntimeError(
            "The requested holdout needs history from %s, but the loaded trips start at %s. "
            "Expand --trip_glob to include the preceding history window."
            % (
                required_history_start.strftime("%Y-%m-%d %H:%M:%S"),
                loaded_hourly_start.strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
    context_mask = hourly_df["datetime"].between(required_history_start, required_target_end)
    hourly_df = hourly_df.loc[context_mask].copy()
    print(
        "Building samples only for required context %s through %s..."
        % (
            required_history_start.strftime("%Y-%m-%d %H:%M:%S"),
            required_target_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
    )
    all_dates = sorted(hourly_df["日期"].dropna().unique())
    daily_feature_df, daily_feature_path, aux_paths, weather_path, aux_merge_summary = load_daily_feature_table(
        source_dir=str(asset_dir),
        station_names=station_names,
        all_dates=all_dates,
        daily_feature_file="snapshot_daily_features_topk.csv",
        aux_temporal_files="station_static_features_topk.csv",
        weather_file=str(weather_file),
    )
    feature_df = build_hourly_feature_frame(hourly_df, daily_feature_df)
    sample_bundle = build_hourly_samples_for_anchors(
        feature_df=feature_df,
        mapping_df=mapping,
        history_feature_cols=metadata["history_feature_cols"],
        target_cols=metadata["target_cols"],
        known_future_feature_cols=metadata["known_future_feature_cols"],
        hist_len=metadata["hist_len"],
        pred_len=metadata["pred_len"],
        anchor_hours=anchor_hours,
        min_known_future_coverage=0.0,
        target_start_offset=args.target_start_offset,
    )
    sample_bundle, weather_lag_missing_dates = apply_future_weather_lag(
        sample_bundle,
        feature_df,
        weather_file,
        metadata["history_feature_cols"],
        args.future_weather_lag_days,
    )
    if weather_lag_missing_dates:
        raise RuntimeError("Lagged future weather is unavailable for dates: %s" % weather_lag_missing_dates)

    x_values = sample_bundle.pop("x")
    history_feature_count = len(metadata["history_feature_cols"])
    applied_history_cols = apply_log1p_transform_inplace(
        x_values[..., :history_feature_count],
        metadata["history_feature_cols"],
        metadata["log1p_feature_cols"],
    )
    applied_future_cols = apply_log1p_transform_inplace(
        x_values[..., history_feature_count:],
        sample_bundle["known_future_feature_cols"],
        metadata["log1p_feature_cols"],
    )
    y_values = np.asarray(sample_bundle["y"], dtype=np.float32)
    sample_dates = np.asarray(sample_bundle["sample_dates"])
    sample_datetimes = np.asarray(sample_bundle["sample_datetimes"])
    sample_anchor_hours = np.asarray(sample_bundle["anchor_hours"])
    target_start_datetimes = np.asarray(sample_bundle["target_start_datetimes"])

    wanted_dates = pd.date_range(args.start_date, args.end_date, freq="D").strftime("%Y-%m-%d").to_numpy()
    available_mask = np.isin(sample_dates, wanted_dates)
    available_dates = set(sample_dates[available_mask].tolist())
    missing_dates = [date for date in wanted_dates.tolist() if date not in available_dates]
    if missing_dates and not args.allow_missing_dates:
        raise RuntimeError("External holdout has missing dates: %s" % missing_dates)
    if not available_mask.all():
        x_values = x_values[available_mask]
        y_values = y_values[available_mask]
        sample_dates = sample_dates[available_mask]
        sample_datetimes = sample_datetimes[available_mask]
        sample_anchor_hours = sample_anchor_hours[available_mask]
        target_start_datetimes = target_start_datetimes[available_mask]
    if len(sample_dates) == 0:
        raise RuntimeError("No requested holdout samples are evaluable")
    anchor_coverage = audit_anchor_coverage(
        sample_datetimes,
        args.start_date,
        args.end_date,
        parse_anchor_hours(args.anchor_hours),
        args.target_start_offset,
        metadata["pred_len"],
    )
    expected_samples = anchor_coverage["expected_count"]
    if anchor_coverage["duplicate_anchor_count"]:
        raise RuntimeError(
            "External holdout has %d duplicate anchor samples"
            % anchor_coverage["duplicate_anchor_count"]
        )
    missing_anchors = anchor_coverage["missing_anchor_datetimes"]
    if missing_anchors and not args.allow_missing_dates:
        preview = ", ".join(missing_anchors[:12])
        if len(missing_anchors) > 12:
            preview += ", ..."
        raise RuntimeError(
            "Expected %d anchor samples, got %d. Missing anchors (%d): %s. "
            "The latest requested forecast target is %s; loaded trips end at %s."
            % (
                expected_samples,
                len(sample_dates),
                len(missing_anchors),
                preview,
                anchor_coverage["latest_required_target_datetime"],
                pd.Timestamp(full_hours.max()).strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    assert_checkpoint_contract(checkpoint, metadata, graph_audit)
    model, checkpoint_audit = build_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    feature_mean = torch.as_tensor(
        np.asarray(checkpoint["feature_mean"]).reshape(-1), dtype=torch.float32, device=device
    )
    feature_std = torch.as_tensor(
        np.asarray(checkpoint["feature_std"]).reshape(-1), dtype=torch.float32, device=device
    )
    target_mean = torch.as_tensor(checkpoint["target_mean"], dtype=torch.float32, device=device)
    target_std = torch.as_tensor(checkpoint["target_std"], dtype=torch.float32, device=device)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x_values), args.batch_size):
            raw_batch = torch.from_numpy(
                np.asarray(x_values[start : start + args.batch_size], dtype=np.float32)
            ).to(device)
            batch = (raw_batch - feature_mean) / feature_std
            prediction = F.softplus(model(batch) * target_std + target_mean, beta=5.0)
            predictions.append(prediction.cpu().numpy())
    pred_values = np.concatenate(predictions, axis=0)
    if not np.isfinite(pred_values).all() or not np.isfinite(y_values).all():
        raise FloatingPointError("Prediction or target contains NaN/Inf")
    if (pred_values < 0).any():
        raise FloatingPointError("Directional predictions must be non-negative")

    abs_error = np.abs(pred_values - y_values)
    rows = []
    for sample_index, date_value in enumerate(sample_dates):
        for horizon_index in range(y_values.shape[1]):
            target_dt = pd.Timestamp(str(target_start_datetimes[sample_index])) + pd.Timedelta(hours=horizon_index)
            for node_index, station_name in enumerate(station_names):
                rows.append(
                    {
                        "date": str(date_value),
                        "month": str(date_value)[:7],
                        "sample_datetime": str(sample_datetimes[sample_index]),
                        "anchor_hour": int(sample_anchor_hours[sample_index]),
                        "target_start_datetime": str(target_start_datetimes[sample_index]),
                        "horizon": horizon_index + 1,
                        "hour": target_dt.strftime("%H:00"),
                        "Node_ID": int(mapping.loc[node_index, "Node_ID"]),
                        "station_name": station_name,
                        "pred_out": float(pred_values[sample_index, horizon_index, node_index, 0]),
                        "pred_in": float(pred_values[sample_index, horizon_index, node_index, 1]),
                        "true_out": float(y_values[sample_index, horizon_index, node_index, 0]),
                        "true_in": float(y_values[sample_index, horizon_index, node_index, 1]),
                        "abs_error_out": float(abs_error[sample_index, horizon_index, node_index, 0]),
                        "abs_error_in": float(abs_error[sample_index, horizon_index, node_index, 1]),
                        "abs_error_avg": float(abs_error[sample_index, horizon_index, node_index].mean()),
                    }
                )
    prediction_frame = pd.DataFrame(rows)
    key_columns = ["date", "sample_datetime", "target_start_datetime", "anchor_hour", "horizon", "Node_ID"]
    duplicate_rows = int(prediction_frame.duplicated(key_columns).sum())
    if duplicate_rows:
        raise RuntimeError("Duplicate prediction keys: %d" % duplicate_rows)

    tag = args.eval_tag
    prediction_frame.to_csv(output_dir / (tag + "_station_hour_predictions.csv"), index=False, encoding="utf-8-sig")
    horizon_metrics(pred_values, y_values).to_csv(
        output_dir / (tag + "_horizon_metrics.csv"), index=False, encoding="utf-8-sig"
    )
    anchor_metrics(pred_values, y_values, sample_anchor_hours).to_csv(
        output_dir / (tag + "_anchor_metrics.csv"), index=False, encoding="utf-8-sig"
    )
    grouped_month_metrics(prediction_frame).to_csv(
        output_dir / (tag + "_monthly_metrics.csv"), index=False, encoding="utf-8-sig"
    )
    prediction_frame.groupby("date", as_index=False)[
        ["abs_error_avg", "abs_error_out", "abs_error_in"]
    ].mean().to_csv(output_dir / (tag + "_daily_metrics.csv"), index=False, encoding="utf-8-sig")
    prediction_frame.groupby(["Node_ID", "station_name"], as_index=False)[
        ["abs_error_avg", "abs_error_out", "abs_error_in"]
    ].mean().to_csv(output_dir / (tag + "_station_metrics.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "schema_version": 1,
        "model": MODEL_ID,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint": str(checkpoint_path),
        "checkpoint_missing_keys": checkpoint_audit["missing_keys"],
        "checkpoint_unexpected_keys": checkpoint_audit["unexpected_keys"],
        "data_dir": str(data_dir),
        "graph_dir": str(graph_dir),
        "graph_audit": graph_audit,
        "requested_start": args.start_date,
        "requested_end": args.end_date,
        "date_start": str(sample_dates[0]),
        "date_end": str(sample_dates[-1]),
        "missing_dates": missing_dates,
        "num_anchor_samples": int(len(sample_dates)),
        "expected_anchor_samples": int(expected_samples),
        "missing_anchor_datetimes": missing_anchors,
        "latest_required_target_datetime": anchor_coverage["latest_required_target_datetime"],
        "pred_shape": list(pred_values.shape),
        "anchor_hours": parse_anchor_hours(args.anchor_hours),
        "overall": compute_metrics(pred_values, y_values),
        "truth_key_signature": truth_key_signature(prediction_frame),
        "duplicate_prediction_keys": duplicate_rows,
        "nonfinite_prediction_or_target": 0,
        "daily_feature_file": daily_feature_path,
        "weather_file": weather_path,
        "future_weather_regime": (
            "oracle_observed_same_day"
            if args.future_weather_lag_days == 0
            else "lagged_observed_d_minus_%d" % args.future_weather_lag_days
        ),
        "future_weather_lag_days": args.future_weather_lag_days,
        "future_weather_lag_missing_dates": weather_lag_missing_dates,
        "trip_glob": args.trip_glob,
        "hourly_range": [str(full_hours.min()), str(full_hours.max())],
        "sample_build_context_range": [str(required_history_start), str(required_target_end)],
        "aux_paths": aux_paths,
        "aux_merge_summary": aux_merge_summary,
        "applied_log1p_cols": sorted(set(applied_history_cols + applied_future_cols)),
        "target_start_offset": args.target_start_offset,
        "output_dir": str(output_dir),
    }
    save_json(output_dir / (tag + "_holdout_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
