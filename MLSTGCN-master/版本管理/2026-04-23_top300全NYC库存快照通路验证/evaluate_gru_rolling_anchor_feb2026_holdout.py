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
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SOURCE_VERSION_DIR = PROJECT_ROOT / "版本管理" / "2026-04-09_小时级骑入骑出_安全库存区间"
if str(SOURCE_VERSION_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_VERSION_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hourly_pipeline_utils import (  # noqa: E402
    aggregate_hourly_trip_counts,
    apply_log1p_transform,
    build_hourly_feature_frame,
    load_daily_feature_table,
)
from prepare_topk_hourly_dataset_rolling_anchors import (  # noqa: E402
    build_hourly_samples_for_anchors,
    parse_anchor_hours,
)
from train_rolling_anchor_gru_baseline import SharedStationGRU, compute_metrics, horizon_metrics, anchor_metrics  # noqa: E402


ANALYSIS_DIR = "分析结果"
FEB_DATA_DIR = "二月份数据处理"
ORDER_SUBDIR = "纽约单车订单数据"
STATION_COL = "站点名称"


def load_npz_metadata(data_dir):
    train_npz = np.load(data_dir / "train.npz", allow_pickle=True)
    feature_mean = train_npz["x"].mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    feature_std = train_npz["x"].std(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    return {
        "input_feature_cols": [str(v) for v in train_npz["input_feature_cols"].tolist()],
        "history_feature_cols": [str(v) for v in train_npz["history_feature_cols"].tolist()],
        "known_future_feature_cols": [str(v) for v in train_npz["known_future_feature_cols"].tolist()],
        "log1p_feature_cols": [str(v) for v in train_npz["log1p_feature_cols"].tolist()],
        "target_cols": [str(v) for v in train_npz["target_cols"].tolist()],
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "hist_len": int(train_npz["x"].shape[1]),
        "pred_len": int(train_npz["y"].shape[1]),
        "in_dim": int(train_npz["x"].shape[-1]),
        "out_dim": int(train_npz["y"].shape[-1]),
        "num_nodes": int(train_npz["x"].shape[2]),
    }


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/temporal_data/bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20")
    parser.add_argument("--graph_dir", default="data/graph/bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20")
    parser.add_argument("--checkpoint", default="分析结果/2026-05-06_top150_rolling_anchor_gru_baseline/best_gru_baseline.pt")
    parser.add_argument("--mstgcn_baseline_csv", default="分析结果/2026-04-28_top150_predlen6_rolling_anchor_holdout_eval/feb2026_model_vs_naive_baselines.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start_date", default="2026-02-01")
    parser.add_argument("--end_date", default="2026-02-28")
    parser.add_argument("--trip_glob", default="20260[12]-citibike-tripdata_*.csv")
    parser.add_argument("--anchor_hours", default="0,6,12,16,20")
    parser.add_argument("--target_start_offset", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", default="分析结果/2026-05-06_top150_gru_rolling_anchor_feb2026_holdout_eval")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    graph_dir = PROJECT_ROOT / args.graph_dir
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_npz_metadata(data_dir)
    mapping_df = pd.read_csv(graph_dir / "selected_node_mapping.csv").sort_values("Node_ID").reset_index(drop=True)
    station_names = mapping_df[STATION_COL].astype(str).tolist()

    order_dir = PROJECT_ROOT / FEB_DATA_DIR / ORDER_SUBDIR
    asset_dir = PROJECT_ROOT / FEB_DATA_DIR / "nyc_top300_inventory_validation"
    weather_file = PROJECT_ROOT / FEB_DATA_DIR / "weather-get" / "NYC_Weather_2024-01-01_to_2026-02-31.csv"
    trip_files = sorted(glob.glob(str(order_dir / args.trip_glob)))
    if not trip_files:
        raise FileNotFoundError("No trip files matched: %s" % (order_dir / args.trip_glob))

    print("Aggregating hourly trip counts from %d files..." % len(trip_files))
    hourly_df, full_hours = aggregate_hourly_trip_counts(trip_files, station_names)
    all_dates = sorted(hourly_df["日期"].dropna().unique())
    print("Hourly range:", str(full_hours.min()), "->", str(full_hours.max()))

    daily_feature_df, daily_feature_path, aux_paths, weather_path, aux_merge_summary = load_daily_feature_table(
        source_dir=str(asset_dir),
        station_names=station_names,
        all_dates=all_dates,
        daily_feature_file="snapshot_daily_features_topk.csv",
        aux_temporal_files="station_static_features_topk.csv",
        weather_file=str(weather_file),
    )
    feature_df = build_hourly_feature_frame(hourly_df, daily_feature_df)
    pipeline_mapping = mapping_df.rename(columns={STATION_COL: "站点名称"})
    anchor_hours = parse_anchor_hours(args.anchor_hours)
    sample_bundle = build_hourly_samples_for_anchors(
        feature_df=feature_df,
        mapping_df=pipeline_mapping,
        history_feature_cols=metadata["history_feature_cols"],
        target_cols=metadata["target_cols"],
        known_future_feature_cols=metadata["known_future_feature_cols"],
        hist_len=metadata["hist_len"],
        pred_len=metadata["pred_len"],
        anchor_hours=anchor_hours,
        min_known_future_coverage=0.0,
        target_start_offset=args.target_start_offset,
    )

    raw_x_values = sample_bundle["x"]
    x_values = raw_x_values.copy()
    history_len = len(metadata["history_feature_cols"])
    history_x, applied_history_cols = apply_log1p_transform(
        x_values[..., :history_len],
        metadata["history_feature_cols"],
        metadata["log1p_feature_cols"],
    )
    future_x, applied_future_cols = apply_log1p_transform(
        x_values[..., history_len:],
        sample_bundle["known_future_feature_cols"],
        metadata["log1p_feature_cols"],
    )
    x_values = np.concatenate([history_x, future_x], axis=-1)
    y_values = sample_bundle["y"]
    sample_dates = np.array(sample_bundle["sample_dates"])
    sample_datetimes = np.array(sample_bundle["sample_datetimes"])
    sample_anchor_hours = np.array(sample_bundle["anchor_hours"])
    target_start_datetimes = np.array(sample_bundle["target_start_datetimes"])

    wanted_dates = pd.date_range(args.start_date, args.end_date, freq="D").strftime("%Y-%m-%d").to_numpy()
    available_mask = np.isin(sample_dates, wanted_dates)
    missing_dates = [d for d in wanted_dates.tolist() if d not in set(sample_dates.tolist())]
    x_values = x_values[available_mask]
    y_values = y_values[available_mask]
    eval_dates = sample_dates[available_mask]
    sample_datetimes = sample_datetimes[available_mask]
    sample_anchor_hours = sample_anchor_hours[available_mask]
    target_start_datetimes = target_start_datetimes[available_mask]
    if len(eval_dates) == 0:
        raise ValueError("No requested dates are evaluable.")

    x_scaled = ((x_values - metadata["feature_mean"]) / metadata["feature_std"]).astype(np.float32)

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint.get("args", {})
    model = SharedStationGRU(
        in_dim=metadata["in_dim"],
        hidden_dim=int(model_args.get("hidden_dim", 64)),
        pred_len=metadata["pred_len"],
        out_dim=metadata["out_dim"],
        num_nodes=metadata["num_nodes"],
        num_layers=int(model_args.get("num_layers", 1)),
        dropout=float(model_args.get("dropout", 0.0)),
        station_embed_dim=int(model_args.get("station_embed_dim", 16)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_mean = torch.as_tensor(checkpoint["target_mean"], dtype=torch.float32, device=device)
    target_std = torch.as_tensor(checkpoint["target_std"], dtype=torch.float32, device=device)
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_scaled), args.batch_size):
            batch = torch.from_numpy(x_scaled[start:start + args.batch_size]).to(device)
            pred = model(batch)
            pred = pred * target_std + target_mean
            pred = F.softplus(pred, beta=5.0)
            preds.append(pred.cpu().numpy())
    pred_values = np.concatenate(preds, axis=0)
    abs_err = np.abs(pred_values - y_values)

    rows = []
    for sample_idx, date_value in enumerate(eval_dates):
        for horizon_idx in range(y_values.shape[1]):
            target_dt = pd.Timestamp(str(target_start_datetimes[sample_idx])) + pd.Timedelta(hours=horizon_idx)
            for node_idx, station_name in enumerate(station_names):
                rows.append(
                    {
                        "date": date_value,
                        "sample_datetime": str(sample_datetimes[sample_idx]),
                        "anchor_hour": int(sample_anchor_hours[sample_idx]),
                        "target_start_datetime": str(target_start_datetimes[sample_idx]),
                        "horizon": horizon_idx + 1,
                        "hour": target_dt.strftime("%H:00"),
                        "Node_ID": int(mapping_df.loc[node_idx, "Node_ID"]),
                        "station_name": station_name,
                        "pred_out": float(pred_values[sample_idx, horizon_idx, node_idx, 0]),
                        "pred_in": float(pred_values[sample_idx, horizon_idx, node_idx, 1]),
                        "true_out": float(y_values[sample_idx, horizon_idx, node_idx, 0]),
                        "true_in": float(y_values[sample_idx, horizon_idx, node_idx, 1]),
                        "abs_error_out": float(abs_err[sample_idx, horizon_idx, node_idx, 0]),
                        "abs_error_in": float(abs_err[sample_idx, horizon_idx, node_idx, 1]),
                        "abs_error_avg": float(abs_err[sample_idx, horizon_idx, node_idx, :].mean()),
                    }
                )
    station_hour_df = pd.DataFrame(rows)
    station_metrics = (
        station_hour_df.groupby(["Node_ID", "station_name"], as_index=False)
        [["abs_error_avg", "abs_error_out", "abs_error_in"]]
        .mean()
        .rename(columns={"abs_error_avg": "mae_avg", "abs_error_out": "mae_out", "abs_error_in": "mae_in"})
    )
    horizon_df = horizon_metrics(pred_values, y_values)
    anchor_df = anchor_metrics(pred_values, y_values, sample_anchor_hours)
    daily_df = station_hour_df.groupby("date", as_index=False)[["abs_error_avg", "abs_error_out", "abs_error_in"]].mean()

    metrics = compute_metrics(pred_values, y_values)
    baseline_rows = [
        {
            "method": "gru_shared_station_no_graph",
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "note": "GRU external Feb2026 rolling-anchor holdout",
        }
    ]
    mstgcn_baseline_path = PROJECT_ROOT / args.mstgcn_baseline_csv
    if mstgcn_baseline_path.exists():
        existing = pd.read_csv(mstgcn_baseline_path)
        for _, row in existing.iterrows():
            baseline_rows.append(
                {
                    "method": row["method"],
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "note": row.get("note", ""),
                }
            )
    comparison_df = pd.DataFrame(baseline_rows).sort_values("mae")

    station_hour_df.to_csv(output_dir / "gru_feb2026_station_hour_predictions.csv", index=False, encoding="utf-8-sig")
    station_metrics.to_csv(output_dir / "gru_feb2026_station_mae.csv", index=False, encoding="utf-8-sig")
    horizon_df.to_csv(output_dir / "gru_feb2026_horizon_mae.csv", index=False, encoding="utf-8-sig")
    anchor_df.to_csv(output_dir / "gru_feb2026_anchor_hour_mae.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(output_dir / "gru_feb2026_daily_mae.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(output_dir / "gru_vs_mstgcn_and_naive_feb2026_holdout.csv", index=False, encoding="utf-8-sig")

    summary = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "date_start": str(eval_dates[0]),
        "date_end": str(eval_dates[-1]),
        "missing_dates": missing_dates,
        "num_anchor_samples": int(len(eval_dates)),
        "pred_shape": list(pred_values.shape),
        "overall": metrics,
        "anchor_hours": anchor_hours,
        "daily_feature_file": daily_feature_path,
        "weather_file": weather_path,
        "aux_paths": aux_paths,
        "aux_merge_summary": aux_merge_summary,
        "comparison_csv": str(output_dir / "gru_vs_mstgcn_and_naive_feb2026_holdout.csv"),
    }
    with (output_dir / "gru_feb2026_holdout_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Comparison table:", output_dir / "gru_vs_mstgcn_and_naive_feb2026_holdout.csv")


if __name__ == "__main__":
    main()
