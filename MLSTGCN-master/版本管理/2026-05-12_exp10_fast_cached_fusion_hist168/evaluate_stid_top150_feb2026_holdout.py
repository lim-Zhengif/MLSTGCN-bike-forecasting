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
PIPELINE_VERSION_DIR = PROJECT_ROOT / "版本管理" / "2026-04-23_top300全NYC库存快照通路验证"
for source_dir in [PIPELINE_VERSION_DIR, SCRIPT_DIR]:
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

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
from train_stid_top150_baseline import (  # noqa: E402
    STIDBaseline,
    anchor_metrics,
    compute_metrics,
    horizon_metrics,
)


ANALYSIS_DIR = "分析结果"
FEB_DATA_DIR = "二月份数据处理"
ORDER_SUBDIR = "纽约单车订单数据"
STATION_COL = "站点名称"
DEFAULT_WANDB_PROJECT = "top150_rolling6h_model_compare"


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


def init_wandb_run(args, output_dir):
    if args.logger != "wandb":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logger requested, but wandb is unavailable in the current environment.") from exc

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.eval_tag,
        config=vars(args),
        tags=tags or None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/temporal_data/bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20_train2025_hist168")
    parser.add_argument("--graph_dir", default="data/graph/bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_graph_train2025_hist168")
    parser.add_argument("--checkpoint", default="分析结果/2026-05-15_top150_stid_baseline_hist168_pred6_seed0/best_stid_baseline.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start_date", default="2026-02-01")
    parser.add_argument("--end_date", default="2026-02-28")
    parser.add_argument("--trip_glob", default="20260[12]-citibike-tripdata_*.csv")
    parser.add_argument("--order_dir", default=None, help="Optional order-data directory. Defaults to 二月份数据处理/纽约单车订单数据.")
    parser.add_argument("--weather_file", default=None, help="Optional weather CSV. Defaults to the current February-processing weather file.")
    parser.add_argument("--eval_tag", default="stid_feb2026", help="Prefix for output CSV/JSON filenames, e.g. stid_feb_mar2026.")
    parser.add_argument("--anchor_hours", default="0,6,12,16,20")
    parser.add_argument("--target_start_offset", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", default="分析结果/2026-05-15_top150_stid_feb2026_holdout")
    parser.add_argument("--logger", choices=["csv", "wandb"], default="csv")
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_tags", default="stid,holdout")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    graph_dir = PROJECT_ROOT / args.graph_dir
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb_run(args, output_dir)

    metadata = load_npz_metadata(data_dir)
    mapping_df = pd.read_csv(graph_dir / "selected_node_mapping.csv").sort_values("Node_ID").reset_index(drop=True)
    station_names = mapping_df[STATION_COL].astype(str).tolist()

    order_dir = Path(args.order_dir) if args.order_dir else PROJECT_ROOT / FEB_DATA_DIR / ORDER_SUBDIR
    if not order_dir.is_absolute():
        order_dir = PROJECT_ROOT / order_dir
    asset_dir = PROJECT_ROOT / FEB_DATA_DIR / "nyc_top300_inventory_validation"
    weather_file = Path(args.weather_file) if args.weather_file else PROJECT_ROOT / FEB_DATA_DIR / "weather-get" / "NYC_Weather_2024-01-01_to_2026-02-31.csv"
    if not weather_file.is_absolute():
        weather_file = PROJECT_ROOT / weather_file
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
    model = STIDBaseline(
        hist_len=metadata["hist_len"],
        in_dim=metadata["in_dim"],
        pred_len=metadata["pred_len"],
        out_dim=metadata["out_dim"],
        num_nodes=metadata["num_nodes"],
        hidden_dim=int(model_args.get("hidden_dim", 256)),
        node_embed_dim=int(model_args.get("node_embed_dim", 32)),
        time_embed_dim=int(model_args.get("time_embed_dim", 16)),
        weekday_embed_dim=int(model_args.get("weekday_embed_dim", 16)),
        num_layers=int(model_args.get("num_layers", 2)),
        dropout=float(model_args.get("dropout", 0.1)),
        hour_feature_idx=checkpoint.get("hour_feature_idx"),
        weekday_feature_idx=checkpoint.get("weekday_feature_idx"),
        feature_mean=checkpoint["feature_mean"],
        feature_std=checkpoint["feature_std"],
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
            pred = F.softplus(pred * target_std + target_mean, beta=5.0)
            preds.append(pred.cpu().numpy())
    pred_values = np.concatenate(preds, axis=0)
    abs_err = np.abs(pred_values - y_values)

    rows = []
    for sample_idx, date_value in enumerate(eval_dates):
        for horizon_idx in range(y_values.shape[1]):
            target_dt = pd.Timestamp(str(target_start_datetimes[sample_idx])) + pd.Timedelta(hours=horizon_idx)
            for node_idx, station_name in enumerate(station_names):
                rows.append({
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
                })
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
    tag = args.eval_tag
    station_hour_df.to_csv(output_dir / ("%s_station_hour_predictions.csv" % tag), index=False, encoding="utf-8-sig")
    station_metrics.to_csv(output_dir / ("%s_station_mae.csv" % tag), index=False, encoding="utf-8-sig")
    horizon_df.to_csv(output_dir / ("%s_horizon_mae.csv" % tag), index=False, encoding="utf-8-sig")
    anchor_df.to_csv(output_dir / ("%s_anchor_hour_mae.csv" % tag), index=False, encoding="utf-8-sig")
    daily_df.to_csv(output_dir / ("%s_daily_mae.csv" % tag), index=False, encoding="utf-8-sig")

    summary = {
        "model": "STID-style baseline",
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "order_dir": str(order_dir),
        "trip_glob": args.trip_glob,
        "requested_start": args.start_date,
        "requested_end": args.end_date,
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
        "applied_log1p_cols": sorted(set(applied_history_cols + applied_future_cols)),
    }
    (output_dir / ("%s_holdout_summary.json" % tag)).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if wandb_run is not None:
        wandb_run.summary["holdout_mae"] = float(metrics["mae"])
        wandb_run.summary["holdout_rmse"] = float(metrics["rmse"])
        wandb_run.summary["date_start"] = str(eval_dates[0])
        wandb_run.summary["date_end"] = str(eval_dates[-1])
        wandb_run.summary["requested_start"] = args.start_date
        wandb_run.summary["requested_end"] = args.end_date
        wandb_run.summary["num_anchor_samples"] = int(len(eval_dates))
        wandb_run.summary["checkpoint"] = str(checkpoint_path)
        wandb_run.log(
            {
                "holdout/mae": float(metrics["mae"]),
                "holdout/rmse": float(metrics["rmse"]),
                "holdout/num_anchor_samples": int(len(eval_dates)),
            }
        )
        wandb_run.finish()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
