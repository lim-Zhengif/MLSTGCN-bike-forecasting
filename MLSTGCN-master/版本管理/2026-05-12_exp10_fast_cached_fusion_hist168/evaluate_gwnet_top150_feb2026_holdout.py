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
PIPELINE_VERSION_DIR = PROJECT_ROOT / "\u7248\u672c\u7ba1\u7406" / "2026-04-23_top300\u5168NYC\u5e93\u5b58\u5feb\u7167\u901a\u8def\u9a8c\u8bc1"
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
from train_gwnet_top150_baseline import (  # noqa: E402
    GraphWaveNetBaseline,
    anchor_metrics,
    compute_metrics,
    horizon_metrics,
    load_supports,
)


ANALYSIS_DIR = "\u5206\u6790\u7ed3\u679c"
FEB_DATA_DIR = "\u4e8c\u6708\u4efd\u6570\u636e\u5904\u7406"
ORDER_SUBDIR = "\u7ebd\u7ea6\u5355\u8f66\u8ba2\u5355\u6570\u636e"
STATION_COL = "\u7ad9\u70b9\u540d\u79f0"
DEFAULT_WANDB_PROJECT = "top150_rolling6h_model_compare"


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_column(df, candidates, fallback_idx=1):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if len(df.columns) > fallback_idx:
        return df.columns[fallback_idx]
    raise KeyError("Cannot resolve column from: %s" % list(df.columns))


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
        raise RuntimeError("W&B logger requested, but wandb is unavailable.") from exc

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.eval_tag,
        config=vars(args),
        tags=tags or None,
    )


def load_metadata_from_checkpoint(checkpoint):
    return {
        "history_feature_cols": [str(item) for item in checkpoint["history_feature_cols"]],
        "known_future_feature_cols": [str(item) for item in checkpoint["known_future_feature_cols"]],
        "log1p_feature_cols": [str(item) for item in checkpoint["log1p_feature_cols"]],
        "target_cols": [str(item) for item in checkpoint["target_cols"]],
        "feature_mean": checkpoint["feature_mean"],
        "feature_std": checkpoint["feature_std"],
        "target_mean": checkpoint["target_mean"],
        "target_std": checkpoint["target_std"],
        "hist_len": int(checkpoint["hist_len"]),
        "pred_len": int(checkpoint["pred_len"]),
        "in_dim": int(checkpoint["in_dim"]),
        "out_dim": int(checkpoint["out_dim"]),
        "num_nodes": int(checkpoint["num_nodes"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/temporal_data/bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20_train2025_hist168")
    parser.add_argument("--graph_dir", default="data/graph/bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_graph_train2025_hist168")
    parser.add_argument("--checkpoint", default=str(Path(ANALYSIS_DIR) / "2026-05-20_top150_gwnet_baseline_hist168_pred6_seed0" / "best_gwnet_baseline.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start_date", default="2026-02-01")
    parser.add_argument("--end_date", default="2026-02-28")
    parser.add_argument("--trip_glob", default="20260[12]-citibike-tripdata_*.csv")
    parser.add_argument("--order_dir", default=None)
    parser.add_argument("--weather_file", default=None)
    parser.add_argument("--eval_tag", default="gwnet_feb2026")
    parser.add_argument("--anchor_hours", default="0,6,12,16,20")
    parser.add_argument("--target_start_offset", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", default=str(Path(ANALYSIS_DIR) / "2026-05-20_top150_gwnet_feb2026_holdout"))
    parser.add_argument("--logger", choices=["csv", "wandb"], default="csv")
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_tags", default="gwnet,holdout")
    args = parser.parse_args()

    data_dir = resolve_project_path(args.data_dir)
    graph_dir = resolve_project_path(args.graph_dir)
    checkpoint_path = resolve_project_path(args.checkpoint)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb_run(args, output_dir)
    device = resolve_device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = load_metadata_from_checkpoint(checkpoint)
    graph_use = checkpoint.get("graph_use", checkpoint.get("args", {}).get("graph_use", "dist,neighb,distri,tempp,func"))
    if isinstance(graph_use, str):
        graph_use = [item.strip() for item in graph_use.split(",") if item.strip()]

    mapping_df = pd.read_csv(graph_dir / "selected_node_mapping.csv").sort_values("Node_ID").reset_index(drop=True)
    station_col = resolve_column(mapping_df, [STATION_COL, "station_name"])
    station_names = mapping_df[station_col].astype(str).tolist()

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
    date_col = resolve_column(hourly_df, ["\u65e5\u671f"])
    all_dates = sorted(hourly_df[date_col].dropna().unique())
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
    pipeline_mapping = mapping_df.rename(columns={station_col: STATION_COL})
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

    x_values = sample_bundle["x"].copy()
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
    supports = [support.to(device) for support in load_supports(graph_dir, graph_use)]
    model_args = checkpoint.get("args", {})
    model = GraphWaveNetBaseline(
        in_dim=metadata["in_dim"],
        pred_len=metadata["pred_len"],
        out_dim=metadata["out_dim"],
        num_nodes=metadata["num_nodes"],
        supports=supports,
        residual_channels=int(model_args.get("residual_channels", 32)),
        skip_channels=int(model_args.get("skip_channels", 64)),
        end_channels=int(model_args.get("end_channels", 128)),
        layers=int(model_args.get("layers", 8)),
        dilation_cycle=int(model_args.get("dilation_cycle", 4)),
        kernel_size=int(model_args.get("kernel_size", 2)),
        gcn_order=int(model_args.get("gcn_order", 2)),
        dropout=float(model_args.get("dropout", 0.2)),
        adaptive_adj=str(model_args.get("adaptive_adj", "true")).lower() == "true",
        node_embed_dim=int(model_args.get("node_embed_dim", 10)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_mean = torch.as_tensor(metadata["target_mean"], dtype=torch.float32, device=device)
    target_std = torch.as_tensor(metadata["target_std"], dtype=torch.float32, device=device)
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
        "model": "Graph WaveNet-style baseline",
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "graph_dir": str(graph_dir),
        "graph_use": graph_use,
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
