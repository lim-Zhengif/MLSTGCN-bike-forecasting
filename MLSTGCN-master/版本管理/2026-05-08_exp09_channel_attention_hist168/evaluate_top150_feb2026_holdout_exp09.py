import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EXP09_VERSION_DIR = SCRIPT_DIR
PIPELINE_VERSION_DIR = PROJECT_ROOT / "\u7248\u672c\u7ba1\u7406" / "2026-04-23_top300\u5168NYC\u5e93\u5b58\u5feb\u7167\u901a\u8def\u9a8c\u8bc1"
for source_dir in [PIPELINE_VERSION_DIR, EXP09_VERSION_DIR]:
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from hourly_pipeline_utils import (  # noqa: E402
    aggregate_hourly_trip_counts,
    apply_log1p_transform,
    build_hourly_feature_frame,
    build_hourly_samples,
    load_daily_feature_table,
)
from datasets.bike import BikeGraph  # noqa: E402
from models.MSTGCN import MSTGCN_submodule  # noqa: E402
from models.fusiongraph import FusionGraphModel  # noqa: E402
import models.fusiongraph as fusiongraph_module  # noqa: E402
from prepare_topk_hourly_dataset_rolling_anchors import (  # noqa: E402
    build_hourly_samples_for_anchors,
    parse_anchor_hours,
)


STATION_COL = "\u7ad9\u70b9\u540d\u79f0"
PIPELINE_STATION_COL = "\u7ad9\u70b9\u540d\u79f0"
ANALYSIS_DIR = "\u5206\u6790\u7ed3\u679c"
FEB_DATA_DIR = "\u4e8c\u6708\u4efd\u6570\u636e\u5904\u7406"
ORDER_SUBDIR = "\u7ebd\u7ea6\u5355\u8f66\u8ba2\u5355\u6570\u636e"


class EvalWrapper(nn.Module):
    def __init__(self, device, fusiongraph, data_config, model_config, categorical_feature_configs):
        super().__init__()
        self.fusiongraph = fusiongraph
        self.model = MSTGCN_submodule(
            device,
            fusiongraph,
            data_config["in_dim"],
            data_config["hist_len"],
            data_config["pred_len"],
            data_config["out_dim"],
            categorical_feature_configs=categorical_feature_configs,
            cheb_k=model_config["cheb_k"],
            nb_block=model_config["nb_block"],
            nb_chev_filter=model_config["nb_chev_filter"],
            nb_time_filter=model_config["nb_time_filter"],
            time_kernel_size=model_config["time_kernel_size"],
            channel_attention=model_config.get("channel_attention", True),
            channel_attention_reduction=model_config.get("channel_attention_reduction", 4),
        )

    def forward(self, x):
        return self.model(x)


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_npz_metadata(data_dir):
    train_npz = np.load(data_dir / "train.npz", allow_pickle=True)
    return {
        "input_feature_cols": [str(v) for v in train_npz["input_feature_cols"].tolist()],
        "history_feature_cols": [str(v) for v in train_npz["history_feature_cols"].tolist()],
        "known_future_feature_cols": [str(v) for v in train_npz["known_future_feature_cols"].tolist()],
        "log1p_feature_cols": [str(v) for v in train_npz["log1p_feature_cols"].tolist()],
        "target_cols": [str(v) for v in train_npz["target_cols"].tolist()],
        "feature_mean": train_npz["x"].mean(axis=(0, 1, 2), keepdims=True).astype(np.float32),
        "feature_std": np.where(train_npz["x"].std(axis=(0, 1, 2), keepdims=True) == 0, 1.0, train_npz["x"].std(axis=(0, 1, 2), keepdims=True)).astype(np.float32),
        "target_mean": train_npz["y"].mean(axis=(0, 1, 2), keepdims=True).astype(np.float32),
        "target_std": np.where(train_npz["y"].std(axis=(0, 1, 2), keepdims=True) == 0, 1.0, train_npz["y"].std(axis=(0, 1, 2), keepdims=True)).astype(np.float32),
        "hist_len": int(train_npz["x"].shape[1]),
        "pred_len": int(train_npz["y"].shape[1]),
    }


def build_categorical_feature_configs(metadata, weekday_embed_dim):
    if weekday_embed_dim <= 0:
        return []
    feature_mean = metadata["feature_mean"].reshape(-1)
    feature_std = metadata["feature_std"].reshape(-1)
    configs = []
    for idx, name in enumerate(metadata["input_feature_cols"]):
        if name in {"\u661f\u671f\u51e0", "future_\u661f\u671f\u51e0"}:
            configs.append(
                {
                    "index": idx,
                    "num_embeddings": 7,
                    "embedding_dim": weekday_embed_dim,
                    "mean": float(feature_mean[idx]),
                    "std": float(feature_std[idx] if feature_std[idx] != 0 else 1.0),
                    "name": name,
                }
            )
    return configs


def find_best_checkpoint(project_name):
    pattern = PROJECT_ROOT / "logs" / project_name / "version_*" / "checkpoints" / "best-*.ckpt"
    matches = sorted(glob.glob(str(pattern)), key=lambda path: os.path.getmtime(path))
    if not matches:
        raise FileNotFoundError("No best checkpoint matched: %s" % pattern)
    return matches[-1]


def compute_metrics(pred, true):
    err = np.abs(pred - true)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join("data", "temporal_data", "bike_hourly_safe_inventory_top150_nyc_full"))
    parser.add_argument("--graph_dir", default=os.path.join("data", "graph", "bike_hourly_safe_inventory_top150_nyc_full"))
    parser.add_argument("--project", default="bike_hourly_safe_inventory_top150_nyc_full_exp06_transfer_bs4")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start_date", default="2026-02-01")
    parser.add_argument("--end_date", default="2026-02-28")
    parser.add_argument("--trip_glob", default="20260[12]-citibike-tripdata_*.csv")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--anchor_hours", default=None, help="Optional comma-separated decision anchors, e.g. 0,6,12,16,20.")
    parser.add_argument("--target_start_offset", type=int, default=1, help="For rolling anchors, decision t predicts t+offset...")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    graph_dir = PROJECT_ROOT / args.graph_dir
    order_dir = PROJECT_ROOT / FEB_DATA_DIR / ORDER_SUBDIR
    asset_dir = PROJECT_ROOT / FEB_DATA_DIR / "nyc_top300_inventory_validation"
    weather_file = PROJECT_ROOT / FEB_DATA_DIR / "weather-get" / "NYC_Weather_2024-01-01_to_2026-02-31.csv"
    mapping_path = graph_dir / "selected_node_mapping.csv"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / ANALYSIS_DIR / "2026-04-24_top150_202602_holdout_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_npz_metadata(data_dir)
    mapping_df = pd.read_csv(mapping_path).sort_values("Node_ID").reset_index(drop=True)
    station_names = mapping_df[STATION_COL].astype(str).tolist()

    trip_files = sorted(glob.glob(str(order_dir / args.trip_glob)))
    if not trip_files:
        raise FileNotFoundError("No trip files matched: %s" % (order_dir / args.trip_glob))

    print("Aggregating hourly trip counts from %d files..." % len(trip_files))
    hourly_df, full_hours = aggregate_hourly_trip_counts(trip_files, station_names)
    all_dates = sorted(hourly_df["\u65e5\u671f"].dropna().unique())
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
    pipeline_mapping = mapping_df.rename(columns={STATION_COL: PIPELINE_STATION_COL})

    if args.anchor_hours:
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
    else:
        anchor_hours = None
        sample_bundle = build_hourly_samples(
            feature_df=feature_df,
            mapping_df=pipeline_mapping,
            history_feature_cols=metadata["history_feature_cols"],
            target_cols=metadata["target_cols"],
            known_future_feature_cols=metadata["known_future_feature_cols"],
            hist_len=metadata["hist_len"],
            pred_len=metadata["pred_len"],
            min_known_future_coverage=0.0,
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
    sample_datetimes = np.array(sample_bundle.get("sample_datetimes", sample_bundle["sample_dates"]))
    sample_anchor_hours = np.array(sample_bundle.get("anchor_hours", [-1] * len(sample_dates)))
    target_start_datetimes = np.array(sample_bundle.get("target_start_datetimes", sample_bundle["sample_dates"]))

    wanted_dates = pd.date_range(args.start_date, args.end_date, freq="D").strftime("%Y-%m-%d").to_numpy()
    available_mask = np.isin(sample_dates, wanted_dates)
    missing_dates = [d for d in wanted_dates.tolist() if d not in set(sample_dates.tolist())]
    x_values = x_values[available_mask]
    raw_x_values = raw_x_values[available_mask]
    y_values = y_values[available_mask]
    sample_datetimes = sample_datetimes[available_mask]
    sample_anchor_hours = sample_anchor_hours[available_mask]
    target_start_datetimes = target_start_datetimes[available_mask]
    eval_dates = sample_dates[available_mask]
    if len(eval_dates) == 0:
        raise ValueError("No requested February dates are evaluable. Missing examples: %s" % missing_dates[:5])

    feature_mean = metadata["feature_mean"]
    feature_std = metadata["feature_std"]
    x_scaled = ((x_values - feature_mean[0]) / feature_std[0]).astype(np.float32)

    device = resolve_device(args.device)
    graph_config = {
        "use": ["dist", "neighb", "distri", "tempp", "func"],
        "fix_weight": False,
        "tempp_diag_zero": True,
        "matrix_weight": True,
        "distri_type": "exp",
        "func_type": "ours",
        "attention": True,
        "sparsify_mode": "topk",
        "sparsify_topk": 20,
        "sparsify_symmetric": True,
        "sparsify_keep_self": True,
    }
    data_config = {"in_dim": x_scaled.shape[-1], "out_dim": y_values.shape[-1], "hist_len": x_scaled.shape[1], "pred_len": y_values.shape[1], "type": "bike"}
    model_config = {
        "cheb_k": 3,
        "nb_block": 2,
        "nb_chev_filter": 64,
        "nb_time_filter": 64,
        "time_kernel_size": 3,
        "channel_attention": True,
        "channel_attention_reduction": 4,
    }

    fusiongraph_module.PROJECT_ROOT = str(PROJECT_ROOT)
    graph = BikeGraph(str(graph_dir), graph_config, device)
    fusiongraph = FusionGraphModel(graph, device, graph_config, data_config, 24, 6, 0.1)
    categorical_configs = build_categorical_feature_configs(metadata, weekday_embed_dim=8)
    model = EvalWrapper(device, fusiongraph, data_config, model_config, categorical_configs).to(device)
    checkpoint_path = args.checkpoint or find_best_checkpoint(args.project)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [key for key in missing if not key.startswith("metric_lightning.")]
    if real_missing:
        raise RuntimeError("Missing checkpoint keys: %s" % real_missing[:10])
    model.eval()

    preds = []
    batch_size = 4
    target_mean = torch.as_tensor(metadata["target_mean"], dtype=torch.float32, device=device)
    target_std = torch.as_tensor(metadata["target_std"], dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            batch = torch.from_numpy(x_scaled[start:start + batch_size]).to(device)
            pred = model(batch)
            pred = pred * target_std + target_mean
            pred = F.softplus(pred, beta=5.0)
            preds.append(pred.cpu().numpy())
    pred_values = np.concatenate(preds, axis=0)
    abs_err = np.abs(pred_values - y_values)

    pred_len = y_values.shape[1]
    naive_last_value = np.repeat(raw_x_values[:, -1:, :, :2], pred_len, axis=1)
    if raw_x_values.shape[1] >= 24 + pred_len - 1:
        previous_day_same_hours = raw_x_values[:, -24:-24 + pred_len, :, :2]
    elif raw_x_values.shape[1] >= pred_len:
        previous_day_same_hours = raw_x_values[:, -pred_len:, :, :2]
    else:
        previous_day_same_hours = naive_last_value
    train_y = np.load(data_dir / "train.npz", allow_pickle=True)["y"]
    naive_train_mean = np.repeat(train_y.mean(axis=0, keepdims=True), y_values.shape[0], axis=0)
    baseline_rows = [
        {
            "method": "model_exp06_top150",
            "mae": float(abs_err.mean()),
            "rmse": float(np.sqrt(np.mean((pred_values - y_values) ** 2))),
            "note": "current top150 checkpoint",
        },
        {
            "method": "naive_last_1h_repeat",
            "mae": float(np.abs(naive_last_value - y_values).mean()),
            "rmse": float(np.sqrt(np.mean((naive_last_value - y_values) ** 2))),
            "note": "repeat the last observed historical hour",
        },
        {
            "method": "naive_previous_day_same_hours",
            "mae": float(np.abs(previous_day_same_hours - y_values).mean()),
            "rmse": float(np.sqrt(np.mean((previous_day_same_hours - y_values) ** 2))),
            "note": "copy the same target-hour range from the previous day",
        },
        {
            "method": "naive_train_mean_by_horizon",
            "mae": float(np.abs(naive_train_mean - y_values).mean()),
            "rmse": float(np.sqrt(np.mean((naive_train_mean - y_values) ** 2))),
            "note": "train-set mean for each horizon/node/channel",
        },
    ]

    rows = []
    for d_idx, date_value in enumerate(eval_dates):
        for h in range(y_values.shape[1]):
            if args.anchor_hours:
                target_dt = pd.Timestamp(str(target_start_datetimes[d_idx])) + pd.Timedelta(hours=h)
                hour_label = target_dt.strftime("%H:00")
            else:
                hour_label = "%02d:00" % h
            for n_idx, station_name in enumerate(station_names):
                rows.append(
                    {
                        "date": date_value,
                        "sample_datetime": str(sample_datetimes[d_idx]),
                        "anchor_hour": int(sample_anchor_hours[d_idx]),
                        "target_start_datetime": str(target_start_datetimes[d_idx]),
                        "horizon": h + 1,
                        "hour": hour_label,
                        "Node_ID": int(mapping_df.loc[n_idx, "Node_ID"]),
                        "station_name": station_name,
                        "pred_out": float(pred_values[d_idx, h, n_idx, 0]),
                        "pred_in": float(pred_values[d_idx, h, n_idx, 1]),
                        "true_out": float(y_values[d_idx, h, n_idx, 0]),
                        "true_in": float(y_values[d_idx, h, n_idx, 1]),
                        "abs_error_out": float(abs_err[d_idx, h, n_idx, 0]),
                        "abs_error_in": float(abs_err[d_idx, h, n_idx, 1]),
                        "abs_error_avg": float(abs_err[d_idx, h, n_idx, :].mean()),
                    }
                )
    station_hour_df = pd.DataFrame(rows)
    station_metrics = (
        station_hour_df.groupby(["Node_ID", "station_name"], as_index=False)
        [["abs_error_avg", "abs_error_out", "abs_error_in"]]
        .mean()
        .rename(columns={"abs_error_avg": "mae_avg", "abs_error_out": "mae_out", "abs_error_in": "mae_in"})
    )
    horizon_metrics = station_hour_df.groupby("horizon", as_index=False)[["abs_error_avg", "abs_error_out", "abs_error_in"]].mean()
    daily_metrics = station_hour_df.groupby("date", as_index=False)[["abs_error_avg", "abs_error_out", "abs_error_in"]].mean()
    anchor_metrics = station_hour_df.groupby("anchor_hour", as_index=False)[["abs_error_avg", "abs_error_out", "abs_error_in"]].mean()
    anchor_horizon_metrics = station_hour_df.groupby(["anchor_hour", "horizon"], as_index=False)[["abs_error_avg", "abs_error_out", "abs_error_in"]].mean()
    peak_metrics = []
    for name, hours in [("morning_07_10", {7, 8, 9}), ("evening_17_20", {17, 18, 19})]:
        target_horizons = {h + 1 for h in hours if h < pred_len}
        sub = station_hour_df[station_hour_df["horizon"].isin(target_horizons)] if target_horizons else station_hour_df.iloc[0:0]
        peak_metrics.append(
            {
                "window": name,
                "mae_avg": None if sub.empty else float(sub["abs_error_avg"].mean()),
                "mae_out": None if sub.empty else float(sub["abs_error_out"].mean()),
                "mae_in": None if sub.empty else float(sub["abs_error_in"].mean()),
                "covered_horizons": ",".join(str(v) for v in sorted(target_horizons)),
            }
        )
    peak_metrics = pd.DataFrame(peak_metrics)
    baseline_metrics = pd.DataFrame(baseline_rows)

    summary = {
        "checkpoint": checkpoint_path,
        "trip_files": trip_files,
        "date_start": str(eval_dates[0]),
        "date_end": str(eval_dates[-1]),
        "requested_start": args.start_date,
        "requested_end": args.end_date,
        "missing_dates": missing_dates,
        "num_days": int(len(eval_dates)),
        "num_stations": int(len(station_names)),
        "pred_shape": list(pred_values.shape),
        "overall": compute_metrics(pred_values.reshape(-1), y_values.reshape(-1)),
        "hist_len": int(metadata["hist_len"]),
        "pred_len": int(metadata["pred_len"]),
        "anchor_hours": None if anchor_hours is None else anchor_hours,
        "target_start_offset": args.target_start_offset if args.anchor_hours else None,
        "anchor_note": (
            "Rolling decision anchors: horizon 1 starts at anchor + target_start_offset."
            if args.anchor_hours
            else "Daily 00:00 anchor: horizon 1 starts at 00:00 of each evaluated date."
        ),
        "mae_1_6": float(abs_err[:, :min(6, pred_len), :, :].mean()),
        "mae_after_6": None if pred_len <= 6 else float(abs_err[:, 6:, :, :].mean()),
        "morning_07_10_mae": None if pred_len < 10 else float(station_hour_df[station_hour_df["horizon"].isin([8, 9, 10])]["abs_error_avg"].mean()),
        "evening_17_20_mae": None if pred_len < 20 else float(station_hour_df[station_hour_df["horizon"].isin([18, 19, 20])]["abs_error_avg"].mean()),
        "baseline_mae": {row["method"]: row["mae"] for row in baseline_rows},
        "model_improve_vs_last_1h_repeat_pct": float((baseline_rows[1]["mae"] - baseline_rows[0]["mae"]) / baseline_rows[1]["mae"] * 100.0),
        "model_improve_vs_previous_day_same_hours_pct": float((baseline_rows[2]["mae"] - baseline_rows[0]["mae"]) / baseline_rows[2]["mae"] * 100.0),
        "model_improve_vs_train_mean_pct": float((baseline_rows[3]["mae"] - baseline_rows[0]["mae"]) / baseline_rows[3]["mae"] * 100.0),
        "daily_feature_file": daily_feature_path,
        "weather_file": weather_path,
        "aux_paths": aux_paths,
        "aux_merge_summary": aux_merge_summary,
        "applied_log1p_cols": sorted(set(applied_history_cols + applied_future_cols)),
        "known_future_cols": sample_bundle["known_future_feature_cols"],
    }

    station_hour_df.to_csv(output_dir / "feb2026_station_hour_predictions.csv", index=False, encoding="utf-8-sig")
    station_metrics.to_csv(output_dir / "feb2026_station_mae.csv", index=False, encoding="utf-8-sig")
    horizon_metrics.to_csv(output_dir / "feb2026_horizon_mae.csv", index=False, encoding="utf-8-sig")
    daily_metrics.to_csv(output_dir / "feb2026_daily_mae.csv", index=False, encoding="utf-8-sig")
    anchor_metrics.to_csv(output_dir / "feb2026_anchor_hour_mae.csv", index=False, encoding="utf-8-sig")
    anchor_horizon_metrics.to_csv(output_dir / "feb2026_anchor_hour_horizon_mae.csv", index=False, encoding="utf-8-sig")
    peak_metrics.to_csv(output_dir / "feb2026_peak_window_mae.csv", index=False, encoding="utf-8-sig")
    baseline_metrics.to_csv(output_dir / "feb2026_model_vs_naive_baselines.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "feb2026_holdout_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved outputs to:", output_dir)


if __name__ == "__main__":
    main()
