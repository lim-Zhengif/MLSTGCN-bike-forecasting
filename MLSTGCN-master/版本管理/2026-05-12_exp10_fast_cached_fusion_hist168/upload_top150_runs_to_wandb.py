import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd

try:
    import wandb
except ImportError as exc:
    raise RuntimeError("wandb is not installed in the active Python environment.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


MAIN_RUNS = [
    {
        "run_name": "exp06_hist72_seed0",
        "group": "hist_len",
        "model": "MSTGCN-exp06",
        "hist_len": 72,
        "project_dir": "bike_hourly_safe_inventory_top150_exp06_rolling6h_train2025_hist72_bs8_seed0",
        "holdout": "分析结果/2026-05-06_top150_exp06_rolling6h_train2025_hist72_feb2026_holdout/feb2026_holdout_summary.json",
    },
    {
        "run_name": "exp06_hist168_seed0",
        "group": "hist_len",
        "model": "MSTGCN-exp06",
        "hist_len": 168,
        "project_dir": "bike_hourly_safe_inventory_top150_exp06_rolling6h_train2025_hist168_bs8_seed0",
        "holdout": "分析结果/2026-05-06_top150_exp06_rolling6h_train2025_hist168_feb2026_holdout/feb2026_holdout_summary.json",
    },
    {
        "run_name": "exp06_hist336_seed0",
        "group": "hist_len",
        "model": "MSTGCN-exp06",
        "hist_len": 336,
        "project_dir": "bike_hourly_safe_inventory_top150_exp06_rolling6h_train2025_hist336_bs4_seed0",
        "holdout": "分析结果/2026-05-06_top150_exp06_rolling6h_train2025_hist336_feb2026_holdout/feb2026_holdout_summary.json",
    },
    {
        "run_name": "exp09_channel_attention_hist168_seed0",
        "group": "model_variant",
        "model": "MSTGCN-exp09-channel-attention",
        "hist_len": 168,
        "project_dir": "bike_hourly_safe_inventory_top150_exp09_channel_attention_rolling6h_train2025_hist168_bs8_seed0",
        "holdout": "分析结果/2026-05-08_exp09_channel_attention_hist168/feb2026_holdout/feb2026_holdout_summary.json",
    },
    {
        "run_name": "exp10_anchor_hour_od_hist168_seed0",
        "group": "model_variant",
        "model": "MSTGCN-exp10-anchor-hour-od",
        "hist_len": 168,
        "project_dir": "bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_rolling6h_train2025_hist168_bs8_seed0",
        "holdout": "分析结果/2026-05-08_exp10_anchor_hour_od_graph_hist168/feb2026_holdout/feb2026_holdout_summary.json",
    },
]


LEGACY_RUNS = [
    {
        "run_name": "top150_24h_transfer_bs4_seed0",
        "group": "legacy_top150",
        "model": "MSTGCN-exp06-24h",
        "hist_len": 168,
        "project_dir": "bike_hourly_safe_inventory_top150_nyc_full_exp06_transfer_bs4",
        "holdout": None,
    },
    {
        "run_name": "top150_24h_transfer_bs8_seed1",
        "group": "legacy_top150",
        "model": "MSTGCN-exp06-24h",
        "hist_len": 168,
        "project_dir": "bike_hourly_safe_inventory_top150_nyc_full_exp06_transfer_bs8_seed1",
        "holdout": None,
    },
    {
        "run_name": "top150_24h_transfer_bs8_seed2",
        "group": "legacy_top150",
        "model": "MSTGCN-exp06-24h",
        "hist_len": 168,
        "project_dir": "bike_hourly_safe_inventory_top150_nyc_full_exp06_transfer_bs8_seed2",
        "holdout": None,
    },
    {
        "run_name": "top150_predlen6_all_hours_seed0",
        "group": "legacy_predlen6",
        "model": "MSTGCN-exp06-predlen6",
        "hist_len": 168,
        "project_dir": "bike_hourly_safe_inventory_top150_nyc_full_exp06_predlen6_bs8_seed0",
        "holdout": "分析结果/2026-04-28_top150_predlen6_202602_holdout_eval/feb2026_holdout_summary.json",
    },
    {
        "run_name": "top150_predlen6_anchor_00_06_12_16_20_seed0",
        "group": "legacy_predlen6",
        "model": "MSTGCN-exp06-predlen6-anchor",
        "hist_len": 168,
        "project_dir": "bike_hourly_safe_inventory_top150_nyc_full_exp06_predlen6_anchors_00_06_12_16_20_bs8_seed0",
        "holdout": "分析结果/2026-04-28_top150_predlen6_rolling_anchor_holdout_eval/feb2026_holdout_summary.json",
    },
]


def latest_metrics_path(project_dir):
    candidates = sorted(
        (PROJECT_ROOT / "logs" / project_dir).glob("version_*/metrics.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def finite_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def load_holdout_summary(path_value):
    if not path_value:
        return {}
    path = PROJECT_ROOT / path_value
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    overall = data.get("overall", {})
    baseline = data.get("baseline_mae", {})
    return {
        "feb_holdout_mae": finite_float(overall.get("mae")),
        "feb_holdout_rmse": finite_float(overall.get("rmse")),
        "feb_holdout_mae_1_6": finite_float(data.get("mae_1_6")),
        "baseline_last_1h_mae": finite_float(baseline.get("naive_last_1h_repeat")),
        "baseline_previous_day_mae": finite_float(baseline.get("naive_previous_day_same_hours")),
        "baseline_train_mean_mae": finite_float(baseline.get("naive_train_mean_by_horizon")),
        "improve_vs_last_1h_pct": finite_float(data.get("model_improve_vs_last_1h_repeat_pct")),
        "improve_vs_previous_day_pct": finite_float(data.get("model_improve_vs_previous_day_same_hours_pct")),
        "improve_vs_train_mean_pct": finite_float(data.get("model_improve_vs_train_mean_pct")),
    }


def log_metrics_csv(metrics_path):
    df = pd.read_csv(metrics_path)
    for idx, row in df.iterrows():
        payload = {}
        for key, value in row.items():
            value = finite_float(value)
            if value is not None:
                if key == "step":
                    payload["csv_step"] = value
                else:
                    payload[key] = value
        if payload:
            payload["upload_step"] = idx
            wandb.log(payload, step=idx)


def last_finite_in_epoch(epoch_df, column):
    if column not in epoch_df.columns:
        return None
    values = [finite_float(value) for value in epoch_df[column].tolist()]
    values = [value for value in values if value is not None]
    return values[-1] if values else None


def log_epoch_metrics_csv(metrics_path):
    df = pd.read_csv(metrics_path)
    if "epoch" not in df.columns:
        return

    metric_columns = [
        "lr-Adam",
        "train_loss_epoch",
        "train_mae_epoch",
        "val_loss_epoch",
        "val_mae_epoch",
        "test_loss",
        "test_mae",
        "mae_avg",
        "rmse_avg",
    ]
    alias_map = {
        "train_loss_epoch": "epoch/train_loss",
        "train_mae_epoch": "epoch/train_mae",
        "val_loss_epoch": "epoch/val_loss",
        "val_mae_epoch": "epoch/val_mae",
        "test_loss": "epoch/test_loss",
        "test_mae": "epoch/test_mae",
        "mae_avg": "epoch/test_mae_avg",
        "rmse_avg": "epoch/test_rmse_avg",
        "lr-Adam": "epoch/lr",
    }

    epoch_values = [finite_float(value) for value in df["epoch"].tolist()]
    epochs = sorted({int(value) for value in epoch_values if value is not None})
    for epoch in epochs:
        epoch_df = df[df["epoch"].apply(lambda value: finite_float(value) == epoch)]
        payload = {"epoch": epoch}
        for column in metric_columns:
            value = last_finite_in_epoch(epoch_df, column)
            if value is None:
                continue
            payload[column] = value
            payload[alias_map[column]] = value
        if len(payload) > 1:
            wandb.log(payload, step=epoch)


def upload_run(run_spec, wandb_project, dry_run=False, log_mode="epoch"):
    metrics_path = latest_metrics_path(run_spec["project_dir"])
    holdout_summary = load_holdout_summary(run_spec.get("holdout"))
    status = {
        "run_name": run_spec["run_name"],
        "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)) if metrics_path else None,
        "holdout_found": bool(holdout_summary),
    }
    if dry_run:
        print(json.dumps(status, ensure_ascii=False))
        return
    if metrics_path is None:
        print("Skip missing metrics:", run_spec["run_name"])
        return

    config = {
        "source_project_dir": run_spec["project_dir"],
        "model": run_spec.get("model"),
        "hist_len": run_spec.get("hist_len"),
        "group": run_spec.get("group"),
        "source": "uploaded_from_local_csv",
        "upload_log_mode": log_mode,
    }
    run = wandb.init(
        project=wandb_project,
        name=run_spec["run_name"],
        group=run_spec.get("group"),
        config=config,
        reinit=True,
    )
    if log_mode == "raw":
        log_metrics_csv(metrics_path)
    elif log_mode == "epoch":
        log_epoch_metrics_csv(metrics_path)
    else:
        log_epoch_metrics_csv(metrics_path)
        log_metrics_csv(metrics_path)
    for key, value in holdout_summary.items():
        if value is not None:
            run.summary[key] = value
    run.summary["source_metrics_csv"] = str(metrics_path.relative_to(PROJECT_ROOT))
    if run_spec.get("holdout"):
        run.summary["source_holdout_summary"] = run_spec["holdout"]
    run.finish()
    print("Uploaded:", run_spec["run_name"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", default="top150_rolling6h_model_compare")
    parser.add_argument("--include", choices=["main", "legacy", "all"], default="main")
    parser.add_argument("--run_suffix", default="")
    parser.add_argument("--log_mode", choices=["epoch", "raw", "both"], default="epoch")
    parser.add_argument("--only", default=None, help="Upload only runs whose run_name contains this text.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    runs = list(MAIN_RUNS)
    if args.include in {"legacy", "all"}:
        runs.extend(LEGACY_RUNS)
    if args.only:
        runs = [run_spec for run_spec in runs if args.only in run_spec["run_name"]]

    for run_spec in runs:
        if args.run_suffix:
            run_spec = dict(run_spec)
            run_spec["run_name"] = run_spec["run_name"] + args.run_suffix
        upload_run(run_spec, args.wandb_project, dry_run=args.dry_run, log_mode=args.log_mode)


if __name__ == "__main__":
    main()
