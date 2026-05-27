import argparse
import json
from pathlib import Path

import pandas as pd
import wandb


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_HOLDOUT_DIR = (
    PROJECT_ROOT
    / "分析结果"
    / "2026-05-12_exp10_fast_cached_fusion_hist168_pred6_seed0_feb2026_holdout"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def get_avg_col(df):
    if "mae" in df.columns:
        return "mae"
    if "abs_error_avg" in df.columns:
        return "abs_error_avg"
    raise KeyError("No MAE/abs_error_avg column found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="trumanmcmichael62-huaqiao-university")
    parser.add_argument("--wandb_project", default="top150_rolling6h_model_compare")
    parser.add_argument("--run_id", default="zk9nk0vl")
    parser.add_argument("--run_name", default="exp10_fast_cached_fusion_hist168_pred6_seed0")
    parser.add_argument("--holdout_dir", default=str(DEFAULT_HOLDOUT_DIR))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    holdout_dir = Path(args.holdout_dir)
    summary = load_json(holdout_dir / "feb2026_holdout_summary.json")
    anchor_df = pd.read_csv(holdout_dir / "feb2026_anchor_hour_mae.csv")
    horizon_df = pd.read_csv(holdout_dir / "feb2026_horizon_mae.csv")
    anchor_avg_col = get_avg_col(anchor_df)
    horizon_avg_col = get_avg_col(horizon_df)

    overall = summary.get("overall", {})
    baseline = summary.get("baseline_mae", {})
    payload = {
        "feb_holdout_mae": float(overall["mae"]),
        "feb_holdout_rmse": float(overall["rmse"]),
        "feb_holdout_mae_1_6": float(summary["mae_1_6"]),
        "baseline_last_1h_mae": float(baseline["naive_last_1h_repeat"]),
        "baseline_previous_day_mae": float(baseline["naive_previous_day_same_hours"]),
        "baseline_train_mean_mae": float(baseline["naive_train_mean_by_horizon"]),
        "improve_vs_last_1h_pct": float(summary["model_improve_vs_last_1h_repeat_pct"]),
        "improve_vs_previous_day_pct": float(summary["model_improve_vs_previous_day_same_hours_pct"]),
        "improve_vs_train_mean_pct": float(summary["model_improve_vs_train_mean_pct"]),
        "mainline_rank_note": "current_prediction_mainline_as_of_2026-05-13",
        "source_holdout_summary": str((holdout_dir / "feb2026_holdout_summary.json").relative_to(PROJECT_ROOT)),
    }
    for _, row in anchor_df.iterrows():
        payload[f"feb_holdout_anchor_mae/{int(row['anchor_hour']):02d}"] = float(row[anchor_avg_col])
    for _, row in horizon_df.iterrows():
        payload[f"feb_holdout_horizon_mae/h{int(row['horizon'])}"] = float(row[horizon_avg_col])

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    api = wandb.Api()
    run = api.run(f"{args.entity}/{args.wandb_project}/{args.run_id}")
    run.name = args.run_name
    tags = set(run.tags or [])
    tags.update({"current-mainline", "feb2026-holdout", "exp10-fast"})
    run.tags = sorted(tags)
    for key, value in payload.items():
        run.summary[key] = value
    run.config.update(
        {
            "current_prediction_mainline": True,
            "mainline_selected_at": "2026-05-13",
            "mainline_reason": "best Feb 2026 external holdout MAE among current top150 rolling-6h experiments",
        },
        allow_val_change=True,
    )
    run.update()
    print("Updated W&B run:", f"{args.entity}/{args.wandb_project}/{args.run_id}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
