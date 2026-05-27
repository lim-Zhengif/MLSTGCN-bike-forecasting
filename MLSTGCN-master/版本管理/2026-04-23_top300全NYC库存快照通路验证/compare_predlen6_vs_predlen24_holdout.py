import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED24_DIR = PROJECT_ROOT / "分析结果" / "2026-04-24_top150_202602_holdout_eval"
PRED6_DIR = PROJECT_ROOT / "分析结果" / "2026-04-28_top150_predlen6_202602_holdout_eval"
OUT_DIR = PROJECT_ROOT / "分析结果" / "2026-04-28_top150_predlen6_vs_24h_holdout_compare"


def load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path / "feb2026_station_hour_predictions.csv")
    df["model"] = model_name
    df["date"] = pd.to_datetime(df["date"])
    df["true_total"] = df["true_out"] + df["true_in"]
    df["pred_total"] = df["pred_out"] + df["pred_in"]
    df["abs_error_total"] = (df["pred_total"] - df["true_total"]).abs()
    return df


def summarize(df: pd.DataFrame, group_cols):
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            samples=("abs_error_avg", "size"),
            mae_avg=("abs_error_avg", "mean"),
            mae_out=("abs_error_out", "mean"),
            mae_in=("abs_error_in", "mean"),
            mae_total=("abs_error_total", "mean"),
            true_total_mean=("true_total", "mean"),
            pred_total_mean=("pred_total", "mean"),
        )
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred24 = load_predictions(PRED24_DIR, "predlen24_take_h1_h6")
    pred6 = load_predictions(PRED6_DIR, "predlen6_direct_h1_h6")

    pred24_h6 = pred24[pred24["horizon"].between(1, 6)].copy()
    pred6_h6 = pred6[pred6["horizon"].between(1, 6)].copy()
    combined = pd.concat([pred24_h6, pred6_h6], ignore_index=True)

    daily_flow = (
        combined.groupby(["model", "date"], as_index=False)
        .agg(true_total_sum=("true_total", "sum"), pred_total_sum=("pred_total", "sum"))
    )
    suspicious_dates = sorted(
        daily_flow.loc[daily_flow["true_total_sum"].eq(0), "date"].dt.strftime("%Y-%m-%d").unique().tolist()
    )

    overall = summarize(combined, ["model"])
    horizon = summarize(combined, ["model", "horizon"])
    daily = summarize(combined, ["model", "date"])
    station = summarize(combined, ["model", "Node_ID", "station_name"])

    valid = combined[~combined["date"].dt.strftime("%Y-%m-%d").isin(suspicious_dates)].copy()
    valid_overall = summarize(valid, ["model"])
    valid_horizon = summarize(valid, ["model", "horizon"])

    overall.to_csv(OUT_DIR / "overall_h1_h6_compare.csv", index=False, encoding="utf-8-sig")
    horizon.to_csv(OUT_DIR / "horizon_h1_h6_compare.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT_DIR / "daily_h1_h6_compare.csv", index=False, encoding="utf-8-sig")
    station.to_csv(OUT_DIR / "station_h1_h6_compare.csv", index=False, encoding="utf-8-sig")
    valid_overall.to_csv(OUT_DIR / "valid_days_overall_h1_h6_compare.csv", index=False, encoding="utf-8-sig")
    valid_horizon.to_csv(OUT_DIR / "valid_days_horizon_h1_h6_compare.csv", index=False, encoding="utf-8-sig")
    daily_flow.to_csv(OUT_DIR / "daily_flow_check_h1_h6.csv", index=False, encoding="utf-8-sig")

    pivot = overall.pivot_table(index=[], columns="model", values="mae_avg", aggfunc="first")
    valid_pivot = valid_overall.pivot_table(index=[], columns="model", values="mae_avg", aggfunc="first")
    pred24_mae = float(pivot["predlen24_take_h1_h6"].iloc[0])
    pred6_mae = float(pivot["predlen6_direct_h1_h6"].iloc[0])
    valid_pred24_mae = float(valid_pivot["predlen24_take_h1_h6"].iloc[0])
    valid_pred6_mae = float(valid_pivot["predlen6_direct_h1_h6"].iloc[0])

    summary = {
        "comparison_scope": "Feb2026 daily 00:00 anchor, horizons 1-6 only",
        "important_note": "Current sample builder uses one anchor per day at 00:00, so this is not arbitrary hourly rolling prediction.",
        "suspicious_zero_true_flow_dates": suspicious_dates,
        "all_days": {
            "predlen24_h1_h6_mae": pred24_mae,
            "predlen6_h1_h6_mae": pred6_mae,
            "predlen6_minus_predlen24": pred6_mae - pred24_mae,
            "predlen6_relative_change_pct": (pred6_mae - pred24_mae) / pred24_mae * 100.0,
        },
        "valid_days_excluding_zero_true_flow_dates": {
            "predlen24_h1_h6_mae": valid_pred24_mae,
            "predlen6_h1_h6_mae": valid_pred6_mae,
            "predlen6_minus_predlen24": valid_pred6_mae - valid_pred24_mae,
            "predlen6_relative_change_pct": (valid_pred6_mae - valid_pred24_mae) / valid_pred24_mae * 100.0,
        },
    }
    with (OUT_DIR / "predlen6_vs_24h_holdout_compare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
