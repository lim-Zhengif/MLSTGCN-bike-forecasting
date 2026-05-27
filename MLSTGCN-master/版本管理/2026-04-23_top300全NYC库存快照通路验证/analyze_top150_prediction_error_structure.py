import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = PROJECT_ROOT / "分析结果" / "2026-04-24_top150_202602_holdout_eval"
GRAPH_DIR = PROJECT_ROOT / "data" / "graph" / "bike_hourly_safe_inventory_top150_nyc_full"
OUT_DIR = PROJECT_ROOT / "分析结果" / "2026-04-27_top150_预测误差结构分析"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = PRED_DIR / "feb2026_station_hour_predictions.csv"
    mapping_path = GRAPH_DIR / "selected_node_mapping.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)

    pred = pd.read_csv(pred_path)
    mapping = pd.read_csv(mapping_path)
    pred["date"] = pd.to_datetime(pred["date"])
    pred["hour_int"] = pred["hour"].str.slice(0, 2).astype(int)
    pred["weekday"] = pred["date"].dt.dayofweek
    pred["is_weekend"] = pred["weekday"] >= 5
    pred["true_total"] = pred["true_out"] + pred["true_in"]
    pred["pred_total"] = pred["pred_out"] + pred["pred_in"]
    pred["abs_error_total"] = (pred["pred_total"] - pred["true_total"]).abs()

    mapping_cols = ["Node_ID", "total_flow_count", "capacity", "lat", "lon"]
    mapping = mapping[mapping_cols].copy()
    mapping = mapping.sort_values("total_flow_count", ascending=False).reset_index(drop=True)
    mapping["activity_rank"] = np.arange(1, len(mapping) + 1)

    def tier(rank: int) -> str:
        if rank <= 50:
            return "top_1_50_hot"
        if rank <= 100:
            return "top_51_100_mid"
        return "top_101_150_lower"

    mapping["activity_tier"] = mapping["activity_rank"].apply(tier)
    pred = pred.merge(mapping, on="Node_ID", how="left")
    return pred, mapping


def add_period_columns(df: pd.DataFrame) -> pd.DataFrame:
    def peak_window(hour: int) -> str:
        if 7 <= hour <= 10:
            return "morning_peak_07_10"
        if 17 <= hour <= 20:
            return "evening_peak_17_20"
        if 0 <= hour <= 5:
            return "late_night_00_05"
        if 11 <= hour <= 16:
            return "daytime_11_16"
        return "other_06_21_23"

    def horizon_band(h: int) -> str:
        if 1 <= h <= 3:
            return "short_1_3h"
        if 4 <= h <= 6:
            return "short_4_6h"
        if 7 <= h <= 12:
            return "mid_7_12h"
        if 13 <= h <= 18:
            return "long_13_18h"
        return "long_19_24h"

    df = df.copy()
    df["peak_window"] = df["hour_int"].apply(peak_window)
    df["horizon_band"] = df["horizon"].astype(int).apply(horizon_band)
    df["day_type"] = np.where(df["is_weekend"], "weekend", "weekday")
    return df


def mae_summary(grouped: pd.core.groupby.generic.DataFrameGroupBy) -> pd.DataFrame:
    return grouped.agg(
        samples=("abs_error_avg", "size"),
        mae_avg=("abs_error_avg", "mean"),
        mae_out=("abs_error_out", "mean"),
        mae_in=("abs_error_in", "mean"),
        mae_total=("abs_error_total", "mean"),
        true_total_mean=("true_total", "mean"),
        pred_total_mean=("pred_total", "mean"),
    )


def write_tables(df: pd.DataFrame, mapping: pd.DataFrame) -> dict:
    ensure_dir(OUT_DIR)
    tables = {}

    table_specs = {
        "horizon_error.csv": mae_summary(df.groupby("horizon", as_index=False)),
        "horizon_band_error.csv": mae_summary(df.groupby("horizon_band", as_index=False)),
        "peak_window_error.csv": mae_summary(df.groupby("peak_window", as_index=False)),
        "day_type_error.csv": mae_summary(df.groupby("day_type", as_index=False)),
        "weekday_error.csv": mae_summary(df.groupby("weekday", as_index=False)),
        "activity_tier_error.csv": mae_summary(df.groupby("activity_tier", as_index=False)),
        "activity_tier_by_peak_window_error.csv": mae_summary(
            df.groupby(["activity_tier", "peak_window"], as_index=False)
        ),
        "horizon_band_by_peak_window_error.csv": mae_summary(
            df.groupby(["horizon_band", "peak_window"], as_index=False)
        ),
    }

    daily = mae_summary(df.groupby("date", as_index=False)).sort_values("mae_avg", ascending=False)
    daily_coverage = df.groupby("date", as_index=False).agg(
        samples=("true_total", "size"),
        true_total_sum=("true_total", "sum"),
        pred_total_sum=("pred_total", "sum"),
        true_nonzero_rows=("true_total", lambda s: int((s > 0).sum())),
        pred_nonzero_rows=("pred_total", lambda s: int((s > 0).sum())),
        true_total_mean=("true_total", "mean"),
        pred_total_mean=("pred_total", "mean"),
    )
    daily_coverage["true_nonzero_ratio"] = daily_coverage["true_nonzero_rows"] / daily_coverage["samples"]
    median_true_sum = daily_coverage["true_total_sum"].median()
    low_flow_threshold = median_true_sum * 0.2
    daily_coverage["coverage_flag"] = np.select(
        [
            daily_coverage["true_total_sum"].eq(0),
            daily_coverage["true_total_sum"].lt(low_flow_threshold),
        ],
        ["zero_true_flow", "suspicious_low_true_flow"],
        default="ok",
    )
    daily_coverage = daily_coverage.sort_values("true_total_sum")

    station = mae_summary(df.groupby(["Node_ID", "station_name"], as_index=False))
    station = station.merge(
        mapping[["Node_ID", "activity_rank", "activity_tier", "total_flow_count", "capacity"]],
        on="Node_ID",
        how="left",
    ).sort_values("mae_avg", ascending=False)
    station_day = mae_summary(df.groupby(["date", "Node_ID", "station_name"], as_index=False))
    station_day = station_day.merge(
        mapping[["Node_ID", "activity_rank", "activity_tier", "total_flow_count"]],
        on="Node_ID",
        how="left",
    ).sort_values("mae_avg", ascending=False)

    table_specs["daily_error_rank.csv"] = daily
    table_specs["daily_flow_coverage.csv"] = daily_coverage
    table_specs["potential_data_gap_dates.csv"] = daily_coverage.loc[
        daily_coverage["coverage_flag"].ne("ok")
    ].copy()
    ok_dates = set(daily_coverage.loc[daily_coverage["coverage_flag"].eq("ok"), "date"])
    valid_df = df.loc[df["date"].isin(ok_dates)].copy()
    table_specs["valid_days_horizon_error.csv"] = mae_summary(
        valid_df.groupby("horizon", as_index=False)
    )
    table_specs["valid_days_peak_window_error.csv"] = mae_summary(
        valid_df.groupby("peak_window", as_index=False)
    )
    table_specs["valid_days_activity_tier_error.csv"] = mae_summary(
        valid_df.groupby("activity_tier", as_index=False)
    )
    table_specs["station_error_rank.csv"] = station
    table_specs["station_day_error_rank.csv"] = station_day

    for name, table in table_specs.items():
        path = OUT_DIR / name
        table.to_csv(path, index=False, encoding="utf-8-sig")
        tables[name] = str(path)

    return tables


def plot_horizon(horizon: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180)
    ax.plot(horizon["horizon"], horizon["mae_avg"], marker="o", lw=2.2, label="avg of in/out")
    ax.plot(horizon["horizon"], horizon["mae_out"], marker="s", lw=1.5, alpha=0.75, label="outflow")
    ax.plot(horizon["horizon"], horizon["mae_in"], marker="^", lw=1.5, alpha=0.75, label="inflow")
    ax.axvspan(1, 6, color="#dbeafe", alpha=0.45, label="rolling dispatch focus")
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("MAE")
    ax.set_title("Top150 Feb2026 Holdout: Horizon-wise Error")
    ax.set_xticks(range(1, 25))
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_horizon_mae_curve.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_horizon_mae_curve.svg", bbox_inches="tight")
    plt.close(fig)


def plot_bars(table: pd.DataFrame, category: str, value: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=180)
    table = table.sort_values(value, ascending=False)
    colors = plt.cm.Blues(np.linspace(0.45, 0.82, len(table)))
    ax.bar(table[category], table[value], color=colors)
    ax.set_ylabel("MAE")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.28)
    ax.tick_params(axis="x", rotation=25)
    for idx, val in enumerate(table[value]):
        ax.text(idx, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{filename}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{filename}.svg", bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame, tables: dict) -> None:
    horizon = pd.read_csv(OUT_DIR / "horizon_error.csv")
    peak = pd.read_csv(OUT_DIR / "peak_window_error.csv")
    tier = pd.read_csv(OUT_DIR / "activity_tier_error.csv")
    daily = pd.read_csv(OUT_DIR / "daily_error_rank.csv")
    daily_coverage = pd.read_csv(OUT_DIR / "daily_flow_coverage.csv")
    station = pd.read_csv(OUT_DIR / "station_error_rank.csv")

    overall = {
        "samples": int(len(df)),
        "overall_mae_avg": float(df["abs_error_avg"].mean()),
        "overall_mae_out": float(df["abs_error_out"].mean()),
        "overall_mae_in": float(df["abs_error_in"].mean()),
        "overall_mae_total": float(df["abs_error_total"].mean()),
        "short_1_6_mae": float(df.loc[df["horizon"].between(1, 6), "abs_error_avg"].mean()),
        "mid_long_7_24_mae": float(df.loc[df["horizon"].between(7, 24), "abs_error_avg"].mean()),
        "morning_peak_07_10_mae": float(
            df.loc[df["peak_window"].eq("morning_peak_07_10"), "abs_error_avg"].mean()
        ),
        "evening_peak_17_20_mae": float(
            df.loc[df["peak_window"].eq("evening_peak_17_20"), "abs_error_avg"].mean()
        ),
    }
    data_gap_dates = daily_coverage.loc[daily_coverage["coverage_flag"].ne("ok"), "date"].astype(str).tolist()
    valid_df = df.loc[~df["date"].dt.strftime("%Y-%m-%d").isin(data_gap_dates)].copy()
    valid_overall = {
        "excluded_dates": data_gap_dates,
        "samples": int(len(valid_df)),
        "overall_mae_avg": float(valid_df["abs_error_avg"].mean()),
        "overall_mae_out": float(valid_df["abs_error_out"].mean()),
        "overall_mae_in": float(valid_df["abs_error_in"].mean()),
        "overall_mae_total": float(valid_df["abs_error_total"].mean()),
        "short_1_6_mae": float(valid_df.loc[valid_df["horizon"].between(1, 6), "abs_error_avg"].mean()),
        "mid_long_7_24_mae": float(valid_df.loc[valid_df["horizon"].between(7, 24), "abs_error_avg"].mean()),
        "morning_peak_07_10_mae": float(
            valid_df.loc[valid_df["peak_window"].eq("morning_peak_07_10"), "abs_error_avg"].mean()
        ),
        "evening_peak_17_20_mae": float(
            valid_df.loc[valid_df["peak_window"].eq("evening_peak_17_20"), "abs_error_avg"].mean()
        ),
    }

    summary = {
        "input_predictions": str(PRED_DIR / "feb2026_station_hour_predictions.csv"),
        "output_dir": str(OUT_DIR),
        "overall": overall,
        "valid_days_overall_excluding_potential_data_gaps": valid_overall,
        "worst_dates_top5": daily.head(5).to_dict(orient="records"),
        "best_dates_top5": daily.tail(5).sort_values("mae_avg").to_dict(orient="records"),
        "worst_stations_top10": station.head(10).to_dict(orient="records"),
        "horizon_mae": horizon.to_dict(orient="records"),
        "peak_window_mae": peak.to_dict(orient="records"),
        "activity_tier_mae": tier.to_dict(orient="records"),
        "potential_data_gap_dates": daily_coverage.loc[
            daily_coverage["coverage_flag"].ne("ok")
        ].to_dict(orient="records"),
        "tables": tables,
    }
    with (OUT_DIR / "prediction_error_structure_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# Top150 Feb2026 Prediction Error Structure",
        "",
        "## Overall",
        f"- Samples: {overall['samples']}",
        f"- Overall MAE(avg in/out): {overall['overall_mae_avg']:.4f}",
        f"- Outflow MAE: {overall['overall_mae_out']:.4f}",
        f"- Inflow MAE: {overall['overall_mae_in']:.4f}",
        f"- Short horizon t+1 to t+6 MAE: {overall['short_1_6_mae']:.4f}",
        f"- Mid/long horizon t+7 to t+24 MAE: {overall['mid_long_7_24_mae']:.4f}",
        f"- Morning peak 07-10 MAE: {overall['morning_peak_07_10_mae']:.4f}",
        f"- Evening peak 17-20 MAE: {overall['evening_peak_17_20_mae']:.4f}",
        "",
        "## Coverage Check",
        f"- Potential data gap dates: {int(daily_coverage['coverage_flag'].ne('ok').sum())}",
        f"- Valid-days overall MAE after excluding potential gaps: {valid_overall['overall_mae_avg']:.4f}",
        f"- Valid-days short horizon t+1 to t+6 MAE: {valid_overall['short_1_6_mae']:.4f}",
        f"- Valid-days mid/long horizon t+7 to t+24 MAE: {valid_overall['mid_long_7_24_mae']:.4f}",
        "- See daily_flow_coverage.csv and potential_data_gap_dates.csv before interpreting worst-date errors.",
        "",
        "## Main Experimental Interpretation",
        "- The model is strongest on short rolling horizons, especially t+1 to t+6.",
        "- Errors increase during peak-related horizons and long horizons, supporting a rolling dispatch design.",
        "- Activity-tier tables show whether the model remains reliable on the busiest operational stations.",
        "",
        "## Key Tables",
    ]
    for name in tables:
        lines.append(f"- {name}")
    (OUT_DIR / "prediction_error_structure_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    pred, mapping = read_inputs()
    pred = add_period_columns(pred)
    tables = write_tables(pred, mapping)

    horizon = pd.read_csv(OUT_DIR / "horizon_error.csv")
    peak = pd.read_csv(OUT_DIR / "peak_window_error.csv")
    tier = pd.read_csv(OUT_DIR / "activity_tier_error.csv")
    day_type = pd.read_csv(OUT_DIR / "day_type_error.csv")

    plot_horizon(horizon)
    plot_bars(peak, "peak_window", "mae_avg", "Peak Window Error", "fig_peak_window_mae_bar")
    plot_bars(tier, "activity_tier", "mae_avg", "Station Activity Tier Error", "fig_activity_tier_mae_bar")
    plot_bars(day_type, "day_type", "mae_avg", "Weekday vs Weekend Error", "fig_day_type_mae_bar")
    write_summary(pred, tables)

    print(f"Output dir: {OUT_DIR}")
    print(f"Overall MAE: {pred['abs_error_avg'].mean():.4f}")
    print(f"Short t+1..t+6 MAE: {pred.loc[pred['horizon'].between(1, 6), 'abs_error_avg'].mean():.4f}")
    print(f"Mid/long t+7..t+24 MAE: {pred.loc[pred['horizon'].between(7, 24), 'abs_error_avg'].mean():.4f}")


if __name__ == "__main__":
    main()
