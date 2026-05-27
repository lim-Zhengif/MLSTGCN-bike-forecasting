from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "\u5206\u6790\u7ed3\u679c"
EVAL_DIR = ANALYSIS_DIR / "2026-04-24_top150_202602_holdout_eval"
OUT_DIR = ANALYSIS_DIR / "2026-04-24_top150_202602_ppt_figures"
GRAPH_DIR = ROOT / "data" / "graph" / "bike_hourly_safe_inventory_top150_nyc_full"
STATION_COL = "\u7ad9\u70b9\u540d\u79f0"


def choose_representative_date(daily_df):
    daily_df = daily_df.copy()
    daily_df["abs_error_avg"] = daily_df["abs_error_avg"].astype(float)
    median_mae = daily_df["abs_error_avg"].median()
    idx = (daily_df["abs_error_avg"] - median_mae).abs().idxmin()
    return str(daily_df.loc[idx, "date"]), float(daily_df.loc[idx, "abs_error_avg"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(EVAL_DIR / "feb2026_station_hour_predictions.csv")
    daily_df = pd.read_csv(EVAL_DIR / "feb2026_daily_mae.csv")
    mapping_df = pd.read_csv(GRAPH_DIR / "selected_node_mapping.csv")

    rep_date, rep_day_mae = choose_representative_date(daily_df)
    top_stations = (
        mapping_df.sort_values("total_flow_count", ascending=False)
        .head(6)[["Node_ID", STATION_COL, "total_flow_count"]]
        .rename(columns={STATION_COL: "station_name"})
    )
    selected_ids = top_stations["Node_ID"].astype(int).tolist()

    plot_df = pred_df[(pred_df["date"] == rep_date) & (pred_df["Node_ID"].isin(selected_ids))].copy()
    plot_df["true_total"] = plot_df["true_out"].astype(float) + plot_df["true_in"].astype(float)
    plot_df["pred_total"] = plot_df["pred_out"].astype(float) + plot_df["pred_in"].astype(float)

    station_mae = (
        plot_df.groupby(["Node_ID", "station_name"], as_index=False)
        .apply(lambda g: pd.Series({"day_mae_total": (g["pred_total"] - g["true_total"]).abs().mean()}))
        .reset_index(drop=True)
    )
    selected_summary = top_stations.merge(station_mae, on=["Node_ID", "station_name"], how="left")
    selected_summary.to_csv(OUT_DIR / "selected_hot_stations_for_prediction_comparison.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    colors = {
        "true": "#111827",
        "pred": "#2563EB",
        "grid": "#E5E7EB",
        "spine": "#CBD5E1",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.4), sharex=True)
    axes = axes.ravel()

    for ax, (_, station_row) in zip(axes, selected_summary.iterrows()):
        node_id = int(station_row["Node_ID"])
        station_name = station_row["station_name"]
        sub = plot_df[plot_df["Node_ID"] == node_id].sort_values("horizon")
        x = sub["horizon"].astype(int)

        ax.plot(x, sub["true_total"], color=colors["true"], linewidth=1.9, marker="o", markersize=2.8, label="Observed")
        ax.plot(x, sub["pred_total"], color=colors["pred"], linewidth=1.8, linestyle="--", marker="s", markersize=2.6, label="Predicted")
        ax.set_title(
            f"Node {node_id}  {station_name}\nMAE={station_row['day_mae_total']:.2f}, flow={int(station_row['total_flow_count']):,}",
            fontsize=9.5,
            pad=8,
        )
        ax.set_xlim(1, 24)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.grid(True, color=colors["grid"], linewidth=0.8)
        ax.tick_params(labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color(colors["spine"])

    axes[0].legend(loc="upper left", fontsize=8.5, frameon=True)
    for ax in axes[3:]:
        ax.set_xlabel("Horizon / hour", fontsize=9)
    for ax in [axes[0], axes[3]]:
        ax.set_ylabel("Total flow", fontsize=9)

    fig.suptitle(
        f"Top150 Hot Stations: Observed vs Predicted Flow on {rep_date}  |  Representative day MAE={rep_day_mae:.2f}",
        fontsize=13.5,
        weight="bold",
        y=0.975,
    )
    fig.tight_layout(rect=[0.035, 0.03, 0.985, 0.92])

    png_path = OUT_DIR / "fig_top150_hot_stations_pred_vs_true_202602_representative.png"
    svg_path = OUT_DIR / "fig_top150_hot_stations_pred_vs_true_202602_representative.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    # A second stress-case figure for the worst February day can be useful as backup slide material.
    worst_row = daily_df.sort_values("abs_error_avg", ascending=False).iloc[0]
    worst_date = str(worst_row["date"])
    worst_mae = float(worst_row["abs_error_avg"])
    worst_df = pred_df[(pred_df["date"] == worst_date) & (pred_df["Node_ID"].isin(selected_ids))].copy()
    worst_df["true_total"] = worst_df["true_out"].astype(float) + worst_df["true_in"].astype(float)
    worst_df["pred_total"] = worst_df["pred_out"].astype(float) + worst_df["pred_in"].astype(float)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.4), sharex=True)
    axes = axes.ravel()
    for ax, (_, station_row) in zip(axes, selected_summary.iterrows()):
        node_id = int(station_row["Node_ID"])
        station_name = station_row["station_name"]
        sub = worst_df[worst_df["Node_ID"] == node_id].sort_values("horizon")
        day_mae = (sub["pred_total"] - sub["true_total"]).abs().mean()
        ax.plot(sub["horizon"].astype(int), sub["true_total"], color=colors["true"], linewidth=1.9, marker="o", markersize=2.8, label="Observed")
        ax.plot(sub["horizon"].astype(int), sub["pred_total"], color=colors["pred"], linewidth=1.8, linestyle="--", marker="s", markersize=2.6, label="Predicted")
        ax.set_title(f"Node {node_id}  {station_name}\nMAE={day_mae:.2f}", fontsize=9.5, pad=8)
        ax.set_xlim(1, 24)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.grid(True, color=colors["grid"], linewidth=0.8)
        ax.tick_params(labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color(colors["spine"])
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=True)
    for ax in axes[3:]:
        ax.set_xlabel("Horizon / hour", fontsize=9)
    for ax in [axes[0], axes[3]]:
        ax.set_ylabel("Total flow", fontsize=9)
    fig.suptitle(
        f"Top150 Hot Stations: Observed vs Predicted Flow on {worst_date}  |  Highest-error day MAE={worst_mae:.2f}",
        fontsize=13.5,
        weight="bold",
        y=0.975,
    )
    fig.tight_layout(rect=[0.035, 0.03, 0.985, 0.92])
    fig.savefig(OUT_DIR / "fig_top150_hot_stations_pred_vs_true_202602_worst_day.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_top150_hot_stations_pred_vs_true_202602_worst_day.svg", bbox_inches="tight")
    plt.close(fig)

    print("Representative date:", rep_date, "daily MAE:", rep_day_mae)
    print("Worst date:", worst_date, "daily MAE:", worst_mae)
    print("Selected stations:")
    print(selected_summary.to_string(index=False))
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
