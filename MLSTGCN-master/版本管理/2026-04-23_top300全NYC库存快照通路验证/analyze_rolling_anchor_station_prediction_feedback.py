from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = PROJECT_ROOT / "分析结果" / "2026-04-28_top150_predlen6_rolling_anchor_holdout_eval"
GRAPH_DIR = PROJECT_ROOT / "data" / "graph" / "bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20"
OUT_DIR = PROJECT_ROOT / "分析结果" / "2026-04-29_top150_rolling_anchor_station_prediction_feedback"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED_DIR / "feb2026_station_hour_predictions.csv")
    mapping = pd.read_csv(GRAPH_DIR / "selected_node_mapping.csv")

    pred["true_total"] = pred["true_out"] + pred["true_in"]
    pred["pred_total"] = pred["pred_out"] + pred["pred_in"]
    pred["abs_error_total"] = (pred["pred_total"] - pred["true_total"]).abs()

    station_summary = (
        pred.groupby(["Node_ID", "station_name"], as_index=False)
        .agg(
            samples=("abs_error_avg", "size"),
            mae_avg=("abs_error_avg", "mean"),
            mae_out=("abs_error_out", "mean"),
            mae_in=("abs_error_in", "mean"),
            mae_total=("abs_error_total", "mean"),
            true_out_mean=("true_out", "mean"),
            pred_out_mean=("pred_out", "mean"),
            true_in_mean=("true_in", "mean"),
            pred_in_mean=("pred_in", "mean"),
            true_total_mean=("true_total", "mean"),
            pred_total_mean=("pred_total", "mean"),
            true_total_sum=("true_total", "sum"),
            pred_total_sum=("pred_total", "sum"),
        )
    )
    station_summary["bias_out_mean"] = station_summary["pred_out_mean"] - station_summary["true_out_mean"]
    station_summary["bias_in_mean"] = station_summary["pred_in_mean"] - station_summary["true_in_mean"]
    station_summary["bias_total_mean"] = station_summary["pred_total_mean"] - station_summary["true_total_mean"]

    mapping_cols = ["Node_ID", "total_flow_count", "capacity", "lat", "lon"]
    station_summary = station_summary.merge(mapping[mapping_cols], on="Node_ID", how="left")
    station_summary = station_summary.sort_values("mae_avg", ascending=False)

    anchor_summary = (
        pred.groupby(["anchor_hour", "Node_ID", "station_name"], as_index=False)
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
    anchor_summary = anchor_summary.merge(
        mapping[["Node_ID", "total_flow_count", "capacity"]],
        on="Node_ID",
        how="left",
    )

    station_summary.to_csv(OUT_DIR / "station_prediction_vs_true_summary.csv", index=False, encoding="utf-8-sig")
    station_summary.head(15).to_csv(OUT_DIR / "worst15_station_prediction_vs_true.csv", index=False, encoding="utf-8-sig")
    station_summary.tail(15).sort_values("mae_avg").to_csv(
        OUT_DIR / "best15_station_prediction_vs_true.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_summary.sort_values("total_flow_count", ascending=False).head(20).to_csv(
        OUT_DIR / "top20_hot_station_prediction_vs_true.csv",
        index=False,
        encoding="utf-8-sig",
    )
    anchor_summary.to_csv(OUT_DIR / "station_anchor_prediction_vs_true_summary.csv", index=False, encoding="utf-8-sig")

    print("Output dir:", OUT_DIR)
    print("\nTOP10_HOT")
    print(
        station_summary.sort_values("total_flow_count", ascending=False)
        .head(10)[
            [
                "Node_ID",
                "station_name",
                "total_flow_count",
                "mae_avg",
                "mae_out",
                "mae_in",
                "true_total_mean",
                "pred_total_mean",
                "bias_total_mean",
            ]
        ]
        .to_string(index=False)
    )
    print("\nWORST10")
    print(
        station_summary.head(10)[
            [
                "Node_ID",
                "station_name",
                "total_flow_count",
                "mae_avg",
                "mae_out",
                "mae_in",
                "true_total_mean",
                "pred_total_mean",
                "bias_total_mean",
            ]
        ]
        .to_string(index=False)
    )
    print("\nBEST10")
    print(
        station_summary.tail(10)
        .sort_values("mae_avg")[
            [
                "Node_ID",
                "station_name",
                "total_flow_count",
                "mae_avg",
                "mae_out",
                "mae_in",
                "true_total_mean",
                "pred_total_mean",
                "bias_total_mean",
            ]
        ]
        .to_string(index=False)
    )
    print("\nANCHOR_MEAN")
    print(pred.groupby("anchor_hour")[["abs_error_avg", "true_total", "pred_total"]].mean().to_string())


if __name__ == "__main__":
    main()
