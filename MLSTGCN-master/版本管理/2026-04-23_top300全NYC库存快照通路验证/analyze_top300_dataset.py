import argparse
import json
import os

import numpy as np
import pandas as pd


def load_split(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return {
        "y": data["y"],
        "sample_dates": data["sample_dates"].astype(str),
        "target_cols": data["target_cols"].astype(str),
        "history_feature_cols": data["history_feature_cols"].astype(str),
        "known_future_feature_cols": data["known_future_feature_cols"].astype(str),
    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_station_name_col(df):
    for candidate in ["station_name", "站点名称"]:
        if candidate in df.columns:
            return candidate
    raise KeyError("Cannot resolve station name column from %s" % list(df.columns))


def build_split_coverage_rows(split_payloads):
    rows = []
    for split_name, payload in split_payloads.items():
        sample_dates = pd.to_datetime(pd.Series(payload["sample_dates"]))
        rows.append(
            {
                "split": split_name,
                "sample_count": int(len(sample_dates)),
                "date_start": sample_dates.min().strftime("%Y-%m-%d"),
                "date_end": sample_dates.max().strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows)


def build_month_coverage(sample_dates):
    month_series = pd.to_datetime(pd.Series(sample_dates)).dt.strftime("%Y-%m")
    return month_series.value_counts().sort_index().rename_axis("month").reset_index(name="sample_days")


def build_station_flow_distribution(y_values, mapping_df, output_dir):
    station_name_col = resolve_station_name_col(mapping_df)
    outflow = y_values[..., 0]
    inflow = y_values[..., 1]
    total_flow = outflow + inflow
    num_samples = y_values.shape[0]

    station_total_out = outflow.sum(axis=(0, 1))
    station_total_in = inflow.sum(axis=(0, 1))
    station_total = total_flow.sum(axis=(0, 1))
    station_avg_daily_total = station_total / float(num_samples)
    station_nonzero_ratio = (total_flow > 0).mean(axis=(0, 1))

    station_df = mapping_df.copy()
    station_df["total_outflow"] = station_total_out.astype(np.float64)
    station_df["total_inflow"] = station_total_in.astype(np.float64)
    station_df["total_flow"] = station_total.astype(np.float64)
    station_df["avg_daily_total_flow"] = station_avg_daily_total.astype(np.float64)
    station_df["nonzero_hour_ratio"] = station_nonzero_ratio.astype(np.float64)
    station_df = station_df.sort_values(["avg_daily_total_flow", station_name_col], ascending=[False, True]).reset_index(drop=True)
    station_df.to_csv(os.path.join(output_dir, "station_flow_rank.csv"), index=False, encoding="utf-8-sig")

    percentiles = np.percentile(station_avg_daily_total, [0, 25, 50, 75, 90, 95, 99, 100])
    distribution_summary = {
        "avg_daily_total_flow_min": float(percentiles[0]),
        "avg_daily_total_flow_p25": float(percentiles[1]),
        "avg_daily_total_flow_median": float(percentiles[2]),
        "avg_daily_total_flow_p75": float(percentiles[3]),
        "avg_daily_total_flow_p90": float(percentiles[4]),
        "avg_daily_total_flow_p95": float(percentiles[5]),
        "avg_daily_total_flow_p99": float(percentiles[6]),
        "avg_daily_total_flow_max": float(percentiles[7]),
        "avg_nonzero_hour_ratio": float(station_nonzero_ratio.mean()),
    }
    top10 = station_df[[station_name_col, "avg_daily_total_flow", "nonzero_hour_ratio"]].head(10)
    bottom10 = station_df[[station_name_col, "avg_daily_total_flow", "nonzero_hour_ratio"]].tail(10)
    return distribution_summary, top10, bottom10


def build_hourly_profile(y_values, output_dir):
    outflow = y_values[..., 0]
    inflow = y_values[..., 1]
    total_flow = outflow + inflow

    mean_out_by_hour = outflow.sum(axis=2).mean(axis=0)
    mean_in_by_hour = inflow.sum(axis=2).mean(axis=0)
    mean_total_by_hour = total_flow.sum(axis=2).mean(axis=0)

    hourly_df = pd.DataFrame(
        {
            "hour": np.arange(24),
            "mean_system_outflow": mean_out_by_hour.astype(np.float64),
            "mean_system_inflow": mean_in_by_hour.astype(np.float64),
            "mean_system_total_flow": mean_total_by_hour.astype(np.float64),
        }
    )
    hourly_df["mean_station_total_flow"] = hourly_df["mean_system_total_flow"] / float(y_values.shape[2])
    hourly_df.to_csv(os.path.join(output_dir, "hourly_profile.csv"), index=False, encoding="utf-8-sig")

    morning_window = hourly_df[hourly_df["hour"].between(7, 10)]
    evening_window = hourly_df[hourly_df["hour"].between(17, 20)]
    morning_peak_row = morning_window.sort_values("mean_system_total_flow", ascending=False).iloc[0]
    evening_peak_row = evening_window.sort_values("mean_system_total_flow", ascending=False).iloc[0]
    overall_peak_row = hourly_df.sort_values("mean_system_total_flow", ascending=False).iloc[0]
    overall_trough_row = hourly_df.sort_values("mean_system_total_flow", ascending=True).iloc[0]

    summary = {
        "overall_peak_hour": int(overall_peak_row["hour"]),
        "overall_peak_mean_system_total_flow": float(overall_peak_row["mean_system_total_flow"]),
        "overall_trough_hour": int(overall_trough_row["hour"]),
        "overall_trough_mean_system_total_flow": float(overall_trough_row["mean_system_total_flow"]),
        "morning_peak_hour_7_10": int(morning_peak_row["hour"]),
        "morning_peak_mean_system_total_flow": float(morning_peak_row["mean_system_total_flow"]),
        "evening_peak_hour_17_20": int(evening_peak_row["hour"]),
        "evening_peak_mean_system_total_flow": float(evening_peak_row["mean_system_total_flow"]),
        "peak_to_trough_ratio": float(
            overall_peak_row["mean_system_total_flow"] / max(overall_trough_row["mean_system_total_flow"], 1e-6)
        ),
    }
    return summary, hourly_df


def build_sample_coverage_summary(split_coverage_df, sample_dates, y_values, snapshot_daily_df, output_dir):
    sample_dates_series = pd.to_datetime(pd.Series(sample_dates))
    sample_date_start = sample_dates_series.min()
    sample_date_end = sample_dates_series.max()

    snapshot_dates = pd.to_datetime(snapshot_daily_df["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique()
    sample_date_text = sample_dates_series.dt.strftime("%Y-%m-%d")
    snapshot_overlap_days = int(np.isin(sample_date_text, snapshot_dates).sum())

    coverage_summary = {
        "sample_date_start": sample_date_start.strftime("%Y-%m-%d"),
        "sample_date_end": sample_date_end.strftime("%Y-%m-%d"),
        "sample_days": int(len(sample_dates_series)),
        "station_count": int(y_values.shape[2]),
        "pred_hours": int(y_values.shape[1]),
        "target_dim": int(y_values.shape[3]),
        "nonzero_outflow_ratio": float((y_values[..., 0] > 0).mean()),
        "nonzero_inflow_ratio": float((y_values[..., 1] > 0).mean()),
        "snapshot_feature_date_start": str(snapshot_daily_df["date"].min()),
        "snapshot_feature_date_end": str(snapshot_daily_df["date"].max()),
        "snapshot_feature_overlap_sample_days": snapshot_overlap_days,
        "snapshot_feature_overlap_ratio": float(snapshot_overlap_days / max(len(sample_dates_series), 1)),
    }
    split_coverage_df.to_csv(os.path.join(output_dir, "split_coverage.csv"), index=False, encoding="utf-8-sig")
    build_month_coverage(sample_dates).to_csv(
        os.path.join(output_dir, "sample_month_coverage.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return coverage_summary


def write_report(output_dir, report_text, summary):
    report_path = os.path.join(output_dir, "top300_dataset_check_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    summary_path = os.path.join(output_dir, "top300_dataset_check_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return report_path, summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="bike_hourly_safe_inventory_top300_nyc_full")
    parser.add_argument("--asset_dir", default=os.path.join("二月份数据处理", "nyc_top300_inventory_validation"))
    parser.add_argument("--output_dir", default=os.path.join("分析结果", "2026-04-23_top300数据检查报告"))
    args = parser.parse_args()

    dataset_dir = os.path.join("data", "temporal_data", args.dataset_name)
    graph_dir = os.path.join("data", "graph", args.dataset_name)
    asset_dir = args.asset_dir
    output_dir = args.output_dir
    ensure_dir(output_dir)

    train_payload = load_split(os.path.join(dataset_dir, "train.npz"))
    val_payload = load_split(os.path.join(dataset_dir, "val.npz"))
    test_payload = load_split(os.path.join(dataset_dir, "test.npz"))
    split_payloads = {"train": train_payload, "val": val_payload, "test": test_payload}

    y_values = np.concatenate([train_payload["y"], val_payload["y"], test_payload["y"]], axis=0)
    sample_dates = np.concatenate(
        [train_payload["sample_dates"], val_payload["sample_dates"], test_payload["sample_dates"]],
        axis=0,
    )

    mapping_df = pd.read_csv(os.path.join(graph_dir, "selected_node_mapping.csv"))
    snapshot_daily_df = pd.read_csv(os.path.join(asset_dir, "snapshot_daily_features_topk.csv"))
    split_coverage_df = build_split_coverage_rows(split_payloads)
    distribution_summary, top10_df, bottom10_df = build_station_flow_distribution(y_values, mapping_df, output_dir)
    peak_summary, hourly_df = build_hourly_profile(y_values, output_dir)
    coverage_summary = build_sample_coverage_summary(split_coverage_df, sample_dates, y_values, snapshot_daily_df, output_dir)

    top10_df.to_csv(os.path.join(output_dir, "station_top10_avg_daily_flow.csv"), index=False, encoding="utf-8-sig")
    bottom10_df.to_csv(os.path.join(output_dir, "station_bottom10_avg_daily_flow.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "dataset_name": args.dataset_name,
        "history_feature_cols": train_payload["history_feature_cols"].tolist(),
        "known_future_feature_cols": train_payload["known_future_feature_cols"].tolist(),
        "target_cols": train_payload["target_cols"].tolist(),
    }
    summary.update(distribution_summary)
    summary.update(peak_summary)
    summary.update(coverage_summary)

    report_text = f"""# Top300 数据集检查报告

## 1. 数据集概况

- 数据集名称：`{args.dataset_name}`
- 站点数：`{coverage_summary['station_count']}`
- 样本天数：`{coverage_summary['sample_days']}`
- 预测窗口：`{coverage_summary['pred_hours']}` 小时
- 目标维度：`{coverage_summary['target_dim']}`（小时骑出量、小时骑入量）
- 样本日期范围：`{coverage_summary['sample_date_start']}` 至 `{coverage_summary['sample_date_end']}`

## 2. 站点流量分布

- 站点平均日总流量最小值：`{distribution_summary['avg_daily_total_flow_min']:.2f}`
- 站点平均日总流量中位数：`{distribution_summary['avg_daily_total_flow_median']:.2f}`
- 站点平均日总流量 P90：`{distribution_summary['avg_daily_total_flow_p90']:.2f}`
- 站点平均日总流量 P95：`{distribution_summary['avg_daily_total_flow_p95']:.2f}`
- 站点平均日总流量最大值：`{distribution_summary['avg_daily_total_flow_max']:.2f}`
- 平均非零小时占比：`{distribution_summary['avg_nonzero_hour_ratio']:.4f}`

### Top10 站点（按平均日总流量）

{top10_df.to_markdown(index=False)}

### Bottom10 站点（按平均日总流量）

{bottom10_df.to_markdown(index=False)}

## 3. 高峰时段强度

- 全日最强小时：`{peak_summary['overall_peak_hour']:02d}:00`，系统平均总流量 ` {peak_summary['overall_peak_mean_system_total_flow']:.2f} `
- 全日最低小时：`{peak_summary['overall_trough_hour']:02d}:00`，系统平均总流量 ` {peak_summary['overall_trough_mean_system_total_flow']:.2f} `
- 07:00-10:00 窗口最强小时：`{peak_summary['morning_peak_hour_7_10']:02d}:00`，系统平均总流量 ` {peak_summary['morning_peak_mean_system_total_flow']:.2f} `
- 17:00-20:00 窗口最强小时：`{peak_summary['evening_peak_hour_17_20']:02d}:00`，系统平均总流量 ` {peak_summary['evening_peak_mean_system_total_flow']:.2f} `
- 峰谷强度比：`{peak_summary['peak_to_trough_ratio']:.2f}`

> 详细小时轮廓见 `hourly_profile.csv`

## 4. 样本覆盖情况

- 训练集样本：`{int(split_coverage_df.loc[split_coverage_df['split'] == 'train', 'sample_count'].iloc[0])}`
- 验证集样本：`{int(split_coverage_df.loc[split_coverage_df['split'] == 'val', 'sample_count'].iloc[0])}`
- 测试集样本：`{int(split_coverage_df.loc[split_coverage_df['split'] == 'test', 'sample_count'].iloc[0])}`
- 小时骑出非零比例：`{coverage_summary['nonzero_outflow_ratio']:.4f}`
- 小时骑入非零比例：`{coverage_summary['nonzero_inflow_ratio']:.4f}`

### 库存快照特征覆盖

- 库存快照日期范围：`{coverage_summary['snapshot_feature_date_start']}` 至 `{coverage_summary['snapshot_feature_date_end']}`
- 与当前训练样本日期重叠天数：`{coverage_summary['snapshot_feature_overlap_sample_days']}`
- 与当前训练样本日期重叠比例：`{coverage_summary['snapshot_feature_overlap_ratio']:.4f}`

这说明当前 `top300` 全年训练输入里，库存快照类已知未来特征实际上没有进入训练样本；当前保留下来的已知未来特征主要是：

`{", ".join(train_payload["known_future_feature_cols"].tolist())}`

## 5. 说明

- 本报告基于 `train/val/test.npz` 中保存的真实目标张量 `y` 统计，不依赖预测结果。
- 由于样本生成使用 `168` 小时历史窗口，样本日期是“可完整构造输入窗口”的日期集合，而不是原始订单覆盖的所有自然日。
"""

    report_path, summary_path = write_report(output_dir, report_text, summary)

    print("Saved report to:", report_path)
    print("Saved summary to:", summary_path)
    print("Output dir:", os.path.abspath(output_dir))


if __name__ == "__main__":
    main()
