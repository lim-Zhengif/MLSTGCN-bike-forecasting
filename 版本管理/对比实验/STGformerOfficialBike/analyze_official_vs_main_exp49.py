# -*- coding: utf-8 -*-
"""Audit and compare official-core STGformer B0 with the exp49 main model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OFFICIAL_ORACLE = PROJECT_ROOT / "分析结果" / "对比实验" / "STGformerOfficialBike" / (
    "holdout_202603_202606/stgformer_official_b0_202603_202606_station_hour_predictions.csv"
)
DEFAULT_OFFICIAL_LAG1 = PROJECT_ROOT / "分析结果" / "对比实验" / "STGformerOfficialBike" / (
    "holdout_202603_202606_lag1/stgformer_official_b0_202603_202606_lag1_station_hour_predictions.csv"
)
DEFAULT_MAIN_ROOT = PROJECT_ROOT / "第一部分补的实验" / "缺少的核心实验" / "外部评估" / "main_exp49"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "分析结果" / "对比实验" / "STGformerOfficialBike" / (
    "comparison_with_main_exp49"
)

KEY_COLS = [
    "date",
    "sample_datetime",
    "target_start_datetime",
    "anchor_hour",
    "horizon",
    "Node_ID",
]
TRUTH_COLS = ["true_out", "true_in"]
PRED_COLS = ["pred_out", "pred_in"]
USE_COLS = KEY_COLS + ["station_name"] + PRED_COLS + TRUTH_COLS


def absolute_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def atomic_json(payload, path):
    def sanitize(value):
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(value) else float(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_predictions(paths, model, seed, weather_regime):
    frames = [pd.read_csv(path, usecols=USE_COLS) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = frame["date"].astype(str).str[:10]
    frame["month"] = frame["date"].str[:7]
    if frame.duplicated(KEY_COLS).any():
        raise RuntimeError("%s seed%s has duplicate prediction keys" % (model, seed))
    numeric_cols = ["anchor_hour", "horizon", "Node_ID"] + PRED_COLS + TRUTH_COLS
    if not np.isfinite(frame[numeric_cols].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("%s seed%s contains NaN/Inf" % (model, seed))
    frame = frame.sort_values(KEY_COLS).reset_index(drop=True)
    frame["model"] = model
    frame["seed"] = int(seed)
    frame["weather_regime"] = weather_regime
    out_error = frame["pred_out"].to_numpy(dtype=np.float64) - frame["true_out"].to_numpy(dtype=np.float64)
    in_error = frame["pred_in"].to_numpy(dtype=np.float64) - frame["true_in"].to_numpy(dtype=np.float64)
    net_error = (
        frame["pred_in"].to_numpy(dtype=np.float64)
        - frame["pred_out"].to_numpy(dtype=np.float64)
        - frame["true_in"].to_numpy(dtype=np.float64)
        + frame["true_out"].to_numpy(dtype=np.float64)
    )
    frame["error_out"] = out_error
    frame["error_in"] = in_error
    frame["error_net"] = net_error
    frame["abs_error_out"] = np.abs(out_error)
    frame["abs_error_in"] = np.abs(in_error)
    frame["abs_error_net"] = np.abs(net_error)
    frame["abs_error_avg"] = 0.5 * (np.abs(out_error) + np.abs(in_error))
    return frame


def assert_same_truth(reference, candidate, candidate_name):
    if len(reference) != len(candidate):
        raise RuntimeError("%s row count differs: %d vs %d" % (candidate_name, len(candidate), len(reference)))
    if not reference[KEY_COLS].equals(candidate[KEY_COLS]):
        raise RuntimeError("%s prediction keys differ from official oracle" % candidate_name)
    reference_truth = reference[TRUTH_COLS].to_numpy(dtype=np.float64)
    candidate_truth = candidate[TRUTH_COLS].to_numpy(dtype=np.float64)
    if not np.array_equal(reference_truth, candidate_truth):
        max_delta = float(np.max(np.abs(reference_truth - candidate_truth)))
        raise RuntimeError("%s truth differs; max absolute delta=%g" % (candidate_name, max_delta))


def metric_record(group):
    out_error = group["error_out"].to_numpy(dtype=np.float64)
    in_error = group["error_in"].to_numpy(dtype=np.float64)
    net_error = group["error_net"].to_numpy(dtype=np.float64)
    true_out = group["true_out"].to_numpy(dtype=np.float64)
    true_in = group["true_in"].to_numpy(dtype=np.float64)
    directional_error = np.concatenate([out_error, in_error])
    directional_true = np.concatenate([true_out, true_in])
    return {
        "rows": int(len(group)),
        "mae": float(np.mean(np.abs(directional_error))),
        "rmse": float(np.sqrt(np.mean(directional_error ** 2))),
        "mape": float(np.mean(np.abs(directional_error) / np.maximum(np.abs(directional_true), 1.0))),
        "mae_out": float(np.mean(np.abs(out_error))),
        "mae_in": float(np.mean(np.abs(in_error))),
        "mae_net": float(np.mean(np.abs(net_error))),
        "rmse_out": float(np.sqrt(np.mean(out_error ** 2))),
        "rmse_in": float(np.sqrt(np.mean(in_error ** 2))),
        "rmse_net": float(np.sqrt(np.mean(net_error ** 2))),
        "bias_out": float(np.mean(out_error)),
        "bias_in": float(np.mean(in_error)),
        "bias_net": float(np.mean(net_error)),
    }


def grouped_metrics(frame, group_cols):
    identity = {
        "model": str(frame["model"].iloc[0]),
        "seed": int(frame["seed"].iloc[0]),
        "weather_regime": str(frame["weather_regime"].iloc[0]),
    }
    if not group_cols:
        return pd.DataFrame([{**identity, **metric_record(frame)}])
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for values, group in frame.groupby(grouper, sort=True):
        values = (values,) if len(group_cols) == 1 else values
        rows.append({**identity, **dict(zip(group_cols, values)), **metric_record(group)})
    return pd.DataFrame(rows)


def demand_bucket_metrics(frame):
    bins = [-0.5, 0.5, 2.5, 5.5, 10.5, 20.5, np.inf]
    labels = ["0", "1-2", "3-5", "6-10", "11-20", "21+"]
    rows = []
    for flow_type in ["out", "in"]:
        true = frame["true_" + flow_type].to_numpy(dtype=np.float64)
        error = frame["error_" + flow_type].to_numpy(dtype=np.float64)
        buckets = pd.cut(true, bins=bins, labels=labels, include_lowest=True)
        for label in labels:
            mask = np.asarray(buckets == label)
            selected_error = error[mask]
            selected_true = true[mask]
            rows.append({
                "model": str(frame["model"].iloc[0]),
                "seed": int(frame["seed"].iloc[0]),
                "weather_regime": str(frame["weather_regime"].iloc[0]),
                "flow_type": flow_type,
                "demand_bucket": label,
                "rows": int(mask.sum()),
                "mae": float(np.mean(np.abs(selected_error))),
                "rmse": float(np.sqrt(np.mean(selected_error ** 2))),
                "mape": float(np.mean(np.abs(selected_error) / np.maximum(np.abs(selected_true), 1.0))),
                "bias": float(np.mean(selected_error)),
            })
    return pd.DataFrame(rows)


def append_main_demand_mean(frame):
    main = frame[frame["model"] == "main_exp49"]
    rows = []
    for (flow_type, bucket), group in main.groupby(["flow_type", "demand_bucket"], sort=False):
        rows.append({
            "model": "main_exp49_3seed_mean",
            "seed": -1,
            "weather_regime": "oracle_observed_same_day",
            "flow_type": flow_type,
            "demand_bucket": bucket,
            "rows": int(round(group["rows"].mean())),
            "mae": float(group["mae"].mean()),
            "rmse": float(group["rmse"].mean()),
            "mape": float(group["mape"].mean()),
            "bias": float(group["bias"].mean()),
            "mae_seed_std": float(group["mae"].std(ddof=1)),
        })
    output = frame.copy()
    output["mae_seed_std"] = np.nan
    return pd.concat([output, pd.DataFrame(rows)], ignore_index=True)


def append_main_seed_mean(frame):
    main = frame[frame["model"] == "main_exp49"].copy()
    key_cols = [name for name in frame.columns if name not in {
        "model", "seed", "weather_regime", "rows", "mae", "rmse", "mape",
        "mae_out", "mae_in", "mae_net", "rmse_out", "rmse_in", "rmse_net",
        "bias_out", "bias_in", "bias_net",
    }]
    metric_cols = [
        "mae", "rmse", "mape", "mae_out", "mae_in", "mae_net",
        "rmse_out", "rmse_in", "rmse_net", "bias_out", "bias_in", "bias_net",
    ]
    grouped = main.groupby(key_cols, dropna=False, sort=True) if key_cols else [((), main)]
    rows = []
    iterator = grouped if key_cols else grouped
    for keys, group in iterator:
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(key_cols, keys))
        row.update({
            "model": "main_exp49_3seed_mean",
            "seed": -1,
            "weather_regime": "oracle_observed_same_day",
            "rows": int(round(group["rows"].mean())),
        })
        for column in metric_cols:
            row[column] = float(group[column].mean())
        row["mae_seed_std"] = float(group["mae"].std(ddof=1))
        rows.append(row)
    output = frame.copy()
    output["mae_seed_std"] = np.nan
    return pd.concat([output, pd.DataFrame(rows)], ignore_index=True, sort=False)


def bootstrap_ci(values, repetitions, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 128):
        count = min(128, repetitions - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def wilcoxon_greater(values):
    from scipy.stats import wilcoxon

    values = np.asarray(values, dtype=np.float64)
    values = values[values != 0]
    if len(values) < 2:
        return float("nan")
    try:
        return float(wilcoxon(values, alternative="greater", method="auto").pvalue)
    except TypeError:
        return float(wilcoxon(values, alternative="greater").pvalue)


def bh_adjust(p_values):
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = order[reverse_rank]
        running = min(running, values[index] * len(values) / (reverse_rank + 1))
        adjusted[index] = running
    return adjusted


def paired_daily_tests(unit_errors, repetitions):
    official = unit_errors["official_b0_oracle_seed0"].rename(columns={"abs_error_avg": "official"})
    main_names = ["main_exp49_seed0", "main_exp49_seed1", "main_exp49_seed2"]
    main_wide = None
    for name in main_names:
        current = unit_errors[name].rename(columns={"abs_error_avg": name})
        main_wide = current if main_wide is None else main_wide.merge(current, on=["date", "Node_ID"])
    main_wide["main_exp49_3seed_error_mean"] = main_wide[main_names].mean(axis=1)
    rows = []
    comparisons = main_names + ["main_exp49_3seed_error_mean"]
    periods = ["all", "2026-03", "2026-04", "2026-05", "2026-06"]
    for comparison_index, comparison in enumerate(comparisons):
        merged = official.merge(main_wide[["date", "Node_ID", comparison]], on=["date", "Node_ID"])
        merged["month"] = merged["date"].str[:7]
        for period_index, period in enumerate(periods):
            selected = merged if period == "all" else merged[merged["month"] == period]
            daily = selected.groupby("date", sort=True)[["official", comparison]].mean()
            difference = daily[comparison].to_numpy() - daily["official"].to_numpy()
            ci_low, ci_high = bootstrap_ci(
                difference,
                repetitions,
                seed=20260914 + comparison_index * 100 + period_index,
            )
            rows.append({
                "comparison": comparison,
                "period": period,
                "paired_days": int(len(daily)),
                "official_mae": float(daily["official"].mean()),
                "main_mae": float(daily[comparison].mean()),
                "main_minus_official_mae": float(difference.mean()),
                "official_improvement_pct": float(100.0 * difference.mean() / daily[comparison].mean()),
                "date_cluster_bootstrap_ci95_low": ci_low,
                "date_cluster_bootstrap_ci95_high": ci_high,
                "wilcoxon_one_sided_p": wilcoxon_greater(difference),
                "test_unit": "date mean over 150 stations, 8 anchors, 3 horizons, 2 directions",
            })
    result = pd.DataFrame(rows)
    result["p_adj_bh"] = bh_adjust(result["wilcoxon_one_sided_p"].to_numpy())
    return result


def dominance_summary(unit_errors):
    official = unit_errors["official_b0_oracle_seed0"].rename(columns={"abs_error_avg": "official"})
    result = {}
    main_names = ["main_exp49_seed0", "main_exp49_seed1", "main_exp49_seed2"]
    main_wide = None
    for name in main_names:
        current = unit_errors[name].rename(columns={"abs_error_avg": name})
        main_wide = current if main_wide is None else main_wide.merge(current, on=["date", "Node_ID"])
    main_wide["main_exp49_3seed_error_mean"] = main_wide[main_names].mean(axis=1)
    for comparison in ["main_exp49_seed0", "main_exp49_3seed_error_mean"]:
        merged = official.merge(main_wide[["date", "Node_ID", comparison]], on=["date", "Node_ID"])
        merged["official_better"] = merged["official"] < merged[comparison]
        daily = merged.groupby("date")[["official", comparison]].mean()
        station = merged.groupby("Node_ID")[["official", comparison]].mean()
        result[comparison] = {
            "date_station_units_official_better": int(merged["official_better"].sum()),
            "date_station_units_total": int(len(merged)),
            "date_station_units_official_better_pct": float(100.0 * merged["official_better"].mean()),
            "days_official_better": int((daily["official"] < daily[comparison]).sum()),
            "days_total": int(len(daily)),
            "stations_official_better": int((station["official"] < station[comparison]).sum()),
            "stations_total": int(len(station)),
        }
    return result


def build_figure(monthly, horizon, anchor, anchor_horizon, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "official_b0_oracle": "Official B0 (same-day weather)",
        "official_b0_lag1": "Official B0 (weather D-1)",
        "main_exp49": "Main exp49 seed0",
        "main_exp49_3seed_mean": "Main exp49 3-seed mean",
    }
    selected = list(labels)
    colors = ["#1677ff", "#69b1ff", "#d46b08", "#fa8c16"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(top=0.84, hspace=0.34, wspace=0.22)

    for table, x_col, axis, title in [
        (monthly, "month", axes[0, 0], "Monthly directional MAE"),
        (horizon, "horizon", axes[0, 1], "Forecast-horizon directional MAE"),
        (anchor, "anchor_hour", axes[1, 0], "Anchor-hour directional MAE"),
    ]:
        for model, color in zip(selected, colors):
            group = table[table["model"] == model]
            if model == "main_exp49":
                group = group[group["seed"] == 0]
            group = group.sort_values(x_col)
            axis.plot(group[x_col].astype(str), group["mae"], marker="o", linewidth=2, label=labels[model], color=color)
        axis.set_title(title)
        axis.set_xlabel(x_col.replace("_", " "))
        axis.set_ylabel("MAE")
        axis.grid(alpha=0.25)

    official = anchor_horizon[anchor_horizon["model"] == "official_b0_oracle"]
    main = anchor_horizon[anchor_horizon["model"] == "main_exp49_3seed_mean"]
    delta = main.merge(official, on=["anchor_hour", "horizon"], suffixes=("_main", "_official"))
    matrix = delta.pivot(index="anchor_hour", columns="horizon", values="mae_main") - delta.pivot(
        index="anchor_hour", columns="horizon", values="mae_official"
    )
    image = axes[1, 1].imshow(matrix.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)))
    axes[1, 1].set_title("Main 3-seed MAE minus Official B0 MAE")
    axes[1, 1].set_xlabel("horizon")
    axes[1, 1].set_ylabel("anchor hour")
    axes[1, 1].set_xticks(range(len(matrix.columns)), matrix.columns)
    axes[1, 1].set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1, 1].text(column, row, "%.2f" % matrix.iloc[row, column], ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes[1, 1], shrink=0.85, label="Positive = Official B0 lower MAE")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=4, frameon=False)
    fig.suptitle("Official STGformer B0 vs previous main model (Mar-Jun 2026, exact shared truth)", fontsize=15, y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-oracle", default=str(DEFAULT_OFFICIAL_ORACLE))
    parser.add_argument("--official-lag1", default=str(DEFAULT_OFFICIAL_LAG1))
    parser.add_argument("--main-root", default=str(DEFAULT_MAIN_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()

    output_dir = absolute_path(args.output_dir)
    official_oracle = load_predictions(
        [absolute_path(args.official_oracle)], "official_b0_oracle", 0, "oracle_observed_same_day"
    )
    frames = [official_oracle]
    official_lag1 = load_predictions(
        [absolute_path(args.official_lag1)], "official_b0_lag1", 0, "lagged_observed_d_minus_1"
    )
    assert_same_truth(official_oracle, official_lag1, "official_b0_lag1")
    frames.append(official_lag1)

    main_root = absolute_path(args.main_root)
    for seed in [0, 1, 2]:
        paths = [
            main_root / ("seed%d" % seed) / ("2026m%02d" % month) /
            ("main_exp49_seed%d_2026m%02d_station_hour_predictions.csv" % (seed, month))
            for month in [3, 4, 5, 6]
        ]
        if not all(path.exists() for path in paths):
            missing = [str(path) for path in paths if not path.exists()]
            raise FileNotFoundError("Missing main-model prediction files: %s" % missing)
        frame = load_predictions(paths, "main_exp49", seed, "oracle_observed_same_day")
        assert_same_truth(official_oracle, frame, "main_exp49_seed%d" % seed)
        frames.append(frame)

    metric_tables = {name: [] for name in ["overall", "monthly", "horizon", "anchor", "anchor_horizon"]}
    groupings = {
        "overall": [],
        "monthly": ["month"],
        "horizon": ["horizon"],
        "anchor": ["anchor_hour"],
        "anchor_horizon": ["anchor_hour", "horizon"],
    }
    unit_errors = {}
    demand_tables = []
    for frame in frames:
        key = "%s_seed%d" % (frame["model"].iloc[0], frame["seed"].iloc[0])
        unit_errors[key] = frame.groupby(["date", "Node_ID"], sort=True, as_index=False)["abs_error_avg"].mean()
        for name, grouping in groupings.items():
            metric_tables[name].append(grouped_metrics(frame, grouping))
        demand_tables.append(demand_bucket_metrics(frame))

    for name in metric_tables:
        metric_tables[name] = append_main_seed_mean(pd.concat(metric_tables[name], ignore_index=True))
        atomic_csv(metric_tables[name], output_dir / (name + "_metrics.csv"))

    demand_table = append_main_demand_mean(pd.concat(demand_tables, ignore_index=True))
    atomic_csv(demand_table, output_dir / "demand_bucket_metrics.csv")

    tests = paired_daily_tests(unit_errors, args.bootstrap_reps)
    atomic_csv(tests, output_dir / "paired_daily_tests.csv")
    build_figure(
        metric_tables["monthly"],
        metric_tables["horizon"],
        metric_tables["anchor"],
        metric_tables["anchor_horizon"],
        output_dir / "official_vs_main_exp49.png",
    )

    overall = metric_tables["overall"].set_index("model")
    official = overall.loc["official_b0_oracle"]
    lag1 = overall.loc["official_b0_lag1"]
    main_seed0 = metric_tables["overall"].query("model == 'main_exp49' and seed == 0").iloc[0]
    main_mean = overall.loc["main_exp49_3seed_mean"]
    summary = {
        "schema_version": 1,
        "verification_status": "VERIFIED",
        "truth_and_key_audit": {
            "exact_match_across_official_oracle_lag1_and_main_seeds": True,
            "rows_per_run": int(len(official_oracle)),
            "dates": [str(official_oracle["date"].min()), str(official_oracle["date"].max())],
            "anchor_samples": int(official_oracle[["date", "sample_datetime"]].drop_duplicates().shape[0]),
            "duplicate_keys": 0,
            "nonfinite_values": 0,
        },
        "official_b0_oracle": official.to_dict(),
        "official_b0_lag1": lag1.to_dict(),
        "main_exp49_seed0": main_seed0.to_dict(),
        "main_exp49_3seed_metric_mean": main_mean.to_dict(),
        "official_improvement_vs_main_seed0_pct": float(100.0 * (main_seed0["mae"] - official["mae"]) / main_seed0["mae"]),
        "official_improvement_vs_main_3seed_mean_pct": float(100.0 * (main_mean["mae"] - official["mae"]) / main_mean["mae"]),
        "lag1_mae_degradation_vs_oracle_pct": float(100.0 * (lag1["mae"] - official["mae"]) / official["mae"]),
        "dominance": dominance_summary(unit_errors),
        "paired_test_unit": "date-clustered; conditional on available trained checkpoints",
        "bootstrap_reps": int(args.bootstrap_reps),
        "outputs": {
            "overall": str(output_dir / "overall_metrics.csv"),
            "monthly": str(output_dir / "monthly_metrics.csv"),
            "horizon": str(output_dir / "horizon_metrics.csv"),
            "anchor": str(output_dir / "anchor_metrics.csv"),
            "anchor_horizon": str(output_dir / "anchor_horizon_metrics.csv"),
            "demand_bucket": str(output_dir / "demand_bucket_metrics.csv"),
            "paired_tests": str(output_dir / "paired_daily_tests.csv"),
            "figure": str(output_dir / "official_vs_main_exp49.png"),
        },
    }
    atomic_json(summary, output_dir / "comparison_summary.json")
    print(json.dumps({
        "truth_exact": True,
        "official_oracle_mae": official["mae"],
        "official_lag1_mae": lag1["mae"],
        "main_seed0_mae": main_seed0["mae"],
        "main_3seed_mean_mae": main_mean["mae"],
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
