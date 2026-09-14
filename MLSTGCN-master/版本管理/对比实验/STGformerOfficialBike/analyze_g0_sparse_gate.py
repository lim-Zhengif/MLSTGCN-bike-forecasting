# -*- coding: utf-8 -*-
"""Audit G0 against its frozen B0 control and apply the preregistered screen."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import sha256_file


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
RESULT_ROOT = PROJECT_ROOT / "分析结果" / "对比实验" / "STGformerOfficialBike"
DEFAULT_B0 = RESULT_ROOT / "holdout_202603_202606" / (
    "stgformer_official_b0_202603_202606_station_hour_predictions.csv"
)
DEFAULT_G0 = RESULT_ROOT / "g0_holdout_202603_202606_seed0_oracle" / (
    "g0_b0_sparse_gate_seed0_202603_202606_oracle_station_hour_predictions.csv"
)
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "g0_screen_seed0"

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
METRIC_COLS = ["mae", "rmse", "mape", "mae_out", "mae_in", "mae_net"]


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
            return {str(key): sanitize(item) for key, item in value.items()}
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


def load_predictions(path, model):
    if not path.exists():
        raise FileNotFoundError("Missing prediction file: %s" % path)
    frame = pd.read_csv(path, usecols=USE_COLS)
    frame["date"] = frame["date"].astype(str).str[:10]
    frame["month"] = frame["date"].str[:7]
    if frame.duplicated(KEY_COLS).any():
        raise RuntimeError("%s contains duplicate prediction keys" % model)
    numeric = ["anchor_hour", "horizon", "Node_ID"] + PRED_COLS + TRUTH_COLS
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("%s contains NaN/Inf" % model)
    frame = frame.sort_values(KEY_COLS).reset_index(drop=True)
    frame["model"] = model
    out_error = frame["pred_out"].to_numpy(dtype=np.float64) - frame["true_out"].to_numpy(dtype=np.float64)
    in_error = frame["pred_in"].to_numpy(dtype=np.float64) - frame["true_in"].to_numpy(dtype=np.float64)
    frame["error_out"] = out_error
    frame["error_in"] = in_error
    frame["error_net"] = (
        frame["pred_in"].to_numpy(dtype=np.float64)
        - frame["pred_out"].to_numpy(dtype=np.float64)
        - frame["true_in"].to_numpy(dtype=np.float64)
        + frame["true_out"].to_numpy(dtype=np.float64)
    )
    frame["abs_error_avg"] = 0.5 * (np.abs(out_error) + np.abs(in_error))
    return frame


def audit_comparability(b0, g0, expected_rows, expected_anchor_samples):
    if len(b0) != len(g0):
        raise RuntimeError("G0/B0 row count differs: %d vs %d" % (len(g0), len(b0)))
    if expected_rows > 0 and len(b0) != expected_rows:
        raise RuntimeError("Expected %d rows, found %d" % (expected_rows, len(b0)))
    if not b0[KEY_COLS].equals(g0[KEY_COLS]):
        raise RuntimeError("G0 prediction keys differ from frozen B0")
    if not b0["station_name"].fillna("").equals(g0["station_name"].fillna("")):
        raise RuntimeError("G0 station names differ from frozen B0 node mapping")
    b0_truth = b0[TRUTH_COLS].to_numpy(dtype=np.float64)
    g0_truth = g0[TRUTH_COLS].to_numpy(dtype=np.float64)
    if not np.array_equal(b0_truth, g0_truth):
        raise RuntimeError(
            "G0 truth differs from frozen B0; max delta=%g"
            % float(np.max(np.abs(b0_truth - g0_truth)))
        )
    anchors = b0[["sample_datetime", "anchor_hour"]].drop_duplicates()
    if expected_anchor_samples > 0 and len(anchors) != expected_anchor_samples:
        raise RuntimeError(
            "Expected %d anchor samples, found %d"
            % (expected_anchor_samples, len(anchors))
        )
    return {
        "exact_key_match": True,
        "exact_truth_match": True,
        "exact_station_name_match": True,
        "rows_per_run": int(len(b0)),
        "anchor_samples": int(len(anchors)),
        "dates": [str(b0["date"].min()), str(b0["date"].max())],
        "stations": int(b0["Node_ID"].nunique()),
        "duplicate_keys": 0,
        "nonfinite_values": 0,
    }


def metric_record(frame):
    out_error = frame["error_out"].to_numpy(dtype=np.float64)
    in_error = frame["error_in"].to_numpy(dtype=np.float64)
    net_error = frame["error_net"].to_numpy(dtype=np.float64)
    true = np.concatenate([
        frame["true_out"].to_numpy(dtype=np.float64),
        frame["true_in"].to_numpy(dtype=np.float64),
    ])
    error = np.concatenate([out_error, in_error])
    return {
        "rows": int(len(frame)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(np.mean(np.abs(error) / np.maximum(np.abs(true), 1.0))),
        "mae_out": float(np.mean(np.abs(out_error))),
        "mae_in": float(np.mean(np.abs(in_error))),
        "mae_net": float(np.mean(np.abs(net_error))),
        "bias_out": float(np.mean(out_error)),
        "bias_in": float(np.mean(in_error)),
        "bias_net": float(np.mean(net_error)),
    }


def grouped_metrics(frame, group_cols):
    identity = {"model": str(frame["model"].iloc[0])}
    if not group_cols:
        return pd.DataFrame([{**identity, **metric_record(frame)}])
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for values, group in frame.groupby(grouper, sort=True):
        values = (values,) if len(group_cols) == 1 else values
        rows.append({**identity, **dict(zip(group_cols, values)), **metric_record(group)})
    return pd.DataFrame(rows)


def category_metrics(frame):
    categories = [
        ("zero", lambda values: values == 0),
        ("low_1_2", lambda values: (values >= 1) & (values <= 2)),
        ("mid_3_5", lambda values: (values >= 3) & (values <= 5)),
        ("high_ge6", lambda values: values >= 6),
        ("very_high_ge21", lambda values: values >= 21),
    ]
    rows = []
    for flow in ["out", "in"]:
        true = frame["true_" + flow].to_numpy(dtype=np.float64)
        error = frame["error_" + flow].to_numpy(dtype=np.float64)
        for category, selector in categories:
            mask = selector(true)
            selected = error[mask]
            rows.append({
                "model": str(frame["model"].iloc[0]),
                "flow_type": flow,
                "demand_category": category,
                "values": int(mask.sum()),
                "mae": float(np.mean(np.abs(selected))),
                "rmse": float(np.sqrt(np.mean(selected ** 2))),
                "bias": float(np.mean(selected)),
            })
    combined = []
    for category, selector in categories:
        true = np.concatenate([
            frame["true_out"].to_numpy(dtype=np.float64),
            frame["true_in"].to_numpy(dtype=np.float64),
        ])
        error = np.concatenate([
            frame["error_out"].to_numpy(dtype=np.float64),
            frame["error_in"].to_numpy(dtype=np.float64),
        ])
        mask = selector(true)
        selected = error[mask]
        combined.append({
            "model": str(frame["model"].iloc[0]),
            "flow_type": "combined",
            "demand_category": category,
            "values": int(mask.sum()),
            "mae": float(np.mean(np.abs(selected))),
            "rmse": float(np.sqrt(np.mean(selected ** 2))),
            "bias": float(np.mean(selected)),
        })
    return pd.DataFrame(rows + combined)


def improvement_pct(control, candidate):
    return float(100.0 * (control - candidate) / control)


def degradation_pct(control, candidate):
    return float(100.0 * (candidate - control) / control)


def acceptance_decision(b0_metrics, g0_metrics, b0_categories, g0_categories):
    b0_category = b0_categories.set_index("demand_category")["mae"]
    g0_category = g0_categories.set_index("demand_category")["mae"]
    overall_improvement = improvement_pct(b0_metrics["mae"], g0_metrics["mae"])
    zero_improvement = improvement_pct(b0_category["zero"], g0_category["zero"])
    high_degradation = degradation_pct(b0_category["high_ge6"], g0_category["high_ge6"])
    rmse_degradation = degradation_pct(b0_metrics["rmse"], g0_metrics["rmse"])
    net_degradation = degradation_pct(b0_metrics["mae_net"], g0_metrics["mae_net"])
    conditions = {
        "overall_mae_improvement_min_0_5pct": {
            "value_pct": overall_improvement,
            "passed": bool(overall_improvement >= 0.5),
        },
        "zero_demand_mae_improvement_min_15pct": {
            "value_pct": zero_improvement,
            "passed": bool(zero_improvement >= 15.0),
        },
        "high_ge6_mae_degradation_max_1pct": {
            "value_pct": high_degradation,
            "passed": bool(high_degradation <= 1.0),
        },
        "rmse_and_net_mae_not_both_degrade_over_1pct": {
            "rmse_degradation_pct": rmse_degradation,
            "net_mae_degradation_pct": net_degradation,
            "passed": bool(not (rmse_degradation > 1.0 and net_degradation > 1.0)),
        },
    }
    accepted = all(item["passed"] for item in conditions.values())
    return {
        "decision": "ADVANCE_TO_B1" if accepted else "STOP_G0_DO_NOT_ADVANCE",
        "accepted": bool(accepted),
        "conditions": conditions,
    }


def bootstrap_ci(values, repetitions, seed=20260914):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 128):
        count = min(128, repetitions - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def paired_daily_test(b0, g0, repetitions):
    b0_daily = b0.groupby("date", sort=True)["abs_error_avg"].mean()
    g0_daily = g0.groupby("date", sort=True)["abs_error_avg"].mean()
    if not b0_daily.index.equals(g0_daily.index):
        raise RuntimeError("G0/B0 daily keys differ")
    difference = b0_daily.to_numpy() - g0_daily.to_numpy()
    nonzero = difference[difference != 0]
    p_value = float("nan")
    if len(nonzero) >= 2:
        from scipy.stats import wilcoxon

        try:
            p_value = float(wilcoxon(nonzero, alternative="greater", method="auto").pvalue)
        except TypeError:
            p_value = float(wilcoxon(nonzero, alternative="greater").pvalue)
    std = float(np.std(difference, ddof=1))
    return {
        "paired_days": int(len(difference)),
        "b0_daily_mae": float(b0_daily.mean()),
        "g0_daily_mae": float(g0_daily.mean()),
        "b0_minus_g0_daily_mae": float(difference.mean()),
        "g0_improvement_pct": improvement_pct(float(b0_daily.mean()), float(g0_daily.mean())),
        "date_cluster_bootstrap_ci95": bootstrap_ci(difference, repetitions),
        "wilcoxon_one_sided_p": p_value,
        "cohens_dz": float(difference.mean() / std) if std > 0 else float("nan"),
        "days_g0_better": int((difference > 0).sum()),
        "days_tied": int((difference == 0).sum()),
        "days_b0_better": int((difference < 0).sum()),
        "test_unit": "date mean over stations, anchors, horizons, and directions",
    }


def build_figure(overall, monthly, categories, anchor_horizon, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"frozen_b0_seed0": "#1677ff", "g0_sparse_gate_seed0": "#d46b08"}
    labels = {"frozen_b0_seed0": "Frozen B0", "g0_sparse_gate_seed0": "G0 sparse gate"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(top=0.86, hspace=0.34, wspace=0.25)

    metric_names = ["mae", "rmse", "mae_net"]
    x = np.arange(len(metric_names))
    for index, model in enumerate(labels):
        row = overall[overall["model"] == model].iloc[0]
        axes[0, 0].bar(
            x + (index - 0.5) * 0.34,
            [row[name] for name in metric_names],
            width=0.34,
            color=colors[model],
            label=labels[model],
        )
    axes[0, 0].set_xticks(x, ["MAE", "RMSE", "Net MAE"])
    axes[0, 0].set_title("Overall error")
    axes[0, 0].grid(axis="y", alpha=0.25)

    for model in labels:
        group = monthly[monthly["model"] == model].sort_values("month")
        axes[0, 1].plot(
            group["month"], group["mae"], marker="o", linewidth=2,
            color=colors[model], label=labels[model],
        )
    axes[0, 1].set_title("Monthly directional MAE")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].grid(alpha=0.25)

    category_order = ["zero", "low_1_2", "mid_3_5", "high_ge6", "very_high_ge21"]
    combined = categories[categories["flow_type"] == "combined"]
    for index, model in enumerate(labels):
        group = combined[combined["model"] == model].set_index("demand_category").reindex(category_order)
        axes[1, 0].bar(
            x=np.arange(len(category_order)) + (index - 0.5) * 0.34,
            height=group["mae"], width=0.34, color=colors[model], label=labels[model],
        )
    axes[1, 0].set_xticks(np.arange(len(category_order)), ["0", "1-2", "3-5", ">=6", ">=21"])
    axes[1, 0].set_title("MAE by true-demand category")
    axes[1, 0].set_ylabel("MAE")
    axes[1, 0].grid(axis="y", alpha=0.25)

    b0 = anchor_horizon[anchor_horizon["model"] == "frozen_b0_seed0"]
    g0 = anchor_horizon[anchor_horizon["model"] == "g0_sparse_gate_seed0"]
    merged = b0.merge(g0, on=["anchor_hour", "horizon"], suffixes=("_b0", "_g0"))
    matrix = merged.pivot(index="anchor_hour", columns="horizon", values="mae_b0") - merged.pivot(
        index="anchor_hour", columns="horizon", values="mae_g0"
    )
    limit = max(float(np.max(np.abs(matrix.to_numpy()))), 0.01)
    image = axes[1, 1].imshow(matrix.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
    axes[1, 1].set_title("B0 minus G0 MAE")
    axes[1, 1].set_xlabel("horizon")
    axes[1, 1].set_ylabel("anchor hour")
    axes[1, 1].set_xticks(range(len(matrix.columns)), matrix.columns)
    axes[1, 1].set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1, 1].text(column, row, "%.2f" % matrix.iloc[row, column], ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes[1, 1], shrink=0.85, label="Positive = G0 lower MAE")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=2, frameon=False,
    )
    fig.suptitle("G0 sparse-demand gate vs frozen B0 (Mar-Jun 2026)", fontsize=15, y=0.985)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(output_dir, audit, overall, categories, decision, paired):
    b0 = overall[overall["model"] == "frozen_b0_seed0"].iloc[0]
    g0 = overall[overall["model"] == "g0_sparse_gate_seed0"].iloc[0]
    combined = categories[categories["flow_type"] == "combined"].set_index(
        ["model", "demand_category"]
    )
    zero_b0 = combined.loc[("frozen_b0_seed0", "zero"), "mae"]
    zero_g0 = combined.loc[("g0_sparse_gate_seed0", "zero"), "mae"]
    high_b0 = combined.loc[("frozen_b0_seed0", "high_ge6"), "mae"]
    high_g0 = combined.loc[("g0_sparse_gate_seed0", "high_ge6"), "mae"]
    verdict = "通过，进入 B1" if decision["accepted"] else "未通过，停止 G0，不进入 B1"
    lines = [
        "# G0 低需求门控外部筛选报告",
        "",
        "## Material Passport",
        "",
        "- 产物：G0 对冻结 B0 seed0 的预注册外部筛选。",
        "- 数据：2026-03-01 至 2026-06-30，同日实测天气，%d 行、%d 个锚点样本。" % (
            audit["rows_per_run"], audit["anchor_samples"]
        ),
        "- 可比性：预测键与真实值逐元素完全一致；测试集未参与 G0 训练或选模。",
        "- 决策：**%s**。" % verdict,
        "",
        "## 核心结果",
        "",
        "| 指标 | 冻结 B0 | G0 | G0 相对变化 |",
        "|---|---:|---:|---:|",
        "| Overall MAE | %.6f | %.6f | %.3f%% 改善 |" % (
            b0["mae"], g0["mae"], improvement_pct(b0["mae"], g0["mae"])
        ),
        "| RMSE | %.6f | %.6f | %.3f%% 退化 |" % (
            b0["rmse"], g0["rmse"], degradation_pct(b0["rmse"], g0["rmse"])
        ),
        "| Net-flow MAE | %.6f | %.6f | %.3f%% 退化 |" % (
            b0["mae_net"], g0["mae_net"], degradation_pct(b0["mae_net"], g0["mae_net"])
        ),
        "| Zero-demand MAE | %.6f | %.6f | %.3f%% 改善 |" % (
            zero_b0, zero_g0, improvement_pct(zero_b0, zero_g0)
        ),
        "| High-demand (>=6) MAE | %.6f | %.6f | %.3f%% 退化 |" % (
            high_b0, high_g0, degradation_pct(high_b0, high_g0)
        ),
        "",
        "日期配对结果：B0−G0 日均 MAE 差为 %.6f，95%% 日期聚类 bootstrap CI 为 [%.6f, %.6f]；"
        "G0 在 %d/%d 天更优，单侧 Wilcoxon p=%.6g。" % (
            paired["b0_minus_g0_daily_mae"],
            paired["date_cluster_bootstrap_ci95"][0],
            paired["date_cluster_bootstrap_ci95"][1],
            paired["days_g0_better"], paired["paired_days"], paired["wilcoxon_one_sided_p"],
        ),
        "",
        "## 预注册门槛",
        "",
    ]
    for name, condition in decision["conditions"].items():
        lines.append("- %s：%s。" % (name, "通过" if condition["passed"] else "未通过"))
    lines.extend([
        "",
        "判定只回答 G0 是否值得继续；它不能证明门控在其他随机种子、天气设定或数据集上稳定有效。",
        "",
    ])
    (output_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Apply the preregistered G0-vs-B0 external screen")
    parser.add_argument("--b0-predictions", default=str(DEFAULT_B0))
    parser.add_argument("--g0-predictions", default=str(DEFAULT_G0))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--expected-rows", type=int, default=439200)
    parser.add_argument("--expected-anchor-samples", type=int, default=976)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()

    b0_path = absolute_path(args.b0_predictions)
    g0_path = absolute_path(args.g0_predictions)
    output_dir = absolute_path(args.output_dir)
    b0 = load_predictions(b0_path, "frozen_b0_seed0")
    g0 = load_predictions(g0_path, "g0_sparse_gate_seed0")
    audit = audit_comparability(b0, g0, args.expected_rows, args.expected_anchor_samples)

    groupings = {
        "overall": [],
        "monthly": ["month"],
        "horizon": ["horizon"],
        "anchor": ["anchor_hour"],
        "anchor_horizon": ["anchor_hour", "horizon"],
    }
    tables = {}
    for name, grouping in groupings.items():
        tables[name] = pd.concat(
            [grouped_metrics(b0, grouping), grouped_metrics(g0, grouping)],
            ignore_index=True,
        )
        atomic_csv(tables[name], output_dir / (name + "_metrics.csv"))
    categories = pd.concat([category_metrics(b0), category_metrics(g0)], ignore_index=True)
    atomic_csv(categories, output_dir / "demand_category_metrics.csv")

    overall_b0 = metric_record(b0)
    overall_g0 = metric_record(g0)
    combined_categories = categories[categories["flow_type"] == "combined"]
    b0_categories = combined_categories[combined_categories["model"] == "frozen_b0_seed0"]
    g0_categories = combined_categories[combined_categories["model"] == "g0_sparse_gate_seed0"]
    decision = acceptance_decision(overall_b0, overall_g0, b0_categories, g0_categories)
    paired = paired_daily_test(b0, g0, args.bootstrap_reps)

    figure_path = output_dir / "g0_vs_frozen_b0.png"
    build_figure(tables["overall"], tables["monthly"], categories, tables["anchor_horizon"], figure_path)
    summary = {
        "schema_version": 1,
        "verification_status": "VERIFIED",
        "experiment_id": "G0",
        "control": "frozen_B0_seed0",
        "weather_regime": "oracle_observed_same_day",
        "audit": audit,
        "input_artifacts": {
            "b0_predictions": str(b0_path),
            "b0_predictions_sha256": sha256_file(b0_path),
            "g0_predictions": str(g0_path),
            "g0_predictions_sha256": sha256_file(g0_path),
        },
        "b0_overall": overall_b0,
        "g0_overall": overall_g0,
        "acceptance": decision,
        "paired_daily_test": paired,
        "bootstrap_reps": int(args.bootstrap_reps),
        "outputs": {
            "report": str(output_dir / "analysis_report.md"),
            "figure_png": str(figure_path),
            "figure_pdf": str(figure_path.with_suffix(".pdf")),
            "overall_metrics": str(output_dir / "overall_metrics.csv"),
            "monthly_metrics": str(output_dir / "monthly_metrics.csv"),
            "horizon_metrics": str(output_dir / "horizon_metrics.csv"),
            "anchor_metrics": str(output_dir / "anchor_metrics.csv"),
            "anchor_horizon_metrics": str(output_dir / "anchor_horizon_metrics.csv"),
            "demand_category_metrics": str(output_dir / "demand_category_metrics.csv"),
        },
    }
    atomic_json(summary, output_dir / "screen_summary.json")
    write_report(output_dir, audit, tables["overall"], categories, decision, paired)
    print(json.dumps({
        "decision": decision["decision"],
        "accepted": decision["accepted"],
        "b0_mae": overall_b0["mae"],
        "g0_mae": overall_g0["mae"],
        "overall_mae_improvement_pct": improvement_pct(overall_b0["mae"], overall_g0["mae"]),
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
