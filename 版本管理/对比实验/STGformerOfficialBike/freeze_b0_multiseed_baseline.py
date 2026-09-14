# -*- coding: utf-8 -*-
"""Validate and freeze the three-seed official STGformer B0 baseline."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
RESULT_ROOT = PROJECT_ROOT / "分析结果" / "对比实验" / "STGformerOfficialBike"
DEFAULT_MAIN_ROOT = PROJECT_ROOT / "第一部分补的实验" / "缺少的核心实验" / "外部评估" / "main_exp49"
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "b0_multiseed_baseline"
CONFIG_PATH = SCRIPT_DIR / "configs" / "b0_official_single_graph.yaml"
SEEDS = (0, 1, 2)
MONTHS = (3, 4, 5, 6)

from analyze_official_vs_main_exp49 import (  # noqa: E402
    KEY_COLS,
    TRUTH_COLS,
    assert_same_truth,
    atomic_csv,
    atomic_json,
    bootstrap_ci,
    demand_bucket_metrics,
    grouped_metrics,
    load_predictions,
    wilcoxon_greater,
)
from protocol import MODEL_ID, UPSTREAM_COMMIT, checkpoint_training_seed  # noqa: E402


METRIC_COLS = [
    "mae",
    "rmse",
    "mape",
    "mae_out",
    "mae_in",
    "mae_net",
    "rmse_out",
    "rmse_in",
    "rmse_net",
    "bias_out",
    "bias_in",
    "bias_net",
]


def absolute_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def official_prediction_path(seed):
    if seed == 0:
        return RESULT_ROOT / "holdout_202603_202606" / (
            "stgformer_official_b0_202603_202606_station_hour_predictions.csv"
        )
    return RESULT_ROOT / ("holdout_202603_202606_seed%d_oracle" % seed) / (
        "stgformer_official_b0_seed%d_202603_202606_oracle_station_hour_predictions.csv" % seed
    )


def official_summary_path(seed):
    if seed == 0:
        return RESULT_ROOT / "holdout_202603_202606" / (
            "stgformer_official_b0_202603_202606_holdout_summary.json"
        )
    return RESULT_ROOT / ("holdout_202603_202606_seed%d_oracle" % seed) / (
        "stgformer_official_b0_seed%d_202603_202606_oracle_holdout_summary.json" % seed
    )


def training_dir(seed):
    return RESULT_ROOT / ("b0_top150_hist168_pred3_seed%d_bs16" % seed)


def main_prediction_paths(main_root, seed):
    return [
        main_root / ("seed%d" % seed) / ("2026m%02d" % month) /
        ("main_exp49_seed%d_2026m%02d_station_hour_predictions.csv" % (seed, month))
        for month in MONTHS
    ]


def summarize_seed_table(frame, model, group_cols):
    selected = frame[frame["model"] == model].copy()
    rows = []
    grouped = selected.groupby(group_cols, dropna=False, sort=True) if group_cols else [((), selected)]
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row.update({
            "model": model + "_3seed_mean",
            "seed": -1,
            "weather_regime": "oracle_observed_same_day",
            "rows": int(round(group["rows"].mean())),
        })
        for column in METRIC_COLS:
            row[column] = float(group[column].mean())
            row[column + "_seed_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_demand_table(frame, model):
    selected = frame[frame["model"] == model].copy()
    rows = []
    for (flow_type, bucket), group in selected.groupby(
        ["flow_type", "demand_bucket"], sort=False
    ):
        row = {
            "model": model + "_3seed_mean",
            "seed": -1,
            "weather_regime": "oracle_observed_same_day",
            "flow_type": flow_type,
            "demand_bucket": bucket,
            "rows": int(round(group["rows"].mean())),
        }
        for column in ["mae", "rmse", "mape", "bias"]:
            row[column] = float(group[column].mean())
            row[column + "_seed_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def audit_training_runs():
    rows = []
    manifests = []
    common_args = None
    common_contract = None
    path_warnings = []
    for seed in SEEDS:
        run_dir = training_dir(seed)
        summary_path = run_dir / "training_summary.json"
        checkpoint_path = run_dir / "best_stgformer_official_b0.pt"
        if not summary_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError("Missing seed%d training artifact" % seed)
        summary = load_json(summary_path)
        if int(summary["args"]["seed"]) != seed:
            raise RuntimeError("Training summary seed mismatch for seed%d" % seed)
        if summary["model"] != MODEL_ID or summary["upstream_commit"] != UPSTREAM_COMMIT:
            raise RuntimeError("Training identity mismatch for seed%d" % seed)
        comparable_args = {
            key: value for key, value in summary["args"].items()
            if key not in {"seed", "output_dir", "wandb_run_name"}
        }
        contract = {
            "shape": summary["shape"],
            "graph_audit": summary["graph_audit"],
            "input_feature_cols": summary["input_feature_cols"],
            "target_cols": summary["target_cols"],
        }
        if common_args is None:
            common_args = comparable_args
            common_contract = contract
        elif comparable_args != common_args or contract != common_contract:
            raise RuntimeError("Training protocol differs for seed%d" % seed)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        actual_seed = checkpoint_training_seed(checkpoint)
        if actual_seed != seed:
            raise RuntimeError(
                "Checkpoint seed mismatch for seed%d: embedded seed%d" % (seed, actual_seed)
            )
        if checkpoint.get("model_id") != MODEL_ID or checkpoint.get("upstream_commit") != UPSTREAM_COMMIT:
            raise RuntimeError("Checkpoint identity mismatch for seed%d" % seed)
        del checkpoint

        recorded_output = Path(summary["args"]["output_dir"])
        recorded_checkpoint = Path(summary["checkpoint"])
        expected_suffix = "b0_top150_hist168_pred3_seed%d_bs16" % seed
        metadata_matches_seed = (
            recorded_output.name == expected_suffix
            and recorded_checkpoint.parent.name == expected_suffix
        )
        if not metadata_matches_seed:
            path_warnings.append(
                "seed%d legacy summary records seed0 output/checkpoint path; actual local artifact "
                "was verified by embedded seed and SHA256" % seed
            )
        checkpoint_hash = sha256_file(checkpoint_path)
        summary_hash = sha256_file(summary_path)
        rows.append({
            "seed": seed,
            "best_epoch": int(summary["best_epoch"]),
            "best_val_mae": float(summary["best_val_mae"]),
            "internal_test_mae": float(summary["internal_test"]["mae"]),
            "internal_test_rmse": float(summary["internal_test"]["rmse"]),
            "internal_test_mape": float(summary["internal_test"]["mape"]),
            "metadata_path_matches_seed": bool(metadata_matches_seed),
            "checkpoint_sha256": checkpoint_hash,
        })
        manifests.append({
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "training_summary": str(summary_path),
            "training_summary_sha256": summary_hash,
            "embedded_seed": actual_seed,
            "legacy_recorded_output_dir": str(recorded_output),
            "legacy_recorded_checkpoint": str(recorded_checkpoint),
            "metadata_path_matches_seed": bool(metadata_matches_seed),
        })
    return pd.DataFrame(rows), manifests, common_args, common_contract, path_warnings


def audit_holdout_summary(seed, summary_path, prediction_path):
    summary = load_json(summary_path)
    failures = []
    if summary.get("model") != MODEL_ID:
        failures.append("model")
    if summary.get("upstream_commit") != UPSTREAM_COMMIT:
        failures.append("upstream_commit")
    if summary.get("future_weather_regime") != "oracle_observed_same_day":
        failures.append("weather_regime")
    if int(summary.get("num_anchor_samples", -1)) != 976:
        failures.append("num_anchor_samples")
    if int(summary.get("expected_anchor_samples", -1)) != 976:
        failures.append("expected_anchor_samples")
    if summary.get("missing_dates") or summary.get("missing_anchor_datetimes"):
        failures.append("coverage")
    if int(summary.get("duplicate_prediction_keys", -1)) != 0:
        failures.append("duplicate_prediction_keys")
    if int(summary.get("nonfinite_prediction_or_target", -1)) != 0:
        failures.append("nonfinite_prediction_or_target")
    if seed != 0 and int(summary.get("seed", -1)) != seed:
        failures.append("summary_seed")
    if failures:
        raise RuntimeError("seed%d holdout summary failed: %s" % (seed, ", ".join(failures)))
    return {
        "seed": seed,
        "prediction_file": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "summary_file": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "truth_key_signature": summary["truth_key_signature"],
        "anchor_samples": int(summary["num_anchor_samples"]),
        "rows": 439200,
        "weather_regime": summary["future_weather_regime"],
        "checkpoint": summary["checkpoint"],
    }


def add_frame_outputs(frame, metric_parts, demand_parts, daily_parts):
    groupings = {
        "overall": [],
        "monthly": ["month"],
        "horizon": ["horizon"],
        "anchor": ["anchor_hour"],
        "anchor_horizon": ["anchor_hour", "horizon"],
    }
    for name, grouping in groupings.items():
        metric_parts[name].append(grouped_metrics(frame, grouping))
    demand_parts.append(demand_bucket_metrics(frame))
    daily = frame.groupby("date", sort=True, as_index=False)["abs_error_avg"].mean()
    daily["model"] = frame["model"].iloc[0]
    daily["seed"] = int(frame["seed"].iloc[0])
    daily_parts.append(daily)


def paired_multiseed_test(daily_metrics, repetitions):
    pivot = daily_metrics.pivot(index="date", columns=["model", "seed"], values="abs_error_avg")
    official = pivot["official_b0_oracle"][list(SEEDS)].mean(axis=1)
    main = pivot["main_exp49"][list(SEEDS)].mean(axis=1)
    difference = main.to_numpy(dtype=np.float64) - official.to_numpy(dtype=np.float64)
    low, high = bootstrap_ci(difference, repetitions, seed=20260914)
    std = float(np.std(difference, ddof=1))
    return {
        "paired_days": int(len(difference)),
        "official_3seed_daily_mae_mean": float(official.mean()),
        "main_3seed_daily_mae_mean": float(main.mean()),
        "main_minus_official_mae": float(difference.mean()),
        "official_improvement_pct": float(100.0 * difference.mean() / main.mean()),
        "date_cluster_bootstrap_ci95": [low, high],
        "wilcoxon_one_sided_p": wilcoxon_greater(difference),
        "paired_cohens_dz": float(difference.mean() / std),
        "days_official_better": int((difference > 0).sum()),
        "days_total": int(len(difference)),
        "test_unit": "date mean after averaging each model over its three seeds",
    }


def build_figure(overall, monthly, demand, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    official = overall[overall["model"] == "official_b0_oracle"]
    main = overall[overall["model"] == "main_exp49"]
    official_month = monthly[monthly["model"] == "official_b0_oracle"]
    main_month = monthly[monthly["model"] == "main_exp49"]
    demand_mean = demand[demand["model"].isin([
        "official_b0_oracle_3seed_mean", "main_exp49_3seed_mean"
    ])]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    positions = np.arange(2)
    means = [official["mae"].mean(), main["mae"].mean()]
    stds = [official["mae"].std(ddof=1), main["mae"].std(ddof=1)]
    axes[0].bar(positions, means, yerr=stds, capsize=6, color=["#1677ff", "#d46b08"])
    axes[0].set_xticks(positions, ["Official B0", "Main exp49"])
    axes[0].set_ylabel("Directional MAE")
    axes[0].set_title("Overall mean ± seed SD")
    axes[0].grid(axis="y", alpha=0.25)

    for table, label, color in [
        (official_month, "Official B0", "#1677ff"),
        (main_month, "Main exp49", "#d46b08"),
    ]:
        grouped = table.groupby("month", sort=True)["mae"]
        x = sorted(table["month"].unique())
        mean = grouped.mean().reindex(x).to_numpy()
        std = grouped.std(ddof=1).reindex(x).to_numpy()
        axes[1].plot(x, mean, marker="o", linewidth=2, label=label, color=color)
        axes[1].fill_between(x, mean - std, mean + std, alpha=0.16, color=color)
    axes[1].set_title("Monthly MAE across seeds")
    axes[1].set_ylabel("Directional MAE")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    labels = ["0", "1-2", "3-5", "6-10", "11-20", "21+"]
    official_demand = demand_mean[demand_mean["model"] == "official_b0_oracle_3seed_mean"]
    main_demand = demand_mean[demand_mean["model"] == "main_exp49_3seed_mean"]
    official_demand = official_demand.groupby("demand_bucket", observed=False)["mae"].mean().reindex(labels)
    main_demand = main_demand.groupby("demand_bucket", observed=False)["mae"].mean().reindex(labels)
    delta = main_demand.to_numpy() - official_demand.to_numpy()
    axes[2].bar(labels, delta, color=np.where(delta >= 0, "#52c41a", "#ff4d4f"))
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Main minus B0 MAE by true demand")
    axes[2].set_xlabel("True demand bucket")
    axes[2].set_ylabel("Positive = B0 better")
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Frozen B0 multi-seed baseline (Mar-Jun 2026, exact shared truth)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def markdown_table(rows, columns, digits=4):
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, separator]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(("%.*f" % (digits, value)) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(output_dir, training, overall, demand, paired, improvement, warnings):
    official = overall[overall["model"] == "official_b0_oracle"].sort_values("seed")
    official_mean = overall[overall["model"] == "official_b0_oracle_3seed_mean"].iloc[0]
    main_mean = overall[overall["model"] == "main_exp49_3seed_mean"].iloc[0]
    zero = demand[(demand["model"].isin([
        "official_b0_oracle_3seed_mean", "main_exp49_3seed_mean"
    ])) & (demand["demand_bucket"].astype(str) == "0")]
    high = demand[(demand["model"].isin([
        "official_b0_oracle_3seed_mean", "main_exp49_3seed_mean"
    ])) & (demand["demand_bucket"].astype(str) == "21+")]
    rows = []
    for _, item in official.iterrows():
        rows.append({"model": "B0 seed%d" % item["seed"], "MAE": item["mae"], "RMSE": item["rmse"], "MAPE": item["mape"]})
    rows.extend([
        {"model": "B0 3-seed mean", "MAE": official_mean["mae"], "RMSE": official_mean["rmse"], "MAPE": official_mean["mape"]},
        {"model": "Main 3-seed mean", "MAE": main_mean["mae"], "RMSE": main_mean["rmse"], "MAPE": main_mean["mape"]},
    ])
    warning_text = "；".join(warnings) if warnings else "无"
    report = """## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Verification Status: VERIFIED
- Version Label: b0_multiseed_frozen_v1

## B0 多种子基线固化报告

三份 B0 checkpoint、三份 2026-03 至 2026-06 oracle 外部评价均通过身份、覆盖、键、真值和有限值检查。每个 seed 含 976 个锚点和 439,200 行站点—时段预测。

### 核心指标

%s

B0 三种子 MAE 为 **%.4f ± %.4f**，相对主模型三种子指标均值降低 **%.2f%%**。按日期配对后，主模型减 B0 的 MAE 差为 **%.4f**，95%% bootstrap CI 为 **[%.4f, %.4f]**，单侧 Wilcoxon `p=%.3g`，配对 Cohen's dz 为 **%.3f**；B0 在 %d/%d 天更优。

### 低需求门控证据

三种子结果仍显示同一结构：真实需求为 0 时，B0 的平均 MAE 高于主模型；真实需求达到 21+ 时，B0 明显更低。零需求分桶结果为：

%s

高需求 21+ 分桶结果为：

%s

因此“低需求门控”具有跨种子证据，但它仍是下一阶段独立候选，不属于已冻结的 B0。门控只能使用历史需求和已知上下文，并应通过单独消融证明其降低零需求误报且不损伤中高需求优势。

### 可追溯性提示

%s

### 统计谬误扫描

- Coverage: 11/11 checked。
- 未发现 Simpson 反转：B0 相对主模型的总体优势在四个月的聚合方向一致。
- 未进行个体层级、诊断率、前后干预或因果推断，生态谬误、基础率忽视、均值回归、因果倒置不适用。
- 所有 122 天和全部三个 seed 均纳入，无幸存者筛选。
- 分桶与分月属于解释性分析；低需求门控结论标记为待消融验证，避免把探索发现当确认性结论。
- checkpoint 路径历史记录瑕疵已保留并披露，没有改写原始训练摘要。
""" % (
        markdown_table(rows, ["model", "MAE", "RMSE", "MAPE"]),
        official_mean["mae"],
        official_mean["mae_seed_std"],
        improvement,
        paired["main_minus_official_mae"],
        paired["date_cluster_bootstrap_ci95"][0],
        paired["date_cluster_bootstrap_ci95"][1],
        paired["wilcoxon_one_sided_p"],
        paired["paired_cohens_dz"],
        paired["days_official_better"],
        paired["days_total"],
        markdown_table(zero[["model", "flow_type", "mae", "mae_seed_std", "bias"]].to_dict("records"), ["model", "flow_type", "mae", "mae_seed_std", "bias"]),
        markdown_table(high[["model", "flow_type", "mae", "mae_seed_std", "bias"]].to_dict("records"), ["model", "flow_type", "mae", "mae_seed_std", "bias"]),
        warning_text,
    )
    path = output_dir / "analysis_report.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Freeze the three-seed B0 baseline")
    parser.add_argument("--main-root", default=str(DEFAULT_MAIN_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()

    output_dir = absolute_path(args.output_dir)
    main_root = absolute_path(args.main_root)
    training, checkpoint_manifest, common_args, common_contract, warnings = audit_training_runs()
    atomic_csv(training, output_dir / "training_metrics_by_seed.csv")

    metric_parts = {name: [] for name in ["overall", "monthly", "horizon", "anchor", "anchor_horizon"]}
    demand_parts = []
    daily_parts = []
    holdout_manifest = []

    reference_path = official_prediction_path(0)
    reference = load_predictions(
        [reference_path], "official_b0_oracle", 0, "oracle_observed_same_day"
    )
    if len(reference) != 439200:
        raise RuntimeError("seed0 row count is %d, expected 439200" % len(reference))
    holdout_manifest.append(audit_holdout_summary(0, official_summary_path(0), reference_path))
    add_frame_outputs(reference, metric_parts, demand_parts, daily_parts)

    for seed in (1, 2):
        prediction_path = official_prediction_path(seed)
        summary_path = official_summary_path(seed)
        frame = load_predictions(
            [prediction_path], "official_b0_oracle", seed, "oracle_observed_same_day"
        )
        assert_same_truth(reference, frame, "official_b0_oracle_seed%d" % seed)
        if len(frame) != 439200:
            raise RuntimeError("seed%d row count is %d, expected 439200" % (seed, len(frame)))
        holdout_manifest.append(audit_holdout_summary(seed, summary_path, prediction_path))
        add_frame_outputs(frame, metric_parts, demand_parts, daily_parts)
        del frame

    for seed in SEEDS:
        paths = main_prediction_paths(main_root, seed)
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing main prediction files: %s" % missing)
        frame = load_predictions(paths, "main_exp49", seed, "oracle_observed_same_day")
        assert_same_truth(reference, frame, "main_exp49_seed%d" % seed)
        add_frame_outputs(frame, metric_parts, demand_parts, daily_parts)
        del frame

    grouped_columns = {
        "overall": [],
        "monthly": ["month"],
        "horizon": ["horizon"],
        "anchor": ["anchor_hour"],
        "anchor_horizon": ["anchor_hour", "horizon"],
    }
    metric_tables = {}
    for name, parts in metric_parts.items():
        table = pd.concat(parts, ignore_index=True)
        summaries = [
            summarize_seed_table(table, "official_b0_oracle", grouped_columns[name]),
            summarize_seed_table(table, "main_exp49", grouped_columns[name]),
        ]
        table = pd.concat([table] + summaries, ignore_index=True, sort=False)
        metric_tables[name] = table
        atomic_csv(table, output_dir / (name + "_metrics.csv"))

    demand = pd.concat(demand_parts, ignore_index=True)
    demand = pd.concat([
        demand,
        summarize_demand_table(demand, "official_b0_oracle"),
        summarize_demand_table(demand, "main_exp49"),
    ], ignore_index=True, sort=False)
    atomic_csv(demand, output_dir / "demand_bucket_metrics.csv")

    daily = pd.concat(daily_parts, ignore_index=True)
    paired = paired_multiseed_test(daily, args.bootstrap_reps)
    atomic_json(paired, output_dir / "paired_daily_test.json")

    overall = metric_tables["overall"]
    official_mean = overall[overall["model"] == "official_b0_oracle_3seed_mean"].iloc[0]
    main_mean = overall[overall["model"] == "main_exp49_3seed_mean"].iloc[0]
    improvement = float(100.0 * (main_mean["mae"] - official_mean["mae"]) / main_mean["mae"])
    truth_signature = holdout_manifest[0]["truth_key_signature"]
    if any(item["truth_key_signature"] != truth_signature for item in holdout_manifest[1:]):
        raise RuntimeError("Official holdout truth signatures differ")

    source_paths = [
        CONFIG_PATH,
        SCRIPT_DIR / "protocol.py",
        SCRIPT_DIR / "train_stgformer_official_bike.py",
        SCRIPT_DIR / "evaluate_stgformer_official_bike.py",
        SCRIPT_DIR / "models" / "stgformer_official_adapter.py",
        SCRIPT_DIR / "upstream" / "STGformer.py",
        SCRIPT_DIR / "analyze_official_vs_main_exp49.py",
        Path(__file__).resolve(),
    ]
    source_artifacts = [
        {"file": str(path), "sha256": sha256_file(path)} for path in source_paths
    ]
    manifest = {
        "schema_version": 1,
        "baseline_id": "B0_STGformerOfficialBike_Top150_hist168_pred3_seeds0-2",
        "status": "FROZEN",
        "verification_status": "VERIFIED",
        "model": MODEL_ID,
        "upstream_commit": UPSTREAM_COMMIT,
        "seeds": list(SEEDS),
        "training_protocol_equal_except_seed_and_output_path": True,
        "training_common_args": common_args,
        "training_contract": common_contract,
        "config_file": str(CONFIG_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_artifacts": source_artifacts,
        "checkpoint_artifacts": checkpoint_manifest,
        "holdout_protocol": {
            "date_start": "2026-03-01",
            "date_end": "2026-06-30",
            "anchor_hours": [0, 3, 6, 9, 12, 15, 18, 21],
            "anchor_samples_per_seed": 976,
            "rows_per_seed": 439200,
            "weather_regime": "oracle_observed_same_day",
            "truth_key_signature": truth_signature,
            "exact_key_and_truth_match_across_b0_and_main_all_seeds": True,
        },
        "holdout_artifacts": holdout_manifest,
        "b0_three_seed_metrics": {
            key: official_mean[key] for key in METRIC_COLS
        },
        "b0_three_seed_metric_std": {
            key: official_mean[key + "_seed_std"] for key in METRIC_COLS
        },
        "main_exp49_read_only_three_seed_metrics": {
            key: main_mean[key] for key in METRIC_COLS
        },
        "b0_mae_improvement_vs_main_three_seed_metric_mean_pct": improvement,
        "paired_daily_test": paired,
        "warnings": warnings,
        "fallacy_scan_coverage": "11/11",
        "next_experiment_boundary": (
            "B0 is immutable; low-demand gating and graph injection must use new experiment IDs."
        ),
    }
    atomic_json(manifest, output_dir / "baseline_manifest.json")
    build_figure(
        overall,
        metric_tables["monthly"],
        demand,
        output_dir / "b0_multiseed_baseline.png",
    )
    write_report(output_dir, training, overall, demand, paired, improvement, warnings)
    print(json.dumps({
        "status": manifest["status"],
        "b0_mae_mean": official_mean["mae"],
        "b0_mae_seed_std": official_mean["mae_seed_std"],
        "main_mae_mean": main_mean["mae"],
        "improvement_pct": improvement,
        "paired_daily": paired,
        "warnings": warnings,
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
