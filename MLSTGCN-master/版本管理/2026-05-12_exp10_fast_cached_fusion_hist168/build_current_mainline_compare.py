from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "分析结果"
OUT_DIR = ANALYSIS_DIR / "2026-05-13_top150_current_mainline_exp10_fast_compare"


EXPERIMENTS = [
    {
        "name": "exp06-hist72",
        "label": "exp06\nhist72",
        "group": "history length",
        "summary": ANALYSIS_DIR
        / "2026-05-06_top150_exp06_rolling6h_train2025_hist72_feb2026_holdout"
        / "feb2026_holdout_summary.json",
        "best_val_mae": 4.282,
    },
    {
        "name": "exp06-hist168",
        "label": "exp06\nhist168",
        "group": "history length",
        "summary": ANALYSIS_DIR
        / "2026-05-06_top150_exp06_rolling6h_train2025_hist168_feb2026_holdout"
        / "feb2026_holdout_summary.json",
        "best_val_mae": 3.9568,
    },
    {
        "name": "exp06-hist336",
        "label": "exp06\nhist336",
        "group": "history length",
        "summary": ANALYSIS_DIR
        / "2026-05-06_top150_exp06_rolling6h_train2025_hist336_feb2026_holdout"
        / "feb2026_holdout_summary.json",
        "best_val_mae": 5.3312,
    },
    {
        "name": "exp09-channel-attention",
        "label": "exp09\nchannel attn",
        "group": "model variant",
        "summary": ANALYSIS_DIR
        / "2026-05-08_exp09_channel_attention_hist168"
        / "feb2026_holdout"
        / "feb2026_holdout_summary.json",
        "best_val_mae": 3.9422,
    },
    {
        "name": "exp10-anchor-hour-OD",
        "label": "exp10\nanchor OD",
        "group": "model variant",
        "summary": ANALYSIS_DIR
        / "2026-05-08_exp10_anchor_hour_od_graph_hist168"
        / "feb2026_holdout"
        / "feb2026_holdout_summary.json",
        "best_val_mae": 3.9757,
    },
    {
        "name": "exp10-fast-cached-fusion",
        "label": "exp10-fast\nseed0",
        "group": "current mainline",
        "summary": ANALYSIS_DIR
        / "2026-05-12_exp10_fast_cached_fusion_hist168_pred6_seed0_feb2026_holdout"
        / "feb2026_holdout_summary.json",
        "best_val_mae": 3.9506,
        "is_mainline": True,
    },
    {
        "name": "exp10-fast-cached-fusion-seed1",
        "label": "exp10-fast\nseed1",
        "group": "current mainline",
        "summary": ANALYSIS_DIR
        / "2026-05-13_exp10_fast_cached_fusion_hist168_pred6_seed1_feb2026_holdout"
        / "feb2026_holdout_summary.json",
        "best_val_mae": 3.9888,
        "is_mainline": True,
    },
    {
        "name": "exp10-fast-no-distri",
        "label": "exp10-fast\nno distri",
        "group": "ablation",
        "summary": ANALYSIS_DIR
        / "2026-05-13_exp10_fast_no_distri_hist168_pred6_seed0_feb2026_holdout"
        / "feb2026_holdout_summary.json",
        "best_val_mae": 3.9984,
    },
    {
        "name": "exp11-weekly-periodic-features",
        "label": "exp11\nperiodic",
        "group": "feature ablation",
        "summary": ANALYSIS_DIR
        / "2026-05-14_exp11_weekly_periodic_features_hist168_pred6_seed0_feb2026_holdout"
        / "feb2026_holdout_summary.json",
        "best_val_mae": 4.0587,
    },
    {
        "name": "exp12-context-gated-multigraph",
        "label": "exp12\ncontext gate",
        "group": "model variant",
        "summary": ANALYSIS_DIR
        / "2026-05-14_exp12_context_gated_multigraph_hist168_pred6_seed0_feb2026_holdout"
        / "feb2026_holdout_summary.json",
        "best_val_mae": 3.9739,
    },
]


def load_summary(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def build_compare_table():
    rows = []
    for item in EXPERIMENTS:
        summary = load_summary(item["summary"])
        overall = summary.get("overall", {})
        baseline = summary.get("baseline_mae", {})
        rows.append(
            {
                "experiment": item["name"],
                "group": item["group"],
                "is_current_mainline": bool(item.get("is_mainline")),
                "hist_len": summary.get("hist_len"),
                "pred_len": summary.get("pred_len"),
                "anchor_hours": ",".join(map(str, summary.get("anchor_hours", []))),
                "feb_holdout_mae": float(overall.get("mae")),
                "feb_holdout_rmse": float(overall.get("rmse")),
                "best_val_mae": item["best_val_mae"],
                "baseline_last_1h_mae": baseline.get("naive_last_1h_repeat"),
                "baseline_previous_day_mae": baseline.get("naive_previous_day_same_hours"),
                "baseline_train_mean_mae": baseline.get("naive_train_mean_by_horizon"),
                "improve_vs_last_1h_pct": summary.get("model_improve_vs_last_1h_repeat_pct"),
                "improve_vs_previous_day_pct": summary.get(
                    "model_improve_vs_previous_day_same_hours_pct"
                ),
                "improve_vs_train_mean_pct": summary.get("model_improve_vs_train_mean_pct"),
                "summary_path": str(item["summary"].relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows)


def build_figure(compare_df):
    main = compare_df[compare_df["experiment"] == "exp10-fast-cached-fusion-seed1"].iloc[0]
    mainline_mean_mae = compare_df[compare_df["is_current_mainline"]]["feb_holdout_mae"].mean()
    base_exp06 = compare_df[compare_df["experiment"] == "exp06-hist168"].iloc[0]
    base_exp10 = compare_df[compare_df["experiment"] == "exp10-anchor-hour-OD"].iloc[0]

    holdout_dir = ANALYSIS_DIR / "2026-05-12_exp10_fast_cached_fusion_hist168_pred6_seed0_feb2026_holdout"
    anchor_df = pd.read_csv(holdout_dir / "feb2026_anchor_hour_mae.csv")
    horizon_df = pd.read_csv(holdout_dir / "feb2026_horizon_mae.csv")
    anchor_mae_col = "mae" if "mae" in anchor_df.columns else "abs_error_avg"
    horizon_mae_col = "mae" if "mae" in horizon_df.columns else "abs_error_avg"

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    colors = [
        "#94a3b8",
        "#64748b",
        "#94a3b8",
        "#f59e0b",
        "#3b82f6",
        "#ef4444",
        "#dc2626",
        "#a78bfa",
        "#14b8a6",
        "#0f766e",
    ]
    fig = plt.figure(figsize=(14, 8), dpi=180)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.24)

    ax1 = fig.add_subplot(grid[0, :])
    x = np.arange(len(compare_df))
    bars = ax1.bar(
        x,
        compare_df["feb_holdout_mae"],
        color=colors,
        edgecolor="#1f2937",
        linewidth=0.5,
    )
    ax1.set_xticks(x, [item["label"] for item in EXPERIMENTS])
    ax1.set_ylabel("Feb 2026 holdout MAE")
    ax1.set_title("Top150 Rolling 6h: Model Comparison on Feb 2026 External Holdout")
    ax1.grid(axis="y", linestyle="--", alpha=0.28)
    for bar, value in zip(bars, compare_df["feb_holdout_mae"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax1.axhline(mainline_mean_mae, color="#b91c1c", linestyle="--", linewidth=1.1, alpha=0.75)
    ax1.text(
        len(compare_df) - 0.65,
        mainline_mean_mae + 0.025,
        f"mainline mean {mainline_mean_mae:.3f}",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#b91c1c",
    )
    ax1.annotate(
        "best current\ncheckpoint",
        xy=(6, main["feb_holdout_mae"]),
        xytext=(5.25, main["feb_holdout_mae"] + 0.3),
        arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.2),
        color="#b91c1c",
        fontsize=10,
    )

    ax2 = fig.add_subplot(grid[1, 0])
    base_names = ["exp10-fast", "Last 1h", "Prev day", "Train mean"]
    base_vals = [
        main["feb_holdout_mae"],
        main["baseline_last_1h_mae"],
        main["baseline_previous_day_mae"],
        main["baseline_train_mean_mae"],
    ]
    b2 = ax2.bar(
        base_names,
        base_vals,
        color=["#ef4444", "#94a3b8", "#94a3b8", "#94a3b8"],
        edgecolor="#1f2937",
        linewidth=0.5,
    )
    ax2.set_ylabel("MAE")
    ax2.set_title("Against Simple Time-Series Baselines")
    ax2.grid(axis="y", linestyle="--", alpha=0.28)
    for bar, value in zip(b2, base_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax3 = fig.add_subplot(grid[1, 1])
    ax3.plot(anchor_df["anchor_hour"], anchor_df[anchor_mae_col], marker="o", color="#2563eb", label="anchor MAE")
    ax3.plot(horizon_df["horizon"], horizon_df[horizon_mae_col], marker="s", color="#16a34a", label="horizon MAE")
    ax3.set_xlabel("anchor hour / horizon step")
    ax3.set_ylabel("MAE")
    ax3.set_title("exp10-fast Error Structure")
    ax3.grid(True, linestyle="--", alpha=0.28)
    ax3.legend(frameon=False)

    fig.suptitle(
        "Current Prediction Mainline: exp10-fast cached fusion, hist168, rolling 6h",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "MAE improvement: "
        f"{((base_exp06['feb_holdout_mae'] - mainline_mean_mae) / base_exp06['feb_holdout_mae'] * 100):.2f}% "
        "vs exp06 hist168; "
        f"{((base_exp10['feb_holdout_mae'] - mainline_mean_mae) / base_exp10['feb_holdout_mae'] * 100):.2f}% "
        "vs original exp10",
        ha="center",
        fontsize=10,
    )
    fig.savefig(OUT_DIR / "fig_top150_current_mainline_exp10_fast_compare.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_top150_current_mainline_exp10_fast_compare.svg", bbox_inches="tight")
    plt.close(fig)


def write_mainline_note(compare_df):
    main_seed0 = compare_df[compare_df["experiment"] == "exp10-fast-cached-fusion"].iloc[0]
    main_seed1 = compare_df[compare_df["experiment"] == "exp10-fast-cached-fusion-seed1"].iloc[0]
    mainline_mean_mae = compare_df[compare_df["is_current_mainline"]]["feb_holdout_mae"].mean()
    mainline_mean_rmse = compare_df[compare_df["is_current_mainline"]]["feb_holdout_rmse"].mean()
    base_exp06 = compare_df[compare_df["experiment"] == "exp06-hist168"].iloc[0]
    base_exp10 = compare_df[compare_df["experiment"] == "exp10-anchor-hour-OD"].iloc[0]
    no_distri = compare_df[compare_df["experiment"] == "exp10-fast-no-distri"].iloc[0]
    exp11 = compare_df[compare_df["experiment"] == "exp11-weekly-periodic-features"].iloc[0]
    exp12 = compare_df[compare_df["experiment"] == "exp12-context-gated-multigraph"].iloc[0]
    note = f"""# 当前预测主线记录（2026-05-13）

当前预测主线：`exp10_fast_cached_fusion_hist168_pred6`。

选择原因：
- seed0 Feb 2026 外部 holdout MAE = {main_seed0['feb_holdout_mae']:.4f}，RMSE = {main_seed0['feb_holdout_rmse']:.4f}。
- seed1 Feb 2026 外部 holdout MAE = {main_seed1['feb_holdout_mae']:.4f}，RMSE = {main_seed1['feb_holdout_rmse']:.4f}。
- 两个 seed 平均 MAE = {mainline_mean_mae:.4f}，平均 RMSE = {mainline_mean_rmse:.4f}。
- 相比 exp06 hist168，平均 MAE 从 {base_exp06['feb_holdout_mae']:.4f} 降至 {mainline_mean_mae:.4f}，提升 {((base_exp06['feb_holdout_mae'] - mainline_mean_mae) / base_exp06['feb_holdout_mae'] * 100):.2f}%。
- 相比原 exp10，平均 MAE 从 {base_exp10['feb_holdout_mae']:.4f} 降至 {mainline_mean_mae:.4f}，提升 {((base_exp10['feb_holdout_mae'] - mainline_mean_mae) / base_exp10['feb_holdout_mae'] * 100):.2f}%。
- no_distri MAE = {no_distri['feb_holdout_mae']:.4f}，明显差于主线，说明 distri 图仍有独立价值，后续主线保留 distri。
- exp11 显式日/周周期特征 MAE = {exp11['feb_holdout_mae']:.4f}，RMSE = {exp11['feb_holdout_rmse']:.4f}，明显差于主线；说明 hist168 已包含较强周期信息，额外周期特征可能引入噪声，暂不作为主线。
- exp12 上下文感知多图门控 MAE = {exp12['feb_holdout_mae']:.4f}，RMSE = {exp12['feb_holdout_rmse']:.4f}，优于 exp11 但未超过 exp10_fast；保留为 DMGGCN 启发的结构探索，不替换当前主线。

主线配置：
- 任务：top150 NYC rolling-anchor 6h prediction
- 历史窗口：168 小时
- 预测窗口：6 小时
- 决策锚点：00 / 06 / 12 / 16 / 20
- 图配置：dist / neighb / distri / tempp / func + od00 / od06 / od12 / od16 / od20
- 结论：保留 distri；融合图在 forward 内缓存一次，供 MSTGCN blocks 复用

本次输出：
- `top150_rolling6h_model_compare_current_mainline.csv`
- `fig_top150_current_mainline_exp10_fast_compare.png`
- `fig_top150_current_mainline_exp10_fast_compare.svg`
"""
    (OUT_DIR / "CURRENT_PREDICTION_MAINLINE.md").write_text(note, encoding="utf-8")
    (ANALYSIS_DIR / "CURRENT_PREDICTION_MAINLINE.md").write_text(note, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    compare_df = build_compare_table()
    compare_df.to_csv(
        OUT_DIR / "top150_rolling6h_model_compare_current_mainline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_figure(compare_df)
    write_mainline_note(compare_df)
    print("WROTE", OUT_DIR)
    print(
        compare_df[
            ["experiment", "feb_holdout_mae", "feb_holdout_rmse", "best_val_mae", "is_current_mainline"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
