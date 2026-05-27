import argparse
from pathlib import Path
import json

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "分析结果"
DEFAULT_WANDB_PROJECT = "top150_rolling6h_model_compare"


EXPERIMENTS = [
    {
        "name": "exp10-fast-cached-fusion-seed0",
        "label": "exp10-fast\nseed0",
        "group": "mainline",
        "summary": ANALYSIS_DIR / "2026-05-12_exp10_fast_cached_fusion_hist168_pred6_seed0_feb2026_holdout" / "feb2026_holdout_summary.json",
    },
    {
        "name": "exp10-fast-cached-fusion-seed1",
        "label": "exp10-fast\nseed1",
        "group": "mainline",
        "summary": ANALYSIS_DIR / "2026-05-13_exp10_fast_cached_fusion_hist168_pred6_seed1_feb2026_holdout" / "feb2026_holdout_summary.json",
    },
    {
        "name": "stid-baseline-seed0",
        "label": "STID\nseed0",
        "group": "baseline",
        "summary": ANALYSIS_DIR / "2026-05-19_top150_stid_feb2026_holdout_stid_seed0" / "stid_feb2026_holdout_summary.json",
    },
    {
        "name": "gwnet-baseline-seed0",
        "label": "GWNet\nseed0",
        "group": "baseline",
        "summary": ANALYSIS_DIR / "2026-05-20_top150_gwnet_feb2026_holdout_seed0" / "gwnet_feb2026_seed0_holdout_summary.json",
    },
    {
        "name": "exp17-fusion-d2stgnn-style",
        "label": "exp17\nD2STGNN-style",
        "group": "negative",
        "summary": ANALYSIS_DIR / "2026-05-20_exp17_fusion_d2stgnn_feb2026_holdout_seed0" / "feb2026_holdout_summary.json",
    },
    {
        "name": "exp18-peak-anchor-loss",
        "label": "exp18\npeak loss",
        "group": "negative",
        "summary": ANALYSIS_DIR / "2026-05-21_exp18_peak_anchor_loss_seed0_feb2026_holdout" / "feb2026_holdout_summary.json",
    },
    {
        "name": "exp10-fast-no-distri",
        "label": "exp10-fast\nno distri",
        "group": "ablation",
        "summary": ANALYSIS_DIR / "2026-05-13_exp10_fast_no_distri_hist168_pred6_seed0_feb2026_holdout" / "feb2026_holdout_summary.json",
    },
    {
        "name": "exp11-weekly-periodic",
        "label": "exp11\nperiodic",
        "group": "ablation",
        "summary": ANALYSIS_DIR / "2026-05-14_exp11_weekly_periodic_features_hist168_pred6_seed0_feb2026_holdout" / "feb2026_holdout_summary.json",
    },
    {
        "name": "exp12-context-gated",
        "label": "exp12\ncontext gate",
        "group": "ablation",
        "summary": ANALYSIS_DIR / "2026-05-14_exp12_context_gated_multigraph_hist168_pred6_seed0_feb2026_holdout" / "feb2026_holdout_summary.json",
    },
    {
        "name": "exp13-anchor-gated",
        "label": "exp13\nanchor gate",
        "group": "negative",
        "summary": ANALYSIS_DIR / "2026-05-16_exp13_anchor_hour_context_gated_multigraph_hist168_pred6_seed0_feb2026_holdout" / "feb2026_holdout_summary.json",
    },
    {
        "name": "exp14-anchor-prior",
        "label": "exp14\nanchor prior",
        "group": "negative",
        "summary": ANALYSIS_DIR / "2026-05-16_exp14_anchor_prior_homogeneous_gate_hist168_pred6_seed0_feb2026_holdout" / "feb2026_holdout_summary.json",
    },
    {
        "name": "exp15-od-only",
        "label": "exp15\nOD only",
        "group": "negative",
        "summary": ANALYSIS_DIR / "2026-05-16_exp15_od_only_dynamic_gate_hist168_pred6_seed0_feb2026_holdout" / "feb2026_holdout_summary.json",
    },
    {
        "name": "exp15-anchorfix",
        "label": "exp15\nanchorfix",
        "group": "negative",
        "summary": ANALYSIS_DIR / "2026-05-18_exp15_anchorfix_feb_to_mar03_2026_holdout" / "feb_to_mar03_2026_holdout_summary.json",
    },
]


def load_summary(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def build_table():
    rows = []
    missing = []
    for item in EXPERIMENTS:
        if not item["summary"].exists():
            missing.append(str(item["summary"].relative_to(PROJECT_ROOT)))
            continue
        summary = load_summary(item["summary"])
        overall = summary.get("overall", {})
        rows.append(
            {
                "experiment": item["name"],
                "group": item["group"],
                "label": item["label"],
                "mae": overall.get("mae"),
                "rmse": overall.get("rmse"),
                "best_val_mae": summary.get("best_val_mae_epoch") or summary.get("best_val_mae"),
                "date_start": summary.get("date_start"),
                "date_end": summary.get("date_end"),
                "requested_start": summary.get("requested_start"),
                "requested_end": summary.get("requested_end"),
                "summary_path": str(item["summary"].relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows), missing


def format_markdown_table(df):
    if df.empty:
        return "_No summaries found._"
    cols = ["experiment", "group", "label", "mae", "rmse", "best_val_mae", "summary_path"]
    view = df.copy()
    for col in ["mae", "rmse", "best_val_mae"]:
        if col in view.columns:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    view = view[cols]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


def log_table_to_wandb(df, args, csv_path):
    if args.logger != "wandb":
        return
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logger requested, but wandb is unavailable in the current environment.") from exc

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        tags=tags or None,
    )
    run.log({"experiment_compare/table": wandb.Table(dataframe=df)})
    run.summary["num_experiments"] = int(len(df))
    if not df.empty:
        best_idx = df["mae"].astype(float).idxmin()
        run.summary["best_experiment"] = str(df.loc[best_idx, "experiment"])
        run.summary["best_mae"] = float(df.loc[best_idx, "mae"])
        run.summary["best_rmse"] = float(df.loc[best_idx, "rmse"])
    run.summary["csv_path"] = str(csv_path)
    run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logger", choices=["csv", "wandb"], default="csv")
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_run_name", default="experiment_compare_table_2026_05_19")
    parser.add_argument("--wandb_tags", default="compare-table")
    args = parser.parse_args()

    df, missing = build_table()
    out_dir = ANALYSIS_DIR / "2026-05-19_experiment_compare_table"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "experiment_compare_table.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    report_path = out_dir / "experiment_compare_table.md"
    report_lines = [
        "# Experiment Compare Table",
        "",
        "This report aggregates the current mainline and baseline summaries that exist in the repo.",
        "",
        "## Rows",
        "",
        format_markdown_table(df),
        "",
    ]
    if missing:
        report_lines.extend([
            "## Missing",
            "",
            *[f"- {path}" for path in missing],
            "",
        ])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log_table_to_wandb(df, args, csv_path)
    print(df.to_string(index=False))
    if missing:
        print("MISSING:")
        for path in missing:
            print(" -", path)
    print("WROTE", out_dir)


if __name__ == "__main__":
    main()
