import os
import re
import shlex

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SOURCE_XLSX = os.path.join(PROJECT_ROOT, "分析结果", "单车模型结果汇总.xlsx")
OUT_DIR = os.path.join(PROJECT_ROOT, "分析结果", "2026-04-24_阶段汇报实验总表")


DEFAULT_HOURLY = {
    "hist_len": "168",
    "pred_len": "24",
    "batch_size": "8",
    "epochs": "60",
    "lr": "1e-4",
    "weight_decay": "1e-4",
    "loss": "huber",
    "huber_delta": "3.0",
    "graph_sparsify_mode": "topk",
    "graph_topk": "15",
    "graph_attention": "true",
    "matrix_weight": "true",
    "time_kernel_size": "3",
}


EXP06_WRAPPER_DEFAULT = {
    "hist_len": "168",
    "pred_len": "24",
    "batch_size": "8",
    "epochs": "60",
    "lr": "5e-5",
    "weight_decay": "1e-4",
    "loss": "huber",
    "huber_delta": "2.0",
    "graph_sparsify_mode": "topk",
    "graph_topk": "20",
    "graph_attention": "true",
    "matrix_weight": "true",
    "time_kernel_size": "3",
}


PARAM_FLAGS = [
    "hist_len",
    "pred_len",
    "batch_size",
    "epochs",
    "lr",
    "weight_decay",
    "loss",
    "huber_delta",
    "graph_sparsify_mode",
    "graph_topk",
    "graph_attention",
    "matrix_weight",
    "time_kernel_size",
    "project",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def classify_stage(version):
    text = str(version)
    if "优化前基线快照" in text:
        return "基线快照"
    if "鲁棒损失" in text:
        return "鲁棒损失"
    if "小时级" in text:
        return "小时级"
    return "其他"


def infer_data_granularity(row):
    version = str(row.get("版本", ""))
    task = str(row.get("任务目录", ""))
    result_type = str(row.get("结果类型", ""))
    if "小时级" in version or "hourly" in task or "小时级" in task:
        return "小时级骑入/骑出"
    if "库存" in task or "库存" in result_type:
        return "库存推演/安全库存"
    return "日级/滚动预测"


def tokenize_command(command):
    if not isinstance(command, str) or not command.strip():
        return []
    try:
        return shlex.split(command, posix=False)
    except ValueError:
        return command.split()


def parse_command_params(row):
    command = row.get("run_command")
    params = {}
    task = str(row.get("任务目录", ""))
    project = str(row.get("train_project", ""))

    if isinstance(command, str) and "train_bike_hourly_exp06.py" in command:
        params.update(EXP06_WRAPPER_DEFAULT)
    elif isinstance(command, str) and "train_bike_hourly.py" in command:
        params.update(DEFAULT_HOURLY)

    tokens = tokenize_command(command)
    for idx, token in enumerate(tokens):
        if not token.startswith("--"):
            continue
        key = token[2:]
        if key not in PARAM_FLAGS:
            continue
        value = "true"
        if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
            value = tokens[idx + 1]
        params[key] = value

    if not params and "2026-04-07" in str(row.get("版本", "")):
        params["hist_len"] = ""
        params["pred_len"] = ""
        params["batch_size"] = ""
        params["epochs"] = ""
        params["lr"] = ""
        params["weight_decay"] = ""
        params["loss"] = "MAE/原始" if "优化前" in str(row.get("版本", "")) else "鲁棒损失"
        params["huber_delta"] = ""
        params["graph_sparsify_mode"] = "" if "优化前" in str(row.get("版本", "")) else "图稀疏化"
        params["graph_topk"] = ""
        params["graph_attention"] = ""
        params["matrix_weight"] = ""
        params["time_kernel_size"] = ""

    if not params and "安全库存区间" in task:
        params.update({"hist_len": "168", "pred_len": "24"})

    if project and project != "nan":
        params.setdefault("project", project)

    return params


def infer_experiment_name(row, params):
    project = row.get("train_project")
    if isinstance(project, str) and project.strip():
        return project.strip()
    task = str(row.get("任务目录", ""))
    if "bike_hourly_safe_inventory_" in task:
        match = re.search(r"(bike_hourly_safe_inventory_[A-Za-z0-9_]+)", task)
        if match:
            return match.group(1)
    return task


def infer_key_change(row, params):
    version = str(row.get("版本", ""))
    exp_name = str(infer_experiment_name(row, params))
    task = str(row.get("任务目录", ""))
    result_type = str(row.get("结果类型", ""))

    if "优化前基线快照" in version:
        return "原始基线快照，用于对比后续优化收益"
    if "鲁棒损失" in version:
        return "引入鲁棒损失、图稀疏化、星期嵌入等优化"
    if "default_lr1e4" in exp_name:
        return "小时级默认学习率1e-4"
    if "exp01_lr5e5" in exp_name:
        return "学习率降为5e-5"
    if "exp02_lr2e4" in exp_name:
        return "学习率升为2e-4"
    if "exp03_huber2" in exp_name and "lr5e5" not in exp_name:
        return "Huber delta调整为2.0"
    if "exp03_lr5e5_huber2" in exp_name:
        return "5e-5学习率 + Huber delta=2.0"
    if "exp04" in exp_name:
        return "exp03基础上训练轮数扩展到90"
    if "topk10" in exp_name:
        return "图稀疏TopK=10"
    if "topk20" in exp_name and "noattn" not in exp_name:
        return "图稀疏TopK=20，当前exp06主版本"
    if "topk15_or_rowmean" in exp_name:
        return "TopK与row-mean联合稀疏策略"
    if "noattn" in exp_name:
        return "关闭动态图注意力，用于消融对比"
    if "安全库存" in result_type:
        return "预测结果转安全库存区间评估"
    if "预测评估" in result_type and "小时级" in task:
        return "小时级滚动预测评估"
    return ""


def add_readme(out_path, rows):
    lines = [
        "# 阶段汇报实验参数-结果总表说明",
        "",
        "本表由 `分析结果/单车模型结果汇总.xlsx` 整理生成，不覆盖原文件。",
        "",
        "主要整理逻辑：",
        "- 第一列 `实验阶段` 将实验归为 `基线快照`、`鲁棒损失`、`小时级`。",
        "- 参数列优先从 `run_command` 解析；没有命令的早期实验按版本名称补充关键改动。",
        "- 结果列保留训练指标、滚动预测指标和安全库存指标，便于汇报时横向比较。",
        "- 部分早期日级/库存推演实验没有训练命令，因此参数为空或以关键改动说明表示。",
        "",
        "建议PPT使用：",
        "- 方法演进页：使用 `阶段汇报_实验参数结果总表_精简版.xlsx`。",
        "- 附录或答疑页：使用 `阶段汇报_实验参数结果总表.xlsx`。",
        "",
        "记录数：%d" % rows,
    ]
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    ensure_dir(OUT_DIR)
    raw = pd.read_excel(SOURCE_XLSX, sheet_name="结果总表")
    records = []

    for idx, row in raw.iterrows():
        params = parse_command_params(row)
        record = {
            "实验阶段": classify_stage(row.get("版本")),
            "实验名称": infer_experiment_name(row, params),
            "版本": row.get("版本"),
            "任务目录": row.get("任务目录"),
            "结果类型": row.get("结果类型"),
            "数据/任务口径": infer_data_granularity(row),
            "关键改动": infer_key_change(row, params),
            "日期范围": "%s 至 %s" % (
                "" if pd.isna(row.get("date_start")) else row.get("date_start"),
                "" if pd.isna(row.get("date_end")) else row.get("date_end"),
            ),
            "hist_len": params.get("hist_len", ""),
            "pred_len": params.get("pred_len", ""),
            "batch_size": params.get("batch_size", ""),
            "epochs": params.get("epochs", ""),
            "lr": params.get("lr", ""),
            "weight_decay": params.get("weight_decay", ""),
            "loss": params.get("loss", ""),
            "huber_delta": params.get("huber_delta", ""),
            "graph_sparsify_mode": params.get("graph_sparsify_mode", ""),
            "graph_topk": params.get("graph_topk", ""),
            "graph_attention": params.get("graph_attention", ""),
            "matrix_weight": params.get("matrix_weight", ""),
            "time_kernel_size": params.get("time_kernel_size", ""),
            "best_val_mae_epoch": row.get("best_val_mae_epoch"),
            "test_mae": row.get("test_mae"),
            "test_loss": row.get("test_loss"),
            "overall_mae_mean": row.get("overall_mae_mean"),
            "overall_rmse_mean": row.get("overall_rmse_mean"),
            "overall_mape_mean": row.get("overall_mape_mean"),
            "hourly_out_mae_mean": row.get("hourly_out_mae_mean"),
            "hourly_in_mae_mean": row.get("hourly_in_mae_mean"),
            "hourly_net_mae_mean": row.get("hourly_net_mae_mean"),
            "pred_S_min_mae": row.get("pred_S_min_mae"),
            "pred_S_max_mae": row.get("pred_S_max_mae"),
            "pred_end_inventory_mae": row.get("pred_end_inventory_mae"),
            "库存综合误差平均值": row.get("库存综合误差平均值"),
            "总综合误差平均值": row.get("总综合误差平均值"),
            "备注": row.get("备注"),
            "结果路径": row.get("结果路径"),
            "run_command": row.get("run_command"),
            "checkpoint": row.get("checkpoint"),
        }
        records.append(record)

    full = pd.DataFrame(records)
    stage_order = {"基线快照": 0, "鲁棒损失": 1, "小时级": 2, "其他": 9}
    full["_stage_order"] = full["实验阶段"].map(stage_order).fillna(9)
    full = full.sort_values(["_stage_order", "版本", "结果类型", "实验名称"]).drop(columns=["_stage_order"])

    concise_cols = [
        "实验阶段",
        "实验名称",
        "结果类型",
        "数据/任务口径",
        "关键改动",
        "hist_len",
        "pred_len",
        "batch_size",
        "epochs",
        "lr",
        "loss",
        "huber_delta",
        "graph_sparsify_mode",
        "graph_topk",
        "graph_attention",
        "best_val_mae_epoch",
        "test_mae",
        "overall_mae_mean",
        "hourly_out_mae_mean",
        "hourly_in_mae_mean",
        "pred_S_min_mae",
        "总综合误差平均值",
    ]
    concise = full[concise_cols]

    full_path = os.path.join(OUT_DIR, "阶段汇报_实验参数结果总表.xlsx")
    concise_path = os.path.join(OUT_DIR, "阶段汇报_实验参数结果总表_精简版.xlsx")
    csv_path = os.path.join(OUT_DIR, "阶段汇报_实验参数结果总表_精简版.csv")

    with pd.ExcelWriter(full_path, engine="openpyxl") as writer:
        full.to_excel(writer, sheet_name="完整参数结果表", index=False)
        concise.to_excel(writer, sheet_name="PPT精简表", index=False)
        raw.to_excel(writer, sheet_name="原始结果总表", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for cell in sheet[1]:
                cell.style = "Headline 4"
            for column_cells in sheet.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                sheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 10), 42)

    with pd.ExcelWriter(concise_path, engine="openpyxl") as writer:
        concise.to_excel(writer, sheet_name="PPT精简表", index=False)
        sheet = writer.book["PPT精简表"]
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.style = "Headline 4"
        for column_cells in sheet.columns:
            max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
            sheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 10), 40)

    concise.to_csv(csv_path, index=False, encoding="utf-8-sig")
    add_readme(os.path.join(OUT_DIR, "说明.md"), len(full))

    print("Saved:", full_path)
    print("Saved:", concise_path)
    print("Saved:", csv_path)
    print("Rows:", len(full))


if __name__ == "__main__":
    main()
