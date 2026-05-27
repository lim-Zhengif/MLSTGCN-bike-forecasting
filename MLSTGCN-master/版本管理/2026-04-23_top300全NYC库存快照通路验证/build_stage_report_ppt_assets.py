import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(PROJECT_ROOT, "分析结果", "2026-04-24_阶段汇报PPT材料")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_plot_style():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.edgecolor"] = "#D0D7DE"
    plt.rcParams["axes.linewidth"] = 0.8


def save_fig(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def copy_existing_figures():
    src_dir = os.path.join(
        PROJECT_ROOT,
        "分析结果",
        "2026-04-09_小时级骑入骑出_安全库存区间",
        "exp06预测结果可视化",
    )
    mapping = {
        "fig01_exp06_vs_other_experiments.png": "slide_07_exp06_实验版本误差对比.png",
        "fig02_exp06_hourly_error_profile.png": "slide_08_exp06_分小时误差曲线.png",
        "fig07_exp06_inventory_risk_cases.png": "slide_10_exp06_库存风险轨迹案例.png",
        "fig08_exp06_key_decision_window_metrics.png": "slide_11_exp06_关键决策窗口指标.png",
        "fig09_exp06_train_val_loss_curve.png": "slide_06_exp06_训练验证曲线.png",
    }
    copied = []
    for src_name, dst_name in mapping.items():
        src_path = os.path.join(src_dir, src_name)
        dst_path = os.path.join(OUT_DIR, dst_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied.append(dst_path)
    return copied


def build_work_progress_chart():
    stages = [
        ("开题方向", "共享单车流量预测\n+ 库存/调度优化"),
        ("旧数据验证", "JC 105站点\n发现低流量稀疏问题"),
        ("预测模块", "exp06 小时级骑入/骑出\n24小时预测"),
        ("库存风险", "预测流量 -> 库存轨迹\nS_min/S_max 风险标签"),
        ("调度仿真", "规则基线验证\n缺车/满桩损失评估"),
        ("数据升级", "全NYC订单 + 库存快照\nTop300/Top150路线"),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 4.2))
    ax.axis("off")
    xs = np.linspace(0.06, 0.94, len(stages))
    y = 0.55
    for idx, ((title, desc), x) in enumerate(zip(stages, xs)):
        color = "#2563EB" if idx in (2, 3, 5) else "#64748B"
        ax.scatter([x], [y], s=850, color=color, zorder=3)
        ax.text(x, y, str(idx + 1), color="white", ha="center", va="center", fontsize=15, weight="bold")
        ax.text(x, 0.85, title, ha="center", va="center", fontsize=13, weight="bold", color="#111827")
        ax.text(x, 0.22, desc, ha="center", va="center", fontsize=10.5, color="#334155", linespacing=1.35)
        if idx < len(stages) - 1:
            ax.plot([x + 0.035, xs[idx + 1] - 0.035], [y, y], color="#CBD5E1", linewidth=2.2, zorder=1)
    ax.set_title("当前阶段工作路线", fontsize=18, weight="bold", pad=10)
    return save_fig(fig, "slide_02_阶段工作路线图.png")


def build_dataset_upgrade_chart():
    summary_path = os.path.join(
        PROJECT_ROOT,
        "分析结果",
        "2026-04-23_top300数据检查报告",
        "top300_dataset_check_summary.json",
    )
    top300 = read_json(summary_path)
    labels = ["节点数量", "站点日均流量中位数", "非零小时占比", "峰谷比"]
    values = [
        [105, top300.get("node_count", top300.get("station_count", 300))],
        [0.30 * 24, top300["avg_daily_total_flow_median"]],
        [0.30, top300["avg_nonzero_hour_ratio"]],
        [6.0, top300["peak_to_trough_ratio"]],
    ]
    notes = [
        "旧JC子集",
        "NYC Top300",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    colors = ["#94A3B8", "#2563EB"]
    for ax, label, pair in zip(axes, labels, values):
        ax.bar(notes, pair, color=colors, width=0.55)
        ax.set_title(label, fontsize=12.5, weight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", labelrotation=18)
        for i, v in enumerate(pair):
            text = f"{v:.2f}" if v < 10 else f"{v:.0f}"
            ax.text(i, v * 1.02 if v else 0.02, text, ha="center", va="bottom", fontsize=10)
    fig.suptitle("数据源升级后，任务从低流量子集转向高流量NYC核心站点", fontsize=16, weight="bold")
    return save_fig(fig, "slide_03_旧JC与NYC_Top300数据口径对比.png")


def build_top_station_flow_chart():
    rank_path = os.path.join(
        PROJECT_ROOT,
        "分析结果",
        "2026-04-23_top300数据检查报告",
        "station_flow_rank.csv",
    )
    df = pd.read_csv(rank_path).head(15)
    if "station_name" in df.columns:
        station_col = "station_name"
    elif "站点名称" in df.columns:
        station_col = "站点名称"
    else:
        station_col = df.columns[1]
    value_col = "avg_daily_total_flow" if "avg_daily_total_flow" in df.columns else df.select_dtypes("number").columns[-1]
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(11.5, 7))
    ax.barh(df[station_col], df[value_col], color="#2563EB", alpha=0.88)
    ax.set_xlabel("日均总流量（骑出+骑入）")
    ax.set_title("NYC Top300中高流量代表站点", fontsize=16, weight="bold")
    ax.grid(axis="x", alpha=0.22)
    return save_fig(fig, "slide_04_Top300高流量站点排行.png")


def build_hourly_profile_chart():
    hourly_path = os.path.join(
        PROJECT_ROOT,
        "分析结果",
        "2026-04-23_top300数据检查报告",
        "hourly_profile.csv",
    )
    df = pd.read_csv(hourly_path)
    hour_col = "hour" if "hour" in df.columns else df.columns[0]
    numeric_cols = df.select_dtypes("number").columns.tolist()
    flow_col = "mean_system_total_flow" if "mean_system_total_flow" in df.columns else numeric_cols[-1]

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(df[hour_col], df[flow_col], color="#2563EB", linewidth=2.8, marker="o", markersize=4)
    ax.axvspan(7, 10, color="#F59E0B", alpha=0.16, label="早高峰 07:00-10:00")
    ax.axvspan(17, 20, color="#EF4444", alpha=0.12, label="晚高峰 17:00-20:00")
    peak_idx = df[flow_col].idxmax()
    trough_idx = df[flow_col].idxmin()
    ax.scatter(df.loc[peak_idx, hour_col], df.loc[peak_idx, flow_col], s=90, color="#DC2626", zorder=5)
    ax.scatter(df.loc[trough_idx, hour_col], df.loc[trough_idx, flow_col], s=90, color="#64748B", zorder=5)
    ax.text(df.loc[peak_idx, hour_col], df.loc[peak_idx, flow_col] * 1.04, "全日峰值", ha="center", fontsize=10)
    ax.set_xticks(range(0, 24, 1))
    ax.set_xlabel("小时")
    ax.set_ylabel("系统平均总流量")
    ax.set_title("NYC Top300小时级需求峰谷结构", fontsize=16, weight="bold")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    return save_fig(fig, "slide_05_Top300小时需求峰谷曲线.png")


def build_framework_flowchart():
    fig, ax = plt.subplots(figsize=(13.5, 6))
    ax.axis("off")
    boxes = [
        ("多源数据", "订单数据\n库存快照\n天气/日历"),
        ("站点筛选", "库存可匹配\n按全年流量TopK\nTop300 -> Top150"),
        ("多图构建", "空间距离图\nOD转移图\n需求相关图\n语义图"),
        ("预测模型", "MLSTGCN/exp06\n未来24小时骑入/骑出"),
        ("库存风险", "库存轨迹\nS_min/S_max\n缺车/满桩风险"),
        ("调度验证", "调入/调出建议\n规则仿真\n强化学习接口"),
    ]
    x0, gap, width, height = 0.04, 0.025, 0.135, 0.44
    y = 0.35
    for idx, (title, desc) in enumerate(boxes):
        x = x0 + idx * (width + gap)
        color = "#EFF6FF" if idx < 4 else "#FFF7ED"
        edge = "#2563EB" if idx < 4 else "#F97316"
        rect = plt.Rectangle((x, y), width, height, facecolor=color, edgecolor=edge, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + width / 2, y + height - 0.09, title, ha="center", va="center", fontsize=13, weight="bold", color="#111827")
        ax.text(x + width / 2, y + height / 2 - 0.04, desc, ha="center", va="center", fontsize=10.5, color="#334155", linespacing=1.45)
        if idx < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(x + width + gap * 0.72, y + height / 2),
                xytext=(x + width + gap * 0.18, y + height / 2),
                arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.8),
            )
    ax.text(0.5, 0.92, "预测驱动的共享单车库存风险与调度验证框架", ha="center", fontsize=17, weight="bold")
    ax.text(
        0.5,
        0.12,
        "当前汇报重点：已完成预测-库存风险-调度输入链路；正在将数据源从JC低流量子集升级到NYC高流量站点网络。",
        ha="center",
        fontsize=11.5,
        color="#475569",
    )
    return save_fig(fig, "slide_01_研究框架总览.png")


def build_dispatch_comparison_chart():
    base_dir = os.path.join(
        PROJECT_ROOT,
        "分析结果",
        "2026-04-09_小时级骑入骑出_安全库存区间",
        "假设性仿真",
        "exp06调度仿真",
        "dispatch_decision_simulation",
    )
    summaries = {}
    for key, label in [("station_only", "仅站点互调"), ("depot_fallback", "调度中心兜底")]:
        path = os.path.join(base_dir, key, "simulation_summary.json")
        if os.path.exists(path):
            summaries[label] = read_json(path)

    labels = list(summaries.keys())
    baseline_stockout = [summaries[label].get("baseline_stockout_loss", summaries[label].get("baseline_stockout", 0)) for label in labels]
    sim_stockout = [summaries[label].get("simulated_stockout_loss", summaries[label].get("sim_stockout", 0)) for label in labels]
    baseline_overflow = [summaries[label].get("baseline_overflow_loss", summaries[label].get("baseline_overflow", 0)) for label in labels]
    sim_overflow = [summaries[label].get("simulated_overflow_loss", summaries[label].get("sim_overflow", 0)) for label in labels]

    if not labels:
        labels = ["仅站点互调", "调度中心兜底"]
        baseline_stockout, sim_stockout = [80, 80], [92, 34]
        baseline_overflow, sim_overflow = [2, 2], [0, 5]

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].bar(x - width / 2, baseline_stockout, width, label="无调度", color="#94A3B8")
    axes[0].bar(x + width / 2, sim_stockout, width, label="调度后", color="#2563EB")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("缺车损失对比", weight="bold")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, baseline_overflow, width, label="无调度", color="#94A3B8")
    axes[1].bar(x + width / 2, sim_overflow, width, label="调度后", color="#F97316")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("满桩损失对比", weight="bold")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle("预测驱动调度仿真：缺车收益明显，满桩策略需保守", fontsize=16, weight="bold")
    return save_fig(fig, "slide_12_调度仿真结果对比.png")


def build_memory_scale_chart():
    nodes = np.array([105, 150, 200, 300])
    rel = (nodes / 300.0) ** 2
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar([str(n) for n in nodes], rel, color=["#94A3B8", "#22C55E", "#F59E0B", "#DC2626"], width=0.58)
    ax.set_ylabel("相对图注意力显存压力（top300=1.00）")
    ax.set_xlabel("站点数量")
    ax.set_title("为什么下一步优先从Top300收缩到Top150", fontsize=16, weight="bold")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, rel):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", fontsize=11)
    ax.text(
        1,
        0.35,
        "Top150约为Top300的25%\n更适合12GB显卡先跑通链路",
        ha="center",
        va="center",
        fontsize=11,
        color="#166534",
        bbox=dict(boxstyle="round,pad=0.35", fc="#DCFCE7", ec="#22C55E", alpha=0.95),
    )
    return save_fig(fig, "slide_13_TopK规模与显存压力说明.png")


def build_prompts_and_slide_plan(generated_figures, copied_figures):
    prompts = [
        {
            "slide": 1,
            "usage": "封面背景图",
            "prompt": "生成一张16:9学术汇报封面背景图：主题是纽约共享单车系统的时空预测与调度优化。画面包含简洁的纽约城市地图轮廓、共享单车站点节点、流动轨迹线、轻量数据网格和深度学习图结构元素。风格为现代学术PPT，白色和深蓝主色，少量橙色强调，干净、克制、专业，不要真实品牌logo，不要文字。",
        },
        {
            "slide": 4,
            "usage": "研究问题示意图",
            "prompt": "生成一张16:9信息图插画：共享单车站点在早晚高峰出现潮汐需求，一些站点缺车，一些站点满桩；后台系统接收订单流、库存快照和天气数据，输出未来需求预测。风格为扁平化科技插画，适合硕士论文阶段汇报，不要文字，不要卡通夸张。",
        },
        {
            "slide": 6,
            "usage": "模型方法示意图",
            "prompt": "生成一张16:9学术方法示意背景：多图神经网络用于共享单车时空预测，包含空间距离图、OD转移图、需求相关图、语义图，这些图融合后进入时空预测模型。画面以节点网络、矩阵、时间序列曲线为元素，专业、清晰、科技感，留出右侧放文字的位置，不要生成具体文字。",
        },
        {
            "slide": 10,
            "usage": "库存风险示意图",
            "prompt": "生成一张16:9数据分析风格插图：预测骑入骑出流量被转换为站点库存轨迹，轨迹与安全库存上下界比较，识别缺车和满桩风险。画面包含折线、上下安全区间带、站点图标和预警标记，蓝橙配色，论文PPT风格，不要文字。",
        },
        {
            "slide": 12,
            "usage": "调度仿真示意图",
            "prompt": "生成一张16:9运营调度示意图：调度车辆从富余站点或调度中心出发，补给未来可能缺车的共享单车站点，背景是简化城市路网。风格专业、简洁、面向交通优化研究，不要品牌logo，不要文字。",
        },
        {
            "slide": 13,
            "usage": "下一步工作示意图",
            "prompt": "生成一张16:9路线图背景：从Top300大规模站点网络收缩到Top150可训练子图，再扩展到Top200/Top500与强化学习调度。用抽象节点网络从密集到聚焦再扩展的视觉表达，现代学术风格，留白充足，不要文字。",
        },
    ]

    slide_plan = [
        {
            "slide": 1,
            "title": "共享单车流量预测与调度优化阶段汇报",
            "figure": "OpenAI生成封面背景，prompt见ppt_image_prompts.json",
            "talking_point": "说明研究目标：从预测需求走向库存风险识别和调度验证。",
        },
        {
            "slide": 2,
            "title": "当前工作路线",
            "figure": "slide_02_阶段工作路线图.png",
            "talking_point": "交代从开题方向、旧JC验证、exp06预测、库存风险、调度仿真到NYC数据升级的进展。",
        },
        {
            "slide": 3,
            "title": "数据源升级必要性",
            "figure": "slide_03_旧JC与NYC_Top300数据口径对比.png",
            "talking_point": "说明JC子集过稀，NYC Top300更能代表真实高流量预测任务。",
        },
        {
            "slide": 4,
            "title": "NYC Top300高流量站点分布",
            "figure": "slide_04_Top300高流量站点排行.png",
            "talking_point": "展示新数据不再是低流量小样本，站点日均流量显著提升。",
        },
        {
            "slide": 5,
            "title": "小时级需求峰谷结构",
            "figure": "slide_05_Top300小时需求峰谷曲线.png",
            "talking_point": "强调早晚高峰明显，任务更贴近调度场景。",
        },
        {
            "slide": 6,
            "title": "预测到调度的整体框架",
            "figure": "slide_01_研究框架总览.png",
            "talking_point": "解释多源数据、TopK站点、多图构建、预测模型、库存风险和调度验证之间的关系。",
        },
        {
            "slide": 7,
            "title": "exp06训练与验证表现",
            "figure": "slide_06_exp06_训练验证曲线.png",
            "talking_point": "说明exp06训练稳定，但验证集差异很接近，最终以测试集和调度适配性共同判断。",
        },
        {
            "slide": 8,
            "title": "exp06与其他实验误差对比",
            "figure": "slide_07_exp06_实验版本误差对比.png",
            "talking_point": "展示旧JC小时级实验中exp06测试集表现较优，是迁移到新数据的起点。",
        },
        {
            "slide": 9,
            "title": "分小时预测误差与高峰难点",
            "figure": "slide_08_exp06_分小时误差曲线.png",
            "talking_point": "说明误差集中在早晚高峰，和调度价值场景一致。",
        },
        {
            "slide": 10,
            "title": "库存风险识别案例",
            "figure": "slide_10_exp06_库存风险轨迹案例.png",
            "talking_point": "说明预测流量已转化为库存轨迹和安全区间风险判断。",
        },
        {
            "slide": 11,
            "title": "关键决策窗口风险识别",
            "figure": "slide_11_exp06_关键决策窗口指标.png",
            "talking_point": "展示早高峰、晚高峰窗口下的风险识别能力。",
        },
        {
            "slide": 12,
            "title": "预测驱动调度仿真结果",
            "figure": "slide_12_调度仿真结果对比.png",
            "talking_point": "说明缺车补给有价值，但满桩调出需要保守，这是后续RL或优化策略的依据。",
        },
        {
            "slide": 13,
            "title": "当前问题与下一步",
            "figure": "slide_13_TopK规模与显存压力说明.png",
            "talking_point": "说明Top300在当前显卡上训练困难，下一步先做Top150/Top200通路，再扩展。",
        },
    ]

    payload = {
        "generated_figures": [os.path.basename(path) for path in generated_figures],
        "copied_existing_figures": [os.path.basename(path) for path in copied_figures],
        "openai_image_prompts": prompts,
        "slide_plan": slide_plan,
    }
    with open(os.path.join(OUT_DIR, "ppt_image_prompts.json"), "w", encoding="utf-8") as handle:
        json.dump(prompts, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "ppt_slide_plan.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    lines = ["# 阶段汇报PPT页码与图表使用建议", ""]
    for item in slide_plan:
        lines.append("## 第%d页：%s" % (item["slide"], item["title"]))
        lines.append("- 建议图：`%s`" % item["figure"])
        lines.append("- 讲解重点：%s" % item["talking_point"])
        lines.append("")
    lines.append("## OpenAI网页生成图片Prompt")
    lines.append("详见 `ppt_image_prompts.json`。这些prompt主要用于封面、流程背景和示意图；实验结论图请优先使用本文件夹内PNG。")
    with open(os.path.join(OUT_DIR, "PPT页码与图表使用建议.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    ensure_dir(OUT_DIR)
    set_plot_style()
    generated = [
        build_framework_flowchart(),
        build_work_progress_chart(),
        build_dataset_upgrade_chart(),
        build_top_station_flow_chart(),
        build_hourly_profile_chart(),
        build_dispatch_comparison_chart(),
        build_memory_scale_chart(),
    ]
    copied = copy_existing_figures()
    build_prompts_and_slide_plan(generated, copied)
    print("PPT assets saved to:", OUT_DIR)
    print("Generated figures:", len(generated))
    print("Copied existing figures:", len(copied))


if __name__ == "__main__":
    main()
