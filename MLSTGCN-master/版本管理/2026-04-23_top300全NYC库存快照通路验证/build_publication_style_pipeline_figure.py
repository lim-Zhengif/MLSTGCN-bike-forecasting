import os
import math

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(PROJECT_ROOT, "分析结果", "2026-04-24_顶刊风格核心流程图")


COLORS = {
    "ink": "#0F172A",
    "muted": "#475569",
    "line": "#94A3B8",
    "data": "#E0F2FE",
    "data_edge": "#0284C7",
    "graph": "#ECFDF5",
    "graph_edge": "#059669",
    "model": "#EEF2FF",
    "model_edge": "#4F46E5",
    "risk": "#FFF7ED",
    "risk_edge": "#EA580C",
    "soft": "#F8FAFC",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setup_style():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42


def add_box(ax, x, y, w, h, title, subtitle=None, fc="#FFFFFF", ec="#CBD5E1", lw=1.6, fs=11, tag=None):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=fs, weight="bold", color=COLORS["ink"], zorder=3)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.34, subtitle, ha="center", va="center", fontsize=fs - 2, color=COLORS["muted"], linespacing=1.25, zorder=3)
    if tag:
        ax.text(
            x + 0.012,
            y + h - 0.022,
            tag,
            ha="left",
            va="top",
            fontsize=7.3,
            color=ec,
            family="DejaVu Sans Mono",
            zorder=4,
        )
    return box


def arrow(ax, start, end, rad=0.0, color="#64748B", lw=1.6):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(patch)


def draw_tiny_timeseries(ax, x, y, w, h, color="#0284C7"):
    xs = [x + w * i / 7.0 for i in range(8)]
    ys = [y + h * (0.35 + 0.25 * math.sin(i * 1.1) + 0.18 * (i in [2, 5])) for i in range(8)]
    ax.plot(xs, ys, color=color, linewidth=1.7, zorder=4)
    ax.scatter(xs, ys, s=8, color=color, zorder=5)


def draw_tiny_graph(ax, cx, cy, r, color="#059669"):
    pts = []
    for i in range(7):
        angle = 2 * math.pi * i / 7 + 0.25
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    edges = [(0, 1), (0, 3), (1, 4), (2, 4), (2, 6), (3, 5), (5, 6), (1, 6)]
    for i, j in edges:
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color="#86EFAC", linewidth=1.1, zorder=3)
    for px, py in pts:
        ax.add_patch(Circle((px, py), r * 0.12, color=color, zorder=4))


def draw_model_core(ax, x, y, w, h):
    add_box(ax, x, y, w, h, "MLSTGCN", "时空卷积 + 动态多图融合", fc=COLORS["model"], ec=COLORS["model_edge"], fs=12, tag="train_bike.py")
    for i in range(3):
        ax.add_patch(Rectangle((x + 0.035 + i * 0.055, y + 0.035), 0.036, 0.025 + i * 0.018, facecolor="#818CF8", edgecolor="none", zorder=4, alpha=0.9))
    draw_tiny_timeseries(ax, x + w - 0.13, y + 0.035, 0.09, 0.055, color=COLORS["model_edge"])


def draw_inventory_band(ax, x, y, w, h):
    add_box(ax, x, y, w, h, "库存风险层", "轨迹越界检测", fc=COLORS["risk"], ec=COLORS["risk_edge"], fs=11.5, tag="build_dispatch_decision_input.py")
    ax.plot([x + 0.035, x + w - 0.035], [y + 0.055, y + 0.055], color="#FDBA74", linewidth=1.4, zorder=4)
    ax.plot([x + 0.035, x + w - 0.035], [y + h - 0.045, y + h - 0.045], color="#FDBA74", linewidth=1.4, zorder=4)
    xs = [x + 0.04 + (w - 0.08) * i / 8 for i in range(9)]
    ys = [y + 0.08, y + 0.10, y + 0.13, y + 0.18, y + 0.22, y + 0.20, y + 0.15, y + 0.09, y + 0.045]
    ax.plot(xs, ys, color=COLORS["risk_edge"], linewidth=1.7, zorder=4)


def build_figure():
    setup_style()
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "预测驱动的共享单车库存风险与调度实验流程", ha="center", va="center", fontsize=22, weight="bold", color=COLORS["ink"])
    ax.text(0.5, 0.928, "从多源数据处理到动态多图预测，再到库存风险识别与调度仿真", ha="center", va="center", fontsize=11.5, color=COLORS["muted"])

    panel_y = 0.78
    labels = [
        ("(a) Data layer", 0.06),
        ("(b) Graph layer", 0.31),
        ("(c) Prediction layer", 0.56),
        ("(d) Decision layer", 0.79),
    ]
    for label, x in labels:
        ax.text(x, panel_y + 0.095, label, fontsize=11, color=COLORS["muted"], weight="bold")

    add_box(ax, 0.045, 0.67, 0.17, 0.17, "多源数据", "订单CSV\n库存快照\n天气/日历", fc=COLORS["data"], ec=COLORS["data_edge"], tag="raw data")
    draw_tiny_timeseries(ax, 0.067, 0.70, 0.05, 0.05)
    draw_tiny_graph(ax, 0.165, 0.715, 0.034, color=COLORS["data_edge"])

    add_box(ax, 0.255, 0.67, 0.18, 0.17, "站点映射与TopK", "名称匹配库存\n按全年流量排序", fc=COLORS["graph"], ec=COLORS["graph_edge"], tag="build_topk_*")
    ax.text(0.345, 0.697, r"$score_i=out_i+in_i$", ha="center", va="center", fontsize=10.5, color=COLORS["graph_edge"])

    add_box(ax, 0.475, 0.67, 0.18, 0.17, "多图矩阵", "距离图 / OD图\n需求相关图 / 语义图", fc=COLORS["graph"], ec=COLORS["graph_edge"], tag="build_topk_graphs.py")
    for k in range(4):
        draw_tiny_graph(ax, 0.512 + k * 0.035, 0.72, 0.016, color=COLORS["graph_edge"])

    draw_model_core(ax, 0.695, 0.67, 0.19, 0.17)

    arrow(ax, (0.215, 0.755), (0.255, 0.755))
    arrow(ax, (0.435, 0.755), (0.475, 0.755))
    arrow(ax, (0.655, 0.755), (0.695, 0.755))

    add_box(ax, 0.08, 0.43, 0.19, 0.15, "小时级样本构造", r"$X_{t-167:t}$ → $Y_{t+1:t+24}$" + "\n骑出量/骑入量", fc="#FFFFFF", ec=COLORS["data_edge"], fs=11, tag="prepare_topk_hourly_dataset.py")
    add_box(ax, 0.33, 0.43, 0.19, 0.15, "模型输出", "未来24小时\n站点级骑入/骑出", fc="#FFFFFF", ec=COLORS["model_edge"], fs=11, tag="evaluate_*.py")
    draw_inventory_band(ax, 0.58, 0.43, 0.20, 0.15)
    add_box(ax, 0.83, 0.43, 0.12, 0.15, "调度仿真", "补车/调出\n损失对比", fc="#FFFFFF", ec=COLORS["risk_edge"], fs=10.5, tag="simulate_*.py")

    arrow(ax, (0.175, 0.67), (0.175, 0.58), rad=0.0, color=COLORS["data_edge"])
    arrow(ax, (0.270, 0.505), (0.330, 0.505), color=COLORS["line"])
    arrow(ax, (0.520, 0.505), (0.580, 0.505), color=COLORS["line"])
    arrow(ax, (0.780, 0.505), (0.830, 0.505), color=COLORS["line"])
    arrow(ax, (0.790, 0.67), (0.430, 0.58), rad=0.18, color=COLORS["model_edge"])

    ax.text(0.67, 0.345, r"$S_{t+h}=S_t+\sum_{k=1}^{h}(\hat{in}_{t+k}-\hat{out}_{t+k})$", ha="center", va="center", fontsize=13, color=COLORS["risk_edge"])
    ax.text(0.67, 0.307, r"risk = I[$S<S_{min}$] or I[$S>S_{max}$]", ha="center", va="center", fontsize=12, color=COLORS["muted"])

    add_box(
        ax,
        0.08,
        0.13,
        0.28,
        0.12,
        "当前工程调整",
        "Top300训练受显存限制\n先构建Top150高流量子图跑通链路",
        fc="#F0FDF4",
        ec=COLORS["graph_edge"],
        fs=11,
        tag="next experiment",
    )
    add_box(
        ax,
        0.42,
        0.13,
        0.23,
        0.12,
        "保留的研究主线",
        "预测需求 → 库存轨迹\n→ 风险识别 → 调度优化",
        fc="#F8FAFC",
        ec=COLORS["line"],
        fs=11,
    )
    add_box(
        ax,
        0.71,
        0.13,
        0.22,
        0.12,
        "后续扩展",
        "Top200/Top500\n轻量图融合或强化学习调度",
        fc="#FFF7ED",
        ec=COLORS["risk_edge"],
        fs=11,
    )
    arrow(ax, (0.36, 0.19), (0.42, 0.19), color=COLORS["line"])
    arrow(ax, (0.65, 0.19), (0.71, 0.19), color=COLORS["line"])

    ax.text(0.5, 0.055, "注：图中脚本名对应当前工程核心入口；结果图与实验表可作为该流程的验证材料。", ha="center", va="center", fontsize=9.5, color="#64748B")
    return fig


def main():
    ensure_dir(OUT_DIR)
    fig = build_figure()
    png_path = os.path.join(OUT_DIR, "顶刊风格_核心代码操作流程图.png")
    svg_path = os.path.join(OUT_DIR, "顶刊风格_核心代码操作流程图.svg")
    pdf_path = os.path.join(OUT_DIR, "顶刊风格_核心代码操作流程图.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    caption_path = os.path.join(OUT_DIR, "图注_核心代码操作流程图.md")
    with open(caption_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# 图注建议\n\n"
            "图X 预测驱动的共享单车库存风险与调度实验流程。"
            "该流程以订单数据、库存快照、天气与日历特征为输入，"
            "首先完成站点匹配与TopK筛选，并构建空间距离图、OD转移图、需求相关图和语义图。"
            "随后基于MLSTGCN进行未来24小时站点级骑入/骑出预测，"
            "再将预测流量累积为库存轨迹，并结合安全库存区间识别缺车与满桩风险，"
            "最终形成调度仿真输入，用于评估补车与调出策略的效果。\n"
        )
    print("Saved:", OUT_DIR)
    print("PNG:", png_path)
    print("SVG:", svg_path)
    print("PDF:", pdf_path)


if __name__ == "__main__":
    main()
