import os
from html import escape

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(PROJECT_ROOT, "分析结果", "2026-04-24_核心代码操作过程图")


NODES = [
    {
        "id": "raw",
        "x": 0.05,
        "y": 0.62,
        "w": 0.16,
        "h": 0.20,
        "title": "1. 原始数据",
        "body": "订单CSV\n库存快照CSV\n天气/日历",
        "color": "#E0F2FE",
        "edge": "#0284C7",
    },
    {
        "id": "topk",
        "x": 0.27,
        "y": 0.62,
        "w": 0.18,
        "h": 0.20,
        "title": "2. 站点匹配与筛选",
        "body": "build_topk_nyc_inventory_assets.py\n名称匹配库存\n按全年流量选TopK",
        "color": "#DCFCE7",
        "edge": "#16A34A",
    },
    {
        "id": "graph",
        "x": 0.52,
        "y": 0.62,
        "w": 0.18,
        "h": 0.20,
        "title": "3. 多图构建",
        "body": "build_topk_graphs.py\n距离图 / OD图\n相关图 / 语义图",
        "color": "#EFF6FF",
        "edge": "#2563EB",
    },
    {
        "id": "sample",
        "x": 0.77,
        "y": 0.62,
        "w": 0.18,
        "h": 0.20,
        "title": "4. 小时级样本",
        "body": "prepare_topk_hourly_dataset.py\nX: 过去168小时\nY: 未来24小时骑入/骑出",
        "color": "#EFF6FF",
        "edge": "#2563EB",
    },
    {
        "id": "train",
        "x": 0.18,
        "y": 0.26,
        "w": 0.20,
        "h": 0.20,
        "title": "5. 模型训练",
        "body": "train_bike.py\nMSTGCN + FusionGraph\nHuber / TopK / GraphAttention",
        "color": "#F8FAFC",
        "edge": "#475569",
    },
    {
        "id": "eval",
        "x": 0.43,
        "y": 0.26,
        "w": 0.20,
        "h": 0.20,
        "title": "6. 预测评估",
        "body": "evaluate_bike_hourly_date_range.py\nMAE/RMSE\n分小时/高峰窗口",
        "color": "#F8FAFC",
        "edge": "#475569",
    },
    {
        "id": "risk",
        "x": 0.68,
        "y": 0.26,
        "w": 0.20,
        "h": 0.20,
        "title": "7. 库存风险输入",
        "body": "build_dispatch_decision_input.py\n预测流量 -> 库存轨迹\nS_min/S_max + 风险标签",
        "color": "#FFF7ED",
        "edge": "#EA580C",
    },
    {
        "id": "dispatch",
        "x": 0.68,
        "y": 0.03,
        "w": 0.20,
        "h": 0.16,
        "title": "8. 调度仿真",
        "body": "simulate_from_dispatch_decision_input.py\n来源站 -> 目标站\n缺车/满桩损失对比",
        "color": "#FFEDD5",
        "edge": "#EA580C",
    },
    {
        "id": "next",
        "x": 0.06,
        "y": 0.03,
        "w": 0.34,
        "h": 0.16,
        "title": "当前调整：从Top300收缩到Top150",
        "body": "原因：FusionGraph注意力显存近似随N²增长\n目标：先跑通NYC高流量站点训练链路，再扩展Top200/Top300轻量版",
        "color": "#F0FDF4",
        "edge": "#16A34A",
    },
]

EDGES = [
    ("raw", "topk"),
    ("topk", "graph"),
    ("graph", "sample"),
    ("sample", "train"),
    ("train", "eval"),
    ("eval", "risk"),
    ("risk", "dispatch"),
    ("topk", "next"),
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def node_center(node):
    return node["x"] + node["w"] / 2, node["y"] + node["h"] / 2


def draw_matplotlib_png(path):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FAFC")

    ax.text(0.5, 0.94, "核心代码具体操作过程图", ha="center", va="center", fontsize=24, weight="bold", color="#0F172A")
    ax.text(
        0.5,
        0.895,
        "从全NYC订单与库存快照开始，到小时级预测、库存风险识别和调度仿真输出",
        ha="center",
        va="center",
        fontsize=13,
        color="#475569",
    )

    node_map = {node["id"]: node for node in NODES}
    for src, dst in EDGES:
        s = node_map[src]
        d = node_map[dst]
        sx, sy = node_center(s)
        dx, dy = node_center(d)
        arrow = FancyArrowPatch(
            (sx, sy),
            (dx, dy),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color="#64748B",
            shrinkA=70,
            shrinkB=70,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    for node in NODES:
        patch = FancyBboxPatch(
            (node["x"], node["y"]),
            node["w"],
            node["h"],
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=2,
            edgecolor=node["edge"],
            facecolor=node["color"],
        )
        ax.add_patch(patch)
        ax.text(
            node["x"] + node["w"] / 2,
            node["y"] + node["h"] - 0.045,
            node["title"],
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
            color="#111827",
        )
        ax.text(
            node["x"] + node["w"] / 2,
            node["y"] + node["h"] / 2 - 0.025,
            node["body"],
            ha="center",
            va="center",
            fontsize=10.3,
            color="#334155",
            linespacing=1.35,
        )

    ax.text(0.05, 0.86, "数据处理链路", fontsize=12, weight="bold", color="#16A34A")
    ax.text(0.18, 0.50, "预测建模链路", fontsize=12, weight="bold", color="#2563EB")
    ax.text(0.68, 0.50, "库存与调度链路", fontsize=12, weight="bold", color="#EA580C")

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def svg_text(x, y, text, size=18, weight="normal", color="#1F2937", anchor="middle"):
    parts = []
    for i, line in enumerate(str(text).split("\n")):
        dy = i * size * 1.25
        parts.append(
            f'<text x="{x}" y="{y + dy}" text-anchor="{anchor}" '
            f'font-family="Microsoft YaHei, SimHei, Arial" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}">{escape(line)}</text>'
        )
    return "\n".join(parts)


def draw_svg(path):
    width, height = 1600, 900
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1600" height="900" fill="#F8FAFC"/>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#64748B"/></marker></defs>',
        svg_text(800, 70, "核心代码具体操作过程图", 34, "bold", "#0F172A"),
        svg_text(800, 115, "从全NYC订单与库存快照开始，到小时级预测、库存风险识别和调度仿真输出", 18, "normal", "#475569"),
    ]

    node_map = {node["id"]: node for node in NODES}
    for src, dst in EDGES:
        s = node_map[src]
        d = node_map[dst]
        sx, sy = node_center(s)
        dx, dy = node_center(d)
        parts.append(
            f'<line x1="{sx * width:.1f}" y1="{height - sy * height:.1f}" '
            f'x2="{dx * width:.1f}" y2="{height - dy * height:.1f}" '
            f'stroke="#64748B" stroke-width="3" marker-end="url(#arrow)" opacity="0.85"/>'
        )

    for node in NODES:
        x = node["x"] * width
        y = height - (node["y"] + node["h"]) * height
        w = node["w"] * width
        h = node["h"] * height
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="18" fill="{node["color"]}" stroke="{node["edge"]}" stroke-width="3"/>'
        )
        parts.append(svg_text(x + w / 2, y + 42, node["title"], 20, "bold", "#111827"))
        parts.append(svg_text(x + w / 2, y + 88, node["body"], 15, "normal", "#334155"))

    parts.append(svg_text(95, 155, "数据处理链路", 17, "bold", "#16A34A", anchor="start"))
    parts.append(svg_text(285, 450, "预测建模链路", 17, "bold", "#2563EB", anchor="start"))
    parts.append(svg_text(1090, 450, "库存与调度链路", 17, "bold", "#EA580C", anchor="start"))
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))


def write_mermaid(path):
    text = """# 核心代码具体操作过程图

```mermaid
flowchart LR
  A[1. 原始数据<br/>订单CSV / 库存快照 / 天气日历] --> B[2. 站点匹配与筛选<br/>build_topk_nyc_inventory_assets.py<br/>名称匹配库存，按全年流量选TopK]
  B --> C[3. 多图构建<br/>build_topk_graphs.py<br/>距离图、OD图、相关图、语义图]
  C --> D[4. 小时级样本<br/>prepare_topk_hourly_dataset.py<br/>X=过去168小时，Y=未来24小时]
  D --> E[5. 模型训练<br/>train_bike.py<br/>MSTGCN + FusionGraph]
  E --> F[6. 预测评估<br/>evaluate_bike_hourly_date_range.py<br/>MAE/RMSE/高峰窗口]
  F --> G[7. 库存风险输入<br/>build_dispatch_decision_input.py<br/>预测流量转库存轨迹和风险标签]
  G --> H[8. 调度仿真<br/>simulate_from_dispatch_decision_input.py<br/>来源站到目标站，损失对比]
  B --> I[当前调整<br/>Top300 -> Top150<br/>先跑通NYC高流量训练链路]
```
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main():
    ensure_dir(OUT_DIR)
    png_path = os.path.join(OUT_DIR, "核心代码_具体操作过程图.png")
    svg_path = os.path.join(OUT_DIR, "核心代码_具体操作过程图.svg")
    mermaid_path = os.path.join(OUT_DIR, "核心代码_具体操作过程图_Mermaid.md")
    draw_matplotlib_png(png_path)
    draw_svg(svg_path)
    write_mermaid(mermaid_path)
    print("Saved:", OUT_DIR)
    print("PNG:", png_path)
    print("SVG:", svg_path)
    print("Mermaid:", mermaid_path)


if __name__ == "__main__":
    main()
