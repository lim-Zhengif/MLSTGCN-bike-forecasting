import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "temporal_data" / "bike_hourly_safe_inventory" / "test.npz"
OUT_DIR = ROOT / "分析结果" / "2026-04-24_JC数据集历史预测窗口图"

BLUE = "#3159C9"
GREEN = "#2E9D42"
BORDER = "#2F58D5"
TEXT = "#163A9F"
GRID = "#E8EEF9"
PANEL_BG = "#FBFDFF"


def scale_points(values, x0, y0, width, height):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        finite = np.array([0.0, 1.0])
    lo = float(finite.min())
    hi = float(finite.max())
    if abs(hi - lo) < 1e-9:
        hi = lo + 1.0
    xs = np.linspace(x0, x0 + width, len(values))
    ys = y0 + height - (values - lo) / (hi - lo) * height
    return list(zip(xs, ys))


def path_from_points(points):
    if not points:
        return ""
    parts = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    parts.extend(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    return " ".join(parts)


def polyline_path(values_a, values_b, chart):
    x0, y0, w, h = chart
    both = np.concatenate([np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)])
    lo = float(np.nanmin(both))
    hi = float(np.nanmax(both))
    if abs(hi - lo) < 1e-9:
        hi = lo + 1.0

    def project(values):
        values = np.asarray(values, dtype=float)
        xs = np.linspace(x0, x0 + w, len(values))
        ys = y0 + h - (values - lo) / (hi - lo) * h
        return path_from_points(list(zip(xs, ys)))

    return project(values_a), project(values_b)


def make_panel(title, left_ticks, bottom_ticks, out_flow, in_flow, filename, future=False):
    width, height = 380, 168
    chart = (52, 44, 268, 60)
    out_path, in_path = polyline_path(out_flow, in_flow, chart)

    tick_xs = np.linspace(chart[0], chart[0] + chart[2], len(bottom_ticks))
    tick_svg = []
    for x, label in zip(tick_xs, bottom_ticks):
        tick_svg.append(f'<text x="{x:.1f}" y="128" text-anchor="middle" class="tick">{label}</text>')

    dots = " ".join([
        '<circle cx="342" cy="82" r="2.1" fill="#1F2937"/>',
        '<circle cx="352" cy="82" r="2.1" fill="#1F2937"/>',
        '<circle cx="362" cy="82" r="2.1" fill="#1F2937"/>',
    ])

    arrow = ""
    if future:
        arrow = '<path d="M190 10 L190 31" stroke="#1F4FD5" stroke-width="8" stroke-linecap="round"/><path d="M177 28 L190 44 L203 28 Z" fill="#1F4FD5"/>'

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <style>
      .tick {{ font-family: Arial, sans-serif; font-size: 12px; fill: #111827; }}
    </style>
  </defs>
  {arrow}
  <rect x="8" y="18" width="364" height="132" rx="8" fill="{PANEL_BG}" stroke="none"/>
  <rect x="{chart[0]}" y="{chart[1]}" width="{chart[2]}" height="{chart[3]}" fill="white" stroke="#CDD6E5" stroke-width="1.2"/>
  <path d="M {chart[0]:.1f} {chart[1] + chart[3] / 2:.1f} H {chart[0] + chart[2]:.1f}" stroke="{GRID}" stroke-width="1"/>
  <path d="{out_path}" fill="none" stroke="{BLUE}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>
  <path d="{in_path}" fill="none" stroke="{GREEN}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>
  {''.join(tick_svg)}
  {dots}
</svg>
'''
    (OUT_DIR / filename).write_text(svg, encoding="utf-8")


def make_combined():
    h = (OUT_DIR / "jc_history_window_168h.svg").read_text(encoding="utf-8")
    f = (OUT_DIR / "jc_prediction_range_24h.svg").read_text(encoding="utf-8")
    h_inner = h.split(">", 1)[1].rsplit("</svg>", 1)[0]
    f_inner = f.split(">", 1)[1].rsplit("</svg>", 1)[0]
    combined = f'''<svg xmlns="http://www.w3.org/2000/svg" width="780" height="190" viewBox="0 0 780 190">
  <rect width="780" height="190" fill="white"/>
  <g transform="translate(0,8)">
    {h_inner}
  </g>
  <g transform="translate(390,8)">
    {f_inner}
  </g>
</svg>
'''
    (OUT_DIR / "jc_history_prediction_windows_combined.svg").write_text(combined, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA_PATH, allow_pickle=True)
    dates = data["sample_dates"]
    matches = np.where(dates == "2026-02-01")[0]
    sample_idx = int(matches[0]) if len(matches) else len(dates) // 2

    x = data["x"][sample_idx]
    y = data["y"][sample_idx]

    # Historical flow features were saved with log1p transformation.
    hist_out = np.expm1(x[:, :, 0]).mean(axis=1)
    hist_in = np.expm1(x[:, :, 1]).mean(axis=1)
    future_out = y[:, :, 0].mean(axis=1)
    future_in = y[:, :, 1].mean(axis=1)

    make_panel(
        "历史窗口：过去 168 小时",
        ("流出", "流入"),
        ("t-167", "t-120", "t-72", "t-24", "t"),
        hist_out,
        hist_in,
        "jc_history_window_168h.svg",
    )
    make_panel(
        "预测范围：未来 24 小时",
        ("流出", "流入"),
        ("t+1", "t+6", "t+12", "t+18", "t+24"),
        future_out,
        future_in,
        "jc_prediction_range_24h.svg",
        future=True,
    )
    make_combined()

    meta = {
        "data_path": str(DATA_PATH),
        "sample_date": str(dates[sample_idx]),
        "sample_index": sample_idx,
        "node_count": int(x.shape[1]),
        "history_hours": int(x.shape[0]),
        "future_hours": int(y.shape[0]),
        "curve_definition": "station-mean hourly outflow/inflow; historical x flow features are expm1-restored from log1p.",
        "outputs": [
            "jc_history_window_168h.svg",
            "jc_prediction_range_24h.svg",
            "jc_history_prediction_windows_combined.svg",
        ],
    }
    (OUT_DIR / "说明.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
