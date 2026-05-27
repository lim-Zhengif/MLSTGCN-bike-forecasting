import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from hourly_pipeline_utils import aggregate_hourly_trip_counts


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_NAME = "bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_graph_train2025_hist168_tempp_hourly_netflow_pearson"


def detect_project_root(start_dir):
    current = Path(start_dir).resolve()
    while True:
        if (current / "data" / "graph").is_dir() and (current / "版本管理").is_dir():
            return current
        if current.parent == current:
            raise FileNotFoundError("Cannot locate project root from: %s" % start_dir)
        current = current.parent


PROJECT_ROOT = detect_project_root(SCRIPT_DIR)
DEFAULT_BASE_GRAPH_DIR = PROJECT_ROOT / "data" / "graph" / "bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_graph_train2025_hist168"


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def copy_spatial_embedding(base_graph_name, output_graph_name):
    se_dir = PROJECT_ROOT / "data" / "SE"
    source = se_dir / ("se_%s.csv" % base_graph_name)
    target = se_dir / ("se_%s.csv" % output_graph_name)
    if not source.exists():
        return None
    shutil.copy2(source, target)
    return target


def find_default_order_dir(trip_glob):
    matches = sorted(PROJECT_ROOT.rglob(trip_glob))
    if not matches:
        raise FileNotFoundError("No trip files matched under project root: %s" % trip_glob)

    by_parent = {}
    for path in matches:
        by_parent.setdefault(path.parent, 0)
        by_parent[path.parent] += 1
    return max(by_parent.items(), key=lambda item: item[1])[0]


def resolve_station_column(mapping_df):
    for candidate in ["station_name", "站点名称", "绔欑偣鍚嶇О"]:
        if candidate in mapping_df.columns:
            return candidate
    if "Node_ID" in mapping_df.columns and len(mapping_df.columns) > 1:
        return mapping_df.columns[1]
    raise KeyError("Cannot resolve station column from: %s" % list(mapping_df.columns))


def standardize_hourly_columns(hourly_df, station_names):
    columns = list(hourly_df.columns)
    if "datetime" not in columns:
        raise KeyError("aggregate_hourly_trip_counts output has no datetime column: %s" % columns)
    if len(columns) < 5:
        raise KeyError("aggregate_hourly_trip_counts output has too few columns: %s" % columns)

    station_set = set(station_names)
    station_col = None
    best_overlap = -1
    for col in columns:
        if col == "datetime":
            continue
        values = set(hourly_df[col].dropna().astype(str).head(max(len(station_names) * 4, 1000)))
        overlap = len(values & station_set)
        if overlap > best_overlap:
            best_overlap = overlap
            station_col = col

    if station_col is None or best_overlap <= 0:
        station_col = columns[1]

    value_cols = [col for col in columns if col not in {"datetime", station_col}]
    numeric_cols = [
        col for col in value_cols
        if pd.api.types.is_numeric_dtype(hourly_df[col])
    ]
    flow_cols = [col for col in numeric_cols if not hourly_df[col].dropna().between(0, 23).all()]
    if len(flow_cols) < 3:
        flow_cols = numeric_cols[:3]
    if len(flow_cols) < 3:
        raise KeyError("Cannot resolve outflow/inflow/netflow columns from: %s" % columns)

    out_col, in_col, net_col = flow_cols[:3]
    date_col = None
    for col in value_cols:
        if col in flow_cols:
            continue
        if pd.api.types.is_object_dtype(hourly_df[col]) or pd.api.types.is_string_dtype(hourly_df[col]):
            date_col = col
            break

    rename_map = {
        station_col: "station_name",
        out_col: "hourly_outflow",
        in_col: "hourly_inflow",
        net_col: "hourly_netflow",
    }
    if date_col:
        rename_map[date_col] = "date"
    standardized = hourly_df.rename(columns=rename_map).copy()
    if "date" not in standardized.columns:
        standardized["date"] = standardized["datetime"].dt.strftime("%Y-%m-%d")
    return standardized


def normalize_graph(corr):
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    corr = np.clip(corr, 0.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def build_correlation_from_frame(frame, station_names, index_col, value_col):
    pivot = frame.pivot_table(
        index=index_col,
        columns="station_name",
        values=value_col,
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=station_names).fillna(0.0)
    corr = pivot.corr(method="pearson").to_numpy(dtype=np.float32)
    return normalize_graph(corr)


def build_hourly_joint_graph(hourly_df, station_names):
    out_corr = build_correlation_from_frame(hourly_df, station_names, "datetime", "hourly_outflow")
    in_corr = build_correlation_from_frame(hourly_df, station_names, "datetime", "hourly_inflow")
    return normalize_graph(0.5 * out_corr + 0.5 * in_corr)


def build_graph(mode, hourly_df, station_names):
    if mode == "hourly_netflow":
        return build_correlation_from_frame(hourly_df, station_names, "datetime", "hourly_netflow")
    if mode == "hourly_outflow":
        return build_correlation_from_frame(hourly_df, station_names, "datetime", "hourly_outflow")
    if mode == "hourly_inflow":
        return build_correlation_from_frame(hourly_df, station_names, "datetime", "hourly_inflow")
    if mode == "hourly_joint":
        return build_hourly_joint_graph(hourly_df, station_names)
    if mode == "daily_netflow":
        daily_df = hourly_df.groupby(["date", "station_name"], as_index=False)["hourly_netflow"].sum()
        return build_correlation_from_frame(daily_df.rename(columns={"date": "datetime"}), station_names, "datetime", "hourly_netflow")
    raise ValueError("Unsupported mode: %s" % mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_graph_dir", default=str(DEFAULT_BASE_GRAPH_DIR))
    parser.add_argument("--order_dir", default=None)
    parser.add_argument("--trip_glob", default="2025*-citibike-tripdata_*.csv")
    parser.add_argument(
        "--output_name",
        default=DEFAULT_OUTPUT_NAME,
        help="Name of the new graph directory under data/graph.",
    )
    parser.add_argument(
        "--mode",
        choices=["daily_netflow", "hourly_netflow", "hourly_outflow", "hourly_inflow", "hourly_joint"],
        default="hourly_netflow",
        help="Which temporal-demand similarity graph to build for tempp_bike.npy.",
    )
    parser.add_argument("--summary_name", default="tempp_variant_graph_summary.json")
    args = parser.parse_args()

    base_graph_dir = resolve_path(args.base_graph_dir)
    order_dir = resolve_path(args.order_dir) if args.order_dir else find_default_order_dir(args.trip_glob)
    output_graph_dir = PROJECT_ROOT / "data" / "graph" / args.output_name
    output_graph_dir.mkdir(parents=True, exist_ok=True)

    base_mapping_path = base_graph_dir / "selected_node_mapping.csv"
    if not base_mapping_path.exists():
        raise FileNotFoundError("Missing base mapping file: %s" % base_mapping_path)

    mapping_df = pd.read_csv(base_mapping_path).sort_values("Node_ID").reset_index(drop=True)
    station_col = resolve_station_column(mapping_df)
    station_names = mapping_df[station_col].astype(str).tolist()

    trip_files = sorted(order_dir.glob(args.trip_glob))
    if not trip_files:
        raise FileNotFoundError("No trip files matched: %s" % (order_dir / args.trip_glob))

    hourly_df, full_hours = aggregate_hourly_trip_counts([str(path) for path in trip_files], station_names)
    hourly_df = standardize_hourly_columns(hourly_df, station_names)
    graph = build_graph(args.mode, hourly_df, station_names)

    for item in base_graph_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, output_graph_dir / item.name)

    copy_spatial_embedding(base_graph_dir.name, output_graph_dir.name)

    np.save(output_graph_dir / "tempp_bike.npy", graph)
    np.save(output_graph_dir / ("graph_demand_correlation_%s.npy" % args.mode), graph)

    summary = {
        "base_graph_dir": str(base_graph_dir),
        "output_graph_dir": str(output_graph_dir),
        "order_dir": str(order_dir),
        "trip_glob": args.trip_glob,
        "mode": args.mode,
        "node_count": int(len(mapping_df)),
        "hour_range": [str(full_hours.min()), str(full_hours.max())] if len(full_hours) else None,
        "tempp_stats": {
            "min": float(graph.min()),
            "max": float(graph.max()),
            "mean": float(graph.mean()),
            "density": float((graph > 0).mean()),
        },
        "trip_file_count": int(len(trip_files)),
    }
    (output_graph_dir / args.summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
