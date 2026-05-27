import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_anchor_hours(value):
    anchors = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            anchors.append(int(item))
    if not anchors:
        raise ValueError("--anchor_hours must include at least one hour.")
    return anchors


def resolve_station_col(mapping_df):
    for candidate in ["station_name", "站点名称"]:
        if candidate in mapping_df.columns:
            return candidate
    for column in mapping_df.columns:
        if column != "Node_ID" and mapping_df[column].dtype == object:
            return column
    raise ValueError("Cannot resolve station name column in selected_node_mapping.csv.")


def target_hours_for_anchor(anchor_hour, pred_len, target_start_offset):
    return [int((anchor_hour + target_start_offset + horizon) % 24) for horizon in range(pred_len)]


def load_mapping(graph_dir):
    mapping_path = graph_dir / "selected_node_mapping.csv"
    mapping_df = pd.read_csv(mapping_path)
    station_col = resolve_station_col(mapping_df)
    mapping_df = mapping_df.sort_values("Node_ID").reset_index(drop=True)
    station_to_node = {str(row[station_col]): int(row["Node_ID"]) for _, row in mapping_df.iterrows()}
    return mapping_df, station_to_node


def build_anchor_matrices(trip_files, station_to_node, anchor_hours, pred_len, target_start_offset, chunksize):
    node_num = len(station_to_node)
    matrices = {anchor: np.zeros((node_num, node_num), dtype=np.float64) for anchor in anchor_hours}
    anchor_hour_sets = {
        anchor: set(target_hours_for_anchor(anchor, pred_len, target_start_offset))
        for anchor in anchor_hours
    }
    all_target_hours = sorted(set().union(*anchor_hour_sets.values()))

    usecols = ["started_at", "start_station_name", "end_station_name"]
    for trip_file in trip_files:
        print("Reading:", trip_file)
        for chunk in pd.read_csv(trip_file, usecols=usecols, chunksize=chunksize):
            chunk = chunk.dropna(subset=usecols)
            chunk = chunk[
                chunk["start_station_name"].astype(str).isin(station_to_node)
                & chunk["end_station_name"].astype(str).isin(station_to_node)
            ]
            if chunk.empty:
                continue
            started_at = pd.to_datetime(chunk["started_at"], errors="coerce")
            chunk = chunk[started_at.notna()].copy()
            if chunk.empty:
                continue
            chunk["hour"] = started_at[started_at.notna()].dt.hour.to_numpy()
            chunk = chunk[chunk["hour"].isin(all_target_hours)]
            if chunk.empty:
                continue
            chunk["src"] = chunk["start_station_name"].astype(str).map(station_to_node)
            chunk["dst"] = chunk["end_station_name"].astype(str).map(station_to_node)
            grouped = chunk.groupby(["hour", "src", "dst"]).size().reset_index(name="count")

            for anchor, hour_set in anchor_hour_sets.items():
                selected = grouped[grouped["hour"].isin(hour_set)]
                if selected.empty:
                    continue
                np.add.at(
                    matrices[anchor],
                    (selected["src"].to_numpy(dtype=np.int64), selected["dst"].to_numpy(dtype=np.int64)),
                    selected["count"].to_numpy(dtype=np.float64),
                )
    return matrices, anchor_hour_sets


def counts_to_heuristic(count_matrix):
    positive = count_matrix[count_matrix > 0]
    mean_positive = float(np.mean(positive)) if positive.size else 1.0
    graph = (1.0 - np.exp(-count_matrix / max(mean_positive, 1e-6))).astype(np.float32)
    np.fill_diagonal(graph, 1.0)
    return graph


def copy_base_graph_files(base_graph_dir, output_graph_dir):
    output_graph_dir.mkdir(parents=True, exist_ok=True)
    for graph_file in base_graph_dir.glob("*"):
        if graph_file.is_file():
            shutil.copy2(graph_file, output_graph_dir / graph_file.name)


def copy_spatial_embedding(base_graph_name, output_graph_name):
    se_dir = PROJECT_ROOT / "data" / "SE"
    source = se_dir / ("se_%s.csv" % base_graph_name)
    target = se_dir / ("se_%s.csv" % output_graph_name)
    if source.exists():
        shutil.copy2(source, target)
        return str(target)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_graph_dir",
        default="data/graph/bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20_train2025_hist168",
    )
    parser.add_argument("--trip_subdir", default="二月份数据处理/纽约单车订单数据")
    parser.add_argument("--trip_pattern", default="2025*-citibike-tripdata_*.csv")
    parser.add_argument("--output_name", default="bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_graph_train2025_hist168")
    parser.add_argument("--anchor_hours", default="0,6,12,16,20")
    parser.add_argument("--pred_len", type=int, default=6)
    parser.add_argument("--target_start_offset", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=500000)
    args = parser.parse_args()

    base_graph_dir = resolve_project_path(args.base_graph_dir)
    trip_dir = resolve_project_path(args.trip_subdir)
    output_graph_dir = PROJECT_ROOT / "data" / "graph" / args.output_name
    anchor_hours = parse_anchor_hours(args.anchor_hours)

    if not base_graph_dir.exists():
        raise FileNotFoundError("Missing base graph dir: %s" % base_graph_dir)
    trip_files = sorted(glob.glob(str(trip_dir / args.trip_pattern)))
    if not trip_files:
        raise FileNotFoundError("No trip files matched: %s" % (trip_dir / args.trip_pattern))

    mapping_df, station_to_node = load_mapping(base_graph_dir)
    copy_base_graph_files(base_graph_dir, output_graph_dir)
    matrices, anchor_hour_sets = build_anchor_matrices(
        trip_files=trip_files,
        station_to_node=station_to_node,
        anchor_hours=anchor_hours,
        pred_len=args.pred_len,
        target_start_offset=args.target_start_offset,
        chunksize=args.chunksize,
    )

    summary_rows = []
    for anchor in anchor_hours:
        counts = matrices[anchor]
        graph = counts_to_heuristic(counts)
        graph_name = "od%02d" % anchor
        np.save(output_graph_dir / ("%s.npy" % graph_name), graph)
        summary_rows.append(
            {
                "graph": graph_name,
                "anchor_hour": int(anchor),
                "target_hours": sorted(int(hour) for hour in anchor_hour_sets[anchor]),
                "total_trips": float(counts.sum()),
                "nonzero_edges": int((counts > 0).sum()),
                "mean_positive_count": float(counts[counts > 0].mean()) if (counts > 0).any() else 0.0,
            }
        )

    se_path = copy_spatial_embedding(base_graph_dir.name, args.output_name)
    summary = {
        "base_graph_dir": str(base_graph_dir),
        "output_graph_dir": str(output_graph_dir),
        "trip_files": trip_files,
        "node_count": int(len(mapping_df)),
        "anchor_hours": anchor_hours,
        "pred_len": int(args.pred_len),
        "target_start_offset": int(args.target_start_offset),
        "spatial_embedding": se_path,
        "graphs": summary_rows,
    }
    with open(output_graph_dir / "anchor_hour_od_graph_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("Built anchor-hour OD graph dir:", output_graph_dir)
    print("Node count:", len(mapping_df))
    print("Trip files:", len(trip_files))
    for row in summary_rows:
        print(row["graph"], "target_hours=", row["target_hours"], "total_trips=", int(row["total_trips"]))


if __name__ == "__main__":
    main()
