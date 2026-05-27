import argparse
import os

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c


def build_spatial_graph(mapping_df):
    lat = mapping_df["lat"].fillna(mapping_df["lat"].median()).astype(float).to_numpy()
    lon = mapping_df["lon"].fillna(mapping_df["lon"].median()).astype(float).to_numpy()
    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :]).astype(np.float32)
    positive = dist[dist > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    graph = np.exp(-dist / max(scale, 1e-6)).astype(np.float32)
    np.fill_diagonal(graph, 1.0)
    return graph


def build_od_graphs(mapping_df, global_od_df):
    station_to_id = dict(zip(mapping_df["station_name"], mapping_df["Node_ID"]))
    node_num = len(mapping_df)
    od_matrix = np.zeros((node_num, node_num), dtype=np.float32)
    for row in global_od_df.itertuples(index=False):
        src = station_to_id.get(row.start_station_name)
        dst = station_to_id.get(row.end_station_name)
        if src is None or dst is None:
            continue
        od_matrix[src, dst] += np.float32(row.total_trips)
    row_sums = od_matrix.sum(axis=1, keepdims=True)
    transition = np.divide(od_matrix, row_sums, out=np.zeros_like(od_matrix), where=row_sums != 0)
    np.fill_diagonal(transition, 1.0)

    positive = od_matrix[od_matrix > 0]
    mean_positive = float(np.mean(positive)) if positive.size else 1.0
    heuristic = (1.0 - np.exp(-od_matrix / max(mean_positive, 1e-6))).astype(np.float32)
    np.fill_diagonal(heuristic, 1.0)
    return transition, heuristic


def build_demand_correlation_graph(mapping_df, daily_netflow_df):
    pivot = daily_netflow_df.pivot_table(
        index="date",
        columns="station_name",
        values="daily_netflow",
        aggfunc="sum",
    )
    ordered_names = mapping_df["station_name"].tolist()
    pivot = pivot.reindex(columns=ordered_names).fillna(0.0)
    corr = pivot.corr(method="pearson").to_numpy(dtype=np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, 0.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def build_semantic_graph(mapping_df):
    capacity = mapping_df["capacity"].fillna(mapping_df["capacity"].median()).astype(float).to_numpy()
    lat = mapping_df["lat"].fillna(mapping_df["lat"].median()).astype(float).to_numpy()
    lon = mapping_df["lon"].fillna(mapping_df["lon"].median()).astype(float).to_numpy()
    features = np.stack([capacity, lat, lon], axis=1).astype(np.float32)
    features = features - features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    features = features / std
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normalized = features / norm
    semantic = normalized @ normalized.T
    semantic = np.clip((semantic + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(semantic, 1.0)
    return semantic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_dir", default=os.path.join("二月份数据处理", "nyc_top300_inventory_validation"))
    args = parser.parse_args()

    asset_dir = os.path.abspath(args.asset_dir)
    mapping_df = pd.read_csv(os.path.join(asset_dir, "GNN_Node_Mapping_topk.csv"))
    global_od_df = pd.read_csv(os.path.join(asset_dir, "global_od_pairs_topk.csv"))
    daily_netflow_df = pd.read_csv(os.path.join(asset_dir, "daily_netflow_topk.csv"))

    spatial = build_spatial_graph(mapping_df)
    transition, heuristic = build_od_graphs(mapping_df, global_od_df)
    correlation = build_demand_correlation_graph(mapping_df, daily_netflow_df)
    semantic = build_semantic_graph(mapping_df)

    np.save(os.path.join(asset_dir, "graph_spatial_distance.npy"), spatial)
    np.save(os.path.join(asset_dir, "graph_od_transition.npy"), transition)
    np.save(os.path.join(asset_dir, "graph_heuristic.npy"), heuristic)
    np.save(os.path.join(asset_dir, "graph_demand_correlation.npy"), correlation)
    np.save(os.path.join(asset_dir, "graph_poi_semantic.npy"), semantic)

    print("Built top-k graph files.")
    print("Asset dir:", asset_dir)
    print("Node count:", len(mapping_df))


if __name__ == "__main__":
    main()
