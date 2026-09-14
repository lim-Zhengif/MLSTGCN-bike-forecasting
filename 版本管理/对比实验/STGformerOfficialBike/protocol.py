import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


UPSTREAM_COMMIT = "7c141029328a91cfeb78c1d8a0bfaa26997d30a0"
MODEL_ID = "stgformer_official_core_bike_b0"
DEFAULT_WANDB_PROJECT = "top150_rolling6h_model_compare"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def compute_metrics(pred, true, mape_eps=1.0):
    pred = np.asarray(pred)
    true = np.asarray(true)
    error = np.abs(pred - true)
    return {
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "mape": float((error / np.maximum(np.abs(true), float(mape_eps))).mean()),
    }


def horizon_metrics(pred, true, mape_eps=1.0):
    rows = []
    error = np.abs(pred - true)
    squared = (pred - true) ** 2
    denominator = np.maximum(np.abs(true), float(mape_eps))
    for index in range(pred.shape[1]):
        rows.append(
            {
                "horizon": index + 1,
                "mae_avg": float(error[:, index].mean()),
                "mae_out": float(error[:, index, :, 0].mean()),
                "mae_in": float(error[:, index, :, 1].mean()),
                "rmse_avg": float(np.sqrt(squared[:, index].mean())),
                "mape_avg": float((error[:, index] / denominator[:, index]).mean()),
            }
        )
    return pd.DataFrame(rows)


def anchor_metrics(pred, true, anchor_hours):
    rows = []
    error = np.abs(pred - true)
    anchor_hours = np.asarray(anchor_hours)
    for anchor in sorted(set(anchor_hours.tolist())):
        mask = anchor_hours == anchor
        rows.append(
            {
                "anchor_hour": int(anchor),
                "samples": int(mask.sum()),
                "mae_avg": float(error[mask].mean()),
                "mae_out": float(error[mask, :, :, 0].mean()),
                "mae_in": float(error[mask, :, :, 1].mean()),
            }
        )
    return pd.DataFrame(rows)


def audit_anchor_coverage(
    sample_datetimes,
    start_date,
    end_date,
    anchor_hours,
    target_start_offset,
    pred_len,
):
    """Describe exact rolling-anchor coverage and the final target hour required."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must not be earlier than start_date")
    anchors = sorted({int(value) for value in anchor_hours})
    if not anchors or anchors[0] < 0 or anchors[-1] > 23:
        raise ValueError("anchor_hours must contain values from 0 through 23")
    if int(target_start_offset) < 0 or int(pred_len) <= 0:
        raise ValueError("target_start_offset must be non-negative and pred_len must be positive")

    expected = []
    for date_value in pd.date_range(start, end, freq="D"):
        for anchor_hour in anchors:
            expected.append(date_value + pd.Timedelta(hours=anchor_hour))
    # pandas 1.x rejects numpy.str_ even though it accepts the equivalent
    # built-in str. Sample metadata comes from a NumPy string array here.
    provided = [pd.Timestamp(str(value)) for value in sample_datetimes]
    provided_set = set(provided)
    missing = [value for value in expected if value not in provided_set]
    duplicate_count = len(provided) - len(provided_set)
    latest_target = expected[-1] + pd.Timedelta(
        hours=int(target_start_offset) + int(pred_len) - 1
    )
    return {
        "expected_count": len(expected),
        "provided_count": len(provided),
        "missing_anchor_datetimes": [value.strftime("%Y-%m-%d %H:%M:%S") for value in missing],
        "duplicate_anchor_count": int(duplicate_count),
        "latest_requested_anchor_datetime": expected[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "latest_required_target_datetime": latest_target.strftime("%Y-%m-%d %H:%M:%S"),
    }


def init_wandb_run(args, output_dir):
    if args.logger != "wandb":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logger requested, but wandb is unavailable") from exc
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or Path(output_dir).name,
        config=vars(args),
        tags=tags or None,
    )


def compute_stats_float64(x_train, y_train):
    feature_mean = x_train.mean(axis=(0, 1, 2), keepdims=True, dtype=np.float64)
    feature_std = x_train.std(axis=(0, 1, 2), keepdims=True, dtype=np.float64)
    target_mean = y_train.mean(axis=(0, 1, 2), keepdims=True, dtype=np.float64)
    target_std = y_train.std(axis=(0, 1, 2), keepdims=True, dtype=np.float64)
    feature_std = np.where(feature_std == 0.0, 1.0, feature_std)
    target_std = np.where(target_std == 0.0, 1.0, target_std)
    return feature_mean, feature_std, target_mean, target_std


def normalize_features(values, mean, std):
    return ((values.astype(np.float64) - mean) / std).astype(np.float32)


def normalize_targets(values, mean, std):
    return ((values.astype(np.float64) - mean) / std).astype(np.float32)


def apply_log1p_transform_inplace(feature_values, feature_cols, selected_cols):
    """Apply the dataset log transform without allocating another full tensor."""
    if feature_values is None or not selected_cols:
        return []
    if not np.issubdtype(feature_values.dtype, np.floating):
        raise TypeError("feature_values must use a floating dtype for in-place log1p")

    applied_cols = []
    for feature_name in selected_cols:
        if feature_name not in feature_cols:
            continue
        feature_index = feature_cols.index(feature_name)
        feature_slice = feature_values[..., feature_index]
        if np.nanmin(feature_slice) < 0:
            continue
        np.log1p(feature_slice, out=feature_slice)
        applied_cols.append(feature_name)
    return applied_cols


def row_normalize(matrix, add_self=True):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Graph must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Graph contains NaN or Inf")
    matrix = np.maximum(matrix, 0.0)
    if add_self:
        matrix = matrix + np.eye(matrix.shape[0], dtype=np.float64)
    row_sum = matrix.sum(axis=1, keepdims=True)
    return (matrix / np.maximum(row_sum, 1e-12)).astype(np.float32)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_records(records):
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def npz_list(data, key):
    return [str(value) for value in data[key].tolist()] if key in data else []


def load_npz_metadata(data_dir):
    train_path = Path(data_dir) / "train.npz"
    train = np.load(train_path, allow_pickle=True)
    mean, std, target_mean, target_std = compute_stats_float64(train["x"], train["y"])
    return {
        "feature_mean": mean,
        "feature_std": std,
        "target_mean": target_mean,
        "target_std": target_std,
        "hist_len": int(train["x"].shape[1]),
        "num_nodes": int(train["x"].shape[2]),
        "in_dim": int(train["x"].shape[3]),
        "pred_len": int(train["y"].shape[1]),
        "out_dim": int(train["y"].shape[3]),
        "input_feature_cols": npz_list(train, "input_feature_cols"),
        "history_feature_cols": npz_list(train, "history_feature_cols"),
        "known_future_feature_cols": npz_list(train, "known_future_feature_cols"),
        "log1p_feature_cols": npz_list(train, "log1p_feature_cols"),
        "target_cols": npz_list(train, "target_cols"),
        "train_npz_sha256": sha256_file(train_path),
    }


def audit_graph_contract(graph_dir, graph_name, num_nodes):
    graph_dir = Path(graph_dir)
    graph_path = graph_dir / (str(graph_name) + ".npy")
    mapping_path = graph_dir / "selected_node_mapping.csv"
    graph = np.load(graph_path)
    if graph.shape != (num_nodes, num_nodes):
        raise ValueError("Graph shape %s does not match %d nodes" % (graph.shape, num_nodes))
    normalized = row_normalize(graph, add_self=True)
    if not np.allclose(normalized.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Normalized graph rows do not sum to one")
    mapping = pd.read_csv(mapping_path).sort_values("Node_ID").reset_index(drop=True)
    if len(mapping) != num_nodes:
        raise ValueError("Node mapping has %d rows, expected %d" % (len(mapping), num_nodes))
    station_col = "站点名称" if "站点名称" in mapping.columns else None
    records = []
    for _, row in mapping.iterrows():
        record = {"Node_ID": int(row["Node_ID"])}
        if station_col:
            record["station_name"] = str(row[station_col])
        records.append(record)
    return {
        "graph_name": str(graph_name),
        "graph_shape": list(graph.shape),
        "graph_sha256": sha256_file(graph_path),
        "mapping_sha256": sha256_file(mapping_path),
        "node_signature": sha256_records(records),
        "normalized_row_sum_min": float(normalized.sum(axis=1).min()),
        "normalized_row_sum_max": float(normalized.sum(axis=1).max()),
        "b0_graph_role": "node_order_audit_only",
        "model_graph_source": "learned_adaptive_embedding",
    }


def environment_info():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def truth_key_signature(frame):
    columns = [
        "date",
        "sample_datetime",
        "target_start_datetime",
        "anchor_hour",
        "horizon",
        "Node_ID",
        "true_out",
        "true_in",
    ]
    hashes = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype=np.uint64)
    return {
        "sum_uint64": str(int(hashes.sum(dtype=np.uint64))),
        "xor_uint64": str(int(np.bitwise_xor.reduce(hashes))),
    }
