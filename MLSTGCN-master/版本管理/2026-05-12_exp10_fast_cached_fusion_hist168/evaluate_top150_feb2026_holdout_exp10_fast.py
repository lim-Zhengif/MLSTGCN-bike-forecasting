import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EXP10_FAST_VERSION_DIR = SCRIPT_DIR
PIPELINE_VERSION_DIR = PROJECT_ROOT / "\u7248\u672c\u7ba1\u7406" / "2026-04-23_top300\u5168NYC\u5e93\u5b58\u5feb\u7167\u901a\u8def\u9a8c\u8bc1"
for source_dir in [PIPELINE_VERSION_DIR, EXP10_FAST_VERSION_DIR]:
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from hourly_pipeline_utils import (  # noqa: E402
    aggregate_hourly_trip_counts,
    apply_log1p_transform,
    build_hourly_feature_frame,
    build_hourly_samples,
    load_daily_feature_table,
)
from datasets.bike import BikeGraph  # noqa: E402
from models.D2STGNN import D2STGNNFusionBackbone  # noqa: E402
from models.MSTGCN import MSTGCN_submodule  # noqa: E402
from models.fusiongraph import FusionGraphModel  # noqa: E402
import models.fusiongraph as fusiongraph_module  # noqa: E402
from prepare_topk_hourly_dataset_rolling_anchors import (  # noqa: E402
    build_hourly_samples_for_anchors,
    parse_anchor_hours,
)


STATION_COL = "\u7ad9\u70b9\u540d\u79f0"
PIPELINE_STATION_COL = "\u7ad9\u70b9\u540d\u79f0"
ANALYSIS_DIR = "\u5206\u6790\u7ed3\u679c"
FEB_DATA_DIR = "\u4e8c\u6708\u4efd\u6570\u636e\u5904\u7406"
ORDER_SUBDIR = "\u7ebd\u7ea6\u5355\u8f66\u8ba2\u5355\u6570\u636e"


class EvalWrapper(nn.Module):
    def __init__(self, device, fusiongraph, data_config, model_config, categorical_feature_configs):
        super().__init__()
        self.fusiongraph = fusiongraph
        if model_config.get("use", "MSTGCN") == "MSTGCN":
            self.model = MSTGCN_submodule(
                device,
                fusiongraph,
                data_config["in_dim"],
                data_config["hist_len"],
                data_config["pred_len"],
                data_config["out_dim"],
                categorical_feature_configs=categorical_feature_configs,
                cheb_k=model_config["cheb_k"],
                nb_block=model_config["nb_block"],
                nb_chev_filter=model_config["nb_chev_filter"],
                nb_time_filter=model_config["nb_time_filter"],
                time_kernel_size=model_config["time_kernel_size"],
                channel_attention=model_config.get("channel_attention", False),
                channel_attention_reduction=model_config.get("channel_attention_reduction", 4),
                trend_alignment_decoder=model_config.get("trend_alignment_decoder", False),
                trend_time_feature_index=model_config.get("trend_time_feature_index", -1),
                trend_time_feature_mean=model_config.get("trend_time_feature_mean", 0.0),
                trend_time_feature_std=model_config.get("trend_time_feature_std", 1.0),
                trend_time_cycle=model_config.get("trend_time_cycle", 24),
                trend_time_embed_dim=model_config.get("trend_time_embed_dim", 16),
                trend_attention_heads=model_config.get("trend_attention_heads", 4),
                trend_dropout=model_config.get("trend_dropout", 0.1),
                ast_tcn_residual=model_config.get("ast_tcn_residual", False),
                ast_tcn_hidden_dim=model_config.get("ast_tcn_hidden_dim", 32),
                ast_tcn_layers=model_config.get("ast_tcn_layers", 4),
                ast_tcn_kernel_size=model_config.get("ast_tcn_kernel_size", 3),
                ast_tcn_dilation_base=model_config.get("ast_tcn_dilation_base", 2),
                ast_tcn_heads=model_config.get("ast_tcn_heads", 4),
                ast_tcn_dropout=model_config.get("ast_tcn_dropout", 0.1),
                ast_tcn_residual_init=model_config.get("ast_tcn_residual_init", 0.05),
                ast_tcn_bounded_alpha=model_config.get("ast_tcn_bounded_alpha", False),
                ast_tcn_alpha_max=model_config.get("ast_tcn_alpha_max", 0.1),
                ast_tcn_horizon_alpha=model_config.get("ast_tcn_horizon_alpha", False),
                ast_tcn_zero_init=model_config.get("ast_tcn_zero_init", False),
                ast_tcn_residual_gate=model_config.get("ast_tcn_residual_gate", False),
                ast_tcn_residual_gate_hidden_dim=model_config.get("ast_tcn_residual_gate_hidden_dim", 16),
                ast_tcn_residual_gate_init=model_config.get("ast_tcn_residual_gate_init", 0.2),
                ast_tcn_edge_bias=model_config.get("ast_tcn_edge_bias", False),
                ast_tcn_edge_bias_init=model_config.get("ast_tcn_edge_bias_init", 0.1),
                ast_tcn_edge_bias_eps=model_config.get("ast_tcn_edge_bias_eps", 1e-6),
            )
        elif model_config.get("use") == "D2STGNN":
            self.model = D2STGNNFusionBackbone(
                device,
                fusiongraph,
                data_config["in_dim"],
                data_config["hist_len"],
                data_config["pred_len"],
                data_config["out_dim"],
                categorical_feature_configs=categorical_feature_configs,
                hidden_dim=model_config["d2_hidden_dim"],
                num_layers=model_config["d2_num_layers"],
                dropout=model_config["d2_dropout"],
                dilation_cycle=model_config["d2_dilation_cycle"],
                kernel_size=model_config["d2_kernel_size"],
                gcn_order=model_config["d2_gcn_order"],
                node_embed_dim=model_config["d2_node_embed_dim"],
                adaptive_adj=model_config["d2_adaptive_adj"],
                use_reverse=model_config["d2_use_reverse"],
                fusion_init=model_config["d2_fusion_init"],
            )
        else:
            raise NotImplementedError("Unsupported model_use: %s" % model_config.get("use"))

    def forward(self, x, anchor_hours=None):
        if isinstance(self.model, MSTGCN_submodule):
            return self.model(x, anchor_hours=anchor_hours)
        return self.model(x)


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_npz_metadata(data_dir):
    train_npz = np.load(data_dir / "train.npz", allow_pickle=True)
    train_x = train_npz["x"].astype(np.float32)
    train_y = train_npz["y"].astype(np.float32)
    feature_std = train_x.std(axis=(0, 1, 2), keepdims=True)
    target_std = train_y.std(axis=(0, 1, 2), keepdims=True)
    return {
        "input_feature_cols": [str(v) for v in train_npz["input_feature_cols"].tolist()],
        "history_feature_cols": [str(v) for v in train_npz["history_feature_cols"].tolist()],
        "known_future_feature_cols": [str(v) for v in train_npz["known_future_feature_cols"].tolist()],
        "log1p_feature_cols": [str(v) for v in train_npz["log1p_feature_cols"].tolist()],
        "target_cols": [str(v) for v in train_npz["target_cols"].tolist()],
        "feature_mean": train_x.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32),
        "feature_std": np.where(feature_std == 0, 1.0, feature_std).astype(np.float32),
        "target_mean": train_y.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32),
        "target_std": np.where(target_std == 0, 1.0, target_std).astype(np.float32),
        "hist_len": int(train_npz["x"].shape[1]),
        "pred_len": int(train_npz["y"].shape[1]),
    }


def build_categorical_feature_configs(metadata, weekday_embed_dim):
    if weekday_embed_dim <= 0:
        return []
    feature_mean = metadata["feature_mean"].reshape(-1)
    feature_std = metadata["feature_std"].reshape(-1)
    configs = []
    for idx, name in enumerate(metadata["input_feature_cols"]):
        if name in {"\u661f\u671f\u51e0", "future_\u661f\u671f\u51e0"}:
            configs.append(
                {
                    "index": idx,
                    "num_embeddings": 7,
                    "embedding_dim": weekday_embed_dim,
                    "mean": float(feature_mean[idx]),
                    "std": float(feature_std[idx] if feature_std[idx] != 0 else 1.0),
                    "name": name,
                }
            )
    return configs


def resolve_anchor_hour_gate_stats(metadata, requested_index):
    feature_mean = metadata["feature_mean"].reshape(-1)
    feature_std = metadata["feature_std"].reshape(-1)
    hour_index = int(requested_index)
    if hour_index < 0:
        for idx, name in enumerate(metadata["input_feature_cols"]):
            if name == "\u5c0f\u65f6":
                hour_index = idx
                break
    if hour_index < 0 or hour_index >= len(feature_mean):
        raise ValueError(
            "--context_gate_anchor_hour requires an hour feature index; pass --context_gate_anchor_hour_index."
        )
    return {
        "context_gate_anchor_hour_index": hour_index,
        "context_gate_anchor_hour_mean": float(feature_mean[hour_index]),
        "context_gate_anchor_hour_std": float(feature_std[hour_index] if feature_std[hour_index] != 0 else 1.0),
    }


def resolve_trend_time_stats(metadata, requested_index, time_cycle):
    feature_mean = metadata["feature_mean"].reshape(-1)
    feature_std = metadata["feature_std"].reshape(-1)
    time_index = int(requested_index)
    if time_index < 0:
        preferred_tokens = ["\u5c0f\u65f6", "hour"] if int(time_cycle) != 48 else ["slot", "\u534a\u5c0f\u65f6"]
        for token in preferred_tokens:
            for idx, name in enumerate(metadata["input_feature_cols"]):
                if token in name and not name.startswith("future_"):
                    time_index = idx
                    break
            if time_index >= 0:
                break
    if time_index < 0 or time_index >= len(feature_mean):
        raise ValueError("--trend_alignment_decoder requires an intra-day time feature index.")
    return {
        "trend_time_feature_index": time_index,
        "trend_time_feature_mean": float(feature_mean[time_index]),
        "trend_time_feature_std": float(feature_std[time_index] if feature_std[time_index] != 0 else 1.0),
    }


def build_periodic_feature_names(target_cols, pred_len):
    names = []
    for prefix in ["prev_day", "prev_week"]:
        for horizon in range(1, pred_len + 1):
            for target_col in target_cols:
                names.append("periodic_%s_h%d_%s" % (prefix, horizon, target_col))
    return names


def build_anchor_interaction_feature_names():
    return [
        "anchor_interact_hour_norm",
        "anchor_interact_is_morning_06",
        "anchor_interact_is_midday_12",
        "anchor_interact_is_evening_16",
        "anchor_interact_is_night_00_20",
        "anchor_interact_peak_x_future_weekend",
        "anchor_interact_peak_x_future_precip",
        "anchor_interact_peak_x_future_wind",
        "anchor_interact_peak_x_recent_out_mean24",
        "anchor_interact_peak_x_recent_in_mean24",
    ]


def append_periodic_features_if_needed(x_values, metadata, target_start_offset):
    expected_periodic_cols = build_periodic_feature_names(metadata["target_cols"], metadata["pred_len"])
    required_periodic_cols = [
        col for col in metadata["input_feature_cols"]
        if col.startswith("periodic_")
    ]
    if not required_periodic_cols:
        return x_values, []
    missing = [col for col in expected_periodic_cols if col not in required_periodic_cols]
    if missing:
        raise ValueError("Unsupported periodic feature layout. Missing expected cols: %s" % missing[:5])

    hist_len = x_values.shape[1]
    num_targets = len(metadata["target_cols"])
    periodic_blocks = []
    for lag in [24, 168]:
        rel_indices = [hist_len - lag + target_start_offset + horizon for horizon in range(metadata["pred_len"])]
        if min(rel_indices) < 0 or max(rel_indices) >= hist_len:
            raise ValueError("Periodic lag %d is outside history window: %s" % (lag, rel_indices))
        values = x_values[:, rel_indices, :, :num_targets]
        values = values.transpose(0, 2, 1, 3).reshape(
            x_values.shape[0],
            x_values.shape[2],
            metadata["pred_len"] * num_targets,
        )
        periodic_blocks.append(values)

    periodic_values = np.concatenate(periodic_blocks, axis=-1).astype(np.float32)
    periodic_window = np.repeat(periodic_values[:, np.newaxis, :, :], hist_len, axis=1)
    x_aug = np.concatenate([x_values, periodic_window], axis=-1).astype(np.float32)
    if x_aug.shape[-1] != len(metadata["input_feature_cols"]):
        raise ValueError(
            "Feature dimension mismatch after periodic augmentation: got %d, expected %d"
            % (x_aug.shape[-1], len(metadata["input_feature_cols"]))
        )
    return x_aug, required_periodic_cols


def append_anchor_interaction_features_if_needed(x_values, metadata, sample_anchor_hours):
    expected_cols = build_anchor_interaction_feature_names()
    required_cols = [
        col for col in metadata["input_feature_cols"]
        if col.startswith("anchor_interact_")
    ]
    if not required_cols:
        return x_values, []
    if required_cols != expected_cols:
        raise ValueError("Unsupported anchor interaction feature layout: %s" % required_cols)
    if len(sample_anchor_hours) != x_values.shape[0]:
        raise ValueError("sample_anchor_hours length does not match x_values samples.")
    if x_values.shape[1] < 24:
        raise ValueError("hist_len must be at least 24 for anchor interaction features.")

    history_len = len(metadata["history_feature_cols"])
    future_cols = metadata["known_future_feature_cols"]
    required_positions = {
        "weekend": 2,
        "precip": 6,
        "wind": 8,
    }
    if len(future_cols) <= max(required_positions.values()):
        raise ValueError("known_future_feature_cols is too short for exp19 anchor interaction features.")
    weekend_idx = history_len + required_positions["weekend"]
    precip_idx = history_len + required_positions["precip"]
    wind_idx = history_len + required_positions["wind"]

    anchors = sample_anchor_hours.astype(np.float32)
    sample_node = lambda value: np.repeat(value[:, np.newaxis], x_values.shape[2], axis=1)
    hour_norm = sample_node(anchors / 23.0)
    morning = sample_node((sample_anchor_hours == 6).astype(np.float32))
    midday = sample_node((sample_anchor_hours == 12).astype(np.float32))
    evening = sample_node((sample_anchor_hours == 16).astype(np.float32))
    night = sample_node(np.isin(sample_anchor_hours, [0, 20]).astype(np.float32))
    peak = sample_node(np.isin(sample_anchor_hours, [6, 12, 16]).astype(np.float32))

    future_weekend = x_values[:, -1, :, weekend_idx]
    future_precip = x_values[:, -1, :, precip_idx]
    future_wind = x_values[:, -1, :, wind_idx]
    recent_out_mean = x_values[:, -24:, :, 0].mean(axis=1)
    recent_in_mean = x_values[:, -24:, :, 1].mean(axis=1)

    feature_block = np.stack(
        [
            hour_norm,
            morning,
            midday,
            evening,
            night,
            peak * future_weekend,
            peak * future_precip,
            peak * future_wind,
            peak * recent_out_mean,
            peak * recent_in_mean,
        ],
        axis=-1,
    ).astype(np.float32)
    feature_window = np.repeat(feature_block[:, np.newaxis, :, :], x_values.shape[1], axis=1)
    x_aug = np.concatenate([x_values, feature_window], axis=-1).astype(np.float32)
    if x_aug.shape[-1] != len(metadata["input_feature_cols"]):
        raise ValueError(
            "Feature dimension mismatch after anchor interaction augmentation: got %d, expected %d"
            % (x_aug.shape[-1], len(metadata["input_feature_cols"]))
        )
    return x_aug, required_cols


def find_best_checkpoint(project_name):
    search_patterns = []

    latest_record_path = SCRIPT_DIR / "latest_hourly_training_checkpoint.json"
    if latest_record_path.exists():
        with open(latest_record_path, "r", encoding="utf-8") as fp:
            latest_record = json.load(fp)
        if latest_record.get("project") == project_name:
            for key in ("preferred_checkpoint", "best_checkpoint"):
                checkpoint = latest_record.get(key)
                if checkpoint and Path(checkpoint).exists():
                    return str(Path(checkpoint))

    summary_pattern = PROJECT_ROOT / ANALYSIS_DIR / "*" / ("training_result_" + project_name) / "training_summary.json"
    for summary_path in sorted(glob.glob(str(summary_pattern)), key=lambda path: os.path.getmtime(path), reverse=True):
        with open(summary_path, "r", encoding="utf-8") as fp:
            summary = json.load(fp)
        for key in ("best_checkpoint", "preferred_checkpoint"):
            checkpoint = summary.get(key)
            if checkpoint and Path(checkpoint).exists():
                return str(Path(checkpoint))

    search_patterns.append(PROJECT_ROOT / "logs" / project_name / "version_*" / "checkpoints" / "best-*.ckpt")
    search_patterns.append(PROJECT_ROOT / "wandb" / "run-*" / "files" / "*" / "*" / "checkpoints" / "best-*.ckpt")
    search_patterns.append(PROJECT_ROOT / "wandb" / "offline-run-*" / "files" / "*" / "*" / "checkpoints" / "best-*.ckpt")

    matches = []
    for pattern in search_patterns:
        search_pattern = str(pattern)
        matches.extend(glob.glob(search_pattern))
    matches = sorted(set(matches), key=lambda path: os.path.getmtime(path))
    if not matches:
        raise FileNotFoundError(
            "No best checkpoint matched project %s. Searched: %s"
            % (project_name, "; ".join(str(pattern) for pattern in search_patterns))
        )
    return matches[-1]


def find_training_metadata(project_name):
    latest_record_path = SCRIPT_DIR / "latest_hourly_training_checkpoint.json"
    if latest_record_path.exists():
        with open(latest_record_path, "r", encoding="utf-8") as fp:
            latest_record = json.load(fp)
        if latest_record.get("project") == project_name:
            return latest_record

    summary_pattern = PROJECT_ROOT / ANALYSIS_DIR / "*" / ("training_result_" + project_name) / "training_summary.json"
    for summary_path in sorted(glob.glob(str(summary_pattern)), key=lambda path: os.path.getmtime(path), reverse=True):
        with open(summary_path, "r", encoding="utf-8") as fp:
            summary = json.load(fp)
        summary["_training_summary_path"] = str(summary_path)
        return summary
    return {}


def get_cli_value(argv, flag):
    if not argv:
        return None
    for idx, item in enumerate(argv):
        if item == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        prefix = flag + "="
        if isinstance(item, str) and item.startswith(prefix):
            return item[len(prefix):]
    return None


def parse_graph_use(value):
    if not value:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return None


def parse_bool_value(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return default


def cli_value_or_default(training_metadata, flag, default=None):
    value = get_cli_value(training_metadata.get("resolved_train_argv"), flag)
    if value is not None:
        return value
    value = get_cli_value(training_metadata.get("entry_args"), flag)
    if value is not None:
        return value
    return default


def compute_metrics(pred, true):
    err = np.abs(pred - true)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=os.path.join(
            "data",
            "temporal_data",
            "bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20_train2025_hist168",
        ),
    )
    parser.add_argument(
        "--graph_dir",
        default=os.path.join("data", "graph", "bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_graph_train2025_hist168"),
    )
    parser.add_argument("--project", default="bike_hourly_safe_inventory_top150_exp10_fast_cached_fusion_rolling6h_train2025_hist168_bs8_seed0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start_date", default="2026-02-01")
    parser.add_argument("--end_date", default="2026-02-28")
    parser.add_argument("--trip_glob", default="20260[12]-citibike-tripdata_*.csv")
    parser.add_argument("--order_dir", default=None, help="Optional order-data directory. Defaults to 二月份数据处理/纽约单车订单数据.")
    parser.add_argument("--weather_file", default=None, help="Optional weather CSV. Defaults to the current February-processing weather file.")
    parser.add_argument("--eval_tag", default="feb2026", help="Prefix for output CSV/JSON filenames, e.g. feb_mar2026.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--anchor_hours", default="0,6,12,16,20", help="Optional comma-separated decision anchors, e.g. 0,6,12,16,20.")
    parser.add_argument("--target_start_offset", type=int, default=1, help="For rolling anchors, decision t predicts t+offset...")
    parser.add_argument("--graph_use", default=None, help="Optional comma-separated graph list. Defaults to recovered training graph_use.")
    parser.add_argument("--hgaurban_graph_prior_path", default=None, help="Optional path to hgaurban_graph_prior.npy for graph_use=hgaurban.")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size. Use 1 for batch-level dynamic graph gates.")
    parser.add_argument("--context_gate", default=None, help="Optional true/false override for recovered context gate.")
    parser.add_argument("--context_gate_hidden_dim", type=int, default=None)
    parser.add_argument("--context_gate_residual", type=float, default=None)
    parser.add_argument("--context_gate_anchor_hour", default=None, help="Optional true/false override for anchor-hour graph gate.")
    parser.add_argument("--context_gate_anchor_hour_index", type=int, default=None)
    parser.add_argument("--context_gate_anchor_embed_dim", type=int, default=None)
    parser.add_argument("--context_gate_anchor_od_prior", type=float, default=None)
    parser.add_argument("--context_gate_scope", default=None, choices=[None, "all", "od_only", "od_residual_correction", "hard_anchor_od"])
    parser.add_argument("--ast_tcn_residual", default=None, help="Optional true/false override for AST-TCN residual branch.")
    parser.add_argument("--ast_tcn_hidden_dim", type=int, default=None)
    parser.add_argument("--ast_tcn_layers", type=int, default=None)
    parser.add_argument("--ast_tcn_kernel_size", type=int, default=None)
    parser.add_argument("--ast_tcn_dilation_base", type=int, default=None)
    parser.add_argument("--ast_tcn_heads", type=int, default=None)
    parser.add_argument("--ast_tcn_dropout", type=float, default=None)
    parser.add_argument("--ast_tcn_residual_init", type=float, default=None)
    parser.add_argument("--ast_tcn_bounded_alpha", default=None, help="Optional true/false override for bounded AST-TCN residual scale.")
    parser.add_argument("--ast_tcn_alpha_max", type=float, default=None)
    parser.add_argument("--ast_tcn_horizon_alpha", default=None, help="Optional true/false override for horizon-specific AST-TCN residual alpha.")
    parser.add_argument("--ast_tcn_zero_init", default=None, help="Optional true/false override for zero-init AST-TCN output head.")
    parser.add_argument("--ast_tcn_residual_gate", default=None, help="Optional true/false override for dynamic AST-TCN residual gate.")
    parser.add_argument("--ast_tcn_residual_gate_hidden_dim", type=int, default=None)
    parser.add_argument("--ast_tcn_residual_gate_init", type=float, default=None)
    parser.add_argument("--ast_tcn_edge_bias", default=None, help="Optional true/false override for graph edge-bias spatial attention.")
    parser.add_argument("--ast_tcn_edge_bias_init", type=float, default=None)
    parser.add_argument("--ast_tcn_edge_bias_eps", type=float, default=None)
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    graph_dir = PROJECT_ROOT / args.graph_dir
    order_dir = Path(args.order_dir) if args.order_dir else PROJECT_ROOT / FEB_DATA_DIR / ORDER_SUBDIR
    if not order_dir.is_absolute():
        order_dir = PROJECT_ROOT / order_dir
    asset_dir = PROJECT_ROOT / FEB_DATA_DIR / "nyc_top300_inventory_validation"
    weather_file = Path(args.weather_file) if args.weather_file else PROJECT_ROOT / FEB_DATA_DIR / "weather-get" / "NYC_Weather_2024-01-01_to_2026-02-31.csv"
    if not weather_file.is_absolute():
        weather_file = PROJECT_ROOT / weather_file
    mapping_path = graph_dir / "selected_node_mapping.csv"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / ANALYSIS_DIR / "2026-05-12_exp10_fast_cached_fusion_feb2026_holdout"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_npz_metadata(data_dir)
    mapping_df = pd.read_csv(mapping_path).sort_values("Node_ID").reset_index(drop=True)
    station_names = mapping_df[STATION_COL].astype(str).tolist()

    trip_files = sorted(glob.glob(str(order_dir / args.trip_glob)))
    if not trip_files:
        raise FileNotFoundError("No trip files matched: %s" % (order_dir / args.trip_glob))

    print("Aggregating hourly trip counts from %d files..." % len(trip_files))
    hourly_df, full_hours = aggregate_hourly_trip_counts(trip_files, station_names)
    all_dates = sorted(hourly_df["\u65e5\u671f"].dropna().unique())
    print("Hourly range:", str(full_hours.min()), "->", str(full_hours.max()))

    daily_feature_df, daily_feature_path, aux_paths, weather_path, aux_merge_summary = load_daily_feature_table(
        source_dir=str(asset_dir),
        station_names=station_names,
        all_dates=all_dates,
        daily_feature_file="snapshot_daily_features_topk.csv",
        aux_temporal_files="station_static_features_topk.csv",
        weather_file=str(weather_file),
    )
    feature_df = build_hourly_feature_frame(hourly_df, daily_feature_df)
    pipeline_mapping = mapping_df.rename(columns={STATION_COL: PIPELINE_STATION_COL})

    if args.anchor_hours:
        anchor_hours = parse_anchor_hours(args.anchor_hours)
        sample_bundle = build_hourly_samples_for_anchors(
            feature_df=feature_df,
            mapping_df=pipeline_mapping,
            history_feature_cols=metadata["history_feature_cols"],
            target_cols=metadata["target_cols"],
            known_future_feature_cols=metadata["known_future_feature_cols"],
            hist_len=metadata["hist_len"],
            pred_len=metadata["pred_len"],
            anchor_hours=anchor_hours,
            min_known_future_coverage=0.0,
            target_start_offset=args.target_start_offset,
        )
    else:
        anchor_hours = None
        sample_bundle = build_hourly_samples(
            feature_df=feature_df,
            mapping_df=pipeline_mapping,
            history_feature_cols=metadata["history_feature_cols"],
            target_cols=metadata["target_cols"],
            known_future_feature_cols=metadata["known_future_feature_cols"],
            hist_len=metadata["hist_len"],
            pred_len=metadata["pred_len"],
            min_known_future_coverage=0.0,
        )

    raw_x_values = sample_bundle["x"]
    x_values = raw_x_values.copy()
    history_len = len(metadata["history_feature_cols"])
    history_x, applied_history_cols = apply_log1p_transform(
        x_values[..., :history_len],
        metadata["history_feature_cols"],
        metadata["log1p_feature_cols"],
    )
    future_x, applied_future_cols = apply_log1p_transform(
        x_values[..., history_len:],
        sample_bundle["known_future_feature_cols"],
        metadata["log1p_feature_cols"],
    )
    x_values = np.concatenate([history_x, future_x], axis=-1)
    sample_anchor_hours = np.array(sample_bundle.get("anchor_hours", [-1] * len(sample_bundle["sample_dates"])))
    x_values, applied_periodic_cols = append_periodic_features_if_needed(
        x_values,
        metadata,
        target_start_offset=args.target_start_offset,
    )
    x_values, applied_anchor_interaction_cols = append_anchor_interaction_features_if_needed(
        x_values,
        metadata,
        sample_anchor_hours,
    )
    y_values = sample_bundle["y"]
    sample_dates = np.array(sample_bundle["sample_dates"])
    sample_datetimes = np.array(sample_bundle.get("sample_datetimes", sample_bundle["sample_dates"]))
    target_start_datetimes = np.array(sample_bundle.get("target_start_datetimes", sample_bundle["sample_dates"]))

    wanted_dates = pd.date_range(args.start_date, args.end_date, freq="D").strftime("%Y-%m-%d").to_numpy()
    available_mask = np.isin(sample_dates, wanted_dates)
    missing_dates = [d for d in wanted_dates.tolist() if d not in set(sample_dates.tolist())]
    x_values = x_values[available_mask]
    raw_x_values = raw_x_values[available_mask]
    y_values = y_values[available_mask]
    sample_datetimes = sample_datetimes[available_mask]
    sample_anchor_hours = sample_anchor_hours[available_mask]
    target_start_datetimes = target_start_datetimes[available_mask]
    eval_dates = sample_dates[available_mask]
    if len(eval_dates) == 0:
        raise ValueError("No requested February dates are evaluable. Missing examples: %s" % missing_dates[:5])

    feature_mean = metadata["feature_mean"]
    feature_std = metadata["feature_std"]
    x_scaled = ((x_values - feature_mean[0]) / feature_std[0]).astype(np.float32)

    training_metadata = find_training_metadata(args.project)
    recovered_graph_use = None
    if args.graph_use:
        recovered_graph_use = parse_graph_use(args.graph_use)
    if recovered_graph_use is None:
        recovered_graph_use = parse_graph_use(training_metadata.get("graph_use"))
    if recovered_graph_use is None:
        recovered_graph_use = parse_graph_use(get_cli_value(training_metadata.get("resolved_train_argv"), "--graph_use"))
    if recovered_graph_use is None:
        recovered_graph_use = parse_graph_use(get_cli_value(training_metadata.get("entry_args"), "--graph_use"))
    if recovered_graph_use is None:
        recovered_graph_use = ["dist", "neighb", "distri", "tempp", "func", "od00", "od06", "od12", "od16", "od20"]
    hgaurban_graph_prior_path = (
        args.hgaurban_graph_prior_path
        if args.hgaurban_graph_prior_path is not None
        else cli_value_or_default(training_metadata, "--hgaurban_graph_prior_path", "")
    )
    if hgaurban_graph_prior_path:
        hgaurban_graph_prior_path = str(Path(hgaurban_graph_prior_path))
        if not Path(hgaurban_graph_prior_path).is_absolute():
            hgaurban_graph_prior_path = str(PROJECT_ROOT / hgaurban_graph_prior_path)
    context_gate = parse_bool_value(
        args.context_gate if args.context_gate is not None else cli_value_or_default(training_metadata, "--context_gate"),
        default=False,
    )
    context_gate_hidden_dim = int(
        args.context_gate_hidden_dim
        if args.context_gate_hidden_dim is not None
        else cli_value_or_default(training_metadata, "--context_gate_hidden_dim", 32)
    )
    context_gate_residual = float(
        args.context_gate_residual
        if args.context_gate_residual is not None
        else cli_value_or_default(training_metadata, "--context_gate_residual", 0.5)
    )
    context_gate_anchor_hour = parse_bool_value(
        args.context_gate_anchor_hour
        if args.context_gate_anchor_hour is not None
        else cli_value_or_default(training_metadata, "--context_gate_anchor_hour"),
        default=False,
    )
    context_gate_anchor_hour_index = int(
        args.context_gate_anchor_hour_index
        if args.context_gate_anchor_hour_index is not None
        else cli_value_or_default(training_metadata, "--context_gate_anchor_hour_index", -1)
    )
    context_gate_anchor_embed_dim = int(
        args.context_gate_anchor_embed_dim
        if args.context_gate_anchor_embed_dim is not None
        else cli_value_or_default(training_metadata, "--context_gate_anchor_embed_dim", 8)
    )
    context_gate_anchor_od_prior = float(
        args.context_gate_anchor_od_prior
        if args.context_gate_anchor_od_prior is not None
        else cli_value_or_default(training_metadata, "--context_gate_anchor_od_prior", 0.0)
    )
    context_gate_scope = (
        args.context_gate_scope
        if args.context_gate_scope is not None
        else cli_value_or_default(training_metadata, "--context_gate_scope", "all")
    )

    device = resolve_device(args.device)
    graph_config = {
        "use": recovered_graph_use,
        "fix_weight": False,
        "tempp_diag_zero": True,
        "matrix_weight": True,
        "context_gate": context_gate,
        "context_gate_hidden_dim": context_gate_hidden_dim,
        "context_gate_residual": context_gate_residual,
        "context_gate_anchor_hour": context_gate_anchor_hour,
        "context_gate_anchor_hour_index": context_gate_anchor_hour_index,
        "context_gate_anchor_embed_dim": context_gate_anchor_embed_dim,
        "context_gate_anchor_od_prior": context_gate_anchor_od_prior,
        "context_gate_scope": context_gate_scope,
        "hgaurban_graph_prior_path": hgaurban_graph_prior_path,
        "distri_type": "exp",
        "func_type": "ours",
        "attention": True,
        "sparsify_mode": "topk",
        "sparsify_topk": 20,
        "sparsify_symmetric": True,
        "sparsify_keep_self": True,
    }
    if graph_config["context_gate_anchor_hour"]:
        graph_config.update(
            resolve_anchor_hour_gate_stats(
                metadata,
                requested_index=graph_config["context_gate_anchor_hour_index"],
            )
        )
    data_config = {"in_dim": x_scaled.shape[-1], "out_dim": y_values.shape[-1], "hist_len": x_scaled.shape[1], "pred_len": y_values.shape[1], "type": "bike"}
    model_config = {
        "use": cli_value_or_default(training_metadata, "--model_use", "MSTGCN"),
        "cheb_k": 3,
        "nb_block": 2,
        "nb_chev_filter": 64,
        "nb_time_filter": 64,
        "time_kernel_size": 3,
        "channel_attention": False,
        "channel_attention_reduction": 4,
        "trend_alignment_decoder": parse_bool_value(
            cli_value_or_default(training_metadata, "--trend_alignment_decoder"),
            default=False,
        ),
        "trend_time_feature_index": int(cli_value_or_default(training_metadata, "--trend_time_feature_index", -1)),
        "trend_time_cycle": int(cli_value_or_default(training_metadata, "--trend_time_cycle", 24)),
        "trend_time_embed_dim": int(cli_value_or_default(training_metadata, "--trend_time_embed_dim", 16)),
        "trend_attention_heads": int(cli_value_or_default(training_metadata, "--trend_attention_heads", 4)),
        "trend_dropout": float(cli_value_or_default(training_metadata, "--trend_dropout", 0.1)),
        "ast_tcn_residual": parse_bool_value(
            args.ast_tcn_residual
            if args.ast_tcn_residual is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_residual"),
            default=False,
        ),
        "ast_tcn_hidden_dim": int(
            args.ast_tcn_hidden_dim
            if args.ast_tcn_hidden_dim is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_hidden_dim", 32)
        ),
        "ast_tcn_layers": int(
            args.ast_tcn_layers
            if args.ast_tcn_layers is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_layers", 4)
        ),
        "ast_tcn_kernel_size": int(
            args.ast_tcn_kernel_size
            if args.ast_tcn_kernel_size is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_kernel_size", 3)
        ),
        "ast_tcn_dilation_base": int(
            args.ast_tcn_dilation_base
            if args.ast_tcn_dilation_base is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_dilation_base", 2)
        ),
        "ast_tcn_heads": int(
            args.ast_tcn_heads
            if args.ast_tcn_heads is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_heads", 4)
        ),
        "ast_tcn_dropout": float(
            args.ast_tcn_dropout
            if args.ast_tcn_dropout is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_dropout", 0.1)
        ),
        "ast_tcn_residual_init": float(
            args.ast_tcn_residual_init
            if args.ast_tcn_residual_init is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_residual_init", 0.05)
        ),
        "ast_tcn_bounded_alpha": parse_bool_value(
            args.ast_tcn_bounded_alpha
            if args.ast_tcn_bounded_alpha is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_bounded_alpha"),
            default=False,
        ),
        "ast_tcn_alpha_max": float(
            args.ast_tcn_alpha_max
            if args.ast_tcn_alpha_max is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_alpha_max", 0.1)
        ),
        "ast_tcn_horizon_alpha": parse_bool_value(
            args.ast_tcn_horizon_alpha
            if args.ast_tcn_horizon_alpha is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_horizon_alpha"),
            default=False,
        ),
        "ast_tcn_zero_init": parse_bool_value(
            args.ast_tcn_zero_init
            if args.ast_tcn_zero_init is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_zero_init"),
            default=False,
        ),
        "ast_tcn_residual_gate": parse_bool_value(
            args.ast_tcn_residual_gate
            if args.ast_tcn_residual_gate is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_residual_gate"),
            default=False,
        ),
        "ast_tcn_residual_gate_hidden_dim": int(
            args.ast_tcn_residual_gate_hidden_dim
            if args.ast_tcn_residual_gate_hidden_dim is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_residual_gate_hidden_dim", 16)
        ),
        "ast_tcn_residual_gate_init": float(
            args.ast_tcn_residual_gate_init
            if args.ast_tcn_residual_gate_init is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_residual_gate_init", 0.2)
        ),
        "ast_tcn_edge_bias": parse_bool_value(
            args.ast_tcn_edge_bias
            if args.ast_tcn_edge_bias is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_edge_bias"),
            default=False,
        ),
        "ast_tcn_edge_bias_init": float(
            args.ast_tcn_edge_bias_init
            if args.ast_tcn_edge_bias_init is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_edge_bias_init", 0.1)
        ),
        "ast_tcn_edge_bias_eps": float(
            args.ast_tcn_edge_bias_eps
            if args.ast_tcn_edge_bias_eps is not None
            else cli_value_or_default(training_metadata, "--ast_tcn_edge_bias_eps", 1e-6)
        ),
        "d2_hidden_dim": int(cli_value_or_default(training_metadata, "--d2_hidden_dim", 64)),
        "d2_num_layers": int(cli_value_or_default(training_metadata, "--d2_num_layers", 4)),
        "d2_dropout": float(cli_value_or_default(training_metadata, "--d2_dropout", 0.1)),
        "d2_dilation_cycle": int(cli_value_or_default(training_metadata, "--d2_dilation_cycle", 2)),
        "d2_kernel_size": int(cli_value_or_default(training_metadata, "--d2_kernel_size", 2)),
        "d2_gcn_order": int(cli_value_or_default(training_metadata, "--d2_gcn_order", 2)),
        "d2_node_embed_dim": int(cli_value_or_default(training_metadata, "--d2_node_embed_dim", 16)),
        "d2_adaptive_adj": parse_bool_value(
            cli_value_or_default(training_metadata, "--d2_adaptive_adj"),
            default=True,
        ),
        "d2_use_reverse": parse_bool_value(
            cli_value_or_default(training_metadata, "--d2_use_reverse"),
            default=True,
        ),
        "d2_fusion_init": float(cli_value_or_default(training_metadata, "--d2_fusion_init", 1.0)),
    }
    if model_config["trend_alignment_decoder"]:
        model_config.update(
            resolve_trend_time_stats(
                metadata,
                requested_index=model_config["trend_time_feature_index"],
                time_cycle=model_config["trend_time_cycle"],
            )
        )

    fusiongraph_module.PROJECT_ROOT = str(PROJECT_ROOT)
    graph = BikeGraph(str(graph_dir), graph_config, device)
    fusiongraph = FusionGraphModel(graph, device, graph_config, data_config, 24, 6, 0.1)
    categorical_configs = build_categorical_feature_configs(metadata, weekday_embed_dim=8)
    model = EvalWrapper(device, fusiongraph, data_config, model_config, categorical_configs).to(device)
    checkpoint_path = args.checkpoint or find_best_checkpoint(args.project)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [key for key in missing if not key.startswith("metric_lightning.")]
    if real_missing:
        raise RuntimeError("Missing checkpoint keys: %s" % real_missing[:10])
    model.eval()

    preds = []
    batch_size = int(args.batch_size)
    target_mean = torch.as_tensor(metadata["target_mean"], dtype=torch.float32, device=device)
    target_std = torch.as_tensor(metadata["target_std"], dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            batch = torch.from_numpy(x_scaled[start:start + batch_size]).to(device)
            anchor_batch = torch.from_numpy(sample_anchor_hours[start:start + batch_size]).to(device).long()
            pred = model(batch, anchor_hours=anchor_batch)
            pred = pred * target_std + target_mean
            pred = F.softplus(pred, beta=5.0)
            preds.append(pred.cpu().numpy())
    pred_values = np.concatenate(preds, axis=0)
    abs_err = np.abs(pred_values - y_values)

    pred_len = y_values.shape[1]
    naive_last_value = np.repeat(raw_x_values[:, -1:, :, :2], pred_len, axis=1)
    if raw_x_values.shape[1] >= 24 + pred_len - 1:
        previous_day_same_hours = raw_x_values[:, -24:-24 + pred_len, :, :2]
    elif raw_x_values.shape[1] >= pred_len:
        previous_day_same_hours = raw_x_values[:, -pred_len:, :, :2]
    else:
        previous_day_same_hours = naive_last_value
    train_y = np.load(data_dir / "train.npz", allow_pickle=True)["y"]
    naive_train_mean = np.repeat(train_y.mean(axis=0, keepdims=True), y_values.shape[0], axis=0)
    baseline_rows = [
        {
            "method": "model_exp06_top150",
            "mae": float(abs_err.mean()),
            "rmse": float(np.sqrt(np.mean((pred_values - y_values) ** 2))),
            "note": "current top150 checkpoint",
        },
        {
            "method": "naive_last_1h_repeat",
            "mae": float(np.abs(naive_last_value - y_values).mean()),
            "rmse": float(np.sqrt(np.mean((naive_last_value - y_values) ** 2))),
            "note": "repeat the last observed historical hour",
        },
        {
            "method": "naive_previous_day_same_hours",
            "mae": float(np.abs(previous_day_same_hours - y_values).mean()),
            "rmse": float(np.sqrt(np.mean((previous_day_same_hours - y_values) ** 2))),
            "note": "copy the same target-hour range from the previous day",
        },
        {
            "method": "naive_train_mean_by_horizon",
            "mae": float(np.abs(naive_train_mean - y_values).mean()),
            "rmse": float(np.sqrt(np.mean((naive_train_mean - y_values) ** 2))),
            "note": "train-set mean for each horizon/node/channel",
        },
    ]

    rows = []
    for d_idx, date_value in enumerate(eval_dates):
        for h in range(y_values.shape[1]):
            if args.anchor_hours:
                target_dt = pd.Timestamp(str(target_start_datetimes[d_idx])) + pd.Timedelta(hours=h)
                hour_label = target_dt.strftime("%H:00")
            else:
                hour_label = "%02d:00" % h
            for n_idx, station_name in enumerate(station_names):
                rows.append(
                    {
                        "date": date_value,
                        "sample_datetime": str(sample_datetimes[d_idx]),
                        "anchor_hour": int(sample_anchor_hours[d_idx]),
                        "target_start_datetime": str(target_start_datetimes[d_idx]),
                        "horizon": h + 1,
                        "hour": hour_label,
                        "Node_ID": int(mapping_df.loc[n_idx, "Node_ID"]),
                        "station_name": station_name,
                        "pred_out": float(pred_values[d_idx, h, n_idx, 0]),
                        "pred_in": float(pred_values[d_idx, h, n_idx, 1]),
                        "true_out": float(y_values[d_idx, h, n_idx, 0]),
                        "true_in": float(y_values[d_idx, h, n_idx, 1]),
                        "abs_error_out": float(abs_err[d_idx, h, n_idx, 0]),
                        "abs_error_in": float(abs_err[d_idx, h, n_idx, 1]),
                        "abs_error_avg": float(abs_err[d_idx, h, n_idx, :].mean()),
                    }
                )
    station_hour_df = pd.DataFrame(rows)
    station_metrics = (
        station_hour_df.groupby(["Node_ID", "station_name"], as_index=False)
        [["abs_error_avg", "abs_error_out", "abs_error_in"]]
        .mean()
        .rename(columns={"abs_error_avg": "mae_avg", "abs_error_out": "mae_out", "abs_error_in": "mae_in"})
    )
    horizon_metrics = station_hour_df.groupby("horizon", as_index=False)[["abs_error_avg", "abs_error_out", "abs_error_in"]].mean()
    daily_metrics = station_hour_df.groupby("date", as_index=False)[["abs_error_avg", "abs_error_out", "abs_error_in"]].mean()
    anchor_metrics = station_hour_df.groupby("anchor_hour", as_index=False)[["abs_error_avg", "abs_error_out", "abs_error_in"]].mean()
    anchor_horizon_metrics = station_hour_df.groupby(["anchor_hour", "horizon"], as_index=False)[["abs_error_avg", "abs_error_out", "abs_error_in"]].mean()
    peak_metrics = []
    for name, hours in [("morning_07_10", {7, 8, 9}), ("evening_17_20", {17, 18, 19})]:
        target_horizons = {h + 1 for h in hours if h < pred_len}
        sub = station_hour_df[station_hour_df["horizon"].isin(target_horizons)] if target_horizons else station_hour_df.iloc[0:0]
        peak_metrics.append(
            {
                "window": name,
                "mae_avg": None if sub.empty else float(sub["abs_error_avg"].mean()),
                "mae_out": None if sub.empty else float(sub["abs_error_out"].mean()),
                "mae_in": None if sub.empty else float(sub["abs_error_in"].mean()),
                "covered_horizons": ",".join(str(v) for v in sorted(target_horizons)),
            }
        )
    peak_metrics = pd.DataFrame(peak_metrics)
    baseline_metrics = pd.DataFrame(baseline_rows)

    summary = {
        "checkpoint": checkpoint_path,
        "order_dir": str(order_dir),
        "trip_glob": args.trip_glob,
        "graph_use": graph_config["use"],
        "context_gate": graph_config["context_gate"],
        "context_gate_hidden_dim": graph_config["context_gate_hidden_dim"],
        "context_gate_residual": graph_config["context_gate_residual"],
        "context_gate_anchor_hour": graph_config["context_gate_anchor_hour"],
        "context_gate_anchor_hour_index": graph_config["context_gate_anchor_hour_index"],
        "context_gate_anchor_embed_dim": graph_config["context_gate_anchor_embed_dim"],
        "context_gate_anchor_od_prior": graph_config["context_gate_anchor_od_prior"],
        "context_gate_scope": graph_config["context_gate_scope"],
        "trend_alignment_decoder": model_config["trend_alignment_decoder"],
        "ast_tcn_residual": model_config["ast_tcn_residual"],
        "ast_tcn_hidden_dim": model_config["ast_tcn_hidden_dim"],
        "ast_tcn_layers": model_config["ast_tcn_layers"],
        "ast_tcn_kernel_size": model_config["ast_tcn_kernel_size"],
        "ast_tcn_dilation_base": model_config["ast_tcn_dilation_base"],
        "ast_tcn_heads": model_config["ast_tcn_heads"],
        "ast_tcn_dropout": model_config["ast_tcn_dropout"],
        "ast_tcn_residual_init": model_config["ast_tcn_residual_init"],
        "ast_tcn_bounded_alpha": model_config["ast_tcn_bounded_alpha"],
        "ast_tcn_alpha_max": model_config["ast_tcn_alpha_max"],
        "ast_tcn_horizon_alpha": model_config["ast_tcn_horizon_alpha"],
        "ast_tcn_zero_init": model_config["ast_tcn_zero_init"],
        "ast_tcn_residual_gate": model_config["ast_tcn_residual_gate"],
        "ast_tcn_residual_gate_hidden_dim": model_config["ast_tcn_residual_gate_hidden_dim"],
        "ast_tcn_residual_gate_init": model_config["ast_tcn_residual_gate_init"],
        "ast_tcn_edge_bias": model_config["ast_tcn_edge_bias"],
        "ast_tcn_edge_bias_init": model_config["ast_tcn_edge_bias_init"],
        "ast_tcn_edge_bias_eps": model_config["ast_tcn_edge_bias_eps"],
        "trip_files": trip_files,
        "date_start": str(eval_dates[0]),
        "date_end": str(eval_dates[-1]),
        "requested_start": args.start_date,
        "requested_end": args.end_date,
        "missing_dates": missing_dates,
        "num_eval_samples": int(len(eval_dates)),
        "num_stations": int(len(station_names)),
        "pred_shape": list(pred_values.shape),
        "overall": compute_metrics(pred_values.reshape(-1), y_values.reshape(-1)),
        "hist_len": int(metadata["hist_len"]),
        "pred_len": int(metadata["pred_len"]),
        "anchor_hours": None if anchor_hours is None else anchor_hours,
        "target_start_offset": args.target_start_offset if args.anchor_hours else None,
        "anchor_note": (
            "Rolling decision anchors: horizon 1 starts at anchor + target_start_offset."
            if args.anchor_hours
            else "Daily 00:00 anchor: horizon 1 starts at 00:00 of each evaluated date."
        ),
        "mae_1_6": float(abs_err[:, :min(6, pred_len), :, :].mean()),
        "mae_after_6": None if pred_len <= 6 else float(abs_err[:, 6:, :, :].mean()),
        "morning_07_10_mae": None if pred_len < 10 else float(station_hour_df[station_hour_df["horizon"].isin([8, 9, 10])]["abs_error_avg"].mean()),
        "evening_17_20_mae": None if pred_len < 20 else float(station_hour_df[station_hour_df["horizon"].isin([18, 19, 20])]["abs_error_avg"].mean()),
        "baseline_mae": {row["method"]: row["mae"] for row in baseline_rows},
        "model_improve_vs_last_1h_repeat_pct": float((baseline_rows[1]["mae"] - baseline_rows[0]["mae"]) / baseline_rows[1]["mae"] * 100.0),
        "model_improve_vs_previous_day_same_hours_pct": float((baseline_rows[2]["mae"] - baseline_rows[0]["mae"]) / baseline_rows[2]["mae"] * 100.0),
        "model_improve_vs_train_mean_pct": float((baseline_rows[3]["mae"] - baseline_rows[0]["mae"]) / baseline_rows[3]["mae"] * 100.0),
        "daily_feature_file": daily_feature_path,
        "weather_file": weather_path,
        "aux_paths": aux_paths,
        "aux_merge_summary": aux_merge_summary,
        "applied_log1p_cols": sorted(set(applied_history_cols + applied_future_cols)),
        "applied_periodic_cols": applied_periodic_cols,
        "applied_anchor_interaction_cols": applied_anchor_interaction_cols,
        "known_future_cols": sample_bundle["known_future_feature_cols"],
    }

    tag = args.eval_tag
    station_hour_df.to_csv(output_dir / ("%s_station_hour_predictions.csv" % tag), index=False, encoding="utf-8-sig")
    station_metrics.to_csv(output_dir / ("%s_station_mae.csv" % tag), index=False, encoding="utf-8-sig")
    horizon_metrics.to_csv(output_dir / ("%s_horizon_mae.csv" % tag), index=False, encoding="utf-8-sig")
    daily_metrics.to_csv(output_dir / ("%s_daily_mae.csv" % tag), index=False, encoding="utf-8-sig")
    anchor_metrics.to_csv(output_dir / ("%s_anchor_hour_mae.csv" % tag), index=False, encoding="utf-8-sig")
    anchor_horizon_metrics.to_csv(output_dir / ("%s_anchor_hour_horizon_mae.csv" % tag), index=False, encoding="utf-8-sig")
    peak_metrics.to_csv(output_dir / ("%s_peak_window_mae.csv" % tag), index=False, encoding="utf-8-sig")
    baseline_metrics.to_csv(output_dir / ("%s_model_vs_naive_baselines.csv" % tag), index=False, encoding="utf-8-sig")
    with open(output_dir / ("%s_holdout_summary.json" % tag), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved outputs to:", output_dir)


if __name__ == "__main__":
    main()
