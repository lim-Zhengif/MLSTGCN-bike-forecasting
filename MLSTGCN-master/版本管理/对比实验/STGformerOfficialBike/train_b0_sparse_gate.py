# -*- coding: utf-8 -*-
"""Train G0: a history-only sparse-demand gate over a frozen B0 model."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

from models import (  # noqa: E402
    GATE_MODEL_ID,
    FrozenB0SparseGate,
    SparseDemandGate,
    build_model_from_checkpoint,
    build_sparse_gate_from_checkpoint,
)
from protocol import (  # noqa: E402
    MODEL_ID,
    UPSTREAM_COMMIT,
    audit_graph_contract,
    b0_training_run_name,
    checkpoint_training_seed,
    compute_metrics,
    environment_info,
    load_npz_metadata,
    save_json,
    set_seed,
    sha256_file,
)


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "temporal_data" / (
    "bike_hourly_safe_inventory_top150_nyc_full_predlen3_"
    "anchors_00_03_06_09_12_15_18_21_train2025_hist168"
)
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "data" / "graph" / (
    "bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_graph_"
    "train2025_hist168_pred3_8anchors"
)
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "分析结果" / "对比实验" / "STGformerOfficialBike"


class RawPairDataset(Dataset):
    def __init__(self, x, y, feature_mean, feature_std):
        self.x = x
        self.y = y
        self.feature_mean = torch.as_tensor(feature_mean.reshape(-1), dtype=torch.float32)
        self.feature_std = torch.as_tensor(feature_std.reshape(-1), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = torch.from_numpy(np.asarray(self.x[index], dtype=np.float32))
        y = torch.from_numpy(np.asarray(self.y[index], dtype=np.float32))
        return (x - self.feature_mean) / self.feature_std, y


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def gate_run_name(seed):
    return "g0_b0_sparse_demand_gate_seed%d" % int(seed)


def extract_gate_cache(wrapper, x, y, batch_size, num_workers, device):
    loader = DataLoader(
        RawPairDataset(
            x,
            y,
            wrapper.base_model.feature_mean.cpu().numpy(),
            wrapper.base_model.feature_std.cpu().numpy(),
        ),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    feature_parts = []
    base_parts = []
    target_parts = []
    wrapper.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            base_raw = wrapper.base_model(x_batch)
            base_counts = F.softplus(
                base_raw * wrapper.target_std + wrapper.target_mean,
                beta=wrapper.softplus_beta,
            )
            features = wrapper.gate_features(x_batch, base_counts)
            feature_parts.append(features.cpu())
            base_parts.append(base_counts.cpu())
            target_parts.append(y_batch)
    return (
        torch.cat(feature_parts).reshape(-1, len(SparseDemandGate.FEATURE_NAMES)),
        torch.cat(base_parts).reshape(-1),
        torch.cat(target_parts).reshape(-1),
    )


def cache_metrics(gate, features, base, target, batch_size, device):
    predictions = []
    gates = []
    gate.eval()
    with torch.no_grad():
        for start in range(0, len(target), int(batch_size)):
            end = start + int(batch_size)
            gate_value = gate(features[start:end].to(device)).cpu()
            gates.append(gate_value)
            predictions.append(base[start:end] * gate_value)
    prediction = torch.cat(predictions).numpy()
    gate_value = torch.cat(gates).numpy()
    base_np = base.numpy()
    target_np = target.numpy()

    def selected_mae(mask, values):
        return float(np.abs(values[mask] - target_np[mask]).mean()) if mask.any() else None

    zero = target_np == 0
    low = target_np <= 2
    high = target_np >= 6
    result = compute_metrics(prediction, target_np)
    result.update({
        "zero_mae": selected_mae(zero, prediction),
        "low_le2_mae": selected_mae(low, prediction),
        "high_ge6_mae": selected_mae(high, prediction),
        "base_mae": float(np.abs(base_np - target_np).mean()),
        "base_zero_mae": selected_mae(zero, base_np),
        "base_low_le2_mae": selected_mae(low, base_np),
        "base_high_ge6_mae": selected_mae(high, base_np),
        "gate_mean": float(gate_value.mean()),
        "gate_zero_target_mean": float(gate_value[zero].mean()),
        "gate_low_le2_target_mean": float(gate_value[low].mean()),
        "gate_high_ge6_target_mean": float(gate_value[high].mean()),
    })
    return result


def train_gate_epoch(
    gate,
    loader,
    optimizer,
    device,
    low_weight,
    high_preserve_weight,
    identity_weight,
    grad_clip,
):
    gate.train()
    totals = []
    for features, base, target in loader:
        features = features.to(device, non_blocking=True)
        base = base.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        gate_value = gate(features)
        prediction = base * gate_value
        weights = 1.0 + float(low_weight) * (target <= 2).to(target.dtype)
        regression = F.smooth_l1_loss(prediction, target, reduction="none")
        regression = (regression * weights).sum() / weights.sum()
        high_mask = target >= 6
        if bool(high_mask.any()):
            preservation = F.smooth_l1_loss(
                prediction[high_mask], base[high_mask], reduction="mean"
            )
        else:
            preservation = regression.new_tensor(0.0)
        identity = torch.mean((1.0 - gate_value) ** 2)
        loss = (
            regression
            + float(high_preserve_weight) * preservation
            + float(identity_weight) * identity
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite sparse-gate loss")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(gate.parameters(), float(grad_clip))
        optimizer.step()
        totals.append((float(loss.item()), float(regression.item()), float(preservation.item())))
    values = np.asarray(totals, dtype=np.float64)
    return {
        "train_loss": float(values[:, 0].mean()),
        "train_regression": float(values[:, 1].mean()),
        "train_high_preservation": float(values[:, 2].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Train G0 low-demand gate over frozen B0")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--graph_dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--graph_name", default="dist")
    parser.add_argument("--base_checkpoint", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache_batch_size", type=int, default=16)
    parser.add_argument("--gate_batch_size", type=int, default=65536)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--gate_floor", type=float, default=0.1)
    parser.add_argument("--initial_gate", type=float, default=0.98)
    parser.add_argument("--low_weight", type=float, default=2.0)
    parser.add_argument("--high_preserve_weight", type=float, default=0.5)
    parser.add_argument("--identity_weight", type=float, default=0.02)
    parser.add_argument("--zero_selection_weight", type=float, default=0.15)
    parser.add_argument("--high_tolerance_pct", type=float, default=0.5)
    parser.add_argument("--high_penalty_weight", type=float, default=5.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    graph_dir = Path(args.graph_dir)
    if args.base_checkpoint is None:
        args.base_checkpoint = str(
            DEFAULT_RESULT_ROOT / b0_training_run_name(args.seed, 16) /
            "best_stgformer_official_b0.pt"
        )
    if args.output_dir is None:
        args.output_dir = str(DEFAULT_RESULT_ROOT / gate_run_name(args.seed))
    base_checkpoint_path = Path(args.base_checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_npz_metadata(data_dir)
    graph_audit = audit_graph_contract(graph_dir, args.graph_name, metadata["num_nodes"])
    base_checkpoint = torch.load(base_checkpoint_path, map_location=device)
    if base_checkpoint.get("model_id") != MODEL_ID:
        raise RuntimeError("base_checkpoint is not the frozen B0 model")
    if checkpoint_training_seed(base_checkpoint) != int(args.seed):
        raise RuntimeError("base checkpoint seed does not match --seed")
    if base_checkpoint.get("graph_audit", {}).get("node_signature") != graph_audit["node_signature"]:
        raise RuntimeError("base checkpoint node signature mismatch")
    for key in ["feature_mean", "feature_std", "target_mean", "target_std"]:
        if not np.allclose(base_checkpoint[key], metadata[key], rtol=1e-7, atol=1e-8):
            raise RuntimeError("base checkpoint %s mismatch" % key)
    base_model, base_audit = build_model_from_checkpoint(base_checkpoint, device=device)
    gate = SparseDemandGate(
        hidden_dim=args.hidden_dim,
        gate_floor=args.gate_floor,
        initial_gate=args.initial_gate,
    ).to(device)
    wrapper = FrozenB0SparseGate(
        base_model=base_model,
        target_mean=base_checkpoint["target_mean"],
        target_std=base_checkpoint["target_std"],
        gate=gate,
        flow_feature_indices=(0, 1),
        hour_feature_index=base_checkpoint["model_config"]["hour_feature_index"],
    ).to(device)

    caches = {}
    cache_started = time.perf_counter()
    for split in ["train", "val", "test"]:
        data = np.load(data_dir / (split + ".npz"), allow_pickle=True)
        caches[split] = extract_gate_cache(
            wrapper,
            data["x"],
            data["y"],
            args.cache_batch_size,
            args.num_workers,
            device,
        )
    cache_seconds = time.perf_counter() - cache_started

    train_loader = DataLoader(
        TensorDataset(*caches["train"]),
        batch_size=args.gate_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    val_base_metrics = cache_metrics(
        gate, *caches["val"], args.gate_batch_size, device
    )
    history = []
    best_score = float("inf")
    best_epoch = -1
    bad_epochs = 0
    checkpoint_path = output_dir / "best_b0_sparse_demand_gate_g0.pt"

    for epoch in range(args.epochs):
        started = time.perf_counter()
        row = train_gate_epoch(
            gate,
            train_loader,
            optimizer,
            device,
            args.low_weight,
            args.high_preserve_weight,
            args.identity_weight,
            args.grad_clip,
        )
        val_metrics = cache_metrics(
            gate, *caches["val"], args.gate_batch_size, device
        )
        high_limit = val_metrics["base_high_ge6_mae"] * (
            1.0 + args.high_tolerance_pct / 100.0
        )
        high_excess = max(0.0, val_metrics["high_ge6_mae"] - high_limit)
        score = (
            val_metrics["mae"]
            + args.zero_selection_weight * val_metrics["zero_mae"]
            + args.high_penalty_weight * high_excess
        )
        row.update({
            "epoch": epoch,
            "selection_score": float(score),
            "epoch_seconds": float(time.perf_counter() - started),
            **{"val_" + key: value for key, value in val_metrics.items()},
        })
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if score < best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            bad_epochs = 0
            torch.save({
                "schema_version": 1,
                "model_id": GATE_MODEL_ID,
                "upstream_commit": UPSTREAM_COMMIT,
                "model_config": base_checkpoint["model_config"],
                "base_model_state_dict": base_checkpoint["model_state_dict"],
                "gate_config": gate.export_config(),
                "gate_state_dict": gate.state_dict(),
                "feature_mean": base_checkpoint["feature_mean"],
                "feature_std": base_checkpoint["feature_std"],
                "target_mean": base_checkpoint["target_mean"],
                "target_std": base_checkpoint["target_std"],
                "input_feature_cols": base_checkpoint["input_feature_cols"],
                "target_cols": base_checkpoint["target_cols"],
                "graph_audit": graph_audit,
                "flow_feature_indices": [0, 1],
                "hour_feature_index": base_checkpoint["model_config"]["hour_feature_index"],
                "softplus_beta": 5.0,
                "base_checkpoint": str(base_checkpoint_path),
                "base_checkpoint_sha256": sha256_file(base_checkpoint_path),
                "best_epoch": best_epoch,
                "best_selection_score": best_score,
                "args": vars(args),
                "environment": environment_info(),
            }, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stop at epoch %d" % epoch)
                break

    saved = torch.load(checkpoint_path, map_location=device)
    restored, checkpoint_audit = build_sparse_gate_from_checkpoint(saved, device=device)
    test_metrics = cache_metrics(
        restored.gate, *caches["test"], args.gate_batch_size, device
    )
    val_metrics = cache_metrics(
        restored.gate, *caches["val"], args.gate_batch_size, device
    )
    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "schema_version": 1,
        "experiment_id": "G0",
        "model": GATE_MODEL_ID,
        "seed": int(args.seed),
        "checkpoint": str(checkpoint_path),
        "checkpoint_missing_keys": checkpoint_audit["missing_keys"],
        "checkpoint_unexpected_keys": checkpoint_audit["unexpected_keys"],
        "base_checkpoint": str(base_checkpoint_path),
        "base_checkpoint_sha256": saved["base_checkpoint_sha256"],
        "base_checkpoint_audit": base_audit,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "cache_seconds": cache_seconds,
        "validation_before_training": val_base_metrics,
        "validation": val_metrics,
        "internal_test": test_metrics,
        "gate_config": saved["gate_config"],
        "selection_contract": {
            "score": "mae + zero_selection_weight*zero_mae + high_penalty_weight*high_excess",
            "high_limit": "base high-demand MAE * (1 + high_tolerance_pct/100)",
            "external_screen_acceptance": {
                "overall_mae_improvement_min_pct": 0.5,
                "zero_demand_mae_improvement_min_pct": 15.0,
                "high_demand_mae_degradation_max_pct": 1.0,
                "rmse_and_net_mae_degradation_max_pct": 1.0,
            },
        },
        "data_contract": {
            "train_npz_sha256": metadata["train_npz_sha256"],
            "node_signature": graph_audit["node_signature"],
            "history_only_features": list(SparseDemandGate.FEATURE_NAMES),
            "holdout_used_for_training_or_selection": False,
        },
        "args": vars(args),
        "environment": environment_info(),
    }
    save_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
