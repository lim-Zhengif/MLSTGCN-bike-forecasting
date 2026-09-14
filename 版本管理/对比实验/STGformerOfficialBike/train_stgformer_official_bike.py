# -*- coding: utf-8 -*-
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
for source_dir in [SCRIPT_DIR]:
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from protocol import (  # noqa: E402
    DEFAULT_WANDB_PROJECT,
    MODEL_ID,
    UPSTREAM_COMMIT,
    anchor_metrics,
    audit_graph_contract,
    compute_metrics,
    compute_stats_float64,
    environment_info,
    horizon_metrics,
    init_wandb_run,
    npz_list,
    resolve_device,
    save_json,
    set_seed,
)
from models import STGformerOfficialBike, build_model_from_checkpoint  # noqa: E402


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "temporal_data" / (
    "bike_hourly_safe_inventory_top150_nyc_full_predlen3_"
    "anchors_00_03_06_09_12_15_18_21_train2025_hist168"
)
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "data" / "graph" / (
    "bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_graph_"
    "train2025_hist168_pred3_8anchors"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "分析结果" / "对比实验" / "STGformerOfficialBike" / (
    "b0_top150_hist168_pred3_seed0_bs16"
)


class NormalizedPairDataset(Dataset):
    def __init__(self, x, y, feature_mean, feature_std, target_mean, target_std):
        self.x = x
        self.y = y
        self.feature_mean = torch.as_tensor(feature_mean.reshape(-1), dtype=torch.float32)
        self.feature_std = torch.as_tensor(feature_std.reshape(-1), dtype=torch.float32)
        self.target_mean = torch.as_tensor(target_mean.reshape(-1), dtype=torch.float32)
        self.target_std = torch.as_tensor(target_std.reshape(-1), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = torch.from_numpy(np.asarray(self.x[index], dtype=np.float32))
        y = torch.from_numpy(np.asarray(self.y[index], dtype=np.float32))
        return (x - self.feature_mean) / self.feature_std, (y - self.target_mean) / self.target_std


class NormalizedFeatureDataset(Dataset):
    def __init__(self, x, feature_mean, feature_std):
        self.x = x
        self.feature_mean = torch.as_tensor(feature_mean.reshape(-1), dtype=torch.float32)
        self.feature_std = torch.as_tensor(feature_std.reshape(-1), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = torch.from_numpy(np.asarray(self.x[index], dtype=np.float32))
        return (x - self.feature_mean) / self.feature_std


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def denormalize_prediction(prediction, target_mean, target_std):
    return F.softplus(prediction * target_std + target_mean, beta=5.0)


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip, max_batches):
    model.train()
    losses = []
    started = time.perf_counter()
    for batch_index, (x_batch, y_batch) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_batch)
        loss = loss_fn(prediction, y_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss at batch %d" % batch_index)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("Training loader produced no batches")
    return float(np.mean(losses)), time.perf_counter() - started


def validate(model, loader, loss_fn, device, target_mean, target_std, max_batches, mape_eps):
    model.eval()
    losses = []
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_index, (x_batch, y_batch) in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            prediction = model(x_batch)
            losses.append(float(loss_fn(prediction, y_batch).item()))
            predictions.append(denormalize_prediction(prediction, target_mean, target_std).cpu().numpy())
            targets.append((y_batch * target_std + target_mean).cpu().numpy())
    if not predictions:
        raise RuntimeError("Validation loader produced no batches")
    pred = np.concatenate(predictions, axis=0)
    true = np.concatenate(targets, axis=0)
    return float(np.mean(losses)), compute_metrics(pred, true, mape_eps=mape_eps)


def run_prediction(model, x, feature_mean, feature_std, batch_size, num_workers, device, target_mean, target_std):
    loader = make_loader(
        NormalizedFeatureDataset(x, feature_mean, feature_std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    outputs = []
    model.eval()
    with torch.no_grad():
        for x_batch in loader:
            prediction = model(x_batch.to(device, non_blocking=True))
            outputs.append(denormalize_prediction(prediction, target_mean, target_std).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Train B0 official-core STGformer bike adapter")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--graph_dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--graph_name", default="dist")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--input_embedding_dim", type=int, default=24)
    parser.add_argument("--tod_embedding_dim", type=int, default=12)
    parser.add_argument("--dow_embedding_dim", type=int, default=12)
    parser.add_argument("--adaptive_embedding_dim", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dropout_a", type=float, default=0.3)
    parser.add_argument("--mlp_ratio", type=float, default=2.0)
    parser.add_argument("--kernel_size", type=int, default=1)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--hour_feature_index", type=int, default=3)
    parser.add_argument("--dow_feature_index", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mape_epsilon", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--logger", choices=["csv", "wandb"], default="csv")
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_tags", default="stgformer,official-core,b0,top150,hist168,pred3")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    wandb_run = init_wandb_run(args, output_dir)

    train_npz = np.load(data_dir / "train.npz", allow_pickle=True)
    val_npz = np.load(data_dir / "val.npz", allow_pickle=True)
    x_train, y_train = train_npz["x"], train_npz["y"]
    x_val, y_val = val_npz["x"], val_npz["y"]
    feature_mean, feature_std, target_mean, target_std = compute_stats_float64(x_train, y_train)
    _, hist_len, num_nodes, in_dim = x_train.shape
    pred_len, out_dim = y_train.shape[1], y_train.shape[3]
    graph_audit = audit_graph_contract(graph_dir, args.graph_name, num_nodes)

    train_loader = make_loader(
        NormalizedPairDataset(x_train, y_train, feature_mean, feature_std, target_mean, target_std),
        args.batch_size,
        True,
        args.num_workers,
    )
    val_loader = make_loader(
        NormalizedPairDataset(x_val, y_val, feature_mean, feature_std, target_mean, target_std),
        args.batch_size,
        False,
        args.num_workers,
    )
    model = STGformerOfficialBike(
        num_nodes=num_nodes,
        in_steps=hist_len,
        out_steps=pred_len,
        input_dim=in_dim,
        output_dim=out_dim,
        feature_mean=feature_mean,
        feature_std=feature_std,
        steps_per_day=24,
        hour_feature_index=args.hour_feature_index,
        dow_feature_index=args.dow_feature_index,
        input_embedding_dim=args.input_embedding_dim,
        tod_embedding_dim=args.tod_embedding_dim,
        dow_embedding_dim=args.dow_embedding_dim,
        adaptive_embedding_dim=args.adaptive_embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        dropout_a=args.dropout_a,
        kernel_size=args.kernel_size,
        order=args.order,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    target_mean_t = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    target_std_t = torch.as_tensor(target_std, dtype=torch.float32, device=device)
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    checkpoint_path = output_dir / "best_stgformer_official_b0.pt"
    history = []

    for epoch in range(args.epochs):
        train_loss, epoch_seconds = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, args.grad_clip, args.max_train_batches
        )
        val_loss, val_metrics = validate(
            model,
            val_loader,
            loss_fn,
            device,
            target_mean_t,
            target_std_t,
            args.max_val_batches,
            args.mape_epsilon,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_mape": val_metrics["mape"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if wandb_run is not None:
            wandb_run.log(row, step=epoch)
        if val_metrics["mae"] < best_val:
            best_val = float(val_metrics["mae"])
            best_epoch = int(epoch)
            bad_epochs = 0
            torch.save(
                {
                    "schema_version": 1,
                    "model_id": MODEL_ID,
                    "upstream_commit": UPSTREAM_COMMIT,
                    "model_config": model.export_config(),
                    "model_state_dict": model.state_dict(),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "best_val_mae": best_val,
                    "best_epoch": best_epoch,
                    "input_feature_cols": npz_list(train_npz, "input_feature_cols"),
                    "target_cols": npz_list(train_npz, "target_cols"),
                    "graph_audit": graph_audit,
                    "args": vars(args),
                    "environment": environment_info(),
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stop at epoch %d" % epoch)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, checkpoint_audit = build_model_from_checkpoint(checkpoint, device=device)
    test_npz = np.load(data_dir / "test.npz", allow_pickle=True)
    test_prediction = run_prediction(
        model,
        test_npz["x"],
        feature_mean,
        feature_std,
        args.batch_size,
        args.num_workers,
        device,
        target_mean_t,
        target_std_t,
    )
    y_test = test_npz["y"].astype(np.float32)
    test_metrics = compute_metrics(test_prediction, y_test, mape_eps=args.mape_epsilon)
    test_anchor_hours = test_npz["anchor_hours"] if "anchor_hours" in test_npz else np.full(len(y_test), -1)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    horizon_metrics(test_prediction, y_test, mape_eps=args.mape_epsilon).to_csv(
        output_dir / "internal_test_horizon_metrics.csv", index=False, encoding="utf-8-sig"
    )
    anchor_metrics(test_prediction, y_test, test_anchor_hours).to_csv(
        output_dir / "internal_test_anchor_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "schema_version": 1,
        "model": MODEL_ID,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint": str(checkpoint_path),
        "checkpoint_missing_keys": checkpoint_audit["missing_keys"],
        "checkpoint_unexpected_keys": checkpoint_audit["unexpected_keys"],
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "internal_test": test_metrics,
        "shape": {
            "hist_len": hist_len,
            "pred_len": pred_len,
            "num_nodes": num_nodes,
            "in_dim": in_dim,
            "out_dim": out_dim,
        },
        "graph_audit": graph_audit,
        "input_feature_cols": npz_list(train_npz, "input_feature_cols"),
        "target_cols": npz_list(train_npz, "target_cols"),
        "environment": environment_info(),
        "args": vars(args),
    }
    save_json(output_dir / "training_summary.json", summary)
    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
