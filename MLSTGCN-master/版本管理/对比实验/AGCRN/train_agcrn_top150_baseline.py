import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models import AGCRNBaseline  # noqa: E402


DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "temporal_data"
    / "bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20_train2025_hist168"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "分析结果"
    / "对比实验"
    / "AGCRN"
    / "top150_hist168_pred6_seed0"
)
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


def load_split(data_dir, split):
    data = np.load(data_dir / ("%s.npz" % split), allow_pickle=True)
    return data["x"].astype(np.float32), data["y"].astype(np.float32), data


def make_loader(x, y, batch_size, shuffle):
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def compute_stats(x_train, y_train):
    feature_mean = x_train.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    feature_std = x_train.std(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)

    target_mean = y_train.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    target_std = y_train.std(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    target_std = np.where(target_std == 0, 1.0, target_std)
    return feature_mean, feature_std, target_mean, target_std


def compute_metrics(pred, true, mape_eps=1.0):
    err = np.abs(pred - true)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "mape": float((err / np.maximum(np.abs(true), float(mape_eps))).mean()),
    }


def horizon_metrics(pred, true, mape_eps=1.0):
    rows = []
    err = np.abs(pred - true)
    sq = (pred - true) ** 2
    denom = np.maximum(np.abs(true), float(mape_eps))
    for idx in range(pred.shape[1]):
        rows.append(
            {
                "horizon": idx + 1,
                "mae_avg": float(err[:, idx].mean()),
                "mae_out": float(err[:, idx, :, 0].mean()),
                "mae_in": float(err[:, idx, :, 1].mean()),
                "rmse_avg": float(np.sqrt(sq[:, idx].mean())),
                "mape_avg": float((err[:, idx] / denom[:, idx]).mean()),
            }
        )
    return pd.DataFrame(rows)


def anchor_metrics(pred, true, anchor_hours):
    rows = []
    err = np.abs(pred - true)
    for anchor in sorted(set(anchor_hours.tolist())):
        mask = anchor_hours == anchor
        rows.append(
            {
                "anchor_hour": int(anchor),
                "samples": int(mask.sum()),
                "mae_avg": float(err[mask].mean()),
                "mae_out": float(err[mask, :, :, 0].mean()),
                "mae_in": float(err[mask, :, :, 1].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_prediction(model, x_scaled, batch_size, device, target_mean, target_std):
    loader = DataLoader(torch.from_numpy(x_scaled), batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            pred = model(xb)
            pred = F.softplus(pred * target_std + target_mean, beta=5.0)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def init_wandb_run(args, output_dir):
    if args.logger != "wandb":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logger requested, but wandb is unavailable.") from exc

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or output_dir.name,
        config=vars(args),
        tags=tags or None,
    )


def npz_list(data, key):
    if key not in data:
        return []
    return [str(item) for item in data[key].tolist()]


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip, max_batches):
    model.train()
    losses = []
    for batch_idx, (xb, yb) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def validate(model, loader, loss_fn, device, target_mean_t, target_std_t, max_batches):
    model.eval()
    losses = []
    preds = []
    trues = []
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            losses.append(float(loss_fn(pred, yb).item()))
            pred_raw = F.softplus(pred * target_std_t + target_mean_t, beta=5.0)
            true_raw = yb * target_std_t + target_mean_t
            preds.append(pred_raw.cpu().numpy())
            trues.append(true_raw.cpu().numpy())

    pred_np = np.concatenate(preds, axis=0)
    true_np = np.concatenate(trues, axis=0)
    metrics = compute_metrics(pred_np, true_np)
    return float(np.mean(losses)), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--rnn_units", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=10)
    parser.add_argument("--cheb_k", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mape_epsilon", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--logger", choices=["csv", "wandb"], default="csv")
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_tags", default="agcrn,baseline,compare")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    wandb_run = init_wandb_run(args, output_dir)

    x_train, y_train, train_npz = load_split(data_dir, "train")
    x_val, y_val, val_npz = load_split(data_dir, "val")
    x_test, y_test, test_npz = load_split(data_dir, "test")

    feature_mean, feature_std, target_mean, target_std = compute_stats(x_train, y_train)
    x_train_scaled = (x_train - feature_mean) / feature_std
    x_val_scaled = (x_val - feature_mean) / feature_std
    x_test_scaled = (x_test - feature_mean) / feature_std
    y_train_scaled = (y_train - target_mean) / target_std
    y_val_scaled = (y_val - target_mean) / target_std

    train_loader = make_loader(x_train_scaled, y_train_scaled, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val_scaled, y_val_scaled, args.batch_size, shuffle=False)

    _, hist_len, num_nodes, in_dim = x_train.shape
    pred_len = y_train.shape[1]
    out_dim = y_train.shape[-1]
    model = AGCRNBaseline(
        num_nodes=num_nodes,
        input_dim=in_dim,
        rnn_units=args.rnn_units,
        output_dim=out_dim,
        horizon=pred_len,
        num_layers=args.num_layers,
        cheb_k=args.cheb_k,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    target_mean_t = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    target_std_t = torch.as_tensor(target_std, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_path = output_dir / "best_agcrn_baseline.pt"
    history = []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            args.grad_clip,
            args.max_train_batches,
        )
        val_loss, val_metrics = validate(
            model,
            val_loader,
            loss_fn,
            device,
            target_mean_t,
            target_std_t,
            args.max_val_batches,
        )
        val_mae = val_metrics["mae"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_rmse": val_metrics["rmse"],
            "val_mape": val_metrics["mape"],
        }
        history.append(row)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/mae": val_metrics["mae"],
                    "val/rmse": val_metrics["rmse"],
                    "val/mape": val_metrics["mape"],
                    "best/val_mae_so_far": min(best_val, val_mae),
                },
                step=epoch,
            )
        print(
            "epoch=%03d train_loss=%.4f val_loss=%.4f val_mae=%.4f val_rmse=%.4f"
            % (epoch, train_loss, val_loss, val_metrics["mae"], val_metrics["rmse"])
        )

        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "args": vars(args),
                    "best_val_mae": best_val,
                    "best_epoch": best_epoch,
                    "hist_len": hist_len,
                    "pred_len": pred_len,
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                    "num_nodes": num_nodes,
                    "input_feature_cols": npz_list(train_npz, "input_feature_cols"),
                    "history_feature_cols": npz_list(train_npz, "history_feature_cols"),
                    "known_future_feature_cols": npz_list(train_npz, "known_future_feature_cols"),
                    "log1p_feature_cols": npz_list(train_npz, "log1p_feature_cols"),
                    "target_cols": npz_list(train_npz, "target_cols"),
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stop at epoch %d" % epoch)
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_pred = run_prediction(model, x_test_scaled, args.batch_size, device, target_mean_t, target_std_t)
    test_metrics = compute_metrics(test_pred, y_test, mape_eps=args.mape_epsilon)
    test_anchor_hours = test_npz["anchor_hours"] if "anchor_hours" in test_npz else np.full((len(y_test),), -1)

    pd.DataFrame(history).to_csv(output_dir / "agcrn_training_history.csv", index=False, encoding="utf-8-sig")
    horizon_metrics(test_pred, y_test, mape_eps=args.mape_epsilon).to_csv(
        output_dir / "agcrn_internal_test_horizon_mae.csv",
        index=False,
        encoding="utf-8-sig",
    )
    anchor_metrics(test_pred, y_test, test_anchor_hours).to_csv(
        output_dir / "agcrn_internal_test_anchor_mae.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "model": "AGCRN baseline",
        "upstream_repo": "https://github.com/LeiBAI/AGCRN.git",
        "upstream_commit": read_upstream_commit(),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "checkpoint": str(best_path),
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val),
        "internal_test": test_metrics,
        "shape": {
            "hist_len": int(hist_len),
            "pred_len": int(pred_len),
            "num_nodes": int(num_nodes),
            "in_dim": int(in_dim),
            "out_dim": int(out_dim),
        },
        "feature_cols": npz_list(train_npz, "input_feature_cols"),
        "target_cols": npz_list(train_npz, "target_cols"),
        "args": vars(args),
    }
    (output_dir / "agcrn_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = int(best_epoch)
        wandb_run.summary["best_val_mae"] = float(best_val)
        wandb_run.summary["internal_test_mae"] = float(test_metrics["mae"])
        wandb_run.summary["internal_test_rmse"] = float(test_metrics["rmse"])
        wandb_run.summary["checkpoint"] = str(best_path)
        wandb_run.finish()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def read_upstream_commit():
    head_path = SCRIPT_DIR / "upstream" / ".git" / "HEAD"
    if not head_path.exists():
        return None
    head_value = head_path.read_text(encoding="utf-8").strip()
    if not head_value.startswith("ref: "):
        return head_value
    ref_path = SCRIPT_DIR / "upstream" / ".git" / head_value[5:]
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()
    return None


if __name__ == "__main__":
    main()
