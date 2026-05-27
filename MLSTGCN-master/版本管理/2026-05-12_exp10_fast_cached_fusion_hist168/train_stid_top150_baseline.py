import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "temporal_data" / "bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20_train2025_hist168"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "分析结果" / "2026-05-15_top150_stid_baseline_hist168_pred6_seed0"
DEFAULT_WANDB_PROJECT = "top150_rolling6h_model_compare"


class STIDBaseline(nn.Module):
    """STID-style baseline: history projection + node/time embeddings + MLP decoder."""

    def __init__(
        self,
        hist_len,
        in_dim,
        pred_len,
        out_dim,
        num_nodes,
        hidden_dim=256,
        node_embed_dim=32,
        time_embed_dim=16,
        weekday_embed_dim=16,
        num_layers=2,
        dropout=0.1,
        hour_feature_idx=None,
        weekday_feature_idx=None,
        feature_mean=None,
        feature_std=None,
    ):
        super().__init__()
        self.hist_len = int(hist_len)
        self.in_dim = int(in_dim)
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)
        self.num_nodes = int(num_nodes)
        self.hour_feature_idx = hour_feature_idx
        self.weekday_feature_idx = weekday_feature_idx

        self.history_proj = nn.Linear(self.hist_len * self.in_dim, hidden_dim)
        self.node_embedding = nn.Embedding(self.num_nodes, node_embed_dim)
        self.hour_embedding = nn.Embedding(24, time_embed_dim) if hour_feature_idx is not None and time_embed_dim > 0 else None
        self.weekday_embedding = nn.Embedding(7, weekday_embed_dim) if weekday_feature_idx is not None and weekday_embed_dim > 0 else None

        mlp_in_dim = hidden_dim + node_embed_dim
        if self.hour_embedding is not None:
            mlp_in_dim += time_embed_dim
        if self.weekday_embedding is not None:
            mlp_in_dim += weekday_embed_dim

        layers = []
        current_dim = mlp_in_dim
        for _ in range(max(int(num_layers), 1)):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, self.pred_len * self.out_dim))
        self.decoder = nn.Sequential(*layers)

        if feature_mean is not None and feature_std is not None:
            self.register_buffer("feature_mean", torch.as_tensor(feature_mean.reshape(-1), dtype=torch.float32))
            self.register_buffer("feature_std", torch.as_tensor(feature_std.reshape(-1), dtype=torch.float32))
        else:
            self.feature_mean = None
            self.feature_std = None

    def _recover_categorical(self, x, feature_idx, modulo):
        feature = x[:, -1, :, feature_idx]
        if self.feature_mean is not None and self.feature_std is not None:
            feature = feature * self.feature_std[feature_idx] + self.feature_mean[feature_idx]
        return feature.round().long().clamp(0, modulo - 1)

    def forward(self, x):
        # x: [B, T, N, F]
        batch_size, hist_len, num_nodes, in_dim = x.shape
        history = x.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, hist_len * in_dim)
        history_hidden = F.relu(self.history_proj(history))

        station_ids = torch.arange(num_nodes, device=x.device)
        node_emb = self.node_embedding(station_ids).unsqueeze(0).expand(batch_size, -1, -1)
        parts = [history_hidden, node_emb]

        if self.hour_embedding is not None:
            hour_ids = self._recover_categorical(x, self.hour_feature_idx, 24)
            parts.append(self.hour_embedding(hour_ids))
        if self.weekday_embedding is not None:
            weekday_ids = self._recover_categorical(x, self.weekday_feature_idx, 7)
            parts.append(self.weekday_embedding(weekday_ids))

        hidden = torch.cat(parts, dim=-1)
        out = self.decoder(hidden)
        return out.view(batch_size, num_nodes, self.pred_len, self.out_dim).permute(0, 2, 1, 3)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(data_dir, split):
    data = np.load(data_dir / ("%s.npz" % split), allow_pickle=True)
    return data["x"].astype(np.float32), data["y"].astype(np.float32), data


def make_loader(x, y, batch_size, shuffle):
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def compute_metrics(pred, true):
    err = np.abs(pred - true)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
    }


def horizon_metrics(pred, true):
    rows = []
    err = np.abs(pred - true)
    sq = (pred - true) ** 2
    for idx in range(pred.shape[1]):
        rows.append({
            "horizon": idx + 1,
            "mae_avg": float(err[:, idx].mean()),
            "mae_out": float(err[:, idx, :, 0].mean()),
            "mae_in": float(err[:, idx, :, 1].mean()),
            "rmse_avg": float(np.sqrt(sq[:, idx].mean())),
        })
    return pd.DataFrame(rows)


def anchor_metrics(pred, true, anchor_hours):
    rows = []
    err = np.abs(pred - true)
    for anchor in sorted(set(anchor_hours.tolist())):
        mask = anchor_hours == anchor
        rows.append({
            "anchor_hour": int(anchor),
            "samples": int(mask.sum()),
            "mae_avg": float(err[mask].mean()),
            "mae_out": float(err[mask, :, :, 0].mean()),
            "mae_in": float(err[mask, :, :, 1].mean()),
        })
    return pd.DataFrame(rows)


def find_feature_idx(feature_cols, candidates):
    for candidate in candidates:
        if candidate in feature_cols:
            return feature_cols.index(candidate)
    return None


def run_prediction(model, x_scaled, batch_size, device, target_mean, target_std):
    loader = DataLoader(torch.from_numpy(x_scaled), batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            pred = model(xb)
            pred = pred * target_std + target_mean
            pred = F.softplus(pred, beta=5.0)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def init_wandb_run(args, output_dir):
    if args.logger != "wandb":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logger requested, but wandb is unavailable in the current environment.") from exc

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or output_dir.name,
        config=vars(args),
        tags=tags or None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--node_embed_dim", type=int, default=32)
    parser.add_argument("--time_embed_dim", type=int, default=16)
    parser.add_argument("--weekday_embed_dim", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--logger", choices=["csv", "wandb"], default="csv")
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_tags", default="stid,baseline")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb_run(args, output_dir)
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    x_train, y_train, train_npz = load_split(data_dir, "train")
    x_val, y_val, val_npz = load_split(data_dir, "val")
    x_test, y_test, test_npz = load_split(data_dir, "test")

    feature_cols = [str(item) for item in train_npz["input_feature_cols"].tolist()]
    hour_idx = find_feature_idx(feature_cols, ["小时", "future_小时"])
    weekday_idx = find_feature_idx(feature_cols, ["星期几", "future_星期几"])

    feature_mean = x_train.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    feature_std = x_train.std(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    target_mean = y_train.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    target_std = y_train.std(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    target_std = np.where(target_std == 0, 1.0, target_std)

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
    model = STIDBaseline(
        hist_len=hist_len,
        in_dim=in_dim,
        pred_len=pred_len,
        out_dim=out_dim,
        num_nodes=num_nodes,
        hidden_dim=args.hidden_dim,
        node_embed_dim=args.node_embed_dim,
        time_embed_dim=args.time_embed_dim,
        weekday_embed_dim=args.weekday_embed_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        hour_feature_idx=hour_idx,
        weekday_feature_idx=weekday_idx,
        feature_mean=feature_mean,
        feature_std=feature_std,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    target_mean_t = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    target_std_t = torch.as_tensor(target_std, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_path = output_dir / "best_stid_baseline.pt"
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(float(loss_fn(pred, yb).item()))
                pred_raw = F.softplus(pred * target_std_t + target_mean_t, beta=5.0)
                true_raw = yb * target_std_t + target_mean_t
                val_preds.append(pred_raw.cpu().numpy())
                val_trues.append(true_raw.cpu().numpy())

        val_pred_np = np.concatenate(val_preds, axis=0)
        val_true_np = np.concatenate(val_trues, axis=0)
        val_mae = compute_metrics(val_pred_np, val_true_np)["mae"]
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "val_mae": val_mae,
        }
        history.append(row)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": row["train_loss"],
                    "val/loss": row["val_loss"],
                    "val/mae": row["val_mae"],
                    "best/val_mae_so_far": min(best_val, val_mae),
                },
                step=epoch,
            )
        print("epoch=%03d train_loss=%.4f val_loss=%.4f val_mae=%.4f" % (epoch, row["train_loss"], row["val_loss"], row["val_mae"]))

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
                    "feature_cols": feature_cols,
                    "args": vars(args),
                    "best_val_mae": best_val,
                    "best_epoch": best_epoch,
                    "hist_len": hist_len,
                    "pred_len": pred_len,
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                    "num_nodes": num_nodes,
                    "hour_feature_idx": hour_idx,
                    "weekday_feature_idx": weekday_idx,
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
    test_metrics = compute_metrics(test_pred, y_test)
    test_anchor_hours = test_npz["anchor_hours"] if "anchor_hours" in test_npz else np.full((len(y_test),), -1)

    pd.DataFrame(history).to_csv(output_dir / "stid_training_history.csv", index=False, encoding="utf-8-sig")
    horizon_metrics(test_pred, y_test).to_csv(output_dir / "stid_internal_test_horizon_mae.csv", index=False, encoding="utf-8-sig")
    anchor_metrics(test_pred, y_test, test_anchor_hours).to_csv(output_dir / "stid_internal_test_anchor_mae.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": "STID-style baseline",
        "data_dir": str(data_dir),
        "checkpoint": str(best_path),
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val),
        "internal_test": test_metrics,
        "feature_cols": feature_cols,
        "hour_feature_idx": hour_idx,
        "weekday_feature_idx": weekday_idx,
        "args": vars(args),
    }
    (output_dir / "stid_training_summary.json").write_text(
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


if __name__ == "__main__":
    main()
