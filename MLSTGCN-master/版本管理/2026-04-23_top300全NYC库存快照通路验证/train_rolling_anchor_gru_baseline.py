import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "temporal_data" / "bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "分析结果" / "2026-05-06_top150_rolling_anchor_gru_baseline"
DEFAULT_EXISTING_BASELINE = PROJECT_ROOT / "分析结果" / "2026-04-28_top150_predlen6_rolling_anchor_holdout_eval" / "feb2026_model_vs_naive_baselines.csv"
DEFAULT_CURRENT_MODEL_METRICS = PROJECT_ROOT / "logs" / "bike_hourly_safe_inventory_top150_nyc_full_exp06_predlen6_anchors_00_06_12_16_20_bs8_seed0" / "version_0" / "metrics.csv"


class SharedStationGRU(nn.Module):
    def __init__(self, in_dim, hidden_dim, pred_len, out_dim, num_nodes, num_layers=1, dropout=0.0, station_embed_dim=16):
        super().__init__()
        self.pred_len = pred_len
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.station_embed_dim = station_embed_dim
        if station_embed_dim > 0:
            self.station_embedding = nn.Embedding(num_nodes, station_embed_dim)
        else:
            self.station_embedding = None
        gru_input_dim = in_dim + station_embed_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * out_dim),
        )

    def forward(self, x):
        # x: [B, T, N, F]. Each station is modeled as an independent sequence.
        batch_size, hist_len, num_nodes, in_dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, hist_len, in_dim)
        if self.station_embedding is not None:
            station_ids = torch.arange(num_nodes, device=x.device).repeat(batch_size)
            station_emb = self.station_embedding(station_ids).unsqueeze(1).expand(-1, hist_len, -1)
            x = torch.cat([x, station_emb], dim=-1)
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]
        out = self.head(last_hidden)
        out = out.reshape(batch_size, num_nodes, self.pred_len, self.out_dim).permute(0, 2, 1, 3)
        return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(data_dir, split):
    data = np.load(data_dir / f"{split}.npz", allow_pickle=True)
    return data["x"].astype(np.float32), data["y"].astype(np.float32), data


def make_loader(x, y, batch_size, shuffle):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


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
        rows.append(
            {
                "horizon": idx + 1,
                "mae_avg": float(err[:, idx].mean()),
                "mae_out": float(err[:, idx, :, 0].mean()),
                "mae_in": float(err[:, idx, :, 1].mean()),
                "rmse_avg": float(np.sqrt(sq[:, idx].mean())),
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
                "true_total_mean": float(true[mask].sum(axis=-1).mean()),
                "pred_total_mean": float(pred[mask].sum(axis=-1).mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--existing_baseline_csv", default=str(DEFAULT_EXISTING_BASELINE))
    parser.add_argument("--current_model_metrics_csv", default=str(DEFAULT_CURRENT_MODEL_METRICS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--station_embed_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    x_train, y_train, train_npz = load_split(data_dir, "train")
    x_val, y_val, val_npz = load_split(data_dir, "val")
    x_test, y_test, test_npz = load_split(data_dir, "test")

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
    model = SharedStationGRU(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        pred_len=pred_len,
        out_dim=out_dim,
        num_nodes=num_nodes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        station_embed_dim=args.station_embed_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    best_val = float("inf")
    best_epoch = -1
    best_path = output_dir / "best_gru_baseline.pt"
    history = []
    bad_epochs = 0
    target_mean_t = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    target_std_t = torch.as_tensor(target_std, dtype=torch.float32, device=device)

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
        val_preds = []
        val_trues = []
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(float(loss_fn(pred, yb).item()))
                pred_raw = pred * target_std_t + target_mean_t
                true_raw = yb * target_std_t + target_mean_t
                pred_raw = torch.nn.functional.softplus(pred_raw, beta=5.0)
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
        print("epoch=%03d train_loss=%.4f val_loss=%.4f val_mae=%.4f" % (epoch, row["train_loss"], row["val_loss"], row["val_mae"]))

        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "best_val_mae": best_val,
                    "best_epoch": best_epoch,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping at epoch %d" % epoch)
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = make_loader(x_test_scaled, y_test, args.batch_size, shuffle=False)
    test_preds = []
    test_trues = []
    with torch.no_grad():
        for xb, yb_raw in test_loader:
            xb = xb.to(device)
            pred_scaled = model(xb)
            pred_raw = pred_scaled * target_std_t + target_mean_t
            pred_raw = torch.nn.functional.softplus(pred_raw, beta=5.0)
            test_preds.append(pred_raw.cpu().numpy())
            test_trues.append(yb_raw.numpy())

    pred_test = np.concatenate(test_preds, axis=0)
    true_test = np.concatenate(test_trues, axis=0)
    test_metrics = compute_metrics(pred_test, true_test)

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "gru_training_history.csv", index=False, encoding="utf-8-sig")
    horizon_df = horizon_metrics(pred_test, true_test)
    horizon_df.to_csv(output_dir / "gru_horizon_metrics.csv", index=False, encoding="utf-8-sig")

    if "anchor_hours" in test_npz:
        anchors = test_npz["anchor_hours"]
        anchor_df = anchor_metrics(pred_test, true_test, anchors)
        anchor_df.to_csv(output_dir / "gru_anchor_hour_metrics.csv", index=False, encoding="utf-8-sig")
    else:
        anchor_df = pd.DataFrame()

    pd.DataFrame(
        [
            {
                "method": "GRU baseline",
                "mae": test_metrics["mae"],
                "rmse": test_metrics["rmse"],
                "note": "shared station GRU, no graph",
            }
        ]
    ).to_csv(output_dir / "gru_test_metrics.csv", index=False, encoding="utf-8-sig")

    comparison_rows = []
    current_model_metrics_path = Path(args.current_model_metrics_csv)
    if current_model_metrics_path.exists():
        current_metrics = pd.read_csv(current_model_metrics_path)
        current_test = current_metrics[current_metrics.get("test_mae", pd.Series(dtype=object)).notna()]
        if not current_test.empty:
            current_row = current_test.iloc[-1]
            comparison_rows.append(
                {
                    "method": "current_mstgcn_same_test_split",
                    "mae": float(current_row["test_mae"]),
                    "rmse": float(current_row["rmse_avg"]) if "rmse_avg" in current_row and pd.notna(current_row["rmse_avg"]) else np.nan,
                    "note": "current dynamic multi-graph MSTGCN on same rolling-anchor test split",
                }
            )
    existing_path = Path(args.existing_baseline_csv)
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        for _, row in existing.iterrows():
            comparison_rows.append(
                {
                    "method": row["method"],
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "note": row.get("note", ""),
                }
            )
    comparison_rows.append(
        {
            "method": "gru_shared_station_no_graph",
            "mae": test_metrics["mae"],
            "rmse": test_metrics["rmse"],
            "note": "lightweight GRU baseline on rolling-anchor test split",
        }
    )
    comparison_df = pd.DataFrame(comparison_rows).sort_values("mae")
    comparison_df.to_csv(output_dir / "gru_vs_existing_baselines.csv", index=False, encoding="utf-8-sig")

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "shape": {
            "train_x": list(x_train.shape),
            "val_x": list(x_val.shape),
            "test_x": list(x_test.shape),
            "train_y": list(y_train.shape),
            "val_y": list(y_val.shape),
            "test_y": list(y_test.shape),
        },
        "model": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "station_embed_dim": args.station_embed_dim,
            "dropout": args.dropout,
        },
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val),
        "test": test_metrics,
        "best_checkpoint": str(best_path),
    }
    with (output_dir / "gru_baseline_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Comparison table:", output_dir / "gru_vs_existing_baselines.csv")


if __name__ == "__main__":
    main()
