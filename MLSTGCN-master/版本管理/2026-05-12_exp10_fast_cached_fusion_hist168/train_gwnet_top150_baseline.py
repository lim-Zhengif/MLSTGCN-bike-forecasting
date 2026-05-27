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
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "data" / "graph" / "bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_graph_train2025_hist168"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "\u5206\u6790\u7ed3\u679c" / "2026-05-20_top150_gwnet_baseline_hist168_pred6_seed0"
DEFAULT_WANDB_PROJECT = "top150_rolling6h_model_compare"


GRAPH_FILE_MAP = {
    "dist": "dist.npy",
    "neighb": "neigh.npy",
    "distri": "bike_heuristic.npy",
    "tempp": "tempp_bike.npy",
    "func": "func.npy",
    "od00": "od00.npy",
    "od06": "od06.npy",
    "od12": "od12.npy",
    "od16": "od16.npy",
    "od20": "od20.npy",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_graph_use(raw_value):
    graphs = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    unknown = [name for name in graphs if name not in GRAPH_FILE_MAP]
    if unknown:
        raise ValueError("Unsupported graph names: %s" % ",".join(unknown))
    return graphs


def normalize_adj(adj):
    adj = np.nan_to_num(adj.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    adj = np.maximum(adj, 0.0)
    np.fill_diagonal(adj, 1.0)
    rowsum = adj.sum(axis=1, keepdims=True)
    rowsum = np.where(rowsum == 0, 1.0, rowsum)
    return adj / rowsum


def load_supports(graph_dir, graph_use):
    supports = []
    for name in graph_use:
        path = graph_dir / GRAPH_FILE_MAP[name]
        if not path.exists():
            raise FileNotFoundError("Missing graph file: %s" % path)
        supports.append(torch.from_numpy(normalize_adj(np.load(path))))
    return supports


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
    return {
        "mae": float(np.abs(pred - true).mean()),
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


class GraphConv(nn.Module):
    def __init__(self, channels, support_len, order, dropout):
        super().__init__()
        self.order = int(order)
        self.dropout = float(dropout)
        self.proj = nn.Conv2d(channels * (support_len * self.order + 1), channels, kernel_size=(1, 1))

    def forward(self, x, supports):
        out = [x]
        for support in supports:
            x1 = torch.einsum("bcnt,nm->bcmt", x, support)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = torch.einsum("bcnt,nm->bcmt", x1, support)
                out.append(x1)
        h = torch.cat(out, dim=1)
        h = self.proj(h)
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNetBaseline(nn.Module):
    def __init__(
        self,
        in_dim,
        pred_len,
        out_dim,
        num_nodes,
        supports,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        layers=8,
        dilation_cycle=4,
        kernel_size=2,
        gcn_order=2,
        dropout=0.2,
        adaptive_adj=True,
        node_embed_dim=10,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)
        self.num_nodes = int(num_nodes)
        self.dropout = float(dropout)
        self.kernel_size = int(kernel_size)
        self.adaptive_adj = bool(adaptive_adj)
        self.register_buffer_count = 0
        for idx, support in enumerate(supports):
            self.register_buffer("support_%d" % idx, support.float())
            self.register_buffer_count += 1

        if self.adaptive_adj:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, node_embed_dim), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(node_embed_dim, num_nodes), requires_grad=True)
            support_len = len(supports) + 1
        else:
            support_len = len(supports)

        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconvs = nn.ModuleList()

        for layer in range(int(layers)):
            dilation = 2 ** (layer % int(dilation_cycle))
            self.filter_convs.append(nn.Conv2d(residual_channels, residual_channels, (1, kernel_size), dilation=(1, dilation)))
            self.gate_convs.append(nn.Conv2d(residual_channels, residual_channels, (1, kernel_size), dilation=(1, dilation)))
            self.residual_convs.append(nn.Conv2d(residual_channels, residual_channels, kernel_size=(1, 1)))
            self.skip_convs.append(nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channels))
            self.gconvs.append(GraphConv(residual_channels, support_len, gcn_order, dropout))

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, pred_len * out_dim, kernel_size=(1, 1))

    def get_supports(self):
        supports = [getattr(self, "support_%d" % idx) for idx in range(self.register_buffer_count)]
        if self.adaptive_adj:
            adaptive = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            supports = supports + [adaptive]
        return supports

    def forward(self, x):
        # x: [B, T, N, F]
        x = x.permute(0, 3, 2, 1)
        supports = self.get_supports()
        x = self.start_conv(x)
        skip = 0
        for filter_conv, gate_conv, residual_conv, skip_conv, bn, gconv in zip(
            self.filter_convs,
            self.gate_convs,
            self.residual_convs,
            self.skip_convs,
            self.bn,
            self.gconvs,
        ):
            residual = x
            pad = (self.kernel_size - 1) * filter_conv.dilation[1]
            x_padded = F.pad(x, (pad, 0, 0, 0))
            x = torch.tanh(filter_conv(x_padded)) * torch.sigmoid(gate_conv(x_padded))
            s = skip_conv(x)
            skip = s if isinstance(skip, int) else skip[..., -s.size(3):] + s
            x = gconv(x, supports)
            x = residual_conv(x)
            x = x + residual[..., -x.size(3):]
            x = bn(x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x[..., -1]
        return x.view(x.size(0), self.pred_len, self.out_dim, self.num_nodes).permute(0, 1, 3, 2)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--graph_dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--graph_use", default="dist,neighb,distri,tempp,func,od00,od06,od12,od16,od20")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--residual_channels", type=int, default=32)
    parser.add_argument("--skip_channels", type=int, default=64)
    parser.add_argument("--end_channels", type=int, default=128)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--dilation_cycle", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--gcn_order", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--adaptive_adj", choices=["true", "false"], default="true")
    parser.add_argument("--node_embed_dim", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--logger", choices=["csv", "wandb"], default="csv")
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_tags", default="gwnet,baseline")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_use = parse_graph_use(args.graph_use)
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    wandb_run = init_wandb_run(args, output_dir)

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
    supports = [support.to(device) for support in load_supports(graph_dir, graph_use)]
    model = GraphWaveNetBaseline(
        in_dim=in_dim,
        pred_len=pred_len,
        out_dim=out_dim,
        num_nodes=num_nodes,
        supports=supports,
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        layers=args.layers,
        dilation_cycle=args.dilation_cycle,
        kernel_size=args.kernel_size,
        gcn_order=args.gcn_order,
        dropout=args.dropout,
        adaptive_adj=args.adaptive_adj == "true",
        node_embed_dim=args.node_embed_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    target_mean_t = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    target_std_t = torch.as_tensor(target_std, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_path = output_dir / "best_gwnet_baseline.pt"
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
                    "args": vars(args),
                    "graph_use": graph_use,
                    "best_val_mae": best_val,
                    "best_epoch": best_epoch,
                    "hist_len": hist_len,
                    "pred_len": pred_len,
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                    "num_nodes": num_nodes,
                    "input_feature_cols": [str(item) for item in train_npz["input_feature_cols"].tolist()],
                    "history_feature_cols": [str(item) for item in train_npz["history_feature_cols"].tolist()],
                    "known_future_feature_cols": [str(item) for item in train_npz["known_future_feature_cols"].tolist()],
                    "log1p_feature_cols": [str(item) for item in train_npz["log1p_feature_cols"].tolist()],
                    "target_cols": [str(item) for item in train_npz["target_cols"].tolist()],
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

    pd.DataFrame(history).to_csv(output_dir / "gwnet_training_history.csv", index=False, encoding="utf-8-sig")
    horizon_metrics(test_pred, y_test).to_csv(output_dir / "gwnet_internal_test_horizon_mae.csv", index=False, encoding="utf-8-sig")
    anchor_metrics(test_pred, y_test, test_anchor_hours).to_csv(output_dir / "gwnet_internal_test_anchor_mae.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": "Graph WaveNet-style baseline",
        "data_dir": str(data_dir),
        "graph_dir": str(graph_dir),
        "graph_use": graph_use,
        "checkpoint": str(best_path),
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val),
        "internal_test": test_metrics,
        "args": vars(args),
    }
    (output_dir / "gwnet_training_summary.json").write_text(
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
