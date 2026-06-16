import argparse
import json
import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "data" / "graph" / "bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_graph_train2025_hist168_pred3_8anchors"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "分析结果" / "2026-06-15_expCSSG_rw_prior" / "cssg_rw_top150_pred3_8anchors"
DEFAULT_GRAPH_USE = "od00,od03,od06,od09,od12,od15,od18,od21"


def parse_graph_use(value):
    names = [item.strip() for item in str(value).split(",") if item.strip()]
    if not names:
        raise ValueError("--graph_use must include at least one OD graph name.")
    invalid = [name for name in names if not name.startswith("od")]
    if invalid:
        raise ValueError("CSSG random-walk prior expects OD graphs, got: %s" % ", ".join(invalid))
    return names


def resolve_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def row_normalize(matrix, eps=1e-12):
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.maximum(matrix, 0.0)
    row_sum = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, np.maximum(row_sum, eps), out=np.zeros_like(matrix), where=row_sum > eps)


def random_walk_average(transition, order, decay):
    if order < 1:
        raise ValueError("--walk_order must be >= 1.")
    transition = row_normalize(transition)
    current = transition.copy()
    total = np.zeros_like(transition, dtype=np.float64)
    weight_total = 0.0
    for hop in range(1, order + 1):
        weight = float(decay) ** (hop - 1)
        total += weight * current
        weight_total += weight
        if hop < order:
            current = current @ transition
    if weight_total <= 0:
        return total
    return total / weight_total


def topk_rows(matrix, topk, keep_self=False):
    if topk <= 0 or topk >= matrix.shape[1]:
        return matrix
    result = np.zeros_like(matrix)
    for row_idx in range(matrix.shape[0]):
        row = matrix[row_idx].copy()
        if not keep_self:
            row[row_idx] = 0.0
        if np.count_nonzero(row) == 0:
            continue
        keep_count = min(topk, np.count_nonzero(row))
        keep_idx = np.argpartition(row, -keep_count)[-keep_count:]
        result[row_idx, keep_idx] = matrix[row_idx, keep_idx]
        if keep_self:
            result[row_idx, row_idx] = matrix[row_idx, row_idx]
    return result


def load_od_graph(graph_dir, name, diag_zero=True):
    path = graph_dir / ("%s.npy" % name)
    if not path.exists():
        raise FileNotFoundError("Missing OD graph: %s" % path)
    matrix = np.load(path).astype(np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("OD graph must be a square matrix: %s shape=%s" % (path, matrix.shape))
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.maximum(matrix, 0.0)
    if diag_zero:
        np.fill_diagonal(matrix, 0.0)
    return matrix


def build_cssg_rw_prior(graphs, beta, walk_order, walk_decay, forward_weight, backward_weight):
    forward_priors = []
    backward_priors = []
    combined_priors = []
    for matrix in graphs:
        base_forward = row_normalize(matrix)
        base_backward = row_normalize(matrix.T)

        # CSSG-inspired bidirectional coupling: forward walks are nudged by
        # incoming structure and backward walks are nudged by outgoing structure.
        forward_transition = row_normalize(base_forward + beta * base_backward)
        backward_transition = row_normalize(base_backward + beta * base_forward)

        forward_prior = random_walk_average(forward_transition, walk_order, walk_decay)
        backward_prior = random_walk_average(backward_transition, walk_order, walk_decay)
        combined_prior = forward_weight * forward_prior + backward_weight * backward_prior

        forward_priors.append(forward_prior)
        backward_priors.append(backward_prior)
        combined_priors.append(combined_prior)

    forward_prior = row_normalize(np.mean(forward_priors, axis=0))
    backward_prior = row_normalize(np.mean(backward_priors, axis=0))
    graph_prior = row_normalize(np.mean(combined_priors, axis=0))
    return graph_prior, forward_prior, backward_prior


def matrix_stats(matrix):
    offdiag = matrix.copy()
    np.fill_diagonal(offdiag, 0.0)
    nonzero = offdiag[offdiag > 0]
    return {
        "shape": list(matrix.shape),
        "min": float(matrix.min()),
        "max": float(matrix.max()),
        "mean": float(matrix.mean()),
        "nonzero": int(np.count_nonzero(offdiag)),
        "density": float(np.count_nonzero(offdiag) / max(offdiag.size - offdiag.shape[0], 1)),
        "row_sum_min": float(matrix.sum(axis=1).min()),
        "row_sum_max": float(matrix.sum(axis=1).max()),
        "nonzero_min": float(nonzero.min()) if nonzero.size else 0.0,
        "nonzero_median": float(np.median(nonzero)) if nonzero.size else 0.0,
        "nonzero_max": float(nonzero.max()) if nonzero.size else 0.0,
        "asymmetry_l1": float(np.abs(matrix - matrix.T).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Build a CSSG-style bidirectional random-walk graph prior from anchor-hour OD graphs.")
    parser.add_argument("--graph_dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--graph_use", default=DEFAULT_GRAPH_USE)
    parser.add_argument("--beta", type=float, default=0.4, help="Bidirectional coupling strength borrowed from CSSG.")
    parser.add_argument("--walk_order", type=int, default=3, help="Maximum random-walk order.")
    parser.add_argument("--walk_decay", type=float, default=0.7, help="Higher-order walk decay.")
    parser.add_argument("--forward_weight", type=float, default=0.5)
    parser.add_argument("--backward_weight", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=20, help="Keep top-k outgoing edges per row after prior construction. <=0 disables.")
    parser.add_argument("--diag_zero", default="true", choices=["true", "false"])
    args = parser.parse_args()

    if args.beta < 0:
        parser.error("--beta must be >= 0.")
    if args.walk_decay <= 0:
        parser.error("--walk_decay must be > 0.")
    total_direction_weight = args.forward_weight + args.backward_weight
    if total_direction_weight <= 0:
        parser.error("--forward_weight + --backward_weight must be > 0.")
    args.forward_weight = args.forward_weight / total_direction_weight
    args.backward_weight = args.backward_weight / total_direction_weight

    graph_dir = resolve_path(args.graph_dir)
    output_dir = resolve_path(args.output_dir)
    graph_names = parse_graph_use(args.graph_use)
    diag_zero = args.diag_zero == "true"

    graphs = [load_od_graph(graph_dir, name, diag_zero=diag_zero) for name in graph_names]
    graph_prior, forward_prior, backward_prior = build_cssg_rw_prior(
        graphs,
        beta=args.beta,
        walk_order=args.walk_order,
        walk_decay=args.walk_decay,
        forward_weight=args.forward_weight,
        backward_weight=args.backward_weight,
    )

    if args.topk > 0:
        graph_prior = row_normalize(topk_rows(graph_prior, args.topk, keep_self=False))
        forward_prior = row_normalize(topk_rows(forward_prior, args.topk, keep_self=False))
        backward_prior = row_normalize(topk_rows(backward_prior, args.topk, keep_self=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "cssg_rw_graph_prior.npy", graph_prior.astype(np.float32))
    np.save(output_dir / "cssg_rw_forward_prior.npy", forward_prior.astype(np.float32))
    np.save(output_dir / "cssg_rw_backward_prior.npy", backward_prior.astype(np.float32))
    np.savez_compressed(
        output_dir / "cssg_rw_prior_bundle.npz",
        graph_prior=graph_prior.astype(np.float32),
        forward_prior=forward_prior.astype(np.float32),
        backward_prior=backward_prior.astype(np.float32),
        graph_names=np.asarray(graph_names),
    )

    summary = {
        "project": "exp-CSSG-rw-prior",
        "version_tag": "2026-06-15_expCSSG_rw_prior",
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "graph_names": graph_names,
        "num_nodes": int(graph_prior.shape[0]),
        "args": vars(args),
        "outputs": {
            "graph_prior": str((output_dir / "cssg_rw_graph_prior.npy").resolve()),
            "forward_prior": str((output_dir / "cssg_rw_forward_prior.npy").resolve()),
            "backward_prior": str((output_dir / "cssg_rw_backward_prior.npy").resolve()),
            "bundle": str((output_dir / "cssg_rw_prior_bundle.npz").resolve()),
        },
        "stats": {
            "graph_prior": matrix_stats(graph_prior),
            "forward_prior": matrix_stats(forward_prior),
            "backward_prior": matrix_stats(backward_prior),
        },
        "note": (
            "CSSG-style prior built from anchor-hour OD graphs. "
            "The single graph prior averages bidirectionally coupled forward/backward random-walk supports "
            "so it can be injected into the existing FusionGraph pipeline as graph_use=cssg_rw."
        ),
    }
    with open(output_dir / "cssg_rw_prior_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("Saved CSSG-RW prior to:", output_dir)
    print(json.dumps(summary["stats"]["graph_prior"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
