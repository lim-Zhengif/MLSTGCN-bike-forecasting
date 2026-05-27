import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_top150_feb2026_holdout_exp10_fast import (
    ANALYSIS_DIR,
    FEB_DATA_DIR,
    ORDER_SUBDIR,
    PIPELINE_STATION_COL,
    PROJECT_ROOT,
    STATION_COL,
    EvalWrapper,
    aggregate_hourly_trip_counts,
    append_periodic_features_if_needed,
    apply_log1p_transform,
    build_categorical_feature_configs,
    build_hourly_feature_frame,
    build_hourly_samples,
    build_hourly_samples_for_anchors,
    cli_value_or_default,
    find_best_checkpoint,
    find_training_metadata,
    fusiongraph_module,
    load_daily_feature_table,
    load_npz_metadata,
    parse_anchor_hours,
    parse_bool_value,
    parse_graph_use,
    get_cli_value,
    resolve_device,
    BikeGraph,
    FusionGraphModel,
)


def build_february_samples(args, metadata, graph_dir):
    order_dir = PROJECT_ROOT / FEB_DATA_DIR / ORDER_SUBDIR
    asset_dir = PROJECT_ROOT / FEB_DATA_DIR / "nyc_top300_inventory_validation"
    weather_file = PROJECT_ROOT / FEB_DATA_DIR / "weather-get" / "NYC_Weather_2024-01-01_to_2026-02-31.csv"
    mapping_path = graph_dir / "selected_node_mapping.csv"

    mapping_df = pd.read_csv(mapping_path).sort_values("Node_ID").reset_index(drop=True)
    station_names = mapping_df[STATION_COL].astype(str).tolist()
    trip_files = sorted(order_dir.glob(args.trip_glob))
    if not trip_files:
        raise FileNotFoundError("No trip files matched: %s" % (order_dir / args.trip_glob))

    hourly_df, _ = aggregate_hourly_trip_counts([str(path) for path in trip_files], station_names)
    all_dates = sorted(hourly_df["日期"].dropna().unique())
    daily_feature_df, _, _, _, _ = load_daily_feature_table(
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

    x_values = sample_bundle["x"].copy()
    history_len = len(metadata["history_feature_cols"])
    history_x, _ = apply_log1p_transform(
        x_values[..., :history_len],
        metadata["history_feature_cols"],
        metadata["log1p_feature_cols"],
    )
    future_x, _ = apply_log1p_transform(
        x_values[..., history_len:],
        sample_bundle["known_future_feature_cols"],
        metadata["log1p_feature_cols"],
    )
    x_values = np.concatenate([history_x, future_x], axis=-1)
    x_values, _ = append_periodic_features_if_needed(
        x_values,
        metadata,
        target_start_offset=args.target_start_offset,
    )

    sample_dates = np.array(sample_bundle["sample_dates"])
    sample_datetimes = np.array(sample_bundle.get("sample_datetimes", sample_bundle["sample_dates"]))
    sample_anchor_hours = np.array(sample_bundle.get("anchor_hours", [-1] * len(sample_dates)))
    target_start_datetimes = np.array(sample_bundle.get("target_start_datetimes", sample_bundle["sample_dates"]))

    wanted_dates = pd.date_range(args.start_date, args.end_date, freq="D").strftime("%Y-%m-%d").to_numpy()
    available_mask = np.isin(sample_dates, wanted_dates)
    x_values = x_values[available_mask]
    sample_dates = sample_dates[available_mask]
    sample_datetimes = sample_datetimes[available_mask]
    sample_anchor_hours = sample_anchor_hours[available_mask]
    target_start_datetimes = target_start_datetimes[available_mask]
    if len(sample_dates) == 0:
        raise ValueError("No requested dates are evaluable.")

    feature_mean = metadata["feature_mean"]
    feature_std = metadata["feature_std"]
    x_scaled = ((x_values - feature_mean[0]) / feature_std[0]).astype(np.float32)
    return x_scaled, sample_dates, sample_datetimes, sample_anchor_hours, target_start_datetimes


def build_model(args, metadata, graph_dir, x_scaled, training_metadata):
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

    context_gate = parse_bool_value(cli_value_or_default(training_metadata, "--context_gate"), default=False)
    context_gate_hidden_dim = int(cli_value_or_default(training_metadata, "--context_gate_hidden_dim", 32))
    context_gate_residual = float(cli_value_or_default(training_metadata, "--context_gate_residual", 0.5))
    if not context_gate:
        raise ValueError("The selected project does not enable --context_gate true.")

    device = resolve_device(args.device)
    graph_config = {
        "use": recovered_graph_use,
        "fix_weight": False,
        "tempp_diag_zero": True,
        "matrix_weight": True,
        "context_gate": context_gate,
        "context_gate_hidden_dim": context_gate_hidden_dim,
        "context_gate_residual": context_gate_residual,
        "distri_type": "exp",
        "func_type": "ours",
        "attention": True,
        "sparsify_mode": "topk",
        "sparsify_topk": 20,
        "sparsify_symmetric": True,
        "sparsify_keep_self": True,
    }
    data_config = {
        "in_dim": x_scaled.shape[-1],
        "out_dim": len(metadata["target_cols"]),
        "hist_len": x_scaled.shape[1],
        "pred_len": metadata["pred_len"],
        "type": "bike",
    }
    model_config = {
        "cheb_k": 3,
        "nb_block": 2,
        "nb_chev_filter": 64,
        "nb_time_filter": 64,
        "time_kernel_size": 3,
        "channel_attention": False,
        "channel_attention_reduction": 4,
    }

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
    if unexpected:
        raise RuntimeError("Unexpected checkpoint keys: %s" % unexpected[:10])
    model.eval()
    return model, graph_config, checkpoint_path, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/temporal_data/bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20_train2025_hist168")
    parser.add_argument("--graph_dir", default="data/graph/bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_graph_train2025_hist168")
    parser.add_argument("--project", default="bike_hourly_safe_inventory_top150_exp12_context_gated_multigraph_hist168_pred6_bs8_seed0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start_date", default="2026-02-01")
    parser.add_argument("--end_date", default="2026-02-28")
    parser.add_argument("--trip_glob", default="20260[12]-citibike-tripdata_*.csv")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--anchor_hours", default="0,6,12,16,20")
    parser.add_argument("--target_start_offset", type=int, default=1)
    parser.add_argument("--graph_use", default=None)
    parser.add_argument("--batch_size", type=int, default=1, help="Use 1 to inspect per-sample gate weights.")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    graph_dir = PROJECT_ROOT / args.graph_dir
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / ANALYSIS_DIR / "2026-05-14_exp12_context_gate_weight_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_npz_metadata(data_dir)
    training_metadata = find_training_metadata(args.project)
    x_scaled, sample_dates, sample_datetimes, sample_anchor_hours, target_start_datetimes = build_february_samples(
        args,
        metadata,
        graph_dir,
    )
    model, graph_config, checkpoint_path, device = build_model(
        args,
        metadata,
        graph_dir,
        x_scaled,
        training_metadata,
    )

    rows = []
    graph_names = graph_config["use"]
    with torch.no_grad():
        for start in range(0, len(x_scaled), args.batch_size):
            end = min(start + args.batch_size, len(x_scaled))
            batch = torch.from_numpy(x_scaled[start:end]).to(device)
            _ = model(batch)
            weights = model.fusiongraph.context_gate_for_run.detach().cpu().numpy()
            row = {
                "batch_start": int(start),
                "batch_end": int(end),
                "sample_count": int(end - start),
                "sample_date": str(sample_dates[start]) if end - start == 1 else "",
                "sample_datetime": str(sample_datetimes[start]) if end - start == 1 else "",
                "anchor_hour": int(sample_anchor_hours[start]) if end - start == 1 else -1,
                "target_start_datetime": str(target_start_datetimes[start]) if end - start == 1 else "",
            }
            for graph_name, weight in zip(graph_names, weights):
                row[graph_name] = float(weight)
            rows.append(row)

    gate_df = pd.DataFrame(rows)
    gate_path = output_dir / "exp12_context_gate_by_sample.csv"
    gate_df.to_csv(gate_path, index=False, encoding="utf-8-sig")

    graph_cols = graph_names
    by_anchor = (
        gate_df.groupby("anchor_hour")[graph_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    by_anchor.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in by_anchor.columns
    ]
    by_anchor_path = output_dir / "exp12_context_gate_by_anchor.csv"
    by_anchor.to_csv(by_anchor_path, index=False, encoding="utf-8-sig")

    mean_weights = gate_df[graph_cols].mean().sort_values(ascending=False)
    summary = {
        "project": args.project,
        "checkpoint": str(checkpoint_path),
        "graph_use": graph_names,
        "num_samples": int(len(gate_df)),
        "batch_size": int(args.batch_size),
        "overall_mean_weights_desc": {key: float(value) for key, value in mean_weights.items()},
        "output_files": {
            "by_sample": str(gate_path),
            "by_anchor": str(by_anchor_path),
        },
    }
    summary_path = output_dir / "exp12_context_gate_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("WROTE", output_dir)
    print("Overall mean gate weights:")
    print(mean_weights.to_string())


if __name__ == "__main__":
    main()
