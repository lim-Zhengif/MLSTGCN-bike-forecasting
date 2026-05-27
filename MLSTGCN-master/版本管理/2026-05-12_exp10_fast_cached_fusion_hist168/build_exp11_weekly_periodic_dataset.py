import argparse
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def as_str_list(values):
    return [str(value) for value in values.tolist()]


def build_periodic_feature_names(target_cols, pred_len):
    names = []
    for prefix in ["prev_day", "prev_week"]:
        for horizon in range(1, pred_len + 1):
            for target_col in target_cols:
                names.append(f"periodic_{prefix}_h{horizon}_{target_col}")
    return names


def append_periodic_features(x, target_cols, pred_len, target_start_offset=1):
    hist_len = x.shape[1]
    num_targets = len(target_cols)
    if hist_len < 168:
        raise ValueError(f"hist_len must be at least 168 for weekly periodic features, got {hist_len}")
    if x.shape[-1] < num_targets:
        raise ValueError("Input feature dimension is smaller than target count.")

    periodic_blocks = []
    for lag in [24, 168]:
        rel_indices = [hist_len - lag + target_start_offset + horizon for horizon in range(pred_len)]
        if min(rel_indices) < 0 or max(rel_indices) >= hist_len:
            raise ValueError(
                f"Periodic lag {lag} is outside history window: indices={rel_indices}, hist_len={hist_len}"
            )
        values = x[:, rel_indices, :, :num_targets]  # [B, pred_len, N, C]
        values = values.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], pred_len * num_targets)
        periodic_blocks.append(values)

    periodic_values = np.concatenate(periodic_blocks, axis=-1).astype(np.float32)  # [B, N, 2*pred_len*C]
    periodic_window = np.repeat(periodic_values[:, np.newaxis, :, :], hist_len, axis=1)
    return np.concatenate([x, periodic_window], axis=-1).astype(np.float32)


def convert_split(input_dir, output_dir, split_name, target_start_offset):
    src_path = input_dir / f"{split_name}.npz"
    data = np.load(src_path, allow_pickle=True)
    target_cols = as_str_list(data["target_cols"])
    pred_len = int(data["y"].shape[1])
    periodic_names = build_periodic_feature_names(target_cols, pred_len)
    input_feature_cols = as_str_list(data["input_feature_cols"])

    if any(name in input_feature_cols for name in periodic_names):
        raise ValueError(f"{src_path} already appears to contain exp11 periodic features.")

    x_aug = append_periodic_features(
        data["x"].astype(np.float32),
        target_cols=target_cols,
        pred_len=pred_len,
        target_start_offset=target_start_offset,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{split_name}.npz",
        x=x_aug,
        y=data["y"].astype(np.float32),
        input_feature_cols=np.array(input_feature_cols + periodic_names),
        history_feature_cols=data["history_feature_cols"],
        known_future_feature_cols=data["known_future_feature_cols"],
        log1p_feature_cols=data["log1p_feature_cols"],
        target_cols=data["target_cols"],
        sample_dates=data["sample_dates"],
        sample_datetimes=data["sample_datetimes"],
        anchor_hours=data["anchor_hours"],
        target_start_datetimes=data["target_start_datetimes"],
    )
    return {
        "split": split_name,
        "input_shape": tuple(data["x"].shape),
        "output_shape": tuple(x_aug.shape),
        "added_features": len(periodic_names),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default=str(
            PROJECT_ROOT
            / "data"
            / "temporal_data"
            / "bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20_train2025_hist168"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(
            PROJECT_ROOT
            / "data"
            / "temporal_data"
            / "bike_hourly_safe_inventory_top150_exp11_weekly_periodic_features_hist168_pred6"
        ),
    )
    parser.add_argument("--target_start_offset", type=int, default=1)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    summaries = [
        convert_split(input_dir, output_dir, split, args.target_start_offset)
        for split in ["train", "val", "test"]
    ]
    print("Built exp11 weekly periodic dataset:", output_dir)
    for item in summaries:
        print(item)


if __name__ == "__main__":
    main()
