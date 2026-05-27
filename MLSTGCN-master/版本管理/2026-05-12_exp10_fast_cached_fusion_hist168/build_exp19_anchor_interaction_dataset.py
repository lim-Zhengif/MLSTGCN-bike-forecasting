import argparse
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


ANCHOR_INTERACTION_FEATURE_NAMES = [
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


def as_str_list(values):
    return [str(value) for value in values.tolist()]


def known_future_col_index(history_feature_cols, known_future_feature_cols, known_future_position):
    if len(known_future_feature_cols) <= known_future_position:
        raise ValueError(
            "known_future_feature_cols is too short: need position %d, got %d"
            % (known_future_position, len(known_future_feature_cols))
        )
    return len(history_feature_cols) + known_future_position


def append_anchor_interaction_features(x, anchor_hours, history_feature_cols, known_future_feature_cols):
    if len(anchor_hours) != x.shape[0]:
        raise ValueError("anchor_hours length does not match x samples.")
    if x.shape[1] < 24:
        raise ValueError("hist_len must be at least 24 for recent demand interaction features.")

    weekend_idx = known_future_col_index(history_feature_cols, known_future_feature_cols, 2)
    precip_idx = known_future_col_index(history_feature_cols, known_future_feature_cols, 6)
    wind_idx = known_future_col_index(history_feature_cols, known_future_feature_cols, 8)
    if max(weekend_idx, precip_idx, wind_idx) >= x.shape[-1]:
        raise ValueError("Base input feature dimension is too small for exp19 known-future features.")

    anchors = anchor_hours.astype(np.float32)
    morning = (anchor_hours == 6).astype(np.float32)
    midday = (anchor_hours == 12).astype(np.float32)
    evening = (anchor_hours == 16).astype(np.float32)
    night = np.isin(anchor_hours, [0, 20]).astype(np.float32)
    peak = np.isin(anchor_hours, [6, 12, 16]).astype(np.float32)

    sample_node = lambda value: np.repeat(value[:, np.newaxis], x.shape[2], axis=1)
    hour_norm = sample_node(anchors / 23.0)
    morning = sample_node(morning)
    midday = sample_node(midday)
    evening = sample_node(evening)
    night = sample_node(night)
    peak = sample_node(peak)

    future_weekend = x[:, -1, :, weekend_idx]
    future_precip = x[:, -1, :, precip_idx]
    future_wind = x[:, -1, :, wind_idx]
    recent_out_mean = x[:, -24:, :, 0].mean(axis=1)
    recent_in_mean = x[:, -24:, :, 1].mean(axis=1)

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
    feature_window = np.repeat(feature_block[:, np.newaxis, :, :], x.shape[1], axis=1)
    return np.concatenate([x.astype(np.float32), feature_window], axis=-1).astype(np.float32)


def convert_split(input_dir, output_dir, split_name):
    src_path = input_dir / f"{split_name}.npz"
    data = np.load(src_path, allow_pickle=True)
    input_feature_cols = as_str_list(data["input_feature_cols"])
    if any(name in input_feature_cols for name in ANCHOR_INTERACTION_FEATURE_NAMES):
        raise ValueError(f"{src_path} already appears to contain exp19 anchor interaction features.")

    x_aug = append_anchor_interaction_features(
        x=data["x"].astype(np.float32),
        anchor_hours=data["anchor_hours"].astype(np.int64),
        history_feature_cols=as_str_list(data["history_feature_cols"]),
        known_future_feature_cols=as_str_list(data["known_future_feature_cols"]),
    )

    output = {key: data[key] for key in data.files if key != "x"}
    output["x"] = x_aug
    output["input_feature_cols"] = np.array(input_feature_cols + ANCHOR_INTERACTION_FEATURE_NAMES)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{split_name}.npz", **output)
    return {
        "split": split_name,
        "input_shape": tuple(data["x"].shape),
        "output_shape": tuple(x_aug.shape),
        "added_features": len(ANCHOR_INTERACTION_FEATURE_NAMES),
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
            / "bike_hourly_safe_inventory_top150_exp19_anchor_interaction_features_hist168_pred6"
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    summaries = [convert_split(input_dir, output_dir, split) for split in ["train", "val", "test"]]
    print("Built exp19 anchor interaction dataset:", output_dir)
    for item in summaries:
        print(item)


if __name__ == "__main__":
    main()
