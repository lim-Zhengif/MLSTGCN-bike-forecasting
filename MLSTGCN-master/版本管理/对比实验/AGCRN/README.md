# AGCRN Baseline

This directory contains the AGCRN comparison experiment for the Top150 bike
hourly forecasting task.

## Layout

- `upstream/`: official AGCRN source snapshot cloned from
  `https://github.com/LeiBAI/AGCRN.git`.
- `models/agcrn.py`: adapted AGCRN modules for this project input/output shape.
- `train_agcrn_top150_baseline.py`: train, validate, and evaluate AGCRN on the
  existing Top150 `.npz` splits.

The adapted code keeps `upstream/` unchanged. The baseline uses the same data
contract as the current project:

- input `x`: `[samples, 168, 150, 21]`
- target `y`: `[samples, 6, 150, 2]`
- feature normalization: Z-score fitted on training split only
- target normalization: fitted on training split, predictions inverse-scaled
  before MAE/RMSE reporting

## Run

From the project root:

```powershell
python ".\版本管理\对比实验\AGCRN\train_agcrn_top150_baseline.py" --device auto
```

The default output directory is:

```text
分析结果/对比实验/AGCRN/top150_hist168_pred6_seed0
```

Important outputs:

- `best_agcrn_baseline.pt`
- `agcrn_training_history.csv`
- `agcrn_internal_test_horizon_mae.csv`
- `agcrn_internal_test_anchor_mae.csv`
- `agcrn_training_summary.json`
