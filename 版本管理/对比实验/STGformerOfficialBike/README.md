# STGformer Official Core Bike B0

This directory is the auditable B0 adaptation of the official STGformer core
for the local Top-150 bike-flow contract:

```text
x [B, 168, 150, F] -> y_hat [B, 3, 150, 2]
```

The historical `版本管理/对比实验/STGformer` implementation is unchanged and
remains labelled as an STGformer-style baseline. The pinned upstream source and
license are recorded in `UPSTREAM.md` and `upstream/STGformer.py`.

The training entry point is self-contained inside this directory and does not
depend on the historical sibling `版本管理/对比实验/baseline_utils.py`.

B0 preserves the official learned adaptive graph. `dist.npy` is loaded only as
an external node-count/order audit. Multi-relation graph injection is deferred
to B1 so the official baseline and the innovation are not mixed.

Recommended environment:

```powershell
D:\Users\Administrator\miniconda3\envs\mlstgcn\python.exe `
  版本管理/对比实验/STGformerOfficialBike/train_stgformer_official_bike.py
```

No full training is started by repository setup. Run the tests first:

```powershell
D:\Users\Administrator\miniconda3\envs\mlstgcn\python.exe -m unittest discover `
  -s 版本管理/对比实验/STGformerOfficialBike/tests -p "test_*.py"
```

The evaluator labels its default same-day observed weather as
`oracle_observed_same_day`. Use `--future_weather_lag_days 1` for the
leakage-safe lagged-weather sensitivity run; the summary keeps the two regimes
separate.
