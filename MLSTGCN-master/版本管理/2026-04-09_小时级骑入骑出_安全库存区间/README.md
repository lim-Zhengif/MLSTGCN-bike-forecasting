# 2026-04-09 小时级骑入骑出与安全库存区间

这个目录是在 `2026-04-07_优化版_鲁棒损失_图稀疏化_星期嵌入` 基础上复制出的独立版本，目标是新增一条更贴近调度的小时级主流程：

- 预测明天每个小时的骑出量和骑入量
- 根据 24 小时净流量曲线推导安全库存区间 `[S_min, S_max]`

## 主要文件

- `prepare_bike_hourly_training_data.py`
  - 小时级数据准备入口
- `prepare_bike_hourly_data.py`
  - 聚合两年订单流水，生成小时级训练数据
- `train_bike_hourly.py`
  - 小时级训练入口
- `evaluate_bike_hourly_date_range.py`
  - 小时级滚动回测
- `generate_bike_safe_inventory_interval_report.py`
  - 根据小时预测生成 `[S_min, S_max]`
- `优化说明_小时级骑入骑出与安全库存区间.md`
  - 详细说明这个版本改了什么、为什么这样改

目录内也保留了上一版的公共代码副本：

- `datasets/`
- `models/`
- `util.py`
- `prepare_bike_data.py`
- `train_bike.py`
- `evaluate_bike_date_range.py`

这样做的目的是方便你后面直接回滚、对照和复现实验。

## 推荐运行顺序

```bash
python prepare_bike_hourly_training_data.py
python train_bike_hourly.py --device auto --epochs 40 --batch_size 8 --logger csv
python evaluate_bike_hourly_date_range.py --device auto --start_date 2026-02-01 --end_date 2026-02-10
python generate_bike_safe_inventory_interval_report.py --eval_dir outputs/bike_hourly_eval_2026-02-01_to_2026-02-10
```

## 默认输出路径

- `data/temporal_data/bike_hourly_safe_inventory`
- `data/graph/bike_hourly_safe_inventory`
- `data/SE/se_bike_hourly_safe_inventory.csv`

这些都是新的独立输出，不会覆盖原来日级 `bike` 的数据和模型。
