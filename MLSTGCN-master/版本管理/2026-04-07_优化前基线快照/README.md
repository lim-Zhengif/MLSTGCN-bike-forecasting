# 2026-04-07 优化前基线快照

这个文件夹保存的是本轮优化前的源码快照，方便后续回滚测试。

包含文件：

- `util.py`
- `prepare_bike_data.py`
- `prepare_bike_training_data.py`
- `train_bike.py`
- `evaluate_bike_date_range.py`
- `models/fusiongraph.py`
- `models/MSTGCN.py`
- `datasets/bike.py`

适用场景：

- 回到本轮优化前的 bike 训练逻辑
- 对比“优化前 vs 优化后”的训练和回测结果
- 排查是数据处理、图融合还是模型结构导致的差异

回滚方法：

1. 将本目录下对应文件复制回项目根目录原位置。
2. 重新运行 `prepare_bike_training_data.py` 生成基线版数据。
3. 再运行 `train_bike.py` 和 `evaluate_bike_date_range.py` 做基线回测。

说明：

- 这里不额外复制公共原始数据。
- 如果需要完全复现基线结果，建议同时保留基线训练日志和 checkpoint。
