# 2026-04-07 优化版快照

这个文件夹保存的是本轮 bike 优化后的源码快照。

本次优化内容：

1. `鲁棒损失`
   - 训练默认从 `MAE` 切换为 `Huber Loss`
   - `checkpoint / early stopping / lr scheduler` 统一改为监控 `val_mae`
   - 修正了共享单车场景下把 `true=0` 当成空值掩掉的问题
   - `MAPE` 改为带 `epsilon=1.0` 的平滑版本，降低零需求陷阱

2. `图稀疏化`
   - 在 `FusionGraphModel` 的融合图输出后增加 Top-K 稀疏化
   - 当前默认：`--graph_sparsify_mode topk --graph_topk 15`
   - 目标是减少稠密图带来的过度平滑和噪声扩散

3. `星期几实体嵌入`
   - 在 `MSTGCN` 输入端增加类别特征 embedding 适配器
   - 当前默认对 `future_星期几` 使用 `8` 维 embedding
   - 可通过 `--weekday_embed_dim 0` 关闭，兼容旧 checkpoint

4. `输入长尾压缩`
   - 在 `prepare_bike_data.py` 中对部分非负输入特征加入 `log1p`
   - 当前自动落在：`日间骑出量`、`日间骑入量`、`future_总桩数`
   - 目的是压缩长尾，减轻极端需求尺度差异

包含文件：

- `util.py`
- `prepare_bike_data.py`
- `prepare_bike_training_data.py`
- `train_bike.py`
- `evaluate_bike_date_range.py`
- `summarize_bike_date_range_report.py`
- `models/fusiongraph.py`
- `models/MSTGCN.py`
- `datasets/bike.py`

推荐执行顺序：

1. `python prepare_bike_training_data.py --output_name bike --se_strategy preserve`
2. `train_bike.py --device auto --epochs 40 --batch_size 8 --logger csv`
3. `evaluate_bike_date_range.py --device auto --start_date 2026-02-01 --end_date 2026-02-10`
4. `summarize_bike_date_range_report.py --eval_dir 分析结果\\2026-02-01到2026-02-10滚动预测 --inventory_summary_json 分析结果\\2026-02-01单日库存推演对比\\摘要.json`

已完成的轻量验证：

- 语法编译通过
- `prepare_bike_training_data.py` 已成功重新生成 `bike` 数据
- `train_bike.py --epochs 1 --batch_size 8 --logger csv` 已跑通

说明：

- 这里不复制原始公共数据文件。
- 如果要回滚，只需将“优化前基线快照”中的对应文件复制回项目根目录即可。
