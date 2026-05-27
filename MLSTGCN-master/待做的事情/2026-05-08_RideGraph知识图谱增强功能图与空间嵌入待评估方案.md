# RideGraph 知识图谱增强功能图与空间嵌入待评估方案

记录日期：2026-05-08

状态：待评估，暂不实施代码改动。

## 参考论文

论文：`Knowledge graph-augmented stacking for accurate bike-sharing demand forecasting: The RideGraph framework`

本地路径：

`E:\学校\2026年实验后找能够引用的论文\！！Knowledge graph-augmented stacking for accurate bike-sharing demand forecasting The RideGraph framework\1-s2.0-S1566253526000953-main.pdf`

核心可借鉴点：论文将结构化领域知识构造成知识图谱，再提取图特征或图嵌入，用于增强共享单车需求预测。它还使用集成学习与 ANN stacking，但对当前项目最有价值的部分不是 stacking 主框架，而是“知识图谱增强特征表示”。

## 与当前项目的关系

当前模型主线是：

1. 构建多张先验图：距离图、OD/邻接图、功能/POI 图、启发式图、需求相关图。
2. `FusionGraphModel` 学习融合图。
3. `MSTGCN` 基于融合图做小时级骑入/骑出预测。
4. 预测结果进一步服务于安全库存区间和调度仿真。

RideGraph 的知识图谱思想与当前项目最自然的结合点有两个：

1. 增强或替换现有 `func.npy`。
2. 增强或替换现有 `data/SE/se_<dataset>.csv` 空间嵌入。

这比直接照搬 RideGraph-ANN Stack 更合适，因为我们的核心创新仍然是动态多图融合和时空图预测。

## 初步判断

可以引用，也值得作为后续模型增强候选。

但不建议直接把 RideGraph 的完整 stacking 框架作为主模型。原因：

- RideGraph 更偏表格回归和集成学习。
- 当前项目已经有明确的站点级时空图神经网络主线。
- 直接改成 stacking 会偏离当前论文主线，也不利于和已有实验结果衔接。

更推荐把 RideGraph 用作“知识图谱增强 POI/功能图或空间嵌入”的理论依据。

## 建议目标

构建一个站点级知识图谱，把站点周边 POI、容量属性、区域功能、天气敏感性、需求模式、OD 联系等结构化知识统一起来，然后生成：

```text
kg_func.npy
kg_semantic.npy
se_<dataset>_kg.csv
```

后续可选择其中一种接入模型。

## 知识图谱节点设计

建议先做站点级异质图，节点类型包括：

```text
Station: 单车站点
POIType: POI 类型，例如商业餐饮、公共交通、办公教育、休闲娱乐
CapacityBin: 容量分箱，例如 low_capacity / mid_capacity / high_capacity
Region: 区域或行政/空间聚类
DemandPattern: 需求模式簇，例如 morning_outflow、evening_inflow、balanced、low_activity
WeatherSensitivity: 天气敏感性簇，例如 rain_sensitive、wind_sensitive、temperature_sensitive
TimeRole: 时间角色，例如 commuter_peak、leisure_weekend、night_sparse
```

其中 `Station` 是核心节点，其余节点用于给站点增加语义结构。

## 知识图谱边设计

建议边类型包括：

```text
Station -has_poi-> POIType
Station -has_capacity_level-> CapacityBin
Station -belongs_to_region-> Region
Station -has_demand_pattern-> DemandPattern
Station -has_weather_sensitivity-> WeatherSensitivity
Station -has_time_role-> TimeRole
Station -od_connected_to-> Station
Station -spatial_neighbor_of-> Station
Station -similar_poi_to-> Station
Station -similar_demand_to-> Station
```

这些边可以由现有数据生成：

- POI 和经纬度来自 `FINAL_JC_BaseTable_with_POI.csv` 或 Top150/Top300 映射资产。
- OD 联系来自 `graph_od_transition.npy` 或 `neigh.npy`。
- 空间邻近来自 `graph_spatial_distance.npy` 或 `dist.npy`。
- 需求模式来自历史骑入/骑出序列聚类。
- 天气敏感性来自不同天气条件下站点需求变化的相关性或回归系数。

## 接入方案 A：增强现有功能图，优先推荐

生成一个站点-站点知识图谱相似度矩阵：

```text
kg_func.npy: [num_nodes, num_nodes]
```

矩阵含义：两个站点在知识图谱中的语义相似度。

相似度可由以下方式得到：

1. 对知识图谱做 node2vec/metapath2vec，得到每个站点的 KG embedding。
2. 计算站点 embedding 余弦相似度。
3. 归一化并稀疏化为 Top-K 图。

接入方式：

```text
func_kg = alpha * func.npy + (1 - alpha) * kg_func.npy
```

然后保存为实验目录下的 `func.npy`，其余训练代码不变。

优点：

- 改动最小。
- 不需要把 FusionGraph 从五图改成六图。
- 最适合先做实验验证。

风险：

- 如果 `kg_func.npy` 与原 POI 图高度重复，提升可能有限。
- `alpha` 需要消融。

建议消融：

```text
alpha = 1.0  原始 func.npy
alpha = 0.7
alpha = 0.5
alpha = 0.3
alpha = 0.0  纯 KG 功能图
```

## 接入方案 B：替换或融合空间嵌入 SE

当前 FusionGraph 会读取：

```text
data/SE/se_<dataset>.csv
```

可以基于知识图谱生成：

```text
data/SE/se_<dataset>_kg.csv
```

接入方式：

```text
SE_final = beta * SE_current + (1 - beta) * SE_kg
```

或者单独使用 `SE_kg` 做实验。

优点：

- 不改变五张图的文件结构。
- KG 语义会直接进入 FusionGraph 的 `SGEmbedding`。
- 适合表达“知识图谱增强空间表示”。

风险：

- 当前 `SE` 维度需要匹配 `M * d`，默认是 `24 * 6 = 144`。
- 如果 KG embedding 方法输出维度不同，需要投影到 144 维。

建议消融：

```text
beta = 1.0  原始 SE
beta = 0.7
beta = 0.5
beta = 0.3
beta = 0.0  纯 KG SE
```

## 接入方案 C：新增第六张 KG 图，暂不推荐第一步做

理论上可以新增：

```text
graph_use = dist,neighb,distri,tempp,func,kg
```

但当前代码中 `SGEmbedding.FC_ge` 的图身份 one-hot 输入维度写死为 5。如果改成六图，需要同步修改：

- `datasets/bike.py`
- `models/fusiongraph.py`
- `train_bike.py`
- 评估脚本中的 graph config
- 旧 checkpoint 兼容逻辑

这会带来较多代码改动，不适合作为第一轮验证。

建议：只有当方案 A 或方案 B 有明显收益后，再考虑六图版本。

## 最小实现路线

第一阶段只做离线图构建，不改模型主体：

1. 读取当前 Top150/Top300 的站点映射和 POI/容量/经纬度/历史需求特征。
2. 构建站点知识图谱。
3. 提取站点 KG embedding。
4. 生成 `kg_func.npy` 和/或 `se_<dataset>_kg.csv`。
5. 复制一个新数据图目录，例如：

```text
data/graph/bike_hourly_safe_inventory_top150_kg_func_alpha05
data/SE/se_bike_hourly_safe_inventory_top150_kg_beta05.csv
```

6. 使用现有训练入口跑消融，不改 `MSTGCN`。

## 建议实验对象

优先选当前较成熟的 Top150 rolling anchor 数据：

```text
data/temporal_data/bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20_train2025_hist168
data/graph/bike_hourly_safe_inventory_top150_nyc_full_predlen6_anchors_00_06_12_16_20_train2025_hist168
```

理由：

- 节点数 150，训练成本可控。
- `pred_len=6` 更适合做滚动预测和调度评估。
- 已有 GRU baseline、Top150 holdout、误差结构分析等结果可对照。

## 建议评价指标

预测指标：

```text
val_mae_epoch
holdout MAE/RMSE
horizon-wise MAE
station-wise MAE
高峰小时 MAE
低活跃时段 MAE
```

调度指标：

```text
缺车风险次数
满桩风险次数
安全库存区间覆盖率
调度前后库存风险改善
dispatch_moves 数量和总调度量
```

如果 KG 增强主要改善低活跃站点、功能区相似站点或天气敏感站点，即使整体 MAE 小幅下降，也有论文价值。

## 论文写作可用表述

如果后续实现并验证有效，可以写成：

受 RideGraph 框架中知识图谱增强特征表示思想启发，本研究将站点 POI、容量、需求模式、天气敏感性与 OD 联系构造成站点级异质知识图谱，并从中提取语义相似度图或知识图谱空间嵌入，用于增强动态多图融合模块中的功能图和空间表示，从而提高模型对站点语义关系和外部环境影响的表达能力。

如果后续没有实现，只建议放在未来工作中：

后续可进一步引入知识图谱增强的站点语义表示，将 POI、天气敏感性、容量和需求模式等结构化知识纳入多图融合预测框架。

## 当前不立即实施的原因

1. 当前模型已经有 POI 功能图和需求相关图，KG 增强是否带来增益需要消融验证。
2. 构建 KG 需要定义节点、边、嵌入方法和相似度归一化，工作量比普通调参更大。
3. 直接新增第六张图会牵动模型结构和旧实验兼容，第一步不宜这样做。
4. 更适合在当前 Top150/Top300 主线稳定后，作为单独版本实验推进。

## 后续触发条件

满足以下任一条件时，可以考虑正式实现：

1. 当前模型在功能相似但地理距离较远的站点上误差较大。
2. POI 功能图的解释力不足，需要更强的语义结构。
3. 需要为论文增加“知识驱动的站点语义建模”创新点。
4. 已准备好运行至少 `原始 func`、`KG func`、`原始 SE`、`KG SE` 四组消融实验。

