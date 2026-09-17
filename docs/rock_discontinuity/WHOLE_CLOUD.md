# 整体点云分块处理

## 处理边界

输入是 `outputs/root_full_xyz_10/cloud.las`，它来自 `危岩体/Block.osgb` 根模型及其子瓦片。当前没有独立的 SHP/KML/GeoJSON 边界，因此本流程按根模型 LAS 的 XY 外接范围处理，不把外接矩形误称为精确岩体轮廓。本文中的识别对象是可观测节理面候选。

当前源数据约 14.75 亿点、约 29.5 GB。整体处理输出的是候选节理面红色叠加层，
不是解体面或失稳块体数量。`outputs/whole_cloud_detachment` 也是历史目录名，
不代表本流程正在做解体判定。

## 分块策略

- 核心块：默认 25 m × 25 m；与 V2 的局部邻域目标一致。
- 重叠：1 m；为法向邻域和边缘结构面拟合提供上下文。
- 原点：由 LAS header 的 XY 最小值向下取整到 25 m 网格。
- 切块：普通 LAS/LAZ 使用 PDAL `tile` 流式读取；COPC 输入使用 `laspy.copc.CopcReader.spatial_query` 按空间索引查询，再输出压缩 LAZ 中间瓦片。
- 去重：只输出每个核心块的点，内部边界半开，避免重叠区重复。
- 恢复：每个块成功后写状态和报告；失败块保留源瓦片，使用 `--resume` 重试。
- 维度声明：普通 LAS 当前为 `2d_xy_fallback`，完整 Z 保留在局部瓦片中；不会把 XY 网格伪称为三维体素切块。
- 合并：每个块独立建立局部 KDTree 并计算 PCA/动态区域生长；块内先按法向角度、平面偏移、XY 边界间隙和合并后 RMS 合并局部片段，所有块完成后再比较相邻块并生成全局 `GJ-xxxxx` 节理面。跨块比较先使用保守的 XY 包围盒网格索引筛出必要候选，再执行原有法向、平面偏移、足迹间隙和 RMS 精确判据；索引只减少不可能接触的组合，不改变合并阈值。方向组先用稀疏轴向方向连通分量预筛选，再对每个分量进行自适应轴向二分，确保最终组内最大方向偏差不超过 `max_set_deviation_deg`，避免方向链式连接造成的大簇和贪心扫描造成的单例过分割。切块、状态、单块处理、全局聚合、记录准备和全点云写出分别位于 `tiling.py`、`state.py`、`tile_processing.py`、`global_aggregation.py`、`output_records.py` 和 `io/whole_cloud_output.py`；命令解析位于 `app/whole_cloud.py`。

## 本次全局聚合优化

- `global_orientation_sets` 不再建立方向角度的稠密 `N×N` 矩阵，也不再对每个贪心小组重复扫描全部剩余法向量；采用稀疏轴向方向连通分量 + 自适应轴向二分 + 小规模轴向二均值细化。
- 二分只改变方向组的归属，不改变局部平面识别、法向、平面偏移、足迹间隙、RMS 和全局候选质量门槛。最终方向组仍由 `joint_sets.max_set_deviation_deg` 控制。
- 跨瓦片合并保留原有精确判据；XY 网格索引只做必要条件预筛选。`whole_cloud_report.json` 新增 `global_aggregation_diagnostics`，记录空间候选数、各拒绝门槛、初始方向簇规模、最终方向组规模分位数和实际算法版本。
- 因为 `WHOLE_ALGORITHM_VERSION` 未改变，已有瓦片报告可以用 `--resume` 复用；但必须重新执行全局聚合，才能得到新的方向组和诊断字段。

## 工程尺度两级门槛

局部分块继续使用 `detachment` 中的 `0.25 m²` 面积和 `0.5 m` 短轴检测下限，
目的是保留可能在块内被截断、随后需要跨瓦片合并的面片。跨瓦片合并完成后，
`global_candidate_gate` 再对最终染色层和 `joint_planes.csv` 使用工程尺度筛选：

- 观测面积不少于 `2.0 m²`；
- 可见长轴不少于 `3.0 m`；
- 短轴不少于 `0.5 m`；
- 置信度、边界完整度、内点率和法向离散度继承 `detachment` 的质量门槛。

`3 m` 对应 ISRM 1978 延续性分级中“中等延续性”的起点。它在这里只作为
点云可见长轴的工程筛选代理，不冒充已经人工核验的迹长；真正的
`trace_length_m` 仍保持不可用。对既有全量瓦片报告的系统抽样显示，当前 ROI 内
候选面面积中位数约 `0.46 m²`、长轴中位数约 `1.12 m`，新门槛预计保留约
`3.7%`，用于压制局部起伏造成的密集碎面。报告中的 `detachment.tile_local_gate`
和 `detachment.global_engineering_scale_gate` 会记录实际使用的两级门槛。

最终的 `candidate_detachment_points.laz`、`joint_planes.csv` 和
`detachment_planes.csv` 使用同一全局筛选结果；筛选前实例数量另存为
`counts.tile_local_candidate_plane_instances`，便于审计而不会污染最终染色层。

该策略对应 PDAL 官方的 `length`、`origin_x/y` 和 `buffer` 语义；大文件读写使用 laspy 的 `chunk_iterator` 和分块写入方式：[PDAL splitter](https://pdal.io/en/2.9.1/stages/filters.splitter.html)、[laspy 大文件分块读写](https://laspy.readthedocs.io/en/latest/basic.html)。

## 运行

完整任务：

```bash
python3 main.py whole-cloud \
  --input outputs/root_full_xyz_10/cloud.las \
  --output outputs/whole_cloud_joint_planes \
  --tile-size 25 \
  --overlap 1 \
  --workers 2
```

只生成压缩中间瓦片：

```bash
python3 main.py whole-cloud \
  --input outputs/root_full_xyz_10/cloud.las \
  --output outputs/whole_cloud_joint_planes \
  --tile-size 25 \
  --overlap 1 \
  --prepare-only
```

只查看全域边界和预计网格，不读点记录：

```bash
python3 main.py whole-cloud \
  --input outputs/root_full_xyz_10/cloud.las \
  --tile-size 25 \
  --overlap 1 \
  --plan-only
```

中断后继续：

```bash
python3 main.py whole-cloud \
  --input outputs/root_full_xyz_10/cloud.las \
  --output outputs/whole_cloud_joint_planes \
  --resume
```

运行时使用标准 `tqdm` 为各阶段显示进度条：读取分块状态、PDAL/COPC 切块、节理候选识别、读取瓦片报告、
跨瓦片合并、建立全局平面组、全局方向聚类（包括方向切分进度）、全局平面汇总、全局候选筛选、间距与节理组统计、候选点云筛选、
最终报告准备，以及 CSV/JSON 写出。每个阶段完成后保留一行结果；长阶段按约 0.25 秒刷新一次，避免终端 I/O
拖慢全局处理。切块阶段的百分比相对于名义网格；最终以“生成非空 LAZ 瓦片”的计数为准。

如果当前环境尚未安装新增依赖，先执行：

```bash
python3 -m pip install "tqdm>=4.66"
```

如果 PDAL 切块阶段被中断，留下了没有 `split_state.json` 的损坏 LAZ，重新运行
同一命令会先把这些残留移动到 `source_tiles.incomplete*`，再重新切块；已有
`processing_state.json` 或瓦片报告时不会自动覆盖，而是要求显式恢复或换输出目录。

## 输出

```text
whole_cloud_report.json       # 全域汇总和失败块
detachment_planes.csv         # 历史兼容文件名：带 tile_id 的候选节理面实例，含 GJ 全局编号
joint_planes.csv              # 跨瓦片合并后的全局节理面清单
joint_sets.csv                # 全局节理组、倾向/倾角、法向间距统计
spacings.csv                  # 跨瓦片合并后的全局成对间距
tile_spacings.csv             # 仅用于审计的瓦片内间距
data_audit.json               # 源文件审计与分块维度声明
density_report.json           # 整体密度报告入口；局部密度在 tile_reports 中
run.json                      # 完整运行记录
traces.csv                    # 当前明确不可用：缺少高分辨率 Mesh/影像
aperture.csv                  # 当前明确不可用：缺少高分辨率 Mesh/影像
candidate_detachment_points.laz  # 历史兼容文件名：红色候选节理面点叠加层
source_tiles/                  # 压缩中间瓦片；默认成功后删除
tile_reports/                  # 每块报告
processing_state.json          # 可恢复状态
split_state.json               # 分块网格和输入快照
```

在 CloudCompare 中同时打开原始点云和红色叠加层：

```bash
CloudCompare \
  "outputs/root_full_xyz_10/cloud.las" \
  "outputs/whole_cloud_joint_planes/candidate_detachment_points.laz"
```

`candidate_plane_instances` 是局部合并后的瓦片节理面实例总数；每个瓦片报告的 `plane_merge` 记录块内合并前后的数量；`global_joint_plane_count` 是当前几何门限下的跨瓦片合并数量。`run.json` 的 `tiling.parallel_mode=thread_pool` 和 `workers` 记录实际并行配置。
`joint_planes.csv` 中一行对应一个全局合并节理面，`detachment_planes.csv` 仍保留瓦片实例，便于追溯合并来源。
`whole_cloud_report.json` 的 `global_aggregation_diagnostics.merge.rejection_counts` 可用于判断跨瓦片候选主要被哪一道门槛拒绝；`orientation.provisional` 和 `orientation.final` 分别对应预筛选和最终筛选后的方向组统计。

全点云结果可直接用统一验收入口检查表行数、状态、点数和候选 LAZ：

```bash
python3 main.py validate outputs/whole_cloud_joint_planes
```

`detachment_planes.csv` 包含每个候选节理面的倾向、倾角、迹长字段、观测平面延伸、
最近间距和 `center_x/center_y/center_z` ENU 中心坐标。当前 `trace_length_m`
在没有独立暴露面迹线时为 `null`，`apparent_persistence_m` 才是本次点云直接
测得的平面最大延伸。`nearest_spacing_m` 和 `spacings.csv` 使用合并后的全局节理面重新计算；
`tile_spacings.csv` 保留块内原始结果。

普通 LAS 的 `data_audit.json` 会记录 `dimension_mode=2d_xy_fallback`；整体源文件不构建全局 KDTree。
P10 需要明确扫描线，P21 需要已验证迹线；当前没有高分辨率 Mesh/影像，因此整体任务只输出
有限观测面和虚拟扫描线间距，`traces.csv` 与 `aperture.csv` 保持不可用原因。
