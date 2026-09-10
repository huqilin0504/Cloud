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
- 合并：每个块独立建立局部 KDTree 并计算 PCA/动态区域生长；块内先按法向角度、平面偏移、XY 边界间隙和合并后 RMS 合并局部片段，所有块完成后再比较相邻块并生成全局 `GJ-xxxxx` 节理面。切块、状态、单块处理、全局聚合、记录准备和全点云写出分别位于 `tiling.py`、`state.py`、`tile_processing.py`、`global_aggregation.py`、`output_records.py` 和 `io/whole_cloud_output.py`；命令解析位于 `app/whole_cloud.py`。

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

运行时会显示两段终端进度：PDAL 或 COPC 索引切块阶段显示已生成的非空瓦片数，终端中的“结构面识别”阶段
对应候选节理面识别，并显示已完成/总瓦片数。切块阶段的百分比是相对于名义网格的近似值；最终以“生成非空 LAZ 瓦片”
的计数为准。

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

`detachment_planes.csv` 包含每个候选节理面的倾向、倾角、迹长字段、观测平面延伸、
最近间距和 `center_x/center_y/center_z` ENU 中心坐标。当前 `trace_length_m`
在没有独立暴露面迹线时为 `null`，`apparent_persistence_m` 才是本次点云直接
测得的平面最大延伸。`nearest_spacing_m` 和 `spacings.csv` 使用合并后的全局节理面重新计算；
`tile_spacings.csv` 保留块内原始结果。

普通 LAS 的 `data_audit.json` 会记录 `dimension_mode=2d_xy_fallback`；整体源文件不构建全局 KDTree。
P10 需要明确扫描线，P21 需要已验证迹线；当前没有高分辨率 Mesh/影像，因此整体任务只输出
有限观测面和虚拟扫描线间距，`traces.csv` 与 `aperture.csv` 保持不可用原因。
