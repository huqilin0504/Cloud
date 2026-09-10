# LAS 岩体节理面识别与参数计算 V2 实现

这是现有 OSGB→LAS 管线之后的下游分析模块，面向一个明确的 20–50 m ROI。
本项目的任务是从点云中识别可观测的节理面候选，并计算其几何参数；不推断地下结构面、
节理的地质成因或块体稳定性。

代码按职责拆分在 `rock_discontinuity/core`、`processing`、`io`、`config` 和
`validation` 中；测试位于 `rock_discontinuity/tests`，说明文档统一位于
`docs/rock_discontinuity`。目录职责见 [项目结构](../项目结构.md)。

## 运行

日常运行统一使用项目根目录的 `main.py`。它只接收已有 LAS/LAZ，不包含
OSGB 转换：

```bash
python3 main.py --plan-only
```

全点云分块识别（普通 LAS/LAZ 自动走 PDAL；COPC 自动走空间索引）：

```bash
python3 main.py whole-cloud \
  --input outputs/root_full_xyz_10/cloud.las \
  --output outputs/whole_cloud_joint_planes \
  --tile-size 25 --overlap 1 --workers 2
```

`outputs/whole_cloud_detachment` 是历史输出目录名，当前目录内容按候选节理面识别结果解释。

下面的 `python3 -m rock_discontinuity` 命令仍保留，作为兼容的 ROI 入口。

先在 `rock_discontinuity/config/default.yaml` 中设置输入和 ROI，或直接运行：

```bash
python3 -m rock_discontinuity \
  --input outputs/root_full_xyz_10/cloud.las \
  --bbox 524898,524938,3132085,3132125 \
  --output outputs/discontinuity_roi_600
```

也可以直接使用当前示例 ROI 的可复现配置：

```bash
python3 -m rock_discontinuity \
  --config rock_discontinuity/config/roi_example.yaml \
  --force
```

输出包括：

```text
planes.csv
plane_instances.csv              # 合并前的局部平面片段（ROI/单块输出）
joint_sets.csv
spacings.csv
detachment_planes.csv          # 历史兼容文件名，语义为候选节理面清单
detachment_candidates.ply      # 历史兼容文件名，语义为候选节理面染色
rejected_planes.csv
segmented_planes.ply
features.npz
report.json
run.json
data_audit.json
density_report.json
traces.csv
aperture.csv
plane_boundaries.geojson
trace_lines.geojson
run_config.yaml
```

当前程序按约 600 点/m² 标准配置，目标是约 0.5 m 级及以上的可观测节理面候选：0.03 m 体素、0.15 m 法向邻域、0.025–0.05 m 动态区域距离、0.03 m RANSAC 距离、0.25 m² 最小面积和 100 个有效点。程序用局部切平面 kNN 输出 P10/P25/Median/Mean/P75/P90；`DENSITY_TARGET_NOT_MET` 只能说明输入密度门限未满足，不能被忽略。

`detachment_candidates.ply` 会把通过几何质量筛选的观测节理面候选染成红色，其余点为灰色；
`detachment_planes.csv` 保存对应清单。这里的 `detachment_*` 只是历史文件名，红色表示候选节理面，
不表示已经解体或失稳。

`detachment_planes.csv` 现在直接包含候选节理面的 `dip_direction_deg`（倾向）、
`dip_deg`（倾角）、`trace_length_m`（迹长，当前无独立暴露边界时为 null）、
`apparent_persistence_m`（点云观测平面最大延伸）、`nearest_spacing_m`（同一
节理组内最近间距）和 `center_x/center_y/center_z`（ENU 中心坐标）。精确的
成对间距仍在 `spacings.csv` 中保存。

`spacings.csv` 同时包含 `spacing_3d_nonpersistent` 和 `spacing_virtual_scanline`。
`plane_boundaries.geojson` 是有限观测边界；`traces.csv`、`trace_lines.geojson` 和
`aperture.csv` 会明确记录高分辨率 Mesh/影像输入缺失，不用表观延伸替代迹线或张开度。

ROI/单瓦片先按同向、共面、空间相邻和合并后 RMS 做局部平面合并；全点云额外输出
`joint_planes.csv`（局部合并后再跨瓦片合并的全局节理面）、`joint_sets.csv`
（全局节理组和间距统计）以及 `tile_spacings.csv`（瓦片内审计结果）。
`detachment_planes.csv` 保留每个瓦片实例，并通过 `global_plane_id=GJ-xxxxx`
追溯到 `joint_planes.csv`。

输入目录 `outputs/root_full_xyz_10` 是历史目录名；当前正式 LAS 的 `stats.json` 记录为 600 点/m²。程序不根据目录名猜密度，以配置中的 600 标准和运行时密度统计为准。

## 验收阶段

当前版本只包含一个验收阶段：

1. 合成数据真值测试，以及指定 ROI 的读取、预处理、分割、参数计算、输出和重复运行验证。

本版本不使用 CloudCompare 人工标注、现场罗盘或人工测线，不对真实 ROI 宣称现场精度。`report.json` 会明确记录外部真值标定不适用。

## 整个 LAS 分块处理

当前完整 LAS 约 14.75 亿点，不能直接作为一个 ROI 载入。分块入口按输入格式选择 PDAL 或 COPC 空间索引生成压缩 LAZ 源瓦片，再逐块调用同一套识别程序：默认核心块 25 m × 25 m、重叠 1 m，核心区半开边界去重；普通 LAS 报告明确标注 `2d_xy_fallback`，因为当前 PDAL splitter 是 XY 分块而不是伪造的 Z 体素分块。处理后的源瓦片默认在成功处理后删除，候选红点叠加层保留；中断后使用 `--resume` 继续。每块内部先按同向、共面、空间相邻和合并后 RMS 合并局部平面；识别阶段默认使用 2 个线程并行处理独立瓦片；所有块完成后，再按相邻边界、法向、平面偏移、边界间隙和 RMS 合并全局节理面。

```bash
python3 main.py whole-cloud \
  --input outputs/root_full_xyz_10/cloud.las \
  --output outputs/whole_cloud_joint_planes \
  --tile-size 25 \
  --overlap 1
```

试运行可加 `--max-tiles 1`。完成后用原始 LAS 加红色候选叠加层查看：

```bash
CloudCompare \
  "outputs/root_full_xyz_10/cloud.las" \
  "outputs/whole_cloud_joint_planes/candidate_detachment_points.laz"
```

全域输出中的 `candidate_plane_instances` 是局部合并后的瓦片节理面实例数，`plane_merge` 记录单块内合并统计，`global_joint_plane_count` 是跨瓦片合并后的全局数量；
当前只输出节理面几何候选，不输出解体块体或失稳块体数量。
