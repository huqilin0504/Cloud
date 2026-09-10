# 基于 LAS 点云的岩体节理面自动识别与参数计算系统
## 详细技术设计文档 V1.0（V2 实现映射）

> 本文保留 V1 的字段和背景说明；若与 `docs/岩体节理面自动识别_文献驱动技术设计_V2.md` 冲突，以 V2.3 为准。当前代码已按 V2.3 落实局部 KDTree、多尺度法向稳定性、切平面 kNN 密度、动态区域生长、PlaneInstance 块内及跨块层次化合并、有限面间距和数据审计；迹线/张开度仍需高分辨率 Mesh/影像输入。

> 目标：从岩体/边坡 LAS 或 LAZ 点云中自动识别可观测节理面候选，并计算倾向、倾角、表观迹长/延伸长度、节理面间距，同时输出质量指标和可视化结果。

---

## 1. 项目目标

### 1.1 输入

系统输入：

- `.las`
- `.laz`

至少包含：

- X
- Y
- Z

可选字段：

- RGB
- intensity
- classification
- GPS time

### 1.2 输出

系统自动输出：

- 独立节理面候选实例
- 节理面组（J1/J2/J3...）
- 倾向 Dip Direction
- 倾角 Dip
- 表观延伸长度 Apparent Persistence
- 可识别情况下的迹长 Trace Length
- 同组节理面间距 Spacing
- 节理面面积
- 平面拟合 RMS
- Inlier Ratio
- Confidence
- 分割后的彩色点云
- CSV / JSON 统计结果

最终结果示例：

| plane_id | set_id | 倾向 | 倾角 | 表观延伸长度 | 面积 | RMS | confidence |
|---|---|---:|---:|---:|---:|---:|---:|
| J1-001 | J1 | 128.4° | 64.2° | 3.81 m | 7.4 m² | 0.016 m | 0.93 |
| J1-002 | J1 | 126.9° | 65.8° | 4.27 m | 8.9 m² | 0.014 m | 0.95 |
| J2-001 | J2 | 241.2° | 72.5° | 5.06 m | 12.6 m² | 0.021 m | 0.89 |

节理面组统计：

| set_id | 数量 | 平均倾向 | 平均倾角 | 平均间距 | 中位间距 |
|---|---:|---:|---:|---:|---:|
| J1 | 34 | 128° | 64° | 0.48 m | 0.45 m |
| J2 | 21 | 242° | 73° | 0.92 m | 0.84 m |
| J3 | 16 | 017° | 38° | 1.36 m | 1.29 m |

---

# 2. 系统边界

这是本项目最重要的约束。本项目的目标是识别可观测节理面候选，不做解体或失稳判断。

## 2.1 系统负责

系统负责从点云几何中识别近似平面的节理面候选，并计算其产状、面积、延伸和间距。
由于单期点云只提供表面几何，算法结果是“候选节理面”，不能仅凭几何自动证明其地质成因。

## 2.2 系统不负责

V1 明确不做：

- 推断地下未出露节理面
- 预测岩体内部真实延伸长度
- 自动区分“节理/断层/层理”等地质成因类别
- 将候选节理面直接认定为解体面、已解体块体或失稳块体
- 完整解决严重植被遮挡
- 直接计算 JRC
- RQD
- 楔形体稳定性分析
- 块体稳定性分析
- 深度学习语义分类

## 2.3 可观测值与真实地质参数必须区分

LAS 只能反映被扫描/重建到的表面。

因此：

- 倾向/倾角：在平面暴露充分时可视为结构面产状估计
- 表观延伸长度：仅代表当前点云中可见部分
- 迹长：仅代表当前暴露面上的可见迹线
- 间距：仅代表当前观测范围和测线条件下的节理面间距

系统不能把：

“可观测值”

直接称为：

“地下真实值”。

---

# 3. 坐标系边界

系统默认坐标轴：

```text
X = East
Y = North
Z = Up
```

即 ENU。

倾向计算依赖坐标轴方向，因此必须明确 CRS/坐标约定。

推荐配置：

```yaml
coordinate:
  x_axis: east
  y_axis: north
  z_axis: up
  unit: meter
```

如果实际数据是：

```text
X = North
Y = East
```

必须调整倾向公式或先转换坐标。

---

# 4. 总体处理流程

```text
LAS / LAZ
    │
    ▼
点云读取
    │
    ▼
坐标系检查
    │
    ▼
ROI 裁剪
    │
    ▼
Voxel Downsample
    │
    ▼
去噪
    │
    ▼
局部法向量估计
    │
    ▼
局部 PCA 特征
    │
    ▼
平面性筛选
    │
    ▼
法向量方向聚类
    │
    ▼
节理面组 J1 / J2 / J3
    │
    ▼
组内空间连通分割
    │
    ▼
独立节理面候选实例
    │
    ▼
RANSAC 粗拟合
    │
    ▼
PCA/TLS 精拟合
    │
    ▼
二维局部平面投影
    │
    ├── 倾向
    ├── 倾角
    ├── 面积
    ├── 表观延伸长度
    ├── 迹长（条件满足时）
    └── 平面拟合质量
    │
    ▼
同组节理面间距分析
    │
    ▼
质量评分
    │
    ▼
CSV / JSON / PLY / 报告
```

---

# 5. 技术选型

V1 推荐使用 Python。

核心技术栈：

```text
Python 3.11+
laspy
numpy
scipy
scikit-learn
shapely
pandas
pyproj
```

当前实现不依赖 Open3D：点云读取使用 `laspy`，邻域查询、局部 PCA 和稀疏空间分割使用 `scipy`，仅保留 `scikit-learn` 作为显式 DBSCAN 兼容回退。Open3D/CloudCompare 不属于本阶段运行时依赖或验收工具。

可选：

```text
hdbscan
alphashape
mplstereonet
PDAL
```

暂时不需要自己写 C++。

原因：

- LAS 读取、点云邻域、RANSAC、KDTree 已有成熟底层实现
- 主要工作是算法组合和地质参数定义
- Python 更便于迭代和参数实验
- 真正遇到超大规模性能瓶颈后再局部 C++/GPU 优化

---

# 6. 推荐项目结构

```text
rock_discontinuity/
├── main.py
├── config/
│   └── default.yaml
├── io/
│   ├── las_reader.py
│   └── result_writer.py
├── preprocessing/
│   ├── roi.py
│   ├── downsample.py
│   └── denoise.py
├── features/
│   ├── normals.py
│   ├── pca_features.py
│   └── planarity.py
├── segmentation/
│   ├── orientation_cluster.py
│   ├── spatial_cluster.py
│   └── plane_instance.py
├── fitting/
│   ├── ransac_plane.py
│   └── tls_plane.py
├── geology/
│   ├── orientation.py
│   ├── boundary.py
│   ├── persistence.py
│   ├── trace.py
│   └── spacing.py
├── quality/
│   ├── plane_quality.py
│   └── confidence.py
├── visualization/
│   ├── colorize.py
│   └── stereonet.py
└── tests/
```

---

# 7. 输入数据质量要求

建议：

```text
平均点间距 ≤ 最小目标结构面尺度的 1/5～1/10
```

例如希望稳定识别 0.5 m 级结构面，则推荐：

```text
点间距 ≤ 5～10 cm
```

需要定义：

```yaml
segmentation:
  min_plane_points: 200
  min_plane_area: 0.25
  min_plane_extent: 0.5
```

低于阈值的点簇不作为最终结构面输出。

---

# 8. 点云预处理

## 8.1 ROI

支持：

- Bounding Box
- Polygon
- 人工选择区域

大型 LAS 不建议第一版直接整幅处理。

## 8.2 Voxel Downsampling

目的：

- 统一局部点密度
- 降低计算量
- 提高邻域搜索稳定性

例如：

```yaml
preprocess:
  voxel_size: 0.03
```

## 8.3 去噪

第一版使用：

```text
Statistical Outlier Removal
```

去除：

- 飞点
- 孤立点
- 重建错误点

---

# 9. 局部法向量估计

对每个点建立局部邻域。

推荐优先使用：

```text
Radius Search
```

而不是固定 KNN，因为结构面分析与真实尺度有关。

例如：

```yaml
normal:
  radius: 0.15
  min_neighbors: 20
```

对邻域计算协方差矩阵：

\[
C = \frac{1}{N}\sum_i (P_i-\bar P)(P_i-\bar P)^T
\]

特征值排序：

\[
\lambda_1 \le \lambda_2 \le \lambda_3
\]

最小特征值对应的特征向量作为局部法向量：

\[
n=(n_x,n_y,n_z)

### 9.1 V2.3 多尺度法向选择

生产配置使用 0.12 m、0.18 m 和 0.27 m 三个嵌套邻域。三个尺度共享一次最大邻域的
KDTree 查询，再分别计算 PCA。相邻尺度的法向轴向夹角不超过 5° 时记为稳定对；默认选取
最小稳定尺度，避免大邻域跨越窄节理边界。没有稳定对时回退到 planarity/特征熵评分。

不稳定点不能作为区域生长种子，但仍可被已建立的稳定平面吸收。面级输出保留
`normal_scale_m`、`normal_stability_deg`、`normal_stable_fraction`、
`normal_valid_scale_count` 和 `normal_stable`，用于追踪过分割是否与尺度不稳定有关。
\]

---

# 10. 平面性特征

至少计算：

### Planarity

\[
P = \frac{\lambda_2-\lambda_1}{\lambda_3}
\]

### Surface Variation

\[
V = \frac{\lambda_1}{\lambda_1+\lambda_2+\lambda_3}
\]

### Linearity

\[
L = \frac{\lambda_3-\lambda_2}{\lambda_3}
\]

候选节理面点筛选：

```text
planarity > threshold
surface_variation < threshold
neighbors >= threshold
```

初始参数示例：

```yaml
feature:
  min_planarity: 0.55
  max_surface_variation: 0.08
```

这些参数必须根据实际点距、粗糙度和噪声调节。

---

# 11. 法向量统一

法向存在：

```text
n
-n
```

二义性。

统一规则：

```text
nz >= 0
```

如果：

```python
nz < 0:
    n = -n
```

---

# 12. 节理面组聚类

目标：

```text
J1
J2
J3
...
```

第一版推荐：

```text
DBSCAN
```

聚类依据为法向方向。

两个单位法向夹角：

\[
\theta = \arccos(|n_i \cdot n_j|)
\]

可定义：

```yaml
orientation_cluster:
  angle_eps_deg: 8.0
  min_samples: 100
```

---

# 13. 单个节理面候选实例分割

方向相同不代表是同一结构面。

例如：

```text
J1-001
J1-002
J1-003
```

可能都属于 J1，但空间位置不同。

因此需要第二阶段：

```text
方向聚类
+
XYZ 空间聚类
```

组内空间分割可使用 DBSCAN：

```yaml
spatial_cluster:
  eps: 0.15
  min_samples: 20
  normal_angle_eps_deg: 3.0
```

最终得到独立节理面候选实例。

---

# 14. 平面拟合

每个节理面候选实例执行：

```text
RANSAC
↓
删除离群点
↓
TLS/PCA 精拟合
```

平面方程：

\[
ax+by+cz+d=0
\]

并归一化：

\[
a^2+b^2+c^2=1
\]

法向：

\[
n=(a,b,c)
\]

---

# 15. 平面拟合质量

点到平面残差：

\[
r_i=|ax_i+by_i+cz_i+d|
\]

输出：

- RMS
- MAE
- P95 residual
- inlier_ratio

RMS：

\[
RMS=\sqrt{\frac{1}{N}\sum_i r_i^2}
\]

这些指标必须保留，不能只输出最终倾向倾角。

---

# 16. 倾角计算

坐标：

```text
X East
Y North
Z Up
```

法向统一到：

```text
nz >= 0
```

倾角：

\[
dip=
atan2(
\sqrt{n_x^2+n_y^2},
n_z
)
\]

结果范围：

```text
0°～90°
```

其中：

- 0°：水平
- 90°：直立

---

# 17. 倾向计算

采用最大坡降方向。

对于 ENU：

\[
dipdir = atan2(n_x,n_y)
\]

归一化：

```python
dip_direction = (
    np.degrees(np.arctan2(nx, ny)) + 360
) % 360
```

约定：

- 0° 北
- 90° 东
- 180° 南
- 270° 西

最终产状例如：

```text
128° ∠ 64°
```

---

# 18. 局部二维结构面坐标系

为了计算：

- 边界
- 面积
- 表观长度
- 迹长

需要在拟合平面内建立局部基：

```text
u
v
n
```

满足：

```text
u ⟂ v
u ⟂ n
v ⟂ n
```

三维点投影为：

```text
XYZ → UV
```

后续几何分析在 UV 平面完成。

---

# 19. 结构面边界

第一版优先：

```text
Alpha Shape
```

而不是 Convex Hull。

因为真实结构面边界可能：

- 不规则
- 凹陷
- 有缺口

Convex Hull 容易过度包络。

---

# 20. 面积

结构面点投影到 UV 后，基于边界多边形计算面积：

```text
area_m2
```

---

# 21. 表观延伸长度

定义：

> 点云中当前可见结构面在拟合平面内的最大可观测延伸尺度。

字段：

```text
apparent_persistence_m
```

第一版可用：

```text
Maximum Feret Diameter
```

或：

\[
L=\max_{i,j}||p_i-p_j||
\]

注意：

```text
apparent_persistence ≠ 地下真实结构面长度
```

---

# 22. 迹长定义和边界

必须严格区分：

```text
apparent_persistence
```

和：

```text
trace_length
```

迹长定义：

> 结构面在当前岩壁/坡面暴露面上的可观测交线长度。

V1 规则：

- 如果能稳定提取结构面与暴露面的线性迹线，则计算 `trace_length`
- 如果条件不足，则 `trace_length = null`
- 禁止直接拿结构面的最大长轴冒充迹长

后续可通过：

- 边缘点提取
- 曲率突变
- 法向突变
- Skeleton
- Polyline tracing

计算真正的迹线。

---

# 23. 节理面间距

间距只对同一节理面组计算。

例如：

```text
J1:
  Plane 1
  Plane 2
  Plane 3
```

不能直接计算质心欧氏距离。

错误：

```text
distance(centroid1, centroid2)
```

推荐：

```text
沿平均结构面法向
或
沿指定虚拟测线
```

计算。

---

# 24. 平均法向

同组法向：

```text
n1, n2, ..., nn
```

统一半球后：

\[
\bar n = normalize(\sum_i n_i)
\]

---

# 25. 简化间距算法

结构面中心：

```text
c_i
```

投影：

\[
t_i=\bar n \cdot c_i
\]

排序：

```text
t1 < t2 < ... < tn
```

间距：

\[
s_i=t_{i+1}-t_i
\]

---

# 26. 推荐间距算法：虚拟测线法

虚拟测线：

\[
x(t)=x_0+t\bar n
\]

第 i 个结构面：

\[
n_i\cdot x+d_i=0
\]

交点参数：

\[
t_i=
-\frac{n_i\cdot x_0+d_i}
{n_i\cdot\bar n}
\]

排序后：

\[
spacing_i=|t_{i+1}-t_i|
\]

V1 优先实现这一方法。

输出统计：

- min
- max
- mean
- median
- std
- P10
- P25
- P75
- P90

---

# 27. 间距边界

以下情况不计算：

- 同组结构面少于 2 个
- 结构面与虚拟测线几何关系异常
- 法向分散过大
- 当前空间范围不足以形成有代表性的间距统计

此时输出：

```text
spacing_available = false
```

而不是强行给数值。

---

# 28. 质量评分

每个结构面必须有质量指标。

建议综合：

- point_count
- area
- planarity
- inlier_ratio
- RMS
- normal_dispersion
- boundary_completeness

形成：

```text
confidence ∈ [0,1]
```

示例：

```yaml
quality:
  min_points: 200
  min_area: 0.25
  max_rms: 0.05
  min_inlier_ratio: 0.70
  min_confidence: 0.60
```

---

# 29. 输出文件设计

推荐输出：

```text
output/
├── planes.csv
├── joint_planes.csv
├── joint_sets.csv
├── spacings.csv
├── tile_spacings.csv
├── detachment_planes.csv          # 历史兼容文件名：候选节理面清单
├── detachment_candidates.ply      # 历史兼容文件名：候选节理面可视化
├── segmented_planes.ply
├── rejected_planes.csv
├── data_audit.json
├── density_report.json
├── run.json
├── traces.csv
├── aperture.csv
├── plane_boundaries.geojson
├── trace_lines.geojson
├── report.json
└── run_config.yaml
```

---

# 30. planes.csv 字段

```text
plane_id
set_id
point_count

nx
ny
nz
plane_d

dip_direction_deg
dip_deg

centroid_x
centroid_y
centroid_z

area_m2
apparent_persistence_m
trace_length_m

rms_m
mae_m
p95_residual_m
inlier_ratio

confidence
```

---

# 31. joint_sets.csv 字段

```text
set_id
plane_count

mean_dip_direction_deg
mean_dip_deg

mean_normal_x
mean_normal_y
mean_normal_z

mean_spacing_m
median_spacing_m
std_spacing_m
min_spacing_m
max_spacing_m
p10_spacing_m
p90_spacing_m
```

---

## 候选节理面输出字段

`detachment_planes.csv` 是历史兼容文件名，在节理面基础字段上直接输出候选节理面摘要：

```text
dip_direction_deg       # 倾向，ENU 方位角
dip_deg                 # 倾角
trace_length_m          # 迹长；无可靠暴露面迹线时为 null
apparent_persistence_m  # 点云观测平面最大延伸
nearest_spacing_m       # 同一节理组内最近的成对间距
center_x
center_y
center_z                # 候选节理面中心坐标，ENU
```

`spacings.csv` 保存完整的成对间距关系，`nearest_spacing_m` 只是便于按单个
候选面查看的摘要值。全点云分块结果另保留 `tile_spacings.csv` 作为瓦片内审计结果；
`spacings.csv` 使用跨瓦片合并后的全局 `GJ-xxxxx` 节理面重新计算。

# 32. 可视化

至少输出：

```text
segmented_planes.ply
```

不同结构面使用不同颜色。

后续可增加：

- Pole Plot
- Stereonet
- Rose Diagram
- Dip/Dip Direction Scatter
- Spacing Histogram

---

# 33. 参数配置文件

示例：

```yaml
input:
  path: data/input.las

coordinate:
  x_axis: east
  y_axis: north
  z_axis: up
  unit: meter

preprocess:
  voxel_size: 0.03
  remove_outliers: true

normal:
  radius: 0.15
  min_neighbors: 20

feature:
  min_planarity: 0.55
  max_surface_variation: 0.08

orientation_cluster:
  method: dbscan
  angle_eps_deg: 8.0
  min_samples: 100

spatial_cluster:
  eps: 0.12
  min_samples: 50

plane:
  ransac_distance: 0.03
  min_points: 200
  min_area: 0.25
  max_rms: 0.05

boundary:
  method: alpha_shape
  alpha: auto

spacing:
  method: virtual_scanline

output:
  directory: output/
```

---

# 34. 参数尺度自适应

所有空间参数必须与点间距相关。

首先估计：

```text
median_nearest_neighbor_distance = dp
```

然后自动推导：

```text
voxel_size ≈ 1～2 dp
normal_radius ≈ 5～10 dp
spatial_eps ≈ 3～8 dp
ransac_threshold ≈ 1～3 dp
```

这样比写死固定米制参数更可靠。

---

# 35. 大型 LAS 性能边界

如果点数达到：

```text
1 亿+
```

V1 不应一次性全部读入内存。

策略：

```text
ROI
+
空间分块
+
Overlap
+
降采样
```

例如：

```text
50 m × 50 m
overlap = 1 m
```

全局流程在块内识别后，只在相邻块之间按法向夹角、拟合平面距离和投影多边形边界
相接条件合并 `PlaneInstance`。跨块采用完整链接约束避免 A-B-C 链式误合并，随后依据
联合足迹和加权拟合质量重新执行全局候选门槛，并按最终全局 `GJ-xxxxx` 编号过滤红色
LAZ；瓦片实例与全局编号同时输出，保证结果可追溯。

合并条件：

- 法向夹角小
- 平面距离小
- 空间邻近
- 边界相接

---

# 36. MVP 实现目标

第一版必须完成闭环：

```text
LAS
↓
法向量
↓
平面性
↓
节理面组
↓
单节理面候选实例
↓
平面拟合
↓
倾向
↓
倾角
↓
面积
↓
表观延伸长度
↓
间距
↓
质量评分
↓
CSV + PLY
```

---

# 37. MVP 不包含

暂不实现：

- 地下结构面延伸推断
- 高精度自动迹线骨架提取
- 地质成因分类
- JRC
- RQD
- 楔形体稳定性
- 深度学习模型
- 自动植被语义分割

---

# 38. 第二阶段

V2 可增加：

- 真正的 trace extraction
- 多尺度法向量（V2.3 已实现）
- 自动参数尺度估计
- 复杂边界恢复
- 重叠区联合边界去重和更强的跨块合并置信度
- 大型 LAZ Streaming
- PDAL Pipeline
- 更强的植被过滤

V2.3 同时实现了层次化二次合并：对第一阶段严格合并结果再次按完整链接、投影足迹邻接、
平面偏移、合并后 RMS 和 RMS 增长比筛选，避免仅凭同向产状把不同节理粘成一个面。

---

# 39. 第三阶段

V3 再考虑：

```text
Point Transformer
PointNet++
KPConv
SparseConv
```

用于：

- 岩体/植被语义分类
- 复杂粗糙结构面识别
- 结构面类型识别

深度学习不是 V1 的前提。

---

# 40. 验收标准

系统验收不能只靠“看起来不错”。

必须定量验证。

## 40.1 合成数据定量验证

合成场景必须保留已知平面、产状、面积和间距真值，用于验证：

- ENU 倾向和倾角计算方向正确；
- 平面实例能够被分割并拟合；
- 两个及以上同组平面能够计算间距；
- 面积、表观延伸和残差字段有限且可重复。

合成数据是当前版本唯一的定量真值来源。具体阈值随合成噪声、点密度和尺度配置记录在测试中，不从真实 ROI 结果外推现场精度。

## 40.2 真实 ROI 工程验证

真实 ROI 只验证：

- ROI 过滤和分块读取；
- 预处理、法向、候选点和实例分割链路；
- 产状、面积、表观延伸、间距和质量字段输出；
- CSV/JSON/PLY/NPZ 文件结构一致；
- 相同输入和配置能够重复运行。

当前版本不使用 CloudCompare 人工标注、现场罗盘或人工测线，不计算真实 ROI 的 Precision、Recall、F1 或现场 MAE，也不宣称现场测量精度。

`detachment_candidates.ply` 仅提供观测节理面候选的可视化：候选点染为红色，其他点染为灰色。
它不表示已经解体或失稳。

---

# 41. Benchmark 场景

至少建立三类测试场景。

### 场景 A

```text
裸岩
结构面清晰
植被少
```

验证基础精度。

### 场景 B

```text
表面粗糙
节理破碎
```

验证鲁棒性。

### 场景 C

```text
点密度变化
部分遮挡
```

验证复杂环境适应性。

---

# 42. 主要风险

## 风险 1：岩壁主表面被误认为结构面

解决：

- 法向分布
- 空间尺度
- 边界形态
- 曲率
- 地质约束

联合判断。

## 风险 2：粗糙面导致法向离散

解决：

```text
多尺度 PCA
```

## 风险 3：点密度变化

解决：

```text
Voxel + 自适应尺度
```

## 风险 4：植被

V1 只能部分处理。

后续结合：

- RGB
- classification
- 几何特征
- 深度学习

## 风险 5：遮挡

遮挡会造成：

- 面积偏小
- 表观长度偏短
- 迹长不完整

输出中必须保留“observed”语义。

---

# 43. 核心数据对象

```python
@dataclass
class PlaneInstance:
    plane_id: str
    set_id: str | None

    point_indices: np.ndarray

    normal: np.ndarray
    d: float

    centroid: np.ndarray

    dip_direction: float
    dip: float

    area: float
    apparent_persistence: float
    trace_length: float | None

    rms: float
    inlier_ratio: float
    confidence: float
```

节理面组：

```python
@dataclass
class JointSet:
    set_id: str

    plane_ids: list[str]

    mean_normal: np.ndarray

    mean_dip_direction: float
    mean_dip: float

    spacings: np.ndarray
```

---

# 44. CLI 设计

运行方式：

```bash
python3 main.py   --input data/slope.las   --output output/   --config config/default.yaml
```

---

# 45. 开发里程碑

### Milestone 1

```text
LAS 读取 + 点云统计 + 可视化
```

### Milestone 2

```text
法向量 + PCA + 平面性
```

### Milestone 3

```text
法向方向聚类
```

### Milestone 4

```text
空间分割独立结构面
```

### Milestone 5

```text
RANSAC + TLS 精拟合
```

### Milestone 6

```text
倾向 + 倾角
```

### Milestone 7

```text
边界 + 面积 + 表观延伸长度
```

### Milestone 8

```text
同组节理面间距
```

### Milestone 9

```text
质量评分 + CSV + PLY
```

### Milestone 10

```text
合成真值 + ROI 工程回归验证
```

---

# 46. 开发原则

整个系统必须保留：

```text
输入
中间结果
参数
最终结果
质量指标
```

不能只输出最终的：

```text
倾向
倾角
```

否则无法：

- 调参
- 排错
- 做验收
- 复现实验

---

# 47. 最终成功定义

V1 成功的标准不是：

```text
能打开 LAS
```

也不是：

```text
能 RANSAC 出几个平面
```

而是完成：

```text
LAS
↓
自动节理面候选实例
↓
自动结构面分组
↓
倾向/倾角
↓
面积/表观延伸长度
↓
间距
↓
质量评分
↓
可重复运行
```

---

# 48. 最终交付物

代码：

```text
rock_discontinuity/
```

配置：

```text
config/default.yaml
```

输出：

```text
planes.csv
joint_planes.csv
joint_sets.csv
spacings.csv
tile_spacings.csv
detachment_planes.csv          # 历史兼容文件名：候选节理面清单
detachment_candidates.ply      # 历史兼容文件名：候选节理面可视化
segmented_planes.ply
report.json
```

验证：

```text
benchmark/
```

文档：

```text
README.md
ALGORITHM.md
PARAMETERS.md
VALIDATION.md
```

---

# 49. 最终结论

本系统应定义为：

> 从岩体三维点云中识别“可观测节理面候选实例”，并基于几何形态和空间关系计算节理面产状及统计参数。

V1 核心目标：

```text
节理面识别
+
节理面组识别
+
倾向
+
倾角
+
面积
+
表观延伸长度
+
间距
+
质量评估
```

其中：

- 倾向、倾角：平面产状参数
- 面积、表观延伸长度：当前可见节理面几何参数
- 迹长：有明确迹线条件时计算
- 间距：同组节理面的空间统计参数
- confidence：结果可信度参考

系统必须始终区分：

```text
点云中可观测的几何量
```

与：

```text
真实地下地质参数
```

不对未观测信息做未经验证的确定性推断。

# 50. V2.3 三 ROI 实验记录

使用 `main.py roi-compare` 对三个典型 ROI 做同输入消融比较，结果写入
`outputs/roi_compare_v23/roi_comparison.csv` 和 `roi_comparison.json`。最终
`multiscale_hmerge` 相对 `baseline` 的候选点比例为：

| ROI | baseline | multiscale_hmerge | 变化 |
|---|---:|---:|---:|
| ROI1 高碎片 | 30.94% | 28.85% | -2.09 个百分点 |
| ROI2 对照 | 8.12% | 9.23% | +1.11 个百分点 |
| ROI3 高红色比例 | 24.72% | 27.71% | +2.99 个百分点 |

该结果证明多尺度稳定性和二次合并已经接入并能改变局部过分割行为，但不能证明所有 ROI
都变好。当前应把它作为工程稳定性改进和诊断字段，而不是现场精度验证；真实 ROI 仍缺少
人工/现场真值，不能从候选点比例直接推导节理数量或失稳数量。
