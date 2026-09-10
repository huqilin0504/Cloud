# 基于 LAS/LAZ 点云的岩体节理面自动识别与几何参数提取系统
## 完整技术实现设计文档 V1.0

> **项目目标**  
> 从岩体/危岩体三维 LAS/LAZ 点云中，采用成熟、可解释、可验证的点云几何算法，自动识别可观测节理面实例，并提取：
>
> - 节理面实例
> - 节理组（Joint Set）
> - 倾向（Dip Direction）
> - 倾角（Dip）
> - 面积（Observed Area）
> - 表观延伸长度（Apparent Persistence）
> - 节理迹长（Trace Length，在满足可观测条件时）
> - 同组节理间距（Spacing）
> - 平面拟合误差与结果质量等级
>
> V1 **不采用深度学习作为核心识别技术**，优先采用已经经过长期工程实践验证的点云处理、局部 PCA、区域生长、RANSAC、TLS、密度聚类和计算几何技术。

---

# 1. 项目定位

本系统定位为：

> **岩体节理面三维几何自动解译工具。**

系统解决的是：

```text
LAS / LAZ
    ↓
岩体表面点云
    ↓
寻找近似平面状不连续面
    ↓
分割成独立节理面实例
    ↓
节理组分类
    ↓
计算产状和几何参数
```

系统不是一个地质成因分类模型。

仅从 XYZ 几何上识别出的对象，更严格的名称是：

```text
Planar Discontinuity Candidate
平面状不连续面候选
```

在项目区域已经明确主要不连续面为节理、并通过地质调查排除了明显层面/断层面的情况下，可以将通过质量控制后的候选面作为：

```text
Joint Plane
节理面
```

输出。

---

# 2. 系统实现目标

## 2.1 核心目标

V1 必须实现以下闭环：

```text
LAS / LAZ
↓
数据质量检查
↓
ROI
↓
局部点密度检查
↓
噪声处理
↓
法向量与局部 PCA
↓
平面候选点
↓
区域生长分割
↓
独立节理面实例
↓
RANSAC + TLS 精拟合
↓
倾向 / 倾角
↓
节理面边界
↓
面积 / 表观延伸长度
↓
节理组 J1/J2/J3...
↓
同组间距
↓
质量评分
↓
CSV / JSON / 可视化点云
```

## 2.2 V1 最小识别目标

项目当前采用以下工程目标：

```text
目标原始局部表面点密度：约 600 points/m²
最小目标节理面等效尺度：0.5 m
最小目标面积：0.25 m²
最小有效点数：100 points
规划有效点保留率：约 70%
```

理论关系：

\[
N_{effective}=\rho_{raw}A\eta
\]

代入：

\[
600\times0.25\times0.70=105
\]

因此：

> 在局部真实表面密度约 600 点/m²、遮挡和噪声可控的条件下，0.5 m 级节理面理论上可以保留约 100 个有效分析点，作为 V1 最小工程目标。

注意：**70% 是项目规划阶段的工程假设，不是固定物理常数。** 正式运行必须记录 `eta_observed = 有效点数 / 原始局部点数`。

---

# 3. 明确系统边界

## 3.1 V1 能做

在节理面确实被点云观测到的前提下，系统能够：

1. 自动识别空间连续的近似平面区域；
2. 将其分割为独立节理面实例；
3. 计算节理面倾向；
4. 计算节理面倾角；
5. 计算可见节理面面积；
6. 计算表观延伸长度；
7. 按产状将节理面划分成 J1/J2/J3 等优势节理组；
8. 计算同组节理面的法向间距；
9. 在具有有效测线时计算测线间距；
10. 在迹线可明确观测且迹线模块启用时计算迹长；
11. 输出误差、点数、面积、残差、置信等级等质量信息。

## 3.2 V1 明确不做

V1 不负责：

- 从点云恢复地下未暴露节理；
- 推断节理真实地下延伸长度；
- 自动判断“节理 / 层理 / 断层 / 片理”的成因；
- 自动完成复杂植被语义分割；
- 自动推断被植被或遮挡完全覆盖的节理；
- 将所有细裂缝线都解释成节理面；
- 自动计算 JRC；
- 自动计算 RQD；
- 危岩块体三维重建；
- 楔形体稳定性分析；
- 平面滑动稳定性计算；
- 倾倒破坏稳定性计算；
- 深度学习节理语义识别。

这些可以作为后续模块。

---

# 4. 关键地质参数定义

## 4.1 节理面实例

单个独立、空间连续的可见平面区域，例如：

```text
J1-001
J1-002
J1-003
```

## 4.2 节理组

产状相近的一组节理面：

```text
J1
J2
J3
```

节理组和节理实例不能混淆。

## 4.3 倾角

范围 `0°～90°`：

- `0°` = 水平
- `90°` = 直立

## 4.4 倾向

范围 `0°～360°`：

- `0°` = North
- `90°` = East
- `180°` = South
- `270°` = West

## 4.5 表观延伸长度

字段：`apparent_persistence_m`

定义：当前点云中实际观测到的节理面，在其自身拟合平面内的最大可见延伸尺度。

它不是地下真实延伸长度。

## 4.6 节理迹长

字段：`trace_length_m`

定义：节理面与当前暴露岩壁/坡面的可见交迹长度。

必须严格区分：

```text
Trace Length ≠ Facet Maximum Length
```

如果无法可靠得到真实交迹：

```text
trace_length_m = null
```

不能用结构面最大轴长代替。

## 4.7 节理间距

对于同一节理组，相邻节理面沿组平均法向或指定测线方向的间隔。

禁止使用质心三维欧氏距离直接作为节理间距。

---

# 5. 技术选型原则

技术选择遵循：

```text
成熟
可解释
可验证
易维护
大数据可扩展
无训练数据依赖
```

V1 不需要训练模型。

---

# 6. 最终技术栈

## 6.1 PDAL

用途：

- LAS/LAZ 读取；
- ROI 裁剪；
- 大文件流水线；
- 噪声标记；
- 可选 voxel downsampling；
- CRS 信息处理；
- 点云预处理。

PDAL 作为大型点云工程 I/O 和流水线层。

## 6.2 Python 3.11+

作为业务和算法编排层。

## 6.3 NumPy

用途：矩阵、PCA、TLS、平面计算、产状计算、向量化处理。

## 6.4 SciPy

用途：KDTree、空间搜索、Delaunay、Convex Hull、统计分析、计算几何辅助。

## 6.5 Open3D

用途：

- PointCloud 数据结构；
- Hybrid Neighbor Search；
- 法向量估计；
- RANSAC 平面拟合；
- 可视化；
- 底层 C++ 点云运算。

## 6.6 scikit-learn

用途：DBSCAN、聚类工具、Benchmark 指标。

DBSCAN 主要用于节理组方向聚类，以及必要时的碎片空间聚类。

单节理面核心分割不依赖纯 DBSCAN，而采用：

```text
Planarity Seeded Region Growing
```

## 6.7 Shapely

用途：二维 polygon、面积、边界、几何相交、边界后处理。

## 6.8 pandas

用途：CSV、统计结果、参数表、Benchmark。

## 6.9 pyproj

用途：CRS 验证、坐标转换、确认 East/North/Up 约定。

## 6.10 CloudCompare

不作为生产代码依赖，而作为独立人工验证工具和工程基准。使用：

```text
qFacets
Compass
Fit Plane
```

对自动结果独立验证。

---

# 7. 为什么采用区域生长，而不是全局循环 RANSAC

不推荐：

```text
整个坡面
↓
RANSAC 最大平面
↓
删除
↓
继续 RANSAC
```

原因：

- 大岩壁主面容易吞噬局部节理；
- 小节理容易被大面掩盖；
- 相邻近平行节理容易合并；
- 点密度变化会明显影响结果；
- 不能很好表达有限、连续的节理面边界。

V1 采用：

```text
局部 PCA
↓
高平面性种子
↓
空间邻域区域生长
↓
有限平面 patch
↓
RANSAC + TLS 精拟合
```

更符合节理面“局部平面 + 空间连续”的几何特征。

---

# 8. 系统总体架构

```text
                        cloud.las / cloud.laz
                                │
                                ▼
                    ┌────────────────────┐
                    │   Data Audit       │
                    │ CRS / XYZ / density│
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ PDAL Preprocessing │
                    │ ROI / outlier      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Local Geometry     │
                    │ Normal / PCA       │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Planar Candidates  │
                    └─────────┬──────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Region Growing       │
                   │ Plane Instances      │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ RANSAC + TLS         │
                   │ Plane Refinement     │
                   └──────────┬───────────┘
                              │
                 ┌────────────┼─────────────┐
                 ▼            ▼             ▼
             Orientation    Boundary      Quality
                 │            │             │
                 └────────────┼─────────────┘
                              ▼
                   ┌──────────────────────┐
                   │ Joint Set Clustering │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Spacing Analysis     │
                   └──────────┬───────────┘
                              │
                              ▼
                  CSV / JSON / PLY / Report
```

---

# 9. 推荐项目目录

```text
rock_joint_analyzer/
├── pyproject.toml
├── README.md
├── config/
│   ├── default.yaml
│   └── production_600ppm2.yaml
├── src/
│   └── rock_joint/
│       ├── cli.py
│       ├── pipeline.py
│       ├── io/
│       │   ├── las_info.py
│       │   ├── pdal_runner.py
│       │   └── writer.py
│       ├── qc/
│       │   ├── crs_check.py
│       │   ├── density.py
│       │   └── audit.py
│       ├── preprocessing/
│       │   ├── crop.py
│       │   ├── outlier.py
│       │   └── recenter.py
│       ├── geometry/
│       │   ├── neighbors.py
│       │   ├── normals.py
│       │   ├── pca.py
│       │   ├── plane.py
│       │   └── orientation.py
│       ├── segmentation/
│       │   ├── planar_candidates.py
│       │   ├── region_growing.py
│       │   ├── merge.py
│       │   └── reject.py
│       ├── fitting/
│       │   ├── ransac.py
│       │   └── tls.py
│       ├── joints/
│       │   ├── set_clustering.py
│       │   ├── fisher_stats.py
│       │   ├── boundary.py
│       │   ├── persistence.py
│       │   ├── trace.py
│       │   └── spacing.py
│       ├── quality/
│       │   ├── gates.py
│       │   └── score.py
│       └── models/
│           ├── plane_instance.py
│           └── joint_set.py
├── tests/
│   ├── unit/
│   ├── synthetic/
│   └── integration/
├── benchmark/
├── scripts/
└── output/
```

---

# 10. Phase 0：输入数据审计

在执行任何识别算法之前，必须先完成数据审计。

## 10.1 LAS Header

读取：

```text
point count
min XYZ
max XYZ
scale
offset
point format
CRS
```

## 10.2 坐标轴确认

正式计算倾向之前必须确认：

```text
X = East
Y = North
Z = Up
unit = meter
```

如果不满足，必须先转换。

## 10.3 数值稳定性

工程坐标可能为：

```text
X = 500000+
Y = 3000000+
```

算法计算时使用局部坐标：

```text
p_local = p_world - ROI_origin
```

所有最终结果再加回世界坐标，避免大坐标影响局部协方差和拟合精度。

---

# 11. ROI

ROI 是 Region Of Interest。

V1 开发阶段不直接处理整座危岩体。第一阶段推荐 `20～50 m` 量级典型裸岩区域，应尽量满足：

- 节理清晰；
- 植被较少；
- 不贴数据外边界；
- 能看到多个节理面；
- 最好包含 2～3 组产状。

---

# 12. PDAL ROI 处理

支持 3D Bounding Box 或 Polygon。

示例：

```json
[
  "cloud.laz",
  {
    "type": "filters.crop",
    "bounds": "([xmin,xmax],[ymin,ymax],[zmin,zmax])"
  },
  "roi.laz"
]
```

---

# 13. 点密度质量控制

## 13.1 禁止使用简单 XY bbox 密度

不能用：

```text
point_count / XY Bounding Box Area
```

因为岩壁可能近乎直立。

## 13.2 局部表面密度

采用 k 近邻方法估计局部密度。对每个采样点找到第 k 个邻点距离 `r_k`，局部近似：

\[
\rho_i\approx\frac{k}{\pi r_k^2}
\]

推荐：

```text
k = 20
```

最终报告：P10、P25、Median、Mean、P75、P90。

---

# 14. 600 points/m² 工程标准

当前 V1：

```text
target_raw_density = 600 points/m²
```

约对应 4 cm 级平均表面点距。

质量报告至少给出：

```text
median_local_density
p10_local_density
```

建议：

```text
Median >= 600 points/m²
```

并同时观察 P10，避免少量超高密度区域拉高平均值。

---

# 15. 数据密度不足时的行为

系统不能偷偷降低目标。

例如目标：

```text
Lmin = 0.5 m
Nmin = 100
```

当前 ROI 只有 300 points/m²，必须输出：

```text
DENSITY_TARGET_NOT_MET
```

并计算：

\[
L_{stable}=\sqrt{\frac{N_{min}}{\rho_{raw}\eta}}
\]

告诉用户当前点云更适合识别多大尺度的节理面。

---

# 16. 噪声处理

V1 采用成熟的统计离群点算法。

推荐 PDAL：

```text
filters.outlier
method = statistical
```

初始参数：

```text
mean_k = 12～20
multiplier = 2.0～2.5
```

噪声点先标记 `Classification = 7`，再过滤。

---

# 17. Voxel Downsample 策略

600 点/m² 已经是目标分析密度，因此：

> **600 点/m² 附近的数据默认不强制降采样。**

只有局部密度明显高于目标时才执行 voxel downsampling。

建议 voxel cell 不超过 `0.5 × median nearest-neighbor spacing`，任何降采样后必须重新执行 density QC。

---

# 18. 局部邻域

局部几何计算使用 Hybrid Search：

```text
radius + max_nn
```

避免纯 KNN 在稀疏区域搜索过远，也避免纯 Radius 在高密度区邻点过多。

---

# 19. 600 点/m² 下的法向初始参数

平均点距约 4 cm。

0.5 m 最小节理目标下，建议初始：

```text
normal_radius: 0.12 ～ 0.18 m
推荐起始值: 0.15 m
max_nn: 64
```

半径 0.15 m 在 600 点/m² 理想平面上理论包含：

\[
600\times\pi\times0.15^2\approx42
\]

个点，适合作为局部 PCA 起始尺度。

---

# 20. 局部 PCA

邻域协方差矩阵：

\[
C=\frac{1}{N}\sum_i(p_i-\bar p)(p_i-\bar p)^T
\]

特征值：

\[
\lambda_1\le\lambda_2\le\lambda_3
\]

最小特征值对应向量作为局部法向量。

---

# 21. 局部几何特征

## 21.1 Planarity

\[
P=\frac{\lambda_2-\lambda_1}{\lambda_3}
\]

## 21.2 Surface Variation

\[
V=\frac{\lambda_1}{\lambda_1+\lambda_2+\lambda_3}
\]

## 21.3 Linearity

\[
L=\frac{\lambda_3-\lambda_2}{\lambda_3}
\]

---

# 22. 平面候选点

初始推荐：

```yaml
planar_candidate:
  min_planarity: 0.55
  max_surface_variation: 0.08
  min_neighbors: 20
```

这些不是行业固定常数，必须经过 Synthetic Benchmark、CloudCompare Benchmark 和真实 ROI 标定。

---

# 23. 法向方向统一

PCA 法向存在 `n` 和 `-n` 二义性。

对于产状计算统一：

```text
nz >= 0
```

对于近直立面另外记录：

```text
dip_direction_ambiguous
```

避免在 90° 附近过度解释法向方向。

---

# 24. 核心实例分割：Planarity-Seeded Region Growing

这是 V1 的核心算法。

## 24.1 Seed

按照 Planarity 从高到低选择未分配候选点作为种子。

## 24.2 Seed Plane

使用种子局部邻域 PCA 产生初始平面：

\[
n\cdot x+d=0
\]

## 24.3 邻点加入条件

候选邻点必须同时满足：

### 条件 A：法向夹角

\[
\theta=\arccos(|n_p\cdot n_{region}|)
\]

要求：

```text
theta <= angle_threshold
```

建议起始值：

```text
8°
```

### 条件 B：点到当前区域平面距离

\[
distance=|n\cdot p+d|
\]

要求小于动态阈值。

## 24.4 距离阈值不要写死

首先从高平面性区域估计 `sigma_noise`，例如使用局部平面残差的 MAD。

建议：

\[
\tau_d=max(2.5\sigma_{noise},0.5d_{50})
\]

并设置项目允许上限。

600 点/m² 数据可从 `0.02～0.04 m` 范围开始实验。

## 24.5 区域平面动态更新

不能始终使用 seed plane。

区域每增加约 32～64 points，重新 TLS 拟合 region plane，更新 normal、d 和 RMS。

---

# 25. Region Growing 停止条件

当以下任一条件出现时停止：

- 无满足条件邻点；
- 区域 RMS 超阈值；
- 法向变化超过阈值；
- 遇到明显边界。

---

# 26. 小碎片合并

两个 patch 只有同时满足下列条件才允许合并：

```text
normal angle <= 5°
plane orthogonal distance <= merge_distance
boundary spatial gap <= merge_gap
merged RMS 合格
```

否则禁止合并。

---

# 27. 禁止只按方向合并

两个面同方向但空间上分离，应该属于同一节理组，但不是同一个节理面。

因此必须区分：

```text
Plane Merge
```

和：

```text
Joint Set Classification
```

---

# 28. 单节理面硬门槛

V1 默认：

```yaml
plane_gate:
  min_effective_points: 100
  min_area_m2: 0.25
  min_minor_extent_m: 0.15
  min_inlier_ratio: 0.70
```

RMS 上限采用 noise-aware threshold，而不是简单写死。

---

# 29. RANSAC 粗拟合

Region Growing 得到初始 patch 后，对每个 patch 单独进行 RANSAC。

目的：

- 去除少量非平面点；
- 去除边界污染；
- 提高最终 plane 稳定性。

禁止对整个 ROI 全局循环 RANSAC 作为主分割算法。

---

# 30. TLS 精拟合

对 RANSAC inliers 使用 Total Least Squares / PCA 重新拟合最终平面。

最终保存：

```text
nx
ny
nz
d
centroid
rms
mae
p95_residual
inlier_ratio
```

---

# 31. 倾角计算

假设：

```text
X = East
Y = North
Z = Up
nz >= 0
```

则：

\[
dip=atan2(\sqrt{n_x^2+n_y^2},n_z)
\]

转换为 degree。

---

# 32. 倾向计算

最大坡降方向：

\[
dipdir=atan2(n_x,n_y)
\]

归一化到 `[0,360)`。

---

# 33. 近直立面处理

当：

```text
dip >= 89°
```

输出：

```text
near_vertical = true
```

并保留 strike 用于人工检查。

---

# 34. 节理面局部二维坐标

对最终平面构建 `u, v, n` 正交坐标系，将 inliers 投影：

```text
XYZ → UV
```

边界、面积、长度在 UV 中计算。

---

# 35. 边界提取

禁止默认直接使用 Convex Hull，因为会过度包络凹边界。

生产实现推荐：

```text
2D Concave Hull
```

或：

```text
Alpha Shape
```

并提供 Convex Hull 作为稳定 fallback。

---

# 36. 边界参数尺度

边界算法参数应与 median point spacing 相关，例如：

```text
boundary max gap = 2～4 × local point spacing
```

不能对所有数据固定一个绝对数字。

---

# 37. 节理面面积

边界 polygon 位于节理自身 UV 平面，因此二维面积即为该平面可见面积：

```text
observed_area_m2
```

---

# 38. 表观延伸长度

V1 自动计算：

```text
apparent_persistence_m
```

推荐采用 Maximum Feret Diameter，并额外输出：

```text
major_extent_m
minor_extent_m
```

---

# 39. 为什么表观延伸长度不是严格迹长

一个完整暴露节理面的最大长度是 facet extent，并不一定是 fracture trace。

因此 `apparent_persistence_m` 可以自动稳定输出，但 `trace_length_m` 必须单独处理。

---

# 40. 迹长技术路线

为保证工程可靠性，V1 采用：

```text
自动候选 + 人工确认 / 半自动路径追踪
```

而不是宣称全自动可靠迹线识别。

## 40.1 Trace Mode

在岩壁表面建立邻接图：

```text
point / mesh vertex = graph node
```

邻接关系：

```text
kNN / mesh edge
```

## 40.2 Trace Cost

路径代价可综合：

```text
curvature
normal discontinuity
intensity
RGB contrast
local roughness
```

## 40.3 起止点

V1 推荐用户选择 start/end，或算法提出候选端点后由用户确认，再执行 Dijkstra / A* 最小代价路径。

## 40.4 Trace Length

最终 3D polyline：

\[
L=\sum_i\|p_{i+1}-p_i\|
\]

---

# 41. 节理组分类

单节理面实例识别完成后再进行节理组分类。

输入是每个 PlaneInstance 的 normal，而不是所有原始点的 normal。

这样：

- 计算量小；
- 结构面级别更加稳定；
- 不被超大节理面的点数完全支配。

---

# 42. 节理组聚类方法

V1 推荐：

```text
DBSCAN on axial normals
```

两个面方向距离：

\[
\theta_{ij}=acos(|n_i\cdot n_j|)
\]

建议起始：

```text
angular_eps = 8°～12°
```

---

# 43. 为什么采用 DBSCAN

优点：

- 不要求预先知道 J1/J2/J3 数量；
- 可以把离群方向视为 noise；
- 算法成熟；
- 参数含义明确。

---

# 44. 节理组方向统计

每个 Joint Set 计算：

- plane count；
- mean normal；
- mean dip；
- mean dip direction；
- angular dispersion；
- 可选 Fisher concentration statistic。

同时保留每一个单面结果。

---

# 45. 节理组平均法向

所有法向统一同一半球后：

\[
\bar n=normalize(\sum_i w_i n_i)
\]

V1 默认：

```text
one-plane-one-vote
w_i = 1
```

避免一个特别大的面因为点数更多而完全控制节理组方向。

可以额外输出 area-weighted mean 作为辅助结果。

---

# 46. 间距计算：两套结果

系统区分：

```text
normal_spacing
scanline_spacing
```

---

# 47. Normal Spacing

适用于同组近似平行节理。

组平均法向为 `n̄`，选择 ROI 参考点 `x0`，每个节理面：

\[
n_i\cdot x+d_i=0
\]

定义虚拟法向测线：

\[
x(t)=x_0+t\bar n
\]

交点参数：

\[
t_i=-\frac{n_i\cdot x_0+d_i}{n_i\cdot\bar n}
\]

排序后：

\[
s_i=t_{i+1}-t_i
\]

---

# 48. 间距有效性门槛

如果单面法向与 set mean normal 偏差过大，例如 `>10°～15°`，不参与该组 spacing。

如果：

```text
joint_set_plane_count < 2
```

不计算。

如果只得到 1 个 spacing，只输出单值，不输出有统计含义的分布结论。

---

# 49. Scanline Spacing

为了与传统现场测线对比，系统支持用户指定 3D virtual scanline。

仅统计实际与有限节理面/迹线相交的节理。

这是 Benchmark 推荐方法，因为最容易与人工测量对应。

---

# 50. ROI 边缘截断问题

位于 ROI 边缘的节理可能：

- 面积被截短；
- persistence 偏小；
- spacing 样本不完整。

因此每个面输出：

```text
edge_censored: true / false
```

这些面可以用于倾向倾角，但 persistence 和 spacing 统计应谨慎处理。

---

# 51. 质量控制采用“硬门槛 + 软评分”

## 51.1 硬门槛

任何关键条件不满足即 rejected，例如：

```text
point_count < 100
or area < 0.25 m²
or minor_extent < 0.15 m
or inlier_ratio < 0.70
```

## 51.2 Soft Score

通过硬门槛后，再根据：

- normalized RMS；
- planarity；
- inlier ratio；
- point count；
- area；
- boundary completeness；
- normal consistency；

计算 `quality_score`，范围 `0～1`。

---

# 52. Quality Score 不是概率

在未进行统计标定前：

```text
quality_score = 0.9
```

不能解释为“90% 概率是真的节理”。它只是工程质量评分。

只有完成有标签 Benchmark 后才能考虑 probability calibration。

---

# 53. 推荐质量等级

```text
A: quality_score >= 0.85
B: 0.70 ～ 0.85
C: 0.55 ～ 0.70
Rejected: < 0.55 或违反硬门槛
```

具体阈值经 Benchmark 后确定。

---

# 54. 核心数据模型 PlaneInstance

```python
@dataclass
class PlaneInstance:
    plane_id: str
    point_indices: np.ndarray

    normal: np.ndarray
    plane_d: float
    centroid: np.ndarray

    dip_direction_deg: float
    dip_deg: float

    point_count: int

    observed_area_m2: float
    apparent_persistence_m: float
    major_extent_m: float
    minor_extent_m: float

    trace_length_m: float | None

    rms_m: float
    mae_m: float
    p95_residual_m: float
    inlier_ratio: float

    edge_censored: bool

    joint_set_id: str | None
    quality_score: float
    quality_grade: str
```

---

# 55. JointSet 数据模型

```python
@dataclass
class JointSet:
    set_id: str
    plane_ids: list[str]

    mean_normal: np.ndarray
    mean_dip_direction_deg: float
    mean_dip_deg: float
    angular_dispersion_deg: float

    normal_spacings_m: np.ndarray
    scanline_spacings_m: np.ndarray
```

---

# 56. 输出文件

```text
output/
├── run.json
├── density_report.json
├── planes.csv
├── joint_sets.csv
├── spacings.csv
├── traces.csv
├── rejected_planes.csv
├── segmented_planes.ply
├── run_config.yaml
└── logs/
```

---

# 57. planes.csv 字段

至少：

```text
plane_id
joint_set_id
point_count
nx
ny
nz
plane_d
centroid_x
centroid_y
centroid_z
dip_direction_deg
dip_deg
observed_area_m2
apparent_persistence_m
major_extent_m
minor_extent_m
trace_length_m
rms_m
mae_m
p95_residual_m
inlier_ratio
edge_censored
quality_score
quality_grade
```

---

# 58. joint_sets.csv 字段

```text
set_id
plane_count
mean_dip_direction_deg
mean_dip_deg
angular_dispersion_deg
mean_normal_spacing_m
median_normal_spacing_m
std_normal_spacing_m
p10_normal_spacing_m
p90_normal_spacing_m
mean_scanline_spacing_m
median_scanline_spacing_m
```

---

# 59. rejected_planes.csv

所有被过滤面也必须保留：

```text
candidate_id
reason
point_count
area
rms
inlier_ratio
```

便于调参和审计。

---

# 60. 生产配置文件：600 points/m²

```yaml
project:
  version: "1.0"

input:
  path: "cloud.laz"

coordinate:
  convention: "ENU"
  unit: "meter"
  require_crs: true

target:
  raw_density_points_m2: 600
  min_structure_scale_m: 0.50
  min_area_m2: 0.25
  min_minor_extent_m: 0.15
  planning_retention_ratio: 0.70

density_qc:
  knn: 20
  required_median_points_m2: 600
  warning_p10_points_m2: 400

outlier:
  method: "statistical"
  mean_k: 16
  multiplier: 2.2

downsample:
  enabled: false

normal:
  radius_m: 0.15
  max_nn: 64
  min_neighbors: 20

planarity:
  min_planarity: 0.55
  max_surface_variation: 0.08

region_growing:
  normal_angle_deg: 8.0
  refit_every_points: 48
  min_region_points: 100

merge:
  normal_angle_deg: 5.0
  gap_multiplier_of_spacing: 3.0

ransac:
  enabled: true
  ransac_n: 3
  probability: 0.999

plane_gate:
  min_effective_points: 100
  min_area_m2: 0.25
  min_minor_extent_m: 0.15
  min_inlier_ratio: 0.70

boundary:
  method: "concave_hull"
  fallback: "convex_hull"

joint_sets:
  method: "dbscan"
  angular_eps_deg: 10.0
  min_planes: 3

spacing:
  normal_spacing: true
  scanline_spacing: true
  max_set_deviation_deg: 12.0

trace:
  mode: "semi_automatic"

quality:
  output_rejected: true

reproducibility:
  random_seed: 20260909
```

所有 threshold 必须在 Benchmark 后固化，不得把初始值当作最终科学标准。

---

# 61. Pipeline 伪代码

```python
def run(config):
    audit = audit_las(config.input)
    verify_coordinate_system(audit)

    roi = crop_with_pdal(config.input, config.roi)
    points = load_roi(roi)

    local_points, origin = recenter(points)

    density = estimate_local_density(local_points)
    assert_density_or_warn(density, config.target)

    clean_points = remove_outliers(local_points)

    normals, eigenvalues = estimate_geometry(clean_points)

    planar_mask = select_planar_candidates(
        normals,
        eigenvalues,
    )

    regions = region_grow(
        clean_points,
        normals,
        planar_mask,
    )

    regions = merge_fragments(regions)

    planes = []

    for region in regions:
        ransac_inliers = robust_plane_inliers(region)
        plane = tls_refit(ransac_inliers)
        boundary = fit_boundary(ransac_inliers, plane)
        metrics = compute_metrics(plane, boundary)

        if pass_quality_gate(metrics):
            planes.append(...)
        else:
            save_rejected(...)

    joint_sets = cluster_joint_sets(planes)
    spacings = calculate_spacings(planes, joint_sets)

    export_results(planes, joint_sets, spacings)
```

---

# 62. 大型点云处理

如果整个危岩体有几千万甚至上亿点，禁止一次性全部转成 NumPy。

采用：

```text
PDAL crop
+
spatial tile
+
overlap
```

---

# 63. Tile

例如：

```text
20 m × 20 m
```

或：

```text
50 m × 50 m
```

根据点密度、内存和节理长度调整。

---

# 64. Tile Overlap

必须保留 overlap，至少大于：

```text
2 × normal_radius
```

正式项目建议从 `0.5～1.0 m` 起步，再根据节理尺寸调整。

---

# 65. 跨 Tile 节理合并

相邻 tile 中两个面满足：

```text
normal angle 小
plane distance 小
boundary near / overlap
merged RMS 合格
```

才合并。

所有跨块合并必须可追溯。

---

# 66. 并行化

并行单位：

```text
Tile
```

推荐 ProcessPool 或任务调度器“每 Tile 一个 worker”。

注意 Open3D / BLAS 自身线程数，避免多进程乘以多 native threads 导致 oversubscription。

---

# 67. 性能优化顺序

```text
ROI
↓
PDAL streaming
↓
Tile
↓
KDTree
↓
NumPy vectorization
↓
Process-level parallelism
↓
Profile
↓
最后才考虑 C++ / CUDA
```

---

# 68. Trace 模块为什么默认半自动

自动节理面识别和自动迹线识别是两个不同问题。

成熟的面识别依赖 plane geometry；迹线往往依赖裂缝阴影、曲率、颜色、局部形态和暴露面关系。

因此 V1：

```text
plane: automatic
trace: semi-automatic / human verified
```

后续有真实 trace 标注数据后再升级全自动。

---

# 69. Benchmark 体系

正式系统必须拥有三层验证。

## 69.1 Synthetic Test

人工生成已知：

```text
dip direction
dip
noise
density
occlusion
spacing
```

验证公式、法向、RANSAC、TLS、spacing。

## 69.2 CloudCompare Reference

真实 ROI 中使用：

- qFacets 自动提取；
- Fit Plane 人工选面；
- Compass 测迹线。

形成独立参考。

## 69.3 Field / Expert Reference

最终验收使用：

- 地质罗盘；
- 人工点云解释；
- 人工测线。

---

# 70. 产状误差

倾角：

\[
E_{dip}=|dip_{auto}-dip_{ref}|
\]

倾向属于 circular angle：

\[
E_{dir}=min(|\alpha-\beta|,360-|\alpha-\beta|)
\]

---

# 71. V1 项目验收目标

以下是项目目标，不是未经验证的算法保证：

```text
Dip Direction MAE <= 5°
Dip MAE <= 5°
```

高质量裸岩目标：

```text
<= 3°
```

---

# 72. 节理面实例检测验收

人工标注部分 ROI 的 joint plane instances，评价：

```text
Precision
Recall
F1
```

建议 V1 项目目标：

```text
Precision >= 0.80
Recall >= 0.75
```

高质量裸岩：

```text
F1 >= 0.85
```

---

# 73. Plane Matching

自动面和人工面匹配不能只比较中心。

至少综合：

```text
normal angle
plane distance
projected overlap
```

满足阈值后才视为 True Positive。

---

# 74. Persistence 验收

针对人工确认边界完整的面：

```text
manual apparent persistence
vs
automatic apparent persistence
```

统计 MAE 和 Median Relative Error。

---

# 75. Spacing 验收

优先使用人工测线和 scanline_spacing 一一对照。

不要只比较不同方法的 overall mean。

---

# 76. 参数标定原则

参数分三类。

## 76.1 物理目标参数

例如：

```text
min_structure_scale = 0.5 m
```

由项目目标确定。

## 76.2 数据尺度参数

例如：

```text
normal radius
RANSAC distance
region gap
```

根据 point spacing 和 noise 自动或半自动推导。

## 76.3 算法经验参数

例如：

```text
planarity threshold
normal angle threshold
DBSCAN angular eps
```

必须通过 Benchmark 标定。

---

# 77. 禁止参数硬编码

不得在算法内部散落固定阈值。

所有参数进入 YAML config，每次运行完整 config 复制到：

```text
output/run_config.yaml
```

---

# 78. 可复现性

每次运行保存：

```text
run_id
timestamp
software version
git commit
input file metadata
ROI
config
random seed
point counts
density stats
rejection stats
```

---

# 79. run.json 示例

```json
{
  "run_id": "20260909_roi01_v001",
  "input": "cloud.laz",
  "input_points": 52500000,
  "roi_points": 1845000,
  "clean_points": 1792000,
  "median_density": 635.2,
  "candidate_planar_points": 1215000,
  "raw_regions": 137,
  "accepted_planes": 48,
  "rejected_planes": 89,
  "joint_sets": 3
}
```

---

# 80. 日志

至少记录：

```text
INFO
WARNING
ERROR
QC
```

示例：

```text
[QC] CRS validated: EPSG:xxxx
[QC] Median surface density: 635 pts/m²
[QC] P10 density: 428 pts/m²
[INFO] Planar candidates: 1,215,422
[INFO] Region growing produced: 137 regions
[INFO] Accepted planes: 48
[WARNING] 11 planes are edge-censored
```

---

# 81. 单元测试

必须覆盖：

### Geometry

- plane fitting；
- normal orientation；
- dip；
- dip direction；
- vertical plane；
- horizontal plane。

### Spacing

人工生成平行平面，例如：

```text
z = 0
z = 1
z = 2
```

确认 spacing = 1，再测试任意倾斜平面。

### Boundary

矩形、凹多边形、缺口。

### Circular Angle

例如 `359°` 和 `1°` 的误差应为 `2°`，不能是 `358°`。

---

# 82. 合成测试

生成 3 个 joint sets，每组包含：

- 已知 dip；
- 已知 dip direction；
- 已知 spacing；
- 随机噪声；
- 600 points/m²；
- 遮挡；
- 离群点。

自动运行整个 pipeline。

---

# 83. 真实 ROI 冒烟测试

如果暂时没有现场基准，系统结果只能标记为：

```text
Algorithm Functional Test
```

或：

```text
Real ROI Smoke Test
```

不能写“现场精度达到 X°”。

---

# 84. 技术验证基准：CloudCompare qFacets

qFacets 可以自动：

- 提取 planar facets；
- 输出法向；
- 输出 RMS；
- 输出 surface；
- 输出 dip；
- 输出 dip direction；
- 按 orientation / orthogonal distance 分类；
- stereogram。

因此适合作为自动流程的独立工程 reference，而不是绝对真值。

---

# 85. 技术验证基准：CloudCompare Compass

Compass 的 Plane Tool 适合人工选择完整暴露平面并计算 dip / dip direction。

Trace Tool 适合人工/半自动节理迹线。

因此 orientation benchmark 和 trace benchmark 都可以基于 CloudCompare 建立。

---

# 86. 主要风险与解决策略

## Risk 1：岩壁主面误识别

岩壁自身可能平整。

措施：结构面空间尺度、节理组关系、边界、地质 ROI、人工验证。

几何系统无法单独证明“这个平面一定是节理”。

## Risk 2：相邻近平行节理被合并

措施：region growing 必须使用空间连续条件；merge 限制 plane offset；不能只按 normal。

## Risk 3：同一粗糙节理被切成多个面

措施：noise-aware normal threshold、fragment merge、合并后 RMS 检验。

## Risk 4：粗糙度造成法向不稳定

措施：Hybrid Neighborhood；根据 density 选半径；V2 可增加 multi-scale PCA。

## Risk 5：点密度不均

措施：local density map、density QC、scale-aware parameters。

## Risk 6：植被

V1 无法完全解决。可通过 LAS classification、RGB、几何 outlier 减少影响。完整语义植被剔除属于后续模块。

## Risk 7：边界截断

通过 `edge_censored` 明确输出。

## Risk 8：迹长被错误定义

`apparent_persistence` 与 `trace_length` 完全分开。

---

# 87. V1 开发里程碑

## M0 数据审计

```text
LAS info + CRS + ROI + density
```

## M1 局部几何

```text
normal + PCA + planarity + surface variation
```

## M2 节理实例

```text
region growing + fragment merge + RANSAC + TLS
```

## M3 产状

```text
dip + dip direction + RMS + quality
```

## M4 边界

```text
polygon + area + major/minor extent + apparent persistence
```

## M5 节理组

```text
J1/J2/J3 + mean orientation + angular dispersion
```

## M6 间距

```text
normal spacing + scanline spacing
```

## M7 迹线

```text
semi-automatic trace tool + trace length
```

## M8 Benchmark

```text
synthetic + CloudCompare + manual ROI
```

## M9 大数据

```text
tile + overlap + cross-tile merge + parallel execution
```

---

# 88. V1 成功定义

V1 成功必须达到：

```text
输入 LAS/LAZ
↓
自动质量检查
↓
自动识别节理面实例
↓
每个面有倾向倾角
↓
有明确面积和表观尺度
↓
能形成 J1/J2/J3
↓
能计算合理间距
↓
有质量指标
↓
结果可由 CloudCompare 独立验证
↓
同一配置可重复运行
```

而不是只实现 `Open3D segment_plane()`。

---

# 89. 后续 V2

在 V1 几何链路稳定后，再考虑：

```text
multi-scale normals
robust IRLS / IRPF normals
automatic optimal neighborhood
advanced region growing
automatic trace candidate extraction
mesh-assisted trace detection
vegetation classification
```

---

# 90. 后续 V3

有足够人工标注后才考虑：

```text
Point Transformer
SparseConv
KPConv
PointNet++
```

用于植被/岩石、节理/非节理、复杂结构面语义。

深度学习不得代替几何产状计算，最终倾向倾角仍由明确的平面几何拟合得到。

---

# 91. 推荐部署

开发环境：

```text
Ubuntu 22.04 / 24.04
Python 3.11
```

PDAL 使用系统稳定版本安装；Python 算法层使用项目根目录的系统 `python3`。

Python 算法层使用项目根目录的系统 `python3`，依赖按根 `pyproject.toml` 统一管理。

---

# 92. PDAL 建议作为 CLI 服务使用

为避免 Python PDAL binding 的二进制依赖问题，可以：

```text
Python
↓ subprocess
pdal pipeline pipeline.json
↓
roi.laz
```

算法层再由 laspy / Open3D 读取 ROI。

这是一种部署上较稳健的工程方式。

---

# 93. GPU

V1 不要求 GPU，优先 CPU。

只有 Profile 证明 normal / neighbor search 成为真正瓶颈，再考虑 Open3D Tensor CUDA。

---

# 94. 工程技术路线最终确定

```text
                 LAS / LAZ
                    │
                    ▼
        PDAL Data Audit / Crop / Noise
                    │
                    ▼
         Local Surface Density QC
                    │
                    ▼
        Open3D Hybrid Neighborhood
                    │
                    ▼
           PCA Normal + Planarity
                    │
                    ▼
        Planarity Seeded Region Growing
                    │
                    ▼
         Finite Planar Patch Instances
                    │
                    ▼
             RANSAC Outlier Clean
                    │
                    ▼
               TLS Plane Fit
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Dip / Dip Direction    2D Plane Boundary
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
                 Area             Apparent Persistence
          │
          ▼
  DBSCAN Angular Clustering
          │
          ▼
       J1 / J2 / J3
          │
          ▼
  Virtual Normal / Scanline
          │
          ▼
        Spacing
          │
          ▼
  Quality Gate + Benchmark
          │
          ▼
 CSV / JSON / PLY / Report
```

迹线作为独立分支：

```text
Rock Surface
+
joint candidate
↓
Graph / cost field
↓
start/end
↓
Dijkstra / A*
↓
3D Trace Polyline
↓
Trace Length
```

V1 采用半自动确认。

---

# 95. 最终边界声明

正式成果文件应明确写：

> 本系统识别结果代表在当前 LAS/LAZ 数据分辨率、可见性和 ROI 范围内能够通过平面几何特征识别的岩体不连续面。系统计算的倾向和倾角为可见平面的几何产状估计；面积和表观延伸长度仅代表当前点云中的可观测暴露范围；节理迹长仅在交迹能够被有效追踪时输出；间距为当前识别节理组和测量方向下的观测统计量。系统不对未被点云观测到的地下节理、隐藏延伸或地质成因做确定性推断。

---

# 96. 工程结论

对于当前目标：

```text
600 points/m²
+
0.5 m 最小节理尺度
+
LAS / LAZ
```

V1 推荐采用：

```text
PDAL
+
Open3D
+
NumPy / SciPy
+
Planarity Seeded Region Growing
+
RANSAC
+
TLS
+
DBSCAN
+
Shapely
+
CloudCompare Benchmark
```

这条技术路线具备：成熟、无需训练数据、参数可解释、结果可复现、工程易维护、可逐步扩展到大规模点云等特点。

不建议第一版引入自研 C++、深度学习、复杂神经网络和完全自动迹线识别。先把节理面几何识别、产状和间距做准，再扩展高级功能。

---

# 97. 技术依据与参考

1. **CloudCompare qFacets**  
   自动提取地质平面、计算 Normal、RMS、Surface、Dip、Dip Direction，并支持按 orientation / orthogonal distance 分类及 stereogram。  
   https://www.cloudcompare.org/doc/wiki/index.php/Facets_%28plugin%29

2. **Dewez, T.J.B. et al. (2016)**  
   FACETS: A CloudCompare plugin to extract geological planes from unstructured 3D point clouds. ISPRS Archives XLI-B5, 799–804.

3. **CloudCompare Compass**  
   面向 virtual outcrop 的结构地质量测工具，支持 Plane Tool 和半自动 Trace Tool。  
   https://www.cloudcompare.org/doc/wiki/index.php/Compass_%28plugin%29

4. **Thiele, S.T. et al. (2017)**  
   Rapid, semi-automatic fracture and contact mapping for point clouds, images and geophysical data. Solid Earth, 8, 1241–1253.

5. **Open3D Point Cloud Documentation**  
   法向量估计、Hybrid neighborhood、RANSAC plane segmentation。  
   https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html

6. **PDAL filters.normal**  
   基于邻域协方差特征值/特征向量计算 Normal 与 Curvature。  
   https://pdal.io/en/latest/stages/filters.normal.html

7. **PDAL filters.outlier**  
   Statistical / Radius Outlier Filtering。  
   https://pdal.io/en/latest/stages/filters.outlier.html

8. **PDAL filters.crop**  
   支持 bbox / polygon / 3D crop，并支持流式处理。  
   https://pdal.io/en/latest/stages/filters.crop.html

9. **PDAL filters.voxeldownsize**  
   Voxel-based point cloud sampling。  
   https://pdal.io/en/stable/stages/filters.voxeldownsize.html

10. **Automatic identification and characterization of discontinuities in rock masses from 3D point clouds**  
    Engineering Geology。方法链包含 normal calculation、discontinuity-set clustering、segmentation、RANSAC fitting、persistence 与 spacing。

11. **A state-of-the-art review of automated extraction of rock mass discontinuity characteristics using three-dimensional surface models**  
    Journal of Rock Mechanics and Geotechnical Engineering, 2021。综述 joint sets、orientation、persistence、spacing、roughness、block size 等自动化提取技术，并讨论 region-growing 类方法的工程表现。

---

# 98. 下一步实施顺序

建议严格按以下顺序开始编码：

```text
Day 1:
LAS audit + CRS + density report

Day 2:
ROI + outlier + local coordinates

Day 3:
normal + PCA + planarity visualization

Day 4:
region growing v1

Day 5:
RANSAC + TLS + dip/dip direction

Day 6:
boundary + area + persistence

Day 7:
joint set clustering

Day 8:
spacing

Day 9:
CloudCompare benchmark

Day 10:
参数标定 + 第一版报告
```

不要在节理面实例分割尚未验证正确之前开始迹线、深度学习和危岩稳定性分析，因为后续所有参数都依赖前面的几何实例是否正确。
