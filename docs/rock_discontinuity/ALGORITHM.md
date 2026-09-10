# V2 岩体节理面识别算法

本实现以 `docs/岩体节理面自动识别_文献驱动技术设计_V2.md` 为技术基线，目标是识别点云中可观测的节理面候选及其几何参数。输出中的“候选节理面”不等于解体面、失稳块体或地下真实延伸。

## 主链

1. 先做数据审计：记录点数、点格式、CRS、XYZ 范围、scale/offset、维度、最近点距和局部密度。
2. 整体点云按空间块处理。COPC 使用空间索引；普通 LAS/LAZ 使用 PDAL 的 XY `tile` 流式切块，并把 Z 保留在每个局部块内。当前普通 LAS 的报告明确写 `dimension_mode=2d_xy_fallback`，不把 XY 切块伪称为三维体素切块。
3. 每个局部块独立建立 KDTree，执行体素/离群点预处理、PCA 法向和局部特征计算；不构建全局原始点 KDTree。
4. 由 planarity、surface variation/curvature 和邻居数筛选候选点。默认阈值是 planarity≥0.55、surface variation≤0.08、局部邻居数≥20。
5. 以“平面质量=planarity−curvature”排序种子，执行动态区域生长。接受条件同时检查邻接、法向夹角、点到当前 TLS 平面的距离和曲率；默认角度 8°→12°、距离 0.025→0.05 m，并每 48 个点重拟合一次。
6. 每个区域先 RANSAC 抗离群粗拟合，再 TLS 精拟合，计算残差、inlier ratio、法向离散度和有限平面边界。面积使用投影 UV 平面的 alpha shape，缺陷时退回 convex hull；面积、最大 Feret/表观延伸和中心均是观测量。
7. 按 ENU 约定输出 `dip_direction_deg`、`dip_deg`、法向、平面偏移和中心坐标。平面若贴近 ROI/块边界会标记 `edge_censored`；缺少遮挡/数据间隙证据时不伪造 `data_gap_censored`。
8. 在 ROI 或单个瓦片内，先对局部拟合的 `PlaneInstance` 做几何合并。只有法向夹角、平面偏移、XY 投影边界间隙和 RMS 同时通过，且合并点集重新拟合仍通过，才合成一个最终平面；原始片段保存在 `plane_instances.csv` 供追溯。
9. 再在最终 `PlaneInstance` 层按轴向法向聚类为节理组，输出 Fisher concentration `fisher_k`。节理组的最小 3 个平面是统计质量门槛；少于 3 个平面仍保留成对间距作为诊断，但标记 `fewer_than_min_planes_for_set_statistics`。
10. 相邻瓦片完成局部合并后，再在 `PlaneInstance` 层做跨瓦片合并；最终生成全局 `GJ-xxxxx`。不重新建立全局原始点 KDTree。
11. 同时计算两种间距：
   - `spacing_3d_nonpersistent`：有限观测面之间的局部最近点法向投影间距；
   - `spacing_virtual_scanline`：沿平均法向的虚拟扫描线间距。
   两者都不能解释为无限延伸、永久连续的地下节理间距。
实现使用 SciPy `KDTree`/稀疏邻接图完成局部邻域，不强制依赖 Open3D；Open3D 在 V2 中属于可替换的工程实现选项，不是识别方法本身。

## 迹线与张开度边界

`apparent_persistence_m`/`major_extent_m` 是点云观测平面的延伸，不能替代结构面与自由面的真实迹线。V2 的迹线需要高分辨率 Mesh/点云和 Normal Tensor Voting；张开度需要最高分辨率 Mesh/影像，且有效分辨率至少满足约 3 倍采样要求。当前 LAS 主链没有这些输入，因此：

- `trace_length_m` 保持 `null`；`traces.csv` 和 `trace_lines.geojson` 写出不可用原因；
- `aperture.csv` 写出 `available=false` 和 `high_resolution_mesh_or_image_required`；
- 不用表观延伸、法向突变或颜色标签冒充迹线/张开度。

## 密度与密度指标

ROI 密度用局部切平面 kNN 估计 `rho=k/(pi*r_k^2)`，默认 k=20，报告 P10、P25、Median、Mean、P75、P90。`standard_status=DENSITY_TARGET_NOT_MET` 表示没有达到配置的 600 点/m² 带，但程序仍保留结果供功能排查。

P10、P20、P21 作为不同指标输出：P10 需要明确扫描线，P20 是观测平面数/观测足迹面积，P21 是已验证迹线总长/面积。缺少扫描线或迹线时输出 null 和原因，不互相替代。

## 输出语义

- `planes.csv`：几何合并后的最终 ROI `PlaneInstance`，含产状、中心、面积、残差、质量和截断标志。
- `plane_instances.csv`：合并前的局部平面片段，用于解释多个片段被合并的来源。
- `joint_sets.csv`：法向节理组、Fisher K 和两类间距统计。
- `spacings.csv`：逐平面对的两类间距及采样数。
- `plane_boundaries.geojson`：有限观测边界，不是地质边界。
- `segmented_planes.ply`：每个平面彩色标签；`detachment_candidates.ply` 的红色只表示几何候选节理面。
- `data_audit.json`、`density_report.json`、`run.json`：输入审计、密度门限和可复现运行记录。

当前不使用 CloudCompare 人工标注、现场罗盘或人工测线作为验收输入；真实现场精度也不会由几何质量分数自动宣称。
