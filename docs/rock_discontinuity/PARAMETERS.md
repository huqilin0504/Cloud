# 参数说明

当前配置见 `config/default.yaml`。

本项目的识别目标是可观测节理面候选；下文的“结构面尺度”和“结构面暴露”均按
节理面候选的几何识别来理解，不涉及解体或稳定性判断。

当前默认设置对应约 600 点/m²、约 4.1 cm 平均点距和 0.5 m 级目标：

- `voxel_size=0.03 m`
- `normal.radius=0.15 m`
- `normal.min_neighbors=20`
- `normal.knn=64`、`normal.max_neighbors=64`
- `density.knn=20`，局部切平面 kNN；输出 P10/P25/Median/Mean/P75/P90
- `region_growing.radius_m=0.15 m`
- `region_growing.normal_angle_base_deg=8°`、`normal_angle_max_deg=12°`
- `region_growing.distance_base_m=0.025 m`、`distance_max_m=0.05 m`
- `region_growing.max_curvature=0.08`
- `region_growing.refit_every_points=48`
- `plane_gate.min_effective_points=100`
- `plane_gate.min_minor_extent_m=0.15 m`
- `plane.min_area=0.25 m²`
- `plane.ransac_distance=0.03 m`
- `quality.min_inlier_ratio=0.70`
- `plane_merge.normal_angle_deg=5°`
- `plane_merge.plane_offset_m=0.08 m`
- `plane_merge.spatial_gap_m=0.30 m`
- `plane_merge.max_rms_m=0.05 m`
- `whole_cloud.merge_normal_angle_deg=5°`
- `whole_cloud.merge_plane_offset_m=0.08 m`
- `whole_cloud.merge_xy_gap_m=0.30 m`（实际至少覆盖 2 倍 overlap）
- `whole_cloud.workers=2`（瓦片识别线程数；内存允许时可用 `--workers N` 覆盖）
- `tiling.tile_size_m=25 m`、`tiling.overlap_m=1 m`、`target_points_per_tile=1200000`
- `joint_sets.min_planes=3`、`joint_sets.fisher_statistics=true`
- `spacing.method=finite_3d_and_virtual_scanline`
- `reproducibility.random_seed=20260909`

600 点/m² 只是输入密度标准，不等于所有场景都能识别所有 0.5 m 节理面；仍需满足节理面暴露充分、噪声可控和局部密度合格。密度失败时报告 `DENSITY_TARGET_NOT_MET`，不静默降级。参数只通过合成数据和 ROI 重复运行调整，不使用人工标注或现场测量作为标定输入。`spatial_cluster` 保留为旧分割方法的兼容参数；默认主路径是 planarity-seeded region growing。

迹线和张开度没有在 LAS 主链中伪估计：`trace` 需要高分辨率 Mesh/点云，`aperture` 需要最高分辨率 Mesh/影像且有效分辨率达到约 3 倍要求，当前对应输出为不可用状态。
