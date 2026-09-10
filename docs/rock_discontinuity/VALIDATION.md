# 验收记录

## 第一阶段：合成数据

`rock_discontinuity/tests/test_geometry.py` 和 `test_synthetic.py` 覆盖：

- ENU 产状公式。
- TLS 平面拟合。
- UV 投影面积和表观延伸。
- 两块平行合成节理面的方向分组和间距。

运行：

```bash
PYTHONPATH=. python3 -m unittest discover -s rock_discontinuity/tests -v
```

## 真实 ROI 功能验证

当前示例使用一个 40 m × 40 m 的临时 ROI，输入 LAS 按 600 点/m² 标准生成：

```text
x = [524898, 524938]
y = [3132085, 3132125]
```

它只用于验证读取、预处理、分割、参数计算、密度统计、输出和报告链路，不作为外部地质真值样本。

报告中的密度统计使用局部切平面 kNN，并给出 P10、P25、Median、Mean、P75、P90；`standard_status=pass` 只表示 ROI 满足本程序的 600 点/m² 密度带，不表示结构面识别精度已经被外部真值证明。若不满足，状态为 `DENSITY_TARGET_NOT_MET`。

输出还必须包含 `data_audit.json`、`density_report.json`、`run.json`、`plane_boundaries.geojson`、`traces.csv` 和 `aperture.csv`。当前没有高分辨率 Mesh/影像，迹线与张开度应为明确的 unavailable/null，而不是填入表观延伸。

运行：

```bash
PYTHONPATH=. python3 -m rock_discontinuity \
  --input outputs/root_full_xyz_10/cloud.las \
  --bbox 524898,524938,3132085,3132125 \
  --output outputs/discontinuity_roi_600

PYTHONPATH=. python3 -m rock_discontinuity.validate \
  outputs/discontinuity_roi_600
```

## 当前版本边界

- 不使用 CloudCompare 人工标注节理面。
- 不使用现场罗盘或人工测线作为验收基准。
- 真实 ROI 只验证工程链路和输出一致性；真实场景的现场精度不在当前版本结论内。
- 正式 0.5 m 级目标按本项目配置要求局部原始密度约 600 点/m²；`min_effective_points=100` 是平面质量门槛，不是输入点密度。
- `spacings.csv` 同时验收 `spacing_3d_nonpersistent` 和 `spacing_virtual_scanline` 两种方法，不能把无限平面间距作为唯一结果。
- `detachment_candidates.ply` 只验证候选节理面染色和输出完整性；红色点不等于解体面或失稳块体。
