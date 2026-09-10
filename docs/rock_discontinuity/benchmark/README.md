# 阶段 1 基准边界

本目录记录节理面识别 MVP 的可复现基准边界。阶段 1 只证明算法链路和输出结构正确，不证明现场地质真值。

## 合成数据

合成数据测试位于 `../../../rock_discontinuity/tests/`，覆盖：

- ENU 坐标下的倾向、倾角计算；
- TLS 平面拟合及残差；
- 局部 UV 投影面积和表观延伸；
- 两块平行合成节理面的方向分组和虚拟测线间距。

运行：

```bash
PYTHONPATH=. python3 -m unittest discover -s rock_discontinuity/tests -v
```

## 真实 ROI

当前真实数据验证使用 `config/roi_example.yaml` 中的 40 m × 40 m ROI，输入源按 600 点/m² 标准生成。该数据用于 0.5 m 级目标的工程链路验证，不作为外部地质真值样本；报告中的 `roi_confirmation` 标记为 `phase1_not_required`。

运行并校验：

```bash
PYTHONPATH=. python3 -m rock_discontinuity \
  --config rock_discontinuity/config/roi_example.yaml \
  --force

PYTHONPATH=. python3 -m rock_discontinuity.validate \
  outputs/discontinuity_roi_600
```

## 不在阶段 1 的结论

- 不把 `trace_length_m = null` 替换成平面长轴；没有稳定暴露面迹线就保持缺失。
- 不把几何 `confidence` 当成统计概率或地质置信度。
- 不从点云表面外推地下延伸、结构面成因、JRC、RQD 或块体稳定性。
- 不把阶段 1 的单元测试和 ROI 输出当作现场精度或外部真值结果。
- 不使用 CloudCompare 人工标注、现场罗盘或人工测线。

## 后续验收门槛

1. 正式 0.5 m 级目标使用至少 100 点/m² 的数据重新标定邻域半径、体素、RANSAC 距离、最小点数和最小面积。
2. 对合成数据保留已知平面、产状和间距真值，作为可重复的定量回归基准。
3. 对真实 ROI 保留输入摘要、参数快照、输出文件和重复运行结果；不输出现场精度结论。
