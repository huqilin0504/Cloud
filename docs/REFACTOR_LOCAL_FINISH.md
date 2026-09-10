# 局部解耦收尾验证报告

## 范围

本轮只处理 `rock_discontinuity` 的全局结果接口、全点云报告准备、ROI 比较入口和静态导入检查。
`pointcloud_joint_extraction/` 是工作区中独立的未跟踪目录，本轮未读取、修改或纳管；`outputs/`
和原始资料也未修改。

使用系统 `python3`，没有创建或调用虚拟环境。

## 代码结果

- `core.models` 增加 `GlobalAggregationResult`、全局平面/间距/方向组/筛选记录类型，以及
  JSON 恢复边界转换函数。
- `processing.global_aggregation.aggregate_global_results()` 统一执行跨瓦片合并、方向分组、
  全局候选复判、间距和编号映射，保留原有低层函数和返回形状。
- `processing.whole_cloud_reporting` 准备报告、计数和候选覆盖层筛选映射；主流程保留切块、
  状态、线程池、进度和失败传播。
- `app.roi_compare` 承担命令解析，`io.roi_compare_output` 承担汇总读写；旧的
  `processing.roi_compare` 命令通过函数内兼容包装转发。
- 导入契约测试现在能解析包 `__init__`、绝对/相对导入、`from . import 子模块` 和别名；
  函数内延迟导入不计入模块加载循环，并有独立测试覆盖。

## 验证结果

执行过的命令：

```text
python3 -m unittest discover -s rock_discontinuity/tests -q  -> 46 passed
python3 -m compileall -q rock_discontinuity osgb_pipeline  -> 通过
python3 osgb_pipeline/test_pipeline.py                  -> PASS
python3 osgb_pipeline/test_stream_las.py                -> PASS
python3 -m pip check                                     -> No broken requirements found
python3 main.py --help                                   -> 0
python3 main.py roi-compare --help                       -> 0
python3 -m rock_discontinuity.processing.roi_compare --help -> 0
```

固定合成输入 `synthetic_tilted_planes.las`、同一配置和 `tile_size=4、overlap=0.5` 的前后对照位于：

- 旧批次：`outputs/refactor_validation/20260910_system/whole_cloud/`
- 本轮：`outputs/refactor_validation/20260910_local_finish/whole_tilted/`

`detachment_planes.csv`、`joint_planes.csv`、`joint_sets.csv`、`spacings.csv` 和
`tile_spacings.csv` 的行数及字段值完全一致；解码后的候选 LAZ 坐标、RGB 颜色和
`point_source_id` 完全一致。报告中仅输出路径不同。

另外，使用系统 Python 从 wheel 在源码目录外安装后，`default.yaml` 包资源读取到
`whole_cloud.footprint_max_points=2000`，安装命令 `rock-discontinuity --help`、ROI 小样本和
全点云小样本均成功。

## 提交与限制

本轮本地提交：

- `f49b91f refactor: type global aggregation stage`
- `4a1bce3 refactor: separate ROI comparison entrypoint`

没有推送远端。当前 Git 工作区只保留用户未跟踪的 `pointcloud_joint_extraction/`，没有把它加入
暂存或提交。

严格的“最初重构前源码”快照仍不存在，因此本报告证明的是当前已保存批次与本轮重构的行为一致，
不等同于最初版本的完整前后对照，也不证明地质识别精度。
