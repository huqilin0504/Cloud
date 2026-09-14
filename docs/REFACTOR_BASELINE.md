# 重构基线与验证记录

这次工作区的 `.git` 没有历史提交，因此不能把现有代码称作最初重构前版本，也无法提供真实
的“重构前后”源码差异。原始资料和历史 `outputs/` 未修改。当前保存的可重复对照产物位于
`outputs/refactor_validation/20260910_system_closeout-20260910/`，包括固定种子 42 的合成输入校验清单、ROI
结果、四瓦片全点云并行结果、串行结果、恢复运行结果和 `source_snapshot.tar.gz` 源码快照。
此前的 `20260910_system_final/` 保留为上一阶段的独立批次，没有被覆盖。
本轮局部解耦收尾以当前工作区和上述合成结果为对照，新增验证产物放在
`outputs/refactor_validation/20260910_local_finish/`；`pointcloud_joint_extraction/` 是未跟踪的独立目录，
本轮未读取、修改或纳管。

## 当前环境

项目全程使用系统 `python3`，没有创建或调用虚拟环境。安装顺序为：

```bash
python3 -m pip install --user 'setuptools>=68,<80' 'wheel>=0.40'
python3 -m pip install --user --no-build-isolation -c constraints.txt -e '.[osgb,dev]'
python3 -m pip check
```

2026-09-10 的环境为 Python 3.10.12，主要版本为 `numpy 2.2.6`、`scipy 1.15.3`、
`scikit-learn 1.7.2`、`shapely 2.1.2`、`laspy 2.7.0`、`lazrs 0.8.2`、`PyYAML 6.0.3`、
`pyproj 3.7.1`、`pytest 9.1.1` 和 `pluggy 1.6.0`。`pip check` 返回
`No broken requirements found`。系统 requests 会输出一个与仓库无关的发行版警告，但不影响
项目依赖检查和测试结果。

## 已完成的结构变化

- `ProcessingResult`、`TileResult` 及特征、计数、间距和输出记录类型位于 `core.models`；
  `run_points()` 保留字典兼容包装，内部 ROI/瓦片流程使用属性接口。
- `core.geometry` 集中法向对齐、夹角矩阵和距离等纯几何规则；ROI 和跨瓦片策略仍保留
  各自的筛选、权重、排序及缺失记录回退。
- LAS/COPC 瓦片原语位于 `io/las_tiles.py`，全点云写出位于 `io/whole_cloud_output.py`；
  `io.output` 只接收处理层已经准备好的记录，不再因缺少记录而回算。
- `processing/tiling.py`、`state.py`、`tile_processing.py`、`global_aggregation.py` 分别
  负责切块、恢复状态、单瓦片识别和全局聚合；`app/whole_cloud.py` 负责命令解析，
  `processing/whole_cloud.py` 只编排阶段并保留历史兼容函数。
- `processing/output_records.py` 在处理层准备边界、迹线、张开度和密度记录；
  `io/output.py` 与 `io/whole_cloud_output.py` 只序列化已经准备好的记录。
- `processing/state.py` 统一状态版本、分块计划兼容检查、原子保存和瓦片完成/失败标记；
  全局候选质量门槛位于 `processing/global_aggregation.py`，避免主流程继续持有这些规则。
- `GlobalAggregationResult` 统一承载跨瓦片合并、方向组、候选复判、间距和编号映射；
  `processing/whole_cloud_reporting.py` 只准备报告与覆盖层筛选映射，不执行写盘。
- ROI 比较的命令解析位于 `app/roi_compare.py`，汇总读写位于 `io/roi_compare_output.py`；
  `processing/roi_compare.py` 保留实验执行、统计和历史兼容包装。
- YAML 是默认配置唯一来源；`footprint_max_points` 的默认值、缺失字段回退和显式覆盖均由
  回归测试固定。根入口和 `rock-discontinuity` 默认全点云，包入口保留 ROI 默认行为。

## 可重复验证命令

```bash
python3 -m pytest -q
python3 osgb_pipeline/test_pipeline.py
python3 osgb_pipeline/test_stream_las.py
python3 -m compileall -q rock_discontinuity osgb_pipeline
python3 main.py --help
python3 -m rock_discontinuity --help
```

当前结果：本轮识别测试 `46 passed`；两个转换回归脚本均输出 `PASS`；编译和四个帮助入口退出码
均为 0。架构契约测试同时解析绝对和相对导入，确认 `core` 不依赖 `processing/io/app`，
`io` 不依赖 `processing/app`。

OSGB 的现有 `osgb_pipeline/build` 是历史缓存。使用全新临时目录并显式指定本机 OSG 库时，
CMake 配置和三个目标成功：

```bash
cmake -S osgb_pipeline -B /tmp/joint-face-cmake \
  -DOSG_INCLUDE_DIR="$PWD/osgb_pipeline/deps/root/usr/include" \
  -DOSGDB_INCLUDE_DIR="$PWD/osgb_pipeline/deps/root/usr/include" \
  -DOPENTHREADS_INCLUDE_DIR="$PWD/osgb_pipeline/deps/root/usr/include" \
  -DOSG_LIBRARY_RELEASE=/usr/lib/x86_64-linux-gnu/libosg.so.161 \
  -DOSGDB_LIBRARY_RELEASE=/usr/lib/x86_64-linux-gnu/libosgDB.so.161 \
  -DOPENTHREADS_LIBRARY_RELEASE=/usr/lib/x86_64-linux-gnu/libOpenThreads.so.21
cmake --build /tmp/joint-face-cmake -j2
```

## 合成数据和运行对照

固定种子 42 的 `synthetic_tilted_planes.las`（1050 点）和 `synthetic_config.yaml` 用于对照；输入
SHA256、配置 SHA256、依赖版本和比较排除项记录在该批次的 `manifest.json`：

- ROI 运行识别 2 个平面、2 个候选实例，生成 19 个结果文件。
- 全点云四瓦片并行运行（2 workers）处理 4/4 瓦片，失败 0，得到 2 个瓦片实例、1 个全局
  平面组和 1 个全局节理组。
- 同一输入、配置和瓦片参数的串行运行（1 worker）在 detachment/global/joint/spacings/
  tile-spacings 表中行数和字段值一致；报告核心计数完全一致。对已有输出执行 `--resume`
  后仍为 4/4 处理、失败 0。
- 使用当前整幅 `outputs/root_full_xyz_10/cloud.las` 的固定 20 m ROI
  (`525000,525020,3132560,3132580`) 完成了一次独立运行，输入 SHA256、计数和表格行数记录在
  closeout `manifest.json` 的 `real_roi` 中；历史多尺度输出引用的旧源瓦片路径已失效，因此不作
  未经核对的真实数据前后严格比较。
- 边界点半开区间归属、空/损坏瓦片隔离、状态原子保存、状态版本/计划兼容、完成/失败标记、
  失败后恢复、旧状态读取、并行调度、全局候选门槛、错层阈值、缺失足迹、大坐标和
  `footprint_max_points` 已由单元和契约测试覆盖；原始状态 JSON 格式保持可读。
- ROI PLY 已解码比较坐标、平面/组标签和颜色；并行与串行全点云 LAZ 已解码比较坐标和
  `point_source_id`，结果记录在 closeout `manifest.json`。

这些合成结果证明重构后的模块边界和调度语义可运行，不证明地质识别精度。由于缺少可核对的
最初重构前源码和固定真实 ROI 输出，真实数据的严格前后字段比较仍需在具备该快照后执行。

## 安装包与入口

已用现代 setuptools 构建 editable 包并生成 `rock-discontinuity` 入口。需验证发布资源时，
可在源码目录外构建 wheel 并检查 YAML 包资源：

```bash
rm -rf /tmp/joint-face-wheel /tmp/joint-face-wheel-target
mkdir -p /tmp/joint-face-wheel /tmp/joint-face-wheel-target
python3 -m pip wheel . --no-deps --no-build-isolation -w /tmp/joint-face-wheel
python3 -m pip install --target /tmp/joint-face-wheel-target --no-deps /tmp/joint-face-wheel/*.whl
cd /tmp
PYTHONPATH=/tmp/joint-face-wheel-target:/home/huqilin/.local/lib/python3.10/site-packages \
python3 -c 'from rock_discontinuity.config import load_config; print(load_config(None)["whole_cloud"]["workers"])'
```

上述 wheel 已在源码目录外通过 `rock-discontinuity --help`、根入口、包入口 ROI 小样本和
whole-cloud 小样本运行；四条路由均成功，YAML `default.yaml` 可从安装包资源读取。

## Git 状态

当前 Git 身份为 `huqilin <97706693@qq.com>`。本轮提交只包含识别源码、测试和文档；
未跟踪的 `pointcloud_joint_extraction/` 与 `outputs/` 均未纳管。提交记录以 `git log` 为准，
不推送远端。
