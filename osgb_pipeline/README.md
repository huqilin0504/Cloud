# OSGB 根节点递归导出 OBJ + LAS

输入为 `Block.osgb`，通过 OSG API 加载真实 PagedLOD 外部引用。遇到外部细化节点跳过当前粗模；沿父子路径累计 Transform；对同一套变换后的三角形输出 OBJ 和表面采样点。

当前交付为**几何 OBJ（无 MTL/纹理）和 XYZ LAS（无 RGB）**。保留原始 OSGB，后续可扩展 UV/纹理采样。文档中的 OBJ 并非点云中间环节，此工程从几何同时导出两种成果。

结构面识别的 V1 参数标准保存在 `structure_params_600.json`：600 points/m²、0.5 m 最小尺度、0.25 m² 最小面积、100 个有效点、0.70 有效点保留率。

## 结构

```text
osgb_pipeline/
  CMakeLists.txt
  src/extract_osgb.cpp       # 递归、变换、三角化、OBJ、二进制 XYZ
  src/make_fixture.cpp       # 人工父子变换及粗细 LOD 测试数据
  run.py                    # metadata、LAS 分批写入、逐点复读校验
  replace_cloud_las.py      # 仅流式写高密度 LAS，成功后原子替换 cloud.las
  structure_params_600.json # 结构面识别 V1 参数标准
  test_pipeline.py           # 变换与 LOD 回归验证
  test_stream_las.py         # 直接 LAS 流式写出回归验证
  build/                    # 本地编译产物（不纳入版本库）
  deps/                     # 本地 Ubuntu OSG 开发包（不纳入版本库）
```

在项目 `/home/huqilin/项目/节理面` 下运行：

```bash
python3 osgb_pipeline/run.py \
  --output outputs/new_obj_las_run --density 600 --origin add
```

Python 命令统一使用系统 `python3`，依赖由项目根 `pyproject.toml` 管理；需要 OSGB 辅助脚本时安装
`.[osgb]`，C++ OSG/PDAL 等系统依赖仍按本目录的 CMake 说明准备。

输出目录必须不存在，防止旧缓存混入和覆盖。`--max-geometries 6` 仅用于试跑，计数单位是 Geometry，不是瓦片。`--origin keep` 不添加原点，`--origin add` 显式添加一次原点。

正式输出位于 `outputs/root_full_xyz_10/`（目录名为历史遗留名称，当前 `stats.json` 的正式 LAS 密度为 600 points/m²）：

- `model.obj`：全部选中几何的单文件网格；不同 Geometry 分组。
- `cloud.las`：同一几何表面按面积均匀采样，当前正式文件密度为 600 点/平方坐标单位。
- `points.bin`：连续 little-endian float64 XYZ，中间文件。
- `geometry.tsv`：源文件与顶点/三角形数。
- `stats.json`：文件访问、变换、几何、三角形、点数量。
- `qc.json`：坐标范围、原点策略、CRS 状态、LAS 全量复读量化误差。
- `mesh_qc.json`：OBJ 逐行面索引检查、顶点/三角形数、源瓦片覆盖和点云包围盒检查。

本次全量运行访问 4,723 个 OSGB、输出 12,076 个 Geometry、39,451,872 个三角形；当前 600 点/m² LAS 采样 1,475,115,656 点。实际源数据路径上 Transform 数为 0；父子矩阵累计通过单独构造的测试验收。输出 OBJ 约 2.4 GiB，600 点/m² LAS 约 29.5 GiB。跨瓦片是否有原始模型接缝，以及绝对坐标是否符合测量控制点，尚未完成目视/外部测量验收；结构与数值校验不能替代这两项。

按 600 points/m² 标准替换现有 LAS 时使用：

```bash
python3 osgb_pipeline/replace_cloud_las.py
```

该命令只生成 `cloud.las.600.part`，用 laspy 检查 LAS 1.4 头和样本点后，才原子替换 `cloud.las`。它不重新生成 OBJ，也不生成大体积 `points.bin`；600 点/m² 结构面识别应直接使用 `cloud.las`。旧 `points.bin` 若存在，仅属于旧流程的历史中间文件，不参与识别。

## 坐标与限制

实测 metadata 为 LOCAL，原点为 `(527093.2468940044,3130096.4615409155,33.174580665997354)`。加原点用于恢复该项目坐标偏移，不代表已确定 EPSG 或完成控制点配准。当前没有外部控制点，因此绝对位置尚未独立验收。

LAS 用 laspy 写 LAS 1.4，scale=0.001；表示量化间隔，不表示测量精度。OBJ 与 LAS 使用相同偏移。原始激光回波、强度和分类数据不存在，保持未赋值。

此解析器针对实际数据里单外部细化分支的 PagedLOD。多个外部 LOD 候选、常驻 LOD、缺失引用、循环引用和不支持的顶点类型会报错，不静默跳过。失败输出目录应视为不完整。最终成功以 `qc.json` 和退出码 0 为准。

测试执行：

```bash
cmake --build osgb_pipeline/build -j2
python3 osgb_pipeline/test_pipeline.py
python3 osgb_pipeline/test_stream_las.py
```

常规系统安装 `libopenscenegraph-dev` 后可用标准 `cmake -S osgb_pipeline -B osgb_pipeline/build`。本机编译使用 deps 内官方开发包头文件和系统现有 OSG 3.6.5 动态库，具体路径保存在 build/CMakeCache.txt。

## 核对资料

- 用户参考：`/home/huqilin/下载/Block_OSGB批量转LAS_LAZ专用流程.md`
- OSG 官方 PagedLOD 实现：https://github.com/openscenegraph/OpenSceneGraph/blob/master/src/osg/PagedLOD.cpp
- OSG 官方 NodeVisitor：https://github.com/openscenegraph/OpenSceneGraph/blob/master/include/osg/NodeVisitor
- laspy 分批写入：https://laspy.readthedocs.io/en/latest/basic.html#chunked-writing

此前的 `scripts/osgb_to_pointcloud.py` 与 `outputs/weiyan_pointcloud_10ptm2/` 保留供对照，不用于本流程验收。
