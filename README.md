# 岩体节理面识别

项目包含两条相互独立的链路：`osgb_pipeline/` 负责 OSGB 到 LAS/LAZ 的前处理，
`rock_discontinuity/` 负责已有 LAS/LAZ 的 ROI 和全点云节理面识别。原始资料与运行结果
不纳入源码版本库。

## 安装

项目使用系统 Python，不要求创建或激活虚拟环境。依赖安装到当前用户目录：

```bash
python3 -m pip install --user 'setuptools>=68,<80' 'wheel>=0.40'
python3 -m pip install --user --no-build-isolation -c constraints.txt -e '.[osgb]'
# 运行测试时再安装开发依赖
python3 -m pip install --user --no-build-isolation -c constraints.txt -e '.[osgb,dev]'
```

两个子目录中的 `requirements.txt` 也按项目根目录执行；它们只是根安装定义的
兼容入口。

推荐入口是 `python3 main.py` 或安装后的 `rock-discontinuity`。两者默认进入
`whole-cloud`；`python3 -m rock_discontinuity` 保留历史 ROI 默认行为。

```bash
python3 main.py --help
python3 main.py roi --input path/to/cloud.las \
  --bbox xmin,xmax,ymin,ymax --output outputs/example
python3 main.py validate outputs/example
```

全点云流程会保留 `split_state.json`、`processing_state.json` 和瓦片报告，可用
`--resume` 继续中断任务。网格、状态和输出记录准备位于独立模块，核心数值流程通过
`ProcessingResult` 和 `TileResult` 传递结果。

更完整的目录说明、参数和验收边界见 [`docs/项目结构.md`](docs/项目结构.md) 与
[`docs/rock_discontinuity/VALIDATION.md`](docs/rock_discontinuity/VALIDATION.md)。
