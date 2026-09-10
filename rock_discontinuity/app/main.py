"""Unified LAS/LAZ application entry point.

This dispatcher intentionally does not import or execute the OSGB pipeline.
It only routes operations on an existing LAS/LAZ file or its analysis output.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from ..cli import main as roi_main
from .whole_cloud import main as whole_cloud_main
from ..processing.roi_compare import main as roi_compare_main
from ..validation.validate import main as validate_main


Command = Callable[[list[str] | None], int]
COMMANDS: dict[str, Command] = {
    "whole-cloud": whole_cloud_main,
    "roi": roi_main,
    "validate": validate_main,
    "roi-compare": roi_compare_main,
}

HELP = """统一 LAS/LAZ 处理入口（不包含 OSGB 转换）

用法：
  python3 main.py [whole-cloud] [全点云分块参数]
  python3 main.py roi [ROI 识别参数]
  python3 main.py validate <输出目录>
  python3 main.py roi-compare [三个ROI消融比较参数]

默认命令是 whole-cloud，因此以下两种写法等价：
  python3 main.py --plan-only
  python3 main.py whole-cloud --plan-only

示例：
  python3 main.py --plan-only
  python3 main.py whole-cloud --input outputs/root_full_xyz_10/cloud.las \\
    --output outputs/whole_cloud_joint_planes --tile-size 25 --overlap 1
  python3 main.py roi --input outputs/root_full_xyz_10/cloud.las \\
    --bbox 524898,524938,3132085,3132125 \\
    --output outputs/discontinuity_roi_600
  python3 main.py validate outputs/discontinuity_roi_600

也支持用 --mode whole-cloud|roi|validate|roi-compare 选择命令。

兼容入口：项目根目录的 `python3 main.py` 默认执行 whole-cloud；
`python3 -m rock_discontinuity` 保留历史行为，默认执行 ROI 命令。
安装后可使用 `rock-discontinuity`，其行为与根目录 main.py 一致。
"""


def _select_command(arguments: Sequence[str]) -> tuple[str, list[str]]:
    remaining = list(arguments)
    if remaining and remaining[0] in COMMANDS:
        return remaining[0], remaining[1:]
    if remaining and remaining[0] == "--mode":
        if len(remaining) < 2 or remaining[1] not in COMMANDS:
            raise SystemExit("--mode 必须是 whole-cloud、roi、validate 或 roi-compare")
        return remaining[1], remaining[2:]
    if remaining and remaining[0].startswith("--mode="):
        command = remaining[0].split("=", 1)[1]
        if command not in COMMANDS:
            raise SystemExit("--mode 必须是 whole-cloud、roi、validate 或 roi-compare")
        return command, remaining[1:]
    return "whole-cloud", remaining


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"-h", "--help"}:
        print(HELP)
        return 0
    command, forwarded = _select_command(arguments)
    return COMMANDS[command](forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
