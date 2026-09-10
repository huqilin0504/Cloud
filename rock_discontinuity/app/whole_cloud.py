"""Command-line adapter for the whole-cloud workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from ..io.las_tiles import source_info
from ..io.whole_cloud_output import _json_default
from ..processing.tiling import grid_plan
from ..processing.whole_cloud_contract import tiling_parameters
from ..processing.whole_cloud import run_whole_cloud


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对完整 LAS/LAZ/COPC 进行可恢复的分块候选节理面识别")
    parser.add_argument("--input", type=Path, default=Path("outputs/root_full_xyz_10/cloud.las"))
    parser.add_argument("--output", type=Path, default=Path("outputs/whole_cloud_joint_planes"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/default.yaml",
    )
    parser.add_argument("--tile-size", type=float, help="瓦片边长；默认读取 tiling.tile_size_m")
    parser.add_argument("--overlap", type=float, help="瓦片 XY 缓冲；默认读取 tiling.overlap_m")
    parser.add_argument("--resume", action="store_true", help="使用已有分块和处理状态继续")
    parser.add_argument("--prepare-only", action="store_true", help="只生成压缩 LAZ 瓦片，不进行识别")
    parser.add_argument("--keep-source-tiles", action="store_true", help="处理后保留中间源 LAZ 瓦片")
    parser.add_argument("--max-tiles", type=int, help="只处理前 N 个瓦片，用于试运行")
    parser.add_argument(
        "--workers",
        type=int,
        help="节理候选识别并行线程数；默认读取 whole_cloud.workers（默认 2）",
    )
    parser.add_argument("--plan-only", action="store_true", help="只读取 LAS header 并打印分块计划")
    parser.add_argument("--pdal", default="pdal", help="PDAL 可执行文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_tiles is not None and args.max_tiles <= 0:
        raise SystemExit("--max-tiles 必须大于 0")
    if args.workers is not None and args.workers <= 0:
        raise SystemExit("--workers 必须大于 0")
    config = load_config(args.config if args.config.exists() else None)
    tile_size, overlap = tiling_parameters(config, args.tile_size, args.overlap)
    if args.plan_only:
        source = source_info(args.input.expanduser().resolve())
        plan = grid_plan(source, tile_size, overlap)
        print(json.dumps({"input": source, "tiling": plan}, ensure_ascii=False, indent=2))
        return 0
    report = run_whole_cloud(
        args.input,
        args.output,
        config,
        tile_size=tile_size,
        overlap=overlap,
        resume=args.resume,
        prepare_only=args.prepare_only,
        keep_source_tiles=args.keep_source_tiles,
        max_tiles=args.max_tiles,
        workers=args.workers,
        pdal_command=args.pdal,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 2 if report["tiling"]["failed_tiles"] else 0


__all__ = ["build_parser", "main"]
