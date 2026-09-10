from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import config_bbox, dump_config, load_config, parse_bbox
from .processing.pipeline import run_las

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 LAS/LAZ ROI 识别可观测岩体结构面")
    parser.add_argument("--input", type=Path, help="LAS/LAZ 输入文件")
    parser.add_argument("--output", type=Path, help="新输出目录")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/default.yaml")
    parser.add_argument("--bbox", help="xmin,xmax,ymin,ymax[,zmin,zmax]")
    parser.add_argument("--force", action="store_true", help="允许覆盖已存在的输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config if args.config.exists() else None)
    if args.input is not None:
        config["input"]["path"] = str(args.input)
    if args.output is not None:
        config["output"]["directory"] = str(args.output)
    bbox = parse_bbox(args.bbox) if args.bbox else config_bbox(config)
    input_path = config["input"].get("path")
    output_path = config["output"].get("directory")
    if not input_path:
        raise SystemExit("必须提供 --input 或在配置中设置 input.path")
    if not output_path:
        raise SystemExit("必须提供 --output 或在配置中设置 output.directory")
    if bbox is None:
        raise SystemExit("必须提供 20–50m ROI：使用 --bbox 或在配置 input.roi 中设置")
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists() and not args.force:
        raise SystemExit(f"输出目录已存在；请换目录或显式使用 --force：{output_path}")
    if output_path == input_path or input_path in output_path.parents:
        raise SystemExit("输出目录不能位于输入文件所在目录内部")
    output_path.mkdir(parents=True, exist_ok=True)
    config["input"]["path"] = str(input_path)
    config["input"]["roi"] = bbox
    config["output"]["directory"] = str(output_path)
    dump_config(config, output_path / "run_config.yaml")
    paths = run_las(input_path, output_path, config, bbox)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))
    return 0
