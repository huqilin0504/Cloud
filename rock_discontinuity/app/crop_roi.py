"""Command-line adapter for streaming projection-polygon LAS cropping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..processing.roi_crop import crop_las_to_projection_roi, load_projection_roi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按 ENU 投影多边形流式裁剪 LAS/LAZ，完整保留源点属性"
    )
    parser.add_argument("--input", type=Path, required=True, help="原始 LAS/LAZ")
    parser.add_argument("--output", type=Path, required=True, help="裁剪后的 LAS/LAZ")
    parser.add_argument(
        "--roi-config",
        type=Path,
        required=True,
        help="包含 origin_xyz、投影角度和 polygon_uv 的 JSON",
    )
    parser.add_argument("--report", type=Path, help="审计报告；默认与输出同名 .roi.json")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="每批读取点数，默认 1000000",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出")
    parser.add_argument("--quiet", action="store_true", help="关闭进度条")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size 必须大于 0")
    roi = load_projection_roi(args.roi_config.expanduser().resolve())
    output = args.output.expanduser().resolve()
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else output.with_suffix(".roi.json")
    )
    report = crop_las_to_projection_roi(
        args.input.expanduser().resolve(),
        output,
        roi,
        report_path=report_path,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        show_progress=not args.quiet,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
