"""CLI adapter for the ROI comparison experiment suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from ..processing.roi_compare import run_comparison, summarize_existing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="比较三个典型ROI的多尺度法向和层次合并消融结果")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/whole_cloud_detachment/source_tiles/tile_7_22.laz"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/roi_compare_v23"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/default.yaml",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summarize_existing:
        summary = summarize_existing(args.output, args.input)
    else:
        config = load_config(args.config if args.config.exists() else None)
        summary = run_comparison(args.input, args.output, config, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
