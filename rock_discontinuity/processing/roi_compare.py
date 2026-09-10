from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..config import dump_config, load_config
from .pipeline import run_las


DEFAULT_ROIS = {
    "roi_1_high_fragmentation": {"min_x": 524960.0, "max_x": 524980.0, "min_y": 3132480.0, "max_y": 3132500.0},
    "roi_2_control": {"min_x": 525000.0, "max_x": 525020.0, "min_y": 3132560.0, "max_y": 3132580.0},
    "roi_3_high_red_ratio": {"min_x": 524960.0, "max_x": 524980.0, "min_y": 3132680.0, "max_y": 3132700.0},
}

VARIANTS = {
    "baseline": {"multiscale": False, "hierarchical": False},
    "multiscale": {"multiscale": True, "hierarchical": False},
    "multiscale_hmerge": {"multiscale": True, "hierarchical": True},
}

DEFAULT_ROI_INPUTS = {
    "roi_1_high_fragmentation": "outputs/whole_cloud_detachment/source_tiles/tile_7_22.laz",
    "roi_2_control": "outputs/whole_cloud_detachment/source_tiles/tile_8_24.laz",
    "roi_3_high_red_ratio": "outputs/whole_cloud_detachment/source_tiles/tile_7_27.laz",
}


def _variant_config(config: dict[str, Any], variant: str) -> dict[str, Any]:
    value = deepcopy(config)
    value["normal_multiscale"]["enabled"] = VARIANTS[variant]["multiscale"]
    value["plane_merge"]["hierarchical"]["enabled"] = VARIANTS[variant]["hierarchical"]
    value["experiment"] = {"suite": "three_roi_multiscale_hierarchical_ablation", "variant": variant}
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * quantile))))
    return float(values[index])


def _summary_row(roi_name: str, bbox: dict[str, float], variant: str, output_dir: Path) -> dict[str, Any]:
    report = _read_json(output_dir / "report.json")
    counts = report.get("counts", {})
    input_points = int(counts.get("input_points", 0))
    candidate_points = int(report.get("detachment", {}).get("candidate_point_count", 0))
    plane_rows = _read_csv(output_dir / "planes.csv")
    stability_values = [
        float(row["normal_stability_deg"])
        for row in plane_rows
        if row.get("normal_stability_deg") not in {None, ""}
    ]
    return {
        "roi": roi_name,
        "variant": variant,
        **bbox,
        "input_points": input_points,
        "plane_instances": int(counts.get("plane_instances", 0)),
        "accepted_planes": int(counts.get("accepted_planes", 0)),
        "candidate_joint_planes": int(counts.get("candidate_joint_planes", 0)),
        "candidate_points": candidate_points,
        "candidate_fraction": candidate_points / input_points if input_points else None,
        "joint_set_count": len(_read_csv(output_dir / "joint_sets.csv")),
        "spacing_rows": len(_read_csv(output_dir / "spacings.csv")),
        "normal_stability_p50_deg": _quantile(stability_values, 0.50),
        "normal_stability_p90_deg": _quantile(stability_values, 0.90),
        "output_dir": str(output_dir),
    }


def _write_summary(output_root: Path, rows: list[dict[str, Any]], input_path: Path) -> dict[str, Any]:
    fields = list(rows[0].keys()) if rows else []
    summary_csv = output_root / "roi_comparison.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "suite": "three_roi_multiscale_hierarchical_ablation",
        "input": str(input_path),
        "rois": DEFAULT_ROIS,
        "roi_inputs": DEFAULT_ROI_INPUTS,
        "variants": VARIANTS,
        "rows": rows,
        "summary_csv": str(summary_csv),
    }
    (output_root / "roi_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def summarize_existing(output_root: Path, input_path: Path) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    rows = [
        _summary_row(roi_name, bbox, variant, output_root / roi_name / variant)
        for roi_name, bbox in DEFAULT_ROIS.items()
        for variant in VARIANTS
    ]
    return _write_summary(output_root, rows, input_path.expanduser().resolve())


def run_comparison(input_path: Path, output_root: Path, config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for roi_name, bbox in DEFAULT_ROIS.items():
        roi_input = Path(DEFAULT_ROI_INPUTS.get(roi_name, str(input_path)))
        if not roi_input.is_absolute():
            roi_input = Path.cwd() / roi_input
        roi_input = roi_input.expanduser().resolve()
        if not roi_input.is_file():
            raise FileNotFoundError(f"ROI 源瓦片不存在：{roi_input}")
        for variant in VARIANTS:
            output_dir = output_root / roi_name / variant
            if output_dir.exists() and any(output_dir.iterdir()) and not force:
                raise FileExistsError(f"输出已存在；请换目录或使用 --force：{output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            variant_config = _variant_config(config, variant)
            variant_config["input"]["path"] = str(roi_input)
            variant_config["input"]["roi"] = bbox
            variant_config["output"]["directory"] = str(output_dir)
            dump_config(variant_config, output_dir / "run_config.yaml")
            print(f"[roi-compare] {roi_name} / {variant}", flush=True)
            run_las(roi_input, output_dir, variant_config, bbox)
            report_path = output_dir / "report.json"
            report = _read_json(report_path)
            report["experiment"] = variant_config["experiment"]
            report["roi_name"] = roi_name
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(_summary_row(roi_name, bbox, variant, output_dir))
    return _write_summary(output_root, rows, input_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="比较三个典型ROI的多尺度法向和层次合并消融结果")
    parser.add_argument("--input", type=Path, default=Path("outputs/whole_cloud_detachment/source_tiles/tile_7_22.laz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/roi_compare_v23"))
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/default.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summarize_existing:
        summary = summarize_existing(args.output, args.input)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    config = load_config(args.config if args.config.exists() else None)
    summary = run_comparison(args.input, args.output, config, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
