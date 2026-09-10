from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ..config import dump_config
from ..io.roi_compare_output import (
    read_csv_rows,
    read_json_report,
    write_comparison_summary,
    write_experiment_report,
)
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


def _quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * quantile))))
    return float(values[index])


def _summary_row(roi_name: str, bbox: dict[str, float], variant: str, output_dir: Path) -> dict[str, Any]:
    report = read_json_report(output_dir / "report.json")
    counts = report.get("counts", {})
    input_points = int(counts.get("input_points", 0))
    candidate_points = int(report.get("detachment", {}).get("candidate_point_count", 0))
    plane_rows = read_csv_rows(output_dir / "planes.csv")
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
        "joint_set_count": len(read_csv_rows(output_dir / "joint_sets.csv")),
        "spacing_rows": len(read_csv_rows(output_dir / "spacings.csv")),
        "normal_stability_p50_deg": _quantile(stability_values, 0.50),
        "normal_stability_p90_deg": _quantile(stability_values, 0.90),
        "output_dir": str(output_dir),
    }


def summarize_existing(output_root: Path, input_path: Path) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    rows = [
        _summary_row(roi_name, bbox, variant, output_root / roi_name / variant)
        for roi_name, bbox in DEFAULT_ROIS.items()
        for variant in VARIANTS
    ]
    return write_comparison_summary(
        output_root,
        rows,
        input_path.expanduser().resolve(),
        rois=DEFAULT_ROIS,
        roi_inputs=DEFAULT_ROI_INPUTS,
        variants=VARIANTS,
    )


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
            report = read_json_report(report_path)
            report["experiment"] = variant_config["experiment"]
            report["roi_name"] = roi_name
            write_experiment_report(report_path, report)
            rows.append(_summary_row(roi_name, bbox, variant, output_dir))
    return write_comparison_summary(
        output_root,
        rows,
        input_path,
        rois=DEFAULT_ROIS,
        roi_inputs=DEFAULT_ROI_INPUTS,
        variants=VARIANTS,
    )


def build_parser(*args: Any, **kwargs: Any):
    """Compatibility wrapper; parser construction lives in ``app``."""

    from ..app.roi_compare import build_parser as app_build_parser

    return app_build_parser(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    """Compatibility command wrapper with a function-local import."""

    from ..app.roi_compare import main as app_main

    return app_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
