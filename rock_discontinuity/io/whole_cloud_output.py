"""File writers for whole-cloud results.

This module accepts already-computed rows and reports.  It does not perform
plane merging, spacing calculation, or candidate selection.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..core.models import (
    ApertureRecord,
    GlobalJointSetRecord,
    GlobalPlaneRecord,
    PlaneInstance,
    SpacingRecord,
    TilePlaneRecord,
    TraceRecord,
    plane_to_row,
)


# CSV schemas belong to the serialization boundary.  The processing layer
# re-exports these names for callers that used the historical module path.
PLANE_FIELDS = list(plane_to_row(PlaneInstance()).keys())
WHOLE_PLANE_FIELDS = [
    "tile_id",
    "global_plane_id",
    "global_set_id",
    "global_instance_count",
    "global_tile_count",
    "core_red_points",
    "tile_plane_index",
    "status",
    "selection_reason",
    "nearest_spacing_m",
    "center_x",
    "center_y",
    "center_z",
    *PLANE_FIELDS,
]
WHOLE_GLOBAL_PLANE_FIELDS = [
    "global_plane_id",
    "global_set_id",
    "tile_count",
    "instance_count",
    "core_red_points",
    "plane_id",
    "set_id",
    "dip_direction_deg",
    "dip_deg",
    "trace_length_m",
    "observed_area_m2",
    "observed_area_sum_m2",
    "max_instance_area_m2",
    "major_extent_m",
    "minor_extent_m",
    "edge_censored",
    "nearest_spacing_m",
    "center_x",
    "center_y",
    "center_z",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "nx",
    "ny",
    "nz",
    "plane_d",
    "bbox_min_x",
    "bbox_min_y",
    "bbox_min_z",
    "bbox_max_x",
    "bbox_max_y",
    "bbox_max_z",
    "rms_m",
    "mae_m",
    "p95_residual_m",
    "inlier_ratio",
    "planarity",
    "normal_dispersion_deg",
    "normal_scale_m",
    "normal_stability_deg",
    "normal_stable_fraction",
    "normal_valid_scale_count",
    "normal_stable",
    "boundary_completeness",
    "confidence",
    "quality_score",
    "quality_grade",
    "near_vertical",
    "data_gap_censored",
    "trace_verified",
    "aperture_available",
    "aperture_reason",
]
WHOLE_JOINT_SET_FIELDS = [
    "set_id",
    "plane_count",
    "mean_dip_direction_deg",
    "mean_dip_deg",
    "mean_normal_x",
    "mean_normal_y",
    "mean_normal_z",
    "angular_dispersion_deg",
    "spacing_available",
    "spacing_reason",
    "mean_spacing_m",
    "median_spacing_m",
    "std_spacing_m",
    "min_spacing_m",
    "max_spacing_m",
    "p10_spacing_m",
    "p90_spacing_m",
    "spacing_3d_sample_count",
    "spacing_virtual_scanline_sample_count",
    "mean_spacing_3d_m",
    "median_spacing_3d_m",
    "mean_spacing_virtual_scanline_m",
    "median_spacing_virtual_scanline_m",
    "fisher_k",
]
WHOLE_SPACING_FIELDS = [
    "tile_id",
    "plane_id_a",
    "plane_id_b",
    "global_plane_id_a",
    "global_plane_id_b",
    "spacing_m",
    "method",
]

def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    raise TypeError(f"无法序列化类型：{type(value)!r}")


def write_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    progress: Callable[[str, int, int, str], None] | None = None,
    stage: str | None = None,
) -> None:
    progress_stage = stage or f"写出 {path.name}"
    total = max(1, len(rows))
    if progress is not None:
        progress(progress_stage, 0, total, f"准备 {len(rows)} 行")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        chunk_size = 4096
        for start in range(0, len(rows), chunk_size):
            stop = min(len(rows), start + chunk_size)
            writer.writerows(rows[start:stop])
            if progress is not None:
                progress(progress_stage, stop, total, f"{path.name} {stop}/{len(rows)}")
    if not rows and progress is not None:
        progress(progress_stage, total, total, f"{path.name} 为空")


def write_json_reports(
    output_dir: Path,
    report: dict[str, Any],
    *,
    progress: Callable[[str, int, int, str], None] | None = None,
    stage: str = "写出 JSON 报告",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = (
        ("data_audit.json", report.get("data_audit", {})),
        ("density_report.json", report.get("density_report", {})),
        ("run.json", report),
        ("whole_cloud_report.json", report),
    )
    total = max(1, len(items))
    if progress is not None:
        progress(stage, 0, total, f"准备 {len(items)} 个文件")
    for index, (name, value) in enumerate(items, start=1):
        (output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        if progress is not None:
            progress(stage, index, total, f"完成 {name}")


def write_whole_cloud_outputs(
    output_dir: Path,
    report: dict[str, Any],
    *,
    plane_rows: list[TilePlaneRecord],
    global_plane_rows: list[GlobalPlaneRecord],
    joint_set_rows: list[GlobalJointSetRecord],
    spacing_rows: list[SpacingRecord],
    tile_spacing_rows: list[SpacingRecord],
    trace_rows: list[TraceRecord],
    aperture_rows: list[ApertureRecord],
    plane_fields: Sequence[str],
    global_plane_fields: Sequence[str],
    joint_set_fields: Sequence[str],
    spacing_fields: Sequence[str],
    progress: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Write all whole-cloud tables and JSON reports from prepared records."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        output_dir / "detachment_planes.csv",
        plane_rows,
        plane_fields,
        progress=progress,
        stage="写出 detachment_planes.csv",
    )
    write_rows(
        output_dir / "joint_planes.csv",
        global_plane_rows,
        global_plane_fields,
        progress=progress,
        stage="写出 joint_planes.csv",
    )
    write_rows(
        output_dir / "joint_sets.csv",
        joint_set_rows,
        joint_set_fields,
        progress=progress,
        stage="写出 joint_sets.csv",
    )
    write_rows(
        output_dir / "spacings.csv",
        spacing_rows,
        spacing_fields,
        progress=progress,
        stage="写出 spacings.csv",
    )
    write_rows(
        output_dir / "tile_spacings.csv",
        tile_spacing_rows,
        spacing_fields,
        progress=progress,
        stage="写出 tile_spacings.csv",
    )

    trace_fields = [
        "plane_id",
        "available",
        "trace_verified",
        "method",
        "chord_length_m",
        "polyline_length_m",
        "reason",
    ]
    write_rows(
        output_dir / "traces.csv",
        trace_rows,
        trace_fields,
        progress=progress,
        stage="写出 traces.csv",
    )
    aperture_fields = [
        "plane_id",
        "available",
        "source",
        "effective_resolution_m",
        "aperture_m",
        "reason",
    ]
    write_rows(
        output_dir / "aperture.csv",
        aperture_rows,
        aperture_fields,
        progress=progress,
        stage="写出 aperture.csv",
    )
    report = dict(report)
    write_json_reports(output_dir, report, progress=progress)
    return report


__all__ = [
    "PLANE_FIELDS",
    "WHOLE_PLANE_FIELDS",
    "WHOLE_GLOBAL_PLANE_FIELDS",
    "WHOLE_JOINT_SET_FIELDS",
    "WHOLE_SPACING_FIELDS",
    "write_json_reports",
    "write_rows",
    "write_whole_cloud_outputs",
]
