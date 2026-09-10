"""File writers for whole-cloud results.

This module accepts already-computed rows and reports.  It does not perform
plane merging, spacing calculation, or candidate selection.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.models import ApertureRecord, TraceRecord

def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    raise TypeError(f"无法序列化类型：{type(value)!r}")


def write_rows(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json_reports(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("data_audit.json", report.get("data_audit", {})),
        ("density_report.json", report.get("density_report", {})),
        ("run.json", report),
        ("whole_cloud_report.json", report),
    ):
        (output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )


def write_whole_cloud_outputs(
    output_dir: Path,
    report: dict[str, Any],
    *,
    plane_rows: list[dict[str, Any]],
    global_plane_rows: list[dict[str, Any]],
    joint_set_rows: list[dict[str, Any]],
    spacing_rows: list[dict[str, Any]],
    tile_spacing_rows: list[dict[str, Any]],
    trace_rows: list[TraceRecord],
    aperture_rows: list[ApertureRecord],
    plane_fields: Sequence[str],
    global_plane_fields: Sequence[str],
    joint_set_fields: Sequence[str],
    spacing_fields: Sequence[str],
) -> dict[str, Any]:
    """Write all whole-cloud tables and JSON reports from prepared records."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "detachment_planes.csv", plane_rows, plane_fields)
    write_rows(output_dir / "joint_planes.csv", global_plane_rows, global_plane_fields)
    write_rows(output_dir / "joint_sets.csv", joint_set_rows, joint_set_fields)
    write_rows(output_dir / "spacings.csv", spacing_rows, spacing_fields)
    write_rows(output_dir / "tile_spacings.csv", tile_spacing_rows, spacing_fields)

    trace_fields = [
        "plane_id",
        "available",
        "trace_verified",
        "method",
        "chord_length_m",
        "polyline_length_m",
        "reason",
    ]
    write_rows(output_dir / "traces.csv", trace_rows, trace_fields)
    aperture_fields = [
        "plane_id",
        "available",
        "source",
        "effective_resolution_m",
        "aperture_m",
        "reason",
    ]
    write_rows(output_dir / "aperture.csv", aperture_rows, aperture_fields)
    report = dict(report)
    write_json_reports(output_dir, report)
    return report


__all__ = [
    "write_json_reports",
    "write_rows",
    "write_whole_cloud_outputs",
]
