"""Row-level geometric helpers used by cross-tile aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.geometry import (
    align_plane_equation,
    footprint_gap_on_plane,
    normal_angle_deg as axial_normal_angle_deg,
)


def row_float(row: dict[str, Any], field: str) -> float | None:
    try:
        value = float(row.get(field))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def row_normal(row: dict[str, Any]) -> np.ndarray | None:
    values = [row_float(row, field) for field in ("nx", "ny", "nz")]
    if any(value is None for value in values):
        return None
    normal = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    return normal / norm


def row_bbox(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    lower_values = [row_float(row, f"bbox_min_{axis}") for axis in "xyz"]
    upper_values = [row_float(row, f"bbox_max_{axis}") for axis in "xyz"]
    if all(value is not None for value in lower_values + upper_values):
        return (
            np.asarray(lower_values, dtype=np.float64),
            np.asarray(upper_values, dtype=np.float64),
        )
    center_values = [row_float(row, f"center_{axis}") for axis in "xyz"]
    if all(value is not None for value in center_values):
        center = np.asarray(center_values, dtype=np.float64)
        return center.copy(), center.copy()
    return None


def tile_id_indices(tile_id: str) -> tuple[int, int] | None:
    try:
        left, right = str(tile_id).split("_", 1)
        return int(left), int(right)
    except (TypeError, ValueError):
        return None


def planes_can_merge_rows(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    *,
    normal_angle_deg: float,
    plane_offset_m: float,
    xy_gap_m: float,
    max_merged_rms_m: float | None = None,
) -> bool:
    """Apply the cross-tile merge gate to two serialized plane rows."""

    normal_a = row_normal(row_a)
    normal_b = row_normal(row_b)
    d_a = row_float(row_a, "plane_d")
    d_b = row_float(row_b, "plane_d")
    if normal_a is None or normal_b is None or d_a is None or d_b is None:
        return False
    angle = axial_normal_angle_deg(normal_a, normal_b)
    if angle is None or angle > float(normal_angle_deg):
        return False
    aligned = align_plane_equation(normal_a, d_a, normal_b, d_b)
    if aligned is None:
        return False
    normal_a, d_a, _, d_b = aligned
    if abs(d_a - d_b) > plane_offset_m:
        return False
    if max_merged_rms_m is not None:
        rms_a = row_float(row_a, "rms_m")
        rms_b = row_float(row_b, "rms_m")
        if rms_a is not None and rms_a > max_merged_rms_m:
            return False
        if rms_b is not None and rms_b > max_merged_rms_m:
            return False
    bbox_a = row_bbox(row_a)
    bbox_b = row_bbox(row_b)
    if bbox_a is None or bbox_b is None:
        return False
    lower_a, upper_a = bbox_a
    lower_b, upper_b = bbox_b
    footprint_gap = footprint_gap_on_plane(
        row_a.get("_footprint_xyz"),
        row_b.get("_footprint_xyz"),
        np.asarray(
            [
                ((row_float(row_a, f"center_{axis}") or 0.0) + (row_float(row_b, f"center_{axis}") or 0.0)) / 2.0
                for axis in "xyz"
            ],
            dtype=np.float64,
        ),
        normal_a,
    )
    if footprint_gap is not None:
        return footprint_gap <= xy_gap_m
    gap = np.maximum(0.0, np.maximum(lower_a, lower_b) - np.minimum(upper_a, upper_b))
    return float(np.linalg.norm(gap)) <= xy_gap_m

