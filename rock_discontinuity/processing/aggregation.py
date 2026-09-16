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


def row_xy_bounds_array(rows: list[dict[str, Any]]) -> np.ndarray:
    """Return conservative XY bounds for serialized plane rows.

    The bounds enclose both the fitted-row bbox and every valid footprint
    vertex.  This makes them a safe necessary-condition filter for both
    branches of ``planes_can_merge_rows``: a footprint distance or a 3-D
    bbox distance cannot be within the merge gap when the corresponding XY
    bounds are farther apart.
    """

    bounds = np.full((len(rows), 4), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        bbox = row_bbox(row)
        if bbox is None:
            continue
        lower, upper = bbox
        values = np.asarray(
            [lower[0], upper[0], lower[1], upper[1]],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            continue
        values = np.asarray(
            [
                min(float(values[0]), float(values[1])),
                max(float(values[0]), float(values[1])),
                min(float(values[2]), float(values[3])),
                max(float(values[2]), float(values[3])),
            ],
            dtype=np.float64,
        )
        footprint = row.get("_footprint_xyz")
        if footprint:
            for ring in footprint:
                try:
                    points = np.asarray(ring, dtype=np.float64)
                except (TypeError, ValueError):
                    continue
                if points.ndim != 2 or points.shape[1] != 3:
                    continue
                finite_points = points[np.all(np.isfinite(points), axis=1)]
                if not len(finite_points):
                    continue
                values[0] = min(values[0], float(np.min(finite_points[:, 0])))
                values[1] = max(values[1], float(np.max(finite_points[:, 0])))
                values[2] = min(values[2], float(np.min(finite_points[:, 1])))
                values[3] = max(values[3], float(np.max(finite_points[:, 1])))
        bounds[index] = values
    return bounds


class XYBBoxIndex:
    """Small temporary grid index for one neighbouring tile's plane rows."""

    def __init__(
        self,
        bounds: np.ndarray,
        row_indices: list[int],
        *,
        cell_size: float,
        max_cells_per_row: int = 4096,
    ) -> None:
        self.bounds = bounds
        self.row_indices = row_indices
        self.cell_size = max(float(cell_size), 1e-9)
        self.max_cells_per_query = int(max_cells_per_row)
        self.cells: dict[tuple[int, int], list[int]] = {}
        self.large_positions: list[int] = []
        for position, row_index in enumerate(row_indices):
            row_bounds = bounds[row_index]
            if not np.all(np.isfinite(row_bounds)):
                continue
            x0, x1, y0, y1 = self._cell_range(row_bounds)
            cell_count = (x1 - x0 + 1) * (y1 - y0 + 1)
            if cell_count > max_cells_per_row:
                self.large_positions.append(position)
                continue
            for cell_x in range(x0, x1 + 1):
                for cell_y in range(y0, y1 + 1):
                    self.cells.setdefault((cell_x, cell_y), []).append(position)

    def _cell_range(self, row_bounds: np.ndarray) -> tuple[int, int, int, int]:
        return (
            int(np.floor(row_bounds[0] / self.cell_size)),
            int(np.floor(row_bounds[1] / self.cell_size)),
            int(np.floor(row_bounds[2] / self.cell_size)),
            int(np.floor(row_bounds[3] / self.cell_size)),
        )

    def query(self, query_bounds: np.ndarray, gap: float) -> list[int]:
        """Return original row positions whose expanded XY boxes intersect."""

        if not np.all(np.isfinite(query_bounds)):
            return []
        gap = max(float(gap), 0.0)
        expanded = np.asarray(
            [
                query_bounds[0] - gap,
                query_bounds[1] + gap,
                query_bounds[2] - gap,
                query_bounds[3] + gap,
            ],
            dtype=np.float64,
        )
        x0, x1, y0, y1 = self._cell_range(expanded)
        cell_count = (x1 - x0 + 1) * (y1 - y0 + 1)
        if cell_count > self.max_cells_per_query:
            positions = set(range(len(self.row_indices)))
        else:
            positions = set(self.large_positions)
            for cell_x in range(x0, x1 + 1):
                for cell_y in range(y0, y1 + 1):
                    positions.update(self.cells.get((cell_x, cell_y), ()))
        result: list[int] = []
        for position in sorted(positions):
            row_bounds = self.bounds[self.row_indices[position]]
            if (
                row_bounds[0] <= expanded[1]
                and row_bounds[1] >= expanded[0]
                and row_bounds[2] <= expanded[3]
                and row_bounds[3] >= expanded[2]
            ):
                result.append(position)
        return result


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
