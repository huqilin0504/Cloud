from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import KDTree


def _nearest_spacing(points: np.ndarray, sample_size: int = 20_000) -> dict[str, Any]:
    if len(points) < 2:
        return {
            "method": "3d_nearest_neighbor",
            "sample_points": int(len(points)),
            "p10_m": None,
            "median_m": None,
            "p90_m": None,
        }
    count = min(len(points), max(2, int(sample_size))) 
    indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    tree = KDTree(points)
    try:
        distances, _ = tree.query(points[indices], k=2, workers=-1)
    except TypeError:
        distances, _ = tree.query(points[indices], k=2)
    nearest = np.asarray(distances[:, 1], dtype=np.float64)
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-9)]
    if not len(nearest):
        return {
            "method": "3d_nearest_neighbor",
            "sample_points": int(count),
            "p10_m": None,
            "median_m": None,
            "p90_m": None,
        }
    return {
        "method": "3d_nearest_neighbor",
        "sample_points": int(count),
        "p10_m": float(np.quantile(nearest, 0.10)),
        "median_m": float(np.median(nearest)),
        "p90_m": float(np.quantile(nearest, 0.90)),
    }


def audit_points(
    points: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
    density_report: dict[str, Any] | None = None,
    tile_count: int | None = None,
    source_mode: str | None = None,
) -> dict[str, Any]:
    """Build the V2 preflight audit record for a loaded point-cloud subset."""

    raw = np.asarray(points, dtype=np.float64)
    finite_mask = np.isfinite(raw).all(axis=1) if raw.ndim == 2 else np.zeros(0, dtype=bool)
    finite = raw[finite_mask] if raw.ndim == 2 and raw.shape[1] == 3 else np.empty((0, 3))
    bounds = None
    if len(finite):
        bounds = {
            "min": [float(value) for value in np.min(finite, axis=0)],
            "max": [float(value) for value in np.max(finite, axis=0)],
        }
    metadata = metadata or {}
    result: dict[str, Any] = {
        "schema_version": "v2",
        "point_count": int(len(finite)),
        "nonfinite_point_count": int(len(raw) - len(finite)) if raw.ndim == 2 else None,
        "point_format": metadata.get("point_format"),
        "crs": metadata.get("crs"),
        "xyz_bounds": bounds or metadata.get("bounds"),
        "scale": metadata.get("scale"),
        "offset": metadata.get("offset"),
        "dimensions": metadata.get("dimensions"),
        "tile_count": int(tile_count) if tile_count is not None else metadata.get("tile_count"),
        "source_mode": source_mode or metadata.get("source_mode"),
        "source_path": metadata.get("path"),
        "nearest_point_spacing": _nearest_spacing(finite),
        "local_density": density_report or {"status": "not_assessed"},
    }
    return result


def audit_source_metadata(
    metadata: dict[str, Any],
    *,
    tile_count: int | None = None,
    source_mode: str | None = None,
) -> dict[str, Any]:
    """Build an audit record from a LAS/COPC header without loading all points."""

    bounds = metadata.get("bounds")
    minimum = bounds.get("min") if isinstance(bounds, dict) else None
    maximum = bounds.get("max") if isinstance(bounds, dict) else None
    footprint = None
    if minimum and maximum and len(minimum) >= 2 and len(maximum) >= 2:
        footprint = float(maximum[0] - minimum[0]) * float(maximum[1] - minimum[1])
    return {
        "schema_version": "v2",
        "point_count": metadata.get("total_points"),
        "selected_point_count": metadata.get("selected_points"),
        "point_format": metadata.get("point_format"),
        "crs": metadata.get("crs"),
        "xyz_bounds": bounds,
        "scale": metadata.get("scale"),
        "offset": metadata.get("offset"),
        "dimensions": metadata.get("dimensions"),
        "tile_count": int(tile_count) if tile_count is not None else metadata.get("tile_count"),
        "source_mode": source_mode or metadata.get("source_mode"),
        "source_format": metadata.get("source_format"),
        "source_path": metadata.get("path"),
        "footprint_bbox_area_m2": footprint,
        "nearest_point_spacing": {"status": "not_assessed_from_header"},
        "local_density": {"status": "not_assessed_from_header"},
    }
