"""Prepare geometry and report records before they reach the IO layer."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..core.geometry import alpha_shape_geometry, plane_basis, project_to_plane
from ..core.models import (
    ApertureRecord,
    GlobalPlaneRecord,
    PlaneInstance,
    TraceRecord,
)


def plane_boundary_feature(
    plane: PlaneInstance,
    points: np.ndarray,
    *,
    alpha: float | str = "auto",
    max_points: int = 5_000,
) -> dict[str, Any] | None:
    """Build one world-coordinate GeoJSON feature from a fitted plane."""

    if len(plane.point_indices) < 3:
        return None
    plane_points = np.asarray(points[plane.point_indices], dtype=np.float64)
    projected = project_to_plane(plane_points, plane.centroid, plane.normal)
    geometry = alpha_shape_geometry(projected, alpha=alpha, max_points=max_points)
    if geometry is None or geometry.is_empty:
        return None
    u, v = plane_basis(plane.normal)

    def lift(point: tuple[float, float]) -> list[float]:
        world = plane.centroid + u * float(point[0]) + v * float(point[1])
        return [float(world[0]), float(world[1]), float(world[2])]

    if geometry.geom_type == "Polygon":
        coordinates = [[lift(point) for point in geometry.exterior.coords]]
        geometry_type = "Polygon"
    elif geometry.geom_type == "MultiPolygon":
        coordinates = [
            [[lift(point) for point in polygon.exterior.coords]]
            for polygon in geometry.geoms
        ]
        geometry_type = "MultiPolygon"
    else:
        return None
    return {
        "type": "Feature",
        "properties": {
            "plane_id": plane.plane_id,
            "set_id": plane.set_id,
            "observed_area_m2": float(plane.area),
            "edge_censored": bool(plane.edge_censored),
            "data_gap_censored": bool(plane.data_gap_censored),
        },
        "geometry": {"type": geometry_type, "coordinates": coordinates},
    }


def prepare_plane_boundaries(
    planes: list[PlaneInstance],
    points: np.ndarray,
    *,
    alpha: float | str = "auto",
    max_points: int = 5_000,
) -> dict[str, Any]:
    features = [
        feature
        for plane in planes
        if (feature := plane_boundary_feature(plane, points, alpha=alpha, max_points=max_points))
        is not None
    ]
    return {"type": "FeatureCollection", "features": features}


def prepare_trace_rows(
    planes: list[PlaneInstance],
    trace_config: dict[str, Any],
) -> list[TraceRecord]:
    return [
        {
            "plane_id": plane.plane_id,
            "available": False,
            "trace_verified": False,
            "method": trace_config.get("method", "normal_tensor_voting"),
            "chord_length_m": None,
            "polyline_length_m": None,
            "reason": "high_resolution_mesh_or_image_required",
        }
        for plane in planes
    ]


def prepare_aperture_rows(
    planes: list[PlaneInstance],
    aperture_config: dict[str, Any],
) -> list[ApertureRecord]:
    return [
        {
            "plane_id": plane.plane_id,
            "available": False,
            "source": aperture_config.get("source", "highest_resolution_mesh_or_image"),
            "effective_resolution_m": None,
            "aperture_m": None,
            "reason": "high_resolution_mesh_or_image_required",
        }
        for plane in planes
    ]


def prepare_whole_cloud_auxiliary_rows(
    global_plane_rows: list[GlobalPlaneRecord],
) -> tuple[list[TraceRecord], list[ApertureRecord]]:
    """Prepare trace and aperture rows for already-aggregated plane records."""

    trace_rows = [
        {
            "plane_id": row.get("global_plane_id"),
            "available": False,
            "trace_verified": False,
            "method": "normal_tensor_voting",
            "chord_length_m": None,
            "polyline_length_m": None,
            "reason": "high_resolution_mesh_or_image_required",
        }
        for row in global_plane_rows
    ]
    aperture_rows = [
        {
            "plane_id": row.get("global_plane_id"),
            "available": False,
            "source": "highest_resolution_mesh_or_image",
            "effective_resolution_m": None,
            "aperture_m": None,
            "reason": "high_resolution_mesh_or_image_required",
        }
        for row in global_plane_rows
    ]
    return trace_rows, aperture_rows


def summarize_tile_density(
    reports: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Summarize stored per-tile density reports without a global KDTree."""

    local_reports = [
        report.get("density", {}).get("local_surface_density", {})
        for report in reports
    ]
    estimated = [item for item in local_reports if item.get("status") == "estimated"]
    medians = np.asarray(
        [
            float(item["median_points_m2"])
            for item in estimated
            if item.get("median_points_m2") is not None
        ],
        dtype=np.float64,
    )
    statuses: dict[str, int] = defaultdict(int)
    for item in local_reports:
        statuses[str(item.get("standard_status", item.get("status", "unknown")))] += 1
    result: dict[str, Any] = {
        "method": "tile_local_tangent_knn",
        "status": "estimated_from_tile_summaries" if len(medians) else "not_assessed",
        "knn": int(config.get("density", {}).get("knn", 20)),
        "target_raw_density_points_m2": float(
            config.get("density", {}).get("target_raw_density_points_m2", 600.0)
        ),
        "tile_count": len(reports),
        "estimated_tile_count": len(estimated),
        "tile_status_counts": dict(statuses),
        "aggregation_note": "字段是局部瓦片摘要的分布，不是把所有源点放入一个全局 KDTree 后的分位数。",
    }
    if len(medians):
        result.update(
            {
                "tile_median_p10_points_m2": float(np.quantile(medians, 0.10)),
                "tile_median_p25_points_m2": float(np.quantile(medians, 0.25)),
                "tile_median_points_m2": float(np.median(medians)),
                "tile_median_mean_points_m2": float(np.mean(medians)),
                "tile_median_p75_points_m2": float(np.quantile(medians, 0.75)),
                "tile_median_p90_points_m2": float(np.quantile(medians, 0.90)),
            }
        )
    return result


__all__ = [
    "plane_boundary_feature",
    "prepare_aperture_rows",
    "prepare_plane_boundaries",
    "prepare_trace_rows",
    "prepare_whole_cloud_auxiliary_rows",
    "summarize_tile_density",
]
