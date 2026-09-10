from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import KDTree


def _query(tree: KDTree, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        return tree.query(points, k=k, workers=-1)
    except TypeError:
        return tree.query(points, k=k)


def _tangent_knn_density(
    points: np.ndarray,
    sample_indices: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Estimate surface density with the k-th neighbour in a local tangent plane."""

    if len(points) < 3 or not len(sample_indices):
        return np.empty(0, dtype=np.float64)
    k = max(3, int(k))
    effective_k = min(k, len(points) - 1)
    query_k = effective_k + 1
    tree = KDTree(points)
    distances, neighbours = _query(tree, points[sample_indices], query_k)
    if query_k == 1:
        distances = np.asarray(distances)[:, None]
        neighbours = np.asarray(neighbours)[:, None]
    neighbours = np.asarray(neighbours, dtype=np.int64)
    local = points[neighbours]
    centers = points[sample_indices]
    offsets = local - centers[:, None, :]
    centered = offsets - offsets.mean(axis=1, keepdims=True)
    covariance = np.einsum("bki,bkj->bij", centered, centered) / max(query_k, 1)
    _, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    tangent_offsets = offsets - np.einsum("bki,bi->bk", offsets, normals)[:, :, None] * normals[:, None, :]
    radial = np.linalg.norm(tangent_offsets, axis=2)
    radial.sort(axis=1)
    kth = radial[:, min(effective_k, radial.shape[1] - 1)]
    fallback = np.asarray(distances, dtype=np.float64)[:, min(effective_k, query_k - 1)]
    kth = np.where(kth > 1e-9, kth, fallback)
    valid = np.isfinite(kth) & (kth > 1e-9)
    density = np.full(len(kth), np.nan, dtype=np.float64)
    density[valid] = float(effective_k) / (np.pi * np.square(kth[valid]))
    return density[np.isfinite(density)]


def estimate_roi_density(
    points: np.ndarray,
    sample_size: int = 20_000,
    *,
    knn: int = 20,
) -> dict[str, Any]:
    """Estimate local tangent-plane surface density for a deterministic sample."""

    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return {
            "method": "tangent_knn",
            "knn": int(knn),
            "status": "insufficient_points",
            "sample_points": int(len(points)),
            "p10_points_m2": None,
            "p25_points_m2": None,
            "median_points_m2": None,
            "mean_points_m2": None,
            "p75_points_m2": None,
            "p90_points_m2": None,
            "xy_bbox_density_points_m2": None,
        }

    sample_count = min(max(2, int(sample_size)), len(points))
    sample_indices = np.linspace(0, len(points) - 1, sample_count, dtype=np.int64)
    local_density = _tangent_knn_density(points, sample_indices, k=knn)
    xy_extent = np.ptp(points[:, :2], axis=0)
    xy_area = float(xy_extent[0] * xy_extent[1])
    xy_density = float(len(points) / xy_area) if xy_area > 1e-12 else None
    if not len(local_density):
        return {
            "method": "tangent_knn",
            "knn": int(knn),
            "status": "insufficient_nonduplicate_neighbours",
            "sample_points": int(sample_count),
            "p10_points_m2": None,
            "p25_points_m2": None,
            "median_points_m2": None,
            "mean_points_m2": None,
            "p75_points_m2": None,
            "p90_points_m2": None,
            "xy_bbox_density_points_m2": xy_density,
        }

    return {
        "method": "tangent_knn",
        "knn": int(knn),
        "status": "estimated",
        "sample_points": int(sample_count),
        "p10_points_m2": float(np.quantile(local_density, 0.10)),
        "p25_points_m2": float(np.quantile(local_density, 0.25)),
        "median_points_m2": float(np.median(local_density)),
        "mean_points_m2": float(np.mean(local_density)),
        "p75_points_m2": float(np.quantile(local_density, 0.75)),
        "p90_points_m2": float(np.quantile(local_density, 0.90)),
        "xy_bbox_density_points_m2": xy_density,
    }


def assess_density(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Add the V2 density gate without aborting a run."""

    result = dict(report)
    target = float(config.get("target_raw_density_points_m2", 600.0))
    expected_retention = float(config.get("expected_retention_ratio", 0.70))
    required_p10 = float(config.get("min_local_density_p10_points_m2", target * 2.0 / 3.0))
    required_median = float(config.get("min_local_density_median_points_m2", target))
    result["target_raw_density_points_m2"] = target
    result["expected_retention_ratio"] = expected_retention
    result["required_p10_points_m2"] = required_p10
    result["required_median_points_m2"] = required_median
    p10 = result.get("p10_points_m2")
    median = result.get("median_points_m2")
    if p10 is None or median is None:
        result["standard_status"] = "DENSITY_NOT_ASSESSED"
    elif p10 >= required_p10 and median >= required_median:
        result["standard_status"] = "pass"
    else:
        result["standard_status"] = "DENSITY_TARGET_NOT_MET"
    return result


def discontinuity_density_metrics(
    *,
    point_count: int,
    footprint_area_m2: float | None,
    plane_count: int,
    trace_length_sum_m: float | None = None,
    scanline_length_m: float | None = None,
) -> dict[str, Any]:
    """Return distinct P10/P20/P21 observables instead of conflating them."""

    area = float(footprint_area_m2) if footprint_area_m2 is not None else float("nan")
    p20 = float(plane_count / area) if np.isfinite(area) and area > 0 else None
    p10 = (
        float(plane_count / float(scanline_length_m))
        if scanline_length_m is not None and scanline_length_m > 0
        else None
    )
    p21 = (
        float(trace_length_sum_m / area)
        if trace_length_sum_m is not None and np.isfinite(area) and area > 0
        else None
    )
    return {
        "p10_discontinuities_per_m": p10,
        "p20_planes_per_m2": p20,
        "p21_trace_length_per_m2": p21,
        "p10_reason": None if p10 is not None else "scanline_not_supplied",
        "p20_method": "plane_count_over_observed_footprint_bbox",
        "p21_reason": None if p21 is not None else "trace_module_unavailable",
        "point_count": int(point_count),
        "plane_count": int(plane_count),
        "footprint_area_m2": float(area) if np.isfinite(area) else None,
    }
