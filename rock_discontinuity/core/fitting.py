from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import geometry_metrics, project_to_plane
from .models import PlaneInstance


@dataclass
class PlaneFit:
    normal: np.ndarray
    d: float
    centroid: np.ndarray
    inlier_mask: np.ndarray


def orient_normal(normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-15)
    if normal[2] < 0:
        normal = -normal
    return normal


def fit_tls(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    centroid = np.asarray(points, dtype=np.float64).mean(axis=0)
    centered = np.asarray(points, dtype=np.float64) - centroid
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    normal = orient_normal(vectors[-1])
    return normal, float(-np.dot(normal, centroid)), centroid


def plane_residuals(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    return np.abs(np.asarray(points) @ np.asarray(normal) + float(d))


def ransac_plane(
    points: np.ndarray,
    *,
    distance_threshold: float,
    iterations: int,
    seed: int = 42,
) -> PlaneFit | None:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_count = -1
    best_error = float("inf")
    for _ in range(max(1, int(iterations))):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            continue
        normal = orient_normal(normal / norm)
        d = float(-np.dot(normal, sample[0]))
        residual = plane_residuals(points, normal, d)
        mask = residual <= distance_threshold
        count = int(mask.sum())
        error = float(residual[mask].mean()) if count else float("inf")
        if count > best_count or (count == best_count and error < best_error):
            best_mask = mask
            best_count = count
            best_error = error
    if best_mask is None or best_count < 3:
        return None
    normal, d, centroid = fit_tls(points[best_mask])
    refined_mask = plane_residuals(points, normal, d) <= distance_threshold
    if int(refined_mask.sum()) >= 3:
        normal, d, centroid = fit_tls(points[refined_mask])
    return PlaneFit(normal, d, centroid, refined_mask)


def dip_and_dip_direction(normal: np.ndarray) -> tuple[float, float]:
    normal = orient_normal(normal)
    dip = float(np.degrees(np.arctan2(np.hypot(normal[0], normal[1]), normal[2])))
    dip_direction = float((np.degrees(np.arctan2(normal[0], normal[1])) + 360.0) % 360.0)
    return dip_direction, dip


def confidence_score(
    *,
    point_count: int,
    area: float,
    planarity: float,
    inlier_ratio: float,
    rms: float,
    normal_dispersion: float,
    boundary_completeness: float,
    config: dict[str, Any],
) -> float:
    min_points = max(1, int(config["min_points"]))
    min_area = max(1e-12, float(config["min_area"]))
    max_rms = max(1e-12, float(config["max_rms"]))
    angle_limit = max(1.0, float(config.get("normal_dispersion_limit_deg", 15.0)))
    scores = np.array(
        [
            min(1.0, point_count / min_points),
            min(1.0, max(0.0, area) / min_area),
            np.clip(planarity, 0.0, 1.0),
            np.clip(inlier_ratio, 0.0, 1.0),
            np.clip(1.0 - rms / max_rms, 0.0, 1.0),
            np.clip(1.0 - normal_dispersion / angle_limit, 0.0, 1.0),
            np.clip(boundary_completeness, 0.0, 1.0),
        ]
    )
    weights = np.array([0.15, 0.10, 0.15, 0.20, 0.15, 0.15, 0.10])
    return float(np.clip(np.dot(scores, weights), 0.0, 1.0))


def quality_grade(score: float, config: dict[str, Any]) -> str:
    grades = config.get("quality_grades", {})
    grade_a = float(grades.get("A", 0.85))
    grade_b = float(grades.get("B", 0.70))
    grade_c = float(grades.get("C", 0.55))
    if score >= grade_a:
        return "A"
    if score >= grade_b:
        return "B"
    if score >= grade_c:
        return "C"
    return "Rejected"


def fit_plane_instance(
    points: np.ndarray,
    normals: np.ndarray,
    planarity: np.ndarray,
    indices: np.ndarray,
    *,
    raw_set_label: int,
    config: dict[str, Any],
    boundary_config: dict[str, Any],
    target_config: dict[str, Any] | None = None,
    quality_config: dict[str, Any] | None = None,
    source_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    plane_id: str = "",
) -> tuple[PlaneInstance | None, str | None]:
    cluster_points = points[indices]
    fit = ransac_plane(
        cluster_points,
        distance_threshold=float(config["ransac_distance"]),
        iterations=int(config["ransac_iterations"]),
        seed=int(config.get("random_seed", 42)),
    )
    if fit is None:
        return None, "ransac_failed"
    inlier_indices = indices[fit.inlier_mask]
    min_effective_points = int(config["min_points"])
    if quality_config is not None:
        min_effective_points = max(
            min_effective_points,
            int(quality_config.get("min_effective_points", min_effective_points)),
        )
    if len(inlier_indices) < min_effective_points:
        return None, "below_min_points"
    residuals = plane_residuals(points[inlier_indices], fit.normal, fit.d)
    inlier_ratio = float(np.mean(fit.inlier_mask))
    if quality_config is not None and inlier_ratio < float(quality_config.get("min_inlier_ratio", 0.0)):
        return None, "below_min_inlier_ratio"
    rms = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(residuals))
    p95 = float(np.quantile(residuals, 0.95))
    projected = project_to_plane(points[inlier_indices], fit.centroid, fit.normal)
    metrics = geometry_metrics(projected, boundary_config)
    bbox_min = np.min(points[inlier_indices], axis=0)
    bbox_max = np.max(points[inlier_indices], axis=0)
    min_area = float(config["min_area"])
    if target_config is not None:
        min_area = max(min_area, float(target_config.get("min_area_m2", min_area)))
    if metrics["area"] < min_area:
        return None, "below_min_area"
    if target_config is not None and metrics["minor_extent"] < float(target_config.get("min_minor_extent_m", 0.0)):
        return None, "below_min_minor_extent"
    if rms > float(config["max_rms"]):
        return None, "above_max_rms"

    local_normals = normals[inlier_indices]
    dots = np.clip(np.abs(local_normals @ fit.normal), 0.0, 1.0)
    normal_dispersion = float(np.degrees(np.arccos(dots)).mean()) if len(dots) else 90.0
    mean_planarity = float(np.nanmean(planarity[inlier_indices]))
    dip_direction, dip = dip_and_dip_direction(fit.normal)
    edge_censored = False
    if source_bounds is not None and len(inlier_indices):
        lower, upper = source_bounds
        margin = float(boundary_config.get("edge_margin_m", 0.15))
        inlier_lower = np.min(points[inlier_indices], axis=0)
        inlier_upper = np.max(points[inlier_indices], axis=0)
        edge_censored = bool(
            np.any(inlier_lower[:2] <= lower[:2] + margin)
            or np.any(inlier_upper[:2] >= upper[:2] - margin)
        )
    score_config = dict(config)
    score_config["min_points"] = min_effective_points
    score_config["min_area"] = min_area
    confidence = confidence_score(
        point_count=len(inlier_indices),
        area=metrics["area"],
        planarity=mean_planarity,
        inlier_ratio=inlier_ratio,
        rms=rms,
        normal_dispersion=normal_dispersion,
        boundary_completeness=metrics["boundary_completeness"],
        config=score_config,
    )
    if quality_config is not None and confidence < float(quality_config.get("min_confidence", 0.0)):
        return None, "below_min_confidence"
    return (
        PlaneInstance(
            plane_id=plane_id,
            raw_set_label=raw_set_label,
            point_indices=inlier_indices,
            candidate_point_count=len(indices),
            normal=fit.normal,
            d=fit.d,
            centroid=fit.centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            dip_direction=dip_direction,
            dip=dip,
            area=float(metrics["area"]),
            apparent_persistence=float(metrics["apparent_persistence"]),
            major_extent=float(metrics["major_extent"]),
            minor_extent=float(metrics["minor_extent"]),
            trace_length=None,
            rms=rms,
            mae=mae,
            p95_residual=p95,
            inlier_ratio=inlier_ratio,
            planarity=mean_planarity,
            normal_dispersion=normal_dispersion,
            boundary_completeness=float(metrics["boundary_completeness"]),
            edge_censored=edge_censored,
            confidence=confidence,
            quality_score=confidence,
            quality_grade=quality_grade(confidence, quality_config or config),
            near_vertical=bool(dip >= 85.0),
            data_gap_censored=False,
            trace_verified=False,
            aperture_available=False,
            aperture_reason="high_resolution_mesh_or_image_required",
        ),
        None,
    )
