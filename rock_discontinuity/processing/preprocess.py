from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import KDTree


def voxel_downsample(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Return voxel centroids and an original-point -> centroid index map."""

    points = np.asarray(points, dtype=np.float64)
    if voxel_size <= 0:
        return points.copy(), np.arange(len(points), dtype=np.int64)
    origin = np.min(points, axis=0)
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    centroids = np.zeros((len(unique_keys), 3), dtype=np.float64)
    np.add.at(centroids, inverse, points)
    counts = np.bincount(inverse, minlength=len(unique_keys)).astype(np.float64)
    centroids /= counts[:, None]
    return centroids, inverse.astype(np.int64, copy=False)


def statistical_inlier_mask(
    points: np.ndarray,
    neighbors: int,
    std_ratio: float,
    *,
    batch_size: int = 50_000,
) -> np.ndarray:
    if len(points) <= max(3, neighbors):
        return np.ones(len(points), dtype=bool)
    tree = KDTree(points)
    mean_distance = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), max(1, int(batch_size))):
        stop = min(len(points), start + max(1, int(batch_size)))
        try:
            distances, _ = tree.query(points[start:stop], k=neighbors + 1, workers=-1)
        except TypeError:
            distances, _ = tree.query(points[start:stop], k=neighbors + 1)
        mean_distance[start:stop] = distances[:, 1:].mean(axis=1)
    threshold = mean_distance.mean() + std_ratio * mean_distance.std()
    return mean_distance <= threshold


def _estimate_local_features_single_scale(
    points: np.ndarray,
    *,
    radius: float,
    min_neighbors: int,
    knn: int,
    max_neighbors: int | None = None,
    batch_size: int = 50_000,
) -> dict[str, np.ndarray]:
    """Estimate normal, eigenvalues and planarity-like features by local PCA."""

    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    normals = np.full((count, 3), np.nan, dtype=np.float64)
    eigenvalues = np.full((count, 3), np.nan, dtype=np.float64)
    planarity = np.full(count, np.nan, dtype=np.float64)
    surface_variation = np.full(count, np.nan, dtype=np.float64)
    curvature = np.full(count, np.nan, dtype=np.float64)
    neighbor_count = np.zeros(count, dtype=np.int32)
    tree = KDTree(points)

    # Radius search is represented by a bounded nearest-neighbour query.  At
    # 600 points/m² and radius=0.15 m, knn=48 covers the expected local support
    # while avoiding a Python object list for every point.
    if radius > 0:
        # Keep enough neighbours to approximate the complete radius support.
        # The upper bound controls memory for dense 600 points/m² ROIs.
        requested_neighbors = max(int(knn), int(min_neighbors) * 2, int(max_neighbors or 96))
    else:
        requested_neighbors = max(3, int(knn))
    query_k = min(requested_neighbors, count)
    for start in range(0, count, max(1, int(batch_size))):
        stop = min(count, start + max(1, int(batch_size)))
        query_points = points[start:stop]
        try:
            distances, nearest = tree.query(query_points, k=query_k, workers=-1)
        except TypeError:
            distances, nearest = tree.query(query_points, k=query_k)
        if query_k == 1:
            distances = distances[:, None]
            nearest = nearest[:, None]
        distances = np.asarray(distances, dtype=np.float64)
        nearest = np.asarray(nearest, dtype=np.int64)
        valid = np.isfinite(distances) & (nearest >= 0) & (nearest < count)
        if radius > 0:
            valid &= distances <= radius
        neighbor_count[start:stop] = valid.sum(axis=1).astype(np.int32)
        enough = neighbor_count[start:stop] >= min_neighbors
        if not np.any(enough):
            continue

        safe_nearest = np.where(valid, nearest, 0)
        local = points[safe_nearest]
        weights = valid.astype(np.float64)[..., None]
        counts = np.maximum(neighbor_count[start:stop].astype(np.float64), 1.0)
        centers = (local * weights).sum(axis=1) / counts[:, None]
        centered = (local - centers[:, None, :]) * weights
        covariance = np.einsum("bki,bkj->bij", centered, centered) / counts[:, None, None]
        values, vectors = np.linalg.eigh(covariance)
        values = np.maximum(values, 0.0)
        valid_indices = np.flatnonzero(enough)
        batch_values = values[valid_indices]
        batch_vectors = vectors[valid_indices]
        batch_normals = batch_vectors[:, :, 0]
        batch_normals = np.where(batch_normals[:, 2:3] < 0, -batch_normals, batch_normals)
        output_indices = start + valid_indices
        eigenvalues[output_indices] = batch_values
        normals[output_indices] = batch_normals
        # Eigenvalues are ascending.  A locally planar neighbourhood has
        # lambda0 ~= 0 and lambda1 ~= lambda2, so planarity is
        # (lambda1 - lambda0) / lambda2.
        planarity[output_indices] = (batch_values[:, 1] - batch_values[:, 0]) / np.maximum(
            batch_values[:, 2], 1e-15
        )
        surface_variation[output_indices] = batch_values[:, 0] / np.maximum(batch_values.sum(axis=1), 1e-15)
        curvature[output_indices] = surface_variation[output_indices]

    return {
        "normals": normals,
        "eigenvalues": eigenvalues,
        "planarity": planarity,
        "surface_variation": surface_variation,
        "curvature": curvature,
        "neighbor_count": neighbor_count,
        "normal_scale_m": np.full(count, float(radius), dtype=np.float64),
        "normal_stability_deg": np.zeros(count, dtype=np.float64),
        "normal_stable": np.ones(count, dtype=bool),
        "normal_valid_scale_count": np.where(neighbor_count >= min_neighbors, 1, 0).astype(np.int16),
    }


def _estimate_local_features_multiscale(
    points: np.ndarray,
    *,
    radii: list[float],
    min_neighbors: int,
    knn: int,
    max_neighbors: int,
    batch_size: int,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Estimate one selected PCA scale and its cross-scale stability.

    One KDTree query is issued at the largest configured radius.  Nested
    neighborhoods are then evaluated from the same distance/index arrays so
    the multiscale pass does not create one tree per scale.
    """

    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    radii = sorted({float(radius) for radius in radii if float(radius) > 0.0})
    if not radii:
        raise ValueError("normal_multiscale.radii_m 至少需要一个正数")
    scale_count = len(radii)
    all_normals = np.full((scale_count, count, 3), np.nan, dtype=np.float64)
    all_eigenvalues = np.full((scale_count, count, 3), np.nan, dtype=np.float64)
    all_planarity = np.full((scale_count, count), np.nan, dtype=np.float64)
    all_surface = np.full((scale_count, count), np.nan, dtype=np.float64)
    all_entropy = np.full((scale_count, count), np.nan, dtype=np.float64)
    all_neighbors = np.zeros((scale_count, count), dtype=np.int32)
    tree = KDTree(points)
    query_k = min(max(3, int(max_neighbors), int(knn), int(min_neighbors) * 2), count)
    max_radius = max(radii)

    for start in range(0, count, max(1, int(batch_size))):
        stop = min(count, start + max(1, int(batch_size)))
        try:
            distances, nearest = tree.query(points[start:stop], k=query_k, workers=-1)
        except TypeError:
            distances, nearest = tree.query(points[start:stop], k=query_k)
        if query_k == 1:
            distances = distances[:, None]
            nearest = nearest[:, None]
        distances = np.asarray(distances, dtype=np.float64)
        nearest = np.asarray(nearest, dtype=np.int64)
        valid_base = np.isfinite(distances) & (nearest >= 0) & (nearest < count)
        valid_base &= distances <= max_radius
        safe_nearest = np.where(valid_base, nearest, 0)
        local = points[safe_nearest]

        for scale_index, radius in enumerate(radii):
            valid = valid_base & (distances <= radius)
            counts = valid.sum(axis=1).astype(np.int32)
            all_neighbors[scale_index, start:stop] = counts
            enough = counts >= int(min_neighbors)
            if not np.any(enough):
                continue
            weights = valid.astype(np.float64)[..., None]
            safe_counts = np.maximum(counts.astype(np.float64), 1.0)
            centers = (local * weights).sum(axis=1) / safe_counts[:, None]
            centered = (local - centers[:, None, :]) * weights
            covariance = np.einsum("bki,bkj->bij", centered, centered) / safe_counts[:, None, None]
            values, vectors = np.linalg.eigh(covariance)
            values = np.maximum(values, 0.0)
            valid_indices = np.flatnonzero(enough)
            batch_values = values[valid_indices]
            batch_vectors = vectors[valid_indices]
            batch_normals = batch_vectors[:, :, 0]
            batch_normals = np.where(batch_normals[:, 2:3] < 0, -batch_normals, batch_normals)
            output_indices = start + valid_indices
            all_eigenvalues[scale_index, output_indices] = batch_values
            all_normals[scale_index, output_indices] = batch_normals
            all_planarity[scale_index, output_indices] = (
                (batch_values[:, 1] - batch_values[:, 0])
                / np.maximum(batch_values[:, 2], 1e-15)
            )
            all_surface[scale_index, output_indices] = (
                batch_values[:, 0] / np.maximum(batch_values.sum(axis=1), 1e-15)
            )
            normalized = batch_values / np.maximum(batch_values.sum(axis=1, keepdims=True), 1e-15)
            all_entropy[scale_index, output_indices] = -np.sum(
                normalized * np.log(np.maximum(normalized, 1e-15)), axis=1
            )

    min_planarity = float(config.get("selection_min_planarity", 0.35))
    valid = (
        np.isfinite(all_entropy)
        & np.isfinite(all_planarity)
        & (all_planarity >= min_planarity)
    )
    # Eigenentropy selects a geometrically stable scale, with a planarity
    # penalty preventing a one-dimensional/edge neighborhood from winning.
    scores = all_entropy + float(config.get("planarity_penalty", 0.75)) * (
        1.0 - np.clip(np.nan_to_num(all_planarity, nan=0.0), 0.0, 1.0)
    ) + float(config.get("surface_penalty", 0.50)) * np.clip(
        np.nan_to_num(all_surface, nan=1.0), 0.0, 1.0
    )
    scores = np.where(valid, scores, np.inf)
    entropy_index = np.argmin(scores, axis=0)
    selected_index = entropy_index.copy()

    # For plane-boundary preservation, prefer the smallest scale whose normal
    # agrees with the immediately larger scale.  This avoids the common
    # failure mode where an entropy criterion selects the largest support for
    # nearly every point and consequently smooths across a narrow joint.
    # Points without an adjacent stable pair keep the entropy-selected scale.
    scale_normals_valid = np.isfinite(all_normals).all(axis=2)
    selection = str(config.get("selection", "smallest_adjacent_stable"))
    adjacent_stable = np.zeros((max(0, scale_count - 1), count), dtype=bool)
    if scale_count >= 2:
        adjacent_dot = np.abs(np.einsum("sni,sni->sn", all_normals[:-1], all_normals[1:]))
        adjacent_angle = np.degrees(np.arccos(np.clip(adjacent_dot, 0.0, 1.0)))
        adjacent_stable = (
            valid[:-1]
            & valid[1:]
            & scale_normals_valid[:-1]
            & scale_normals_valid[1:]
            & (adjacent_angle <= float(config.get("stable_angle_deg", 5.0)))
        )
    if selection == "smallest_adjacent_stable" and adjacent_stable.shape[0] > 0:
        has_adjacent_stable = np.any(adjacent_stable, axis=0)
        first_stable_pair = np.argmax(adjacent_stable, axis=0)
        selected_index[has_adjacent_stable] = first_stable_pair[has_adjacent_stable]
    selected_score = scores[selected_index, np.arange(count)]
    selected_valid = np.isfinite(selected_score)
    point_indices = np.arange(count)
    normals = all_normals[selected_index, point_indices]
    eigenvalues = all_eigenvalues[selected_index, point_indices]
    planarity = all_planarity[selected_index, point_indices]
    surface = all_surface[selected_index, point_indices]
    neighbors = all_neighbors[selected_index, point_indices]
    normal_scale = np.asarray(radii, dtype=np.float64)[selected_index]

    dot_to_selected = np.abs(np.einsum("sni,ni->sn", all_normals, normals))
    angle_to_selected = np.degrees(np.arccos(np.clip(dot_to_selected, 0.0, 1.0)))
    stability_mask = scale_normals_valid & selected_valid[None, :]
    stability_angles = np.where(stability_mask, angle_to_selected, -np.inf)
    normal_stability = np.max(stability_angles, axis=0)
    normal_stability[~np.any(stability_mask, axis=0)] = np.nan
    valid_scale_count = scale_normals_valid.sum(axis=0).astype(np.int16)
    min_stable_scales = int(config.get("min_stable_scales", 2))
    stable_pair_for_selected = np.zeros(count, dtype=bool)
    for pair_index in range(adjacent_stable.shape[0]):
        selected_pair = (selected_index == pair_index) | (selected_index == pair_index + 1)
        stable_pair_for_selected |= selected_pair & adjacent_stable[pair_index]
    normal_stable = selected_valid & (valid_scale_count >= min_stable_scales) & stable_pair_for_selected
    return {
        "normals": normals,
        "eigenvalues": eigenvalues,
        "planarity": planarity,
        "surface_variation": surface,
        "curvature": surface,
        "neighbor_count": neighbors,
        "normal_scale_m": np.where(selected_valid, normal_scale, np.nan),
        "normal_stability_deg": normal_stability,
        "normal_stable": normal_stable,
        "normal_valid_scale_count": valid_scale_count,
        "multiscale_entropy": selected_score,
        "multiscale_scale_values": np.asarray(radii, dtype=np.float64),
    }


def estimate_local_features(
    points: np.ndarray,
    *,
    radius: float,
    min_neighbors: int,
    knn: int,
    max_neighbors: int | None = None,
    batch_size: int = 50_000,
    multiscale_config: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Estimate local PCA features, optionally selecting a stable scale."""

    multiscale_config = multiscale_config or {}
    if bool(multiscale_config.get("enabled", False)):
        return _estimate_local_features_multiscale(
            points,
            radii=[float(value) for value in multiscale_config.get("radii_m", [radius])],
            min_neighbors=min_neighbors,
            knn=knn,
            max_neighbors=int(multiscale_config.get("max_neighbors", max_neighbors or 96)),
            batch_size=batch_size,
            config=multiscale_config,
        )
    return _estimate_local_features_single_scale(
        points,
        radius=radius,
        min_neighbors=min_neighbors,
        knn=knn,
        max_neighbors=max_neighbors,
        batch_size=batch_size,
    )


def select_planar_candidates(features: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    planarity = features["planarity"]
    surface = features.get("curvature", features["surface_variation"])
    neighbors = features["neighbor_count"]
    return (
        np.isfinite(planarity)
        & np.isfinite(surface)
        & np.isfinite(features["normals"]).all(axis=1)
        & (planarity >= float(config["min_planarity"]))
        & (surface <= float(config["max_surface_variation"]))
        & (neighbors >= int(config.get("min_neighbors", 0)))
    )
