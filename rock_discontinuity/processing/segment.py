from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from ..core.models import SegmentationResult


def _fit_region_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Fit a local TLS plane and return normal, d, centroid and RMS."""

    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(len(points), 1)
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, 0]
    normal /= max(float(np.linalg.norm(normal)), 1e-15)
    if normal[2] < 0:
        normal = -normal
    d = float(-np.dot(normal, centroid))
    residuals = np.abs(points @ normal + d)
    rms = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else float("inf")
    return normal, d, centroid, rms


def _median_nearest_spacing(points: np.ndarray, sample_size: int = 10_000) -> float:
    if len(points) < 2:
        return float("nan")
    sample_count = min(len(points), max(2, int(sample_size)))
    indices = np.linspace(0, len(points) - 1, sample_count, dtype=np.int64)
    tree = KDTree(points)
    distances, _ = tree.query(points[indices], k=2)
    nearest = np.asarray(distances[:, 1], dtype=np.float64)
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-9)]
    return float(np.median(nearest)) if len(nearest) else float("nan")


def _noise_aware_distance_threshold(
    points: np.ndarray,
    features: dict[str, np.ndarray],
    candidate_indices: np.ndarray,
    config: dict[str, Any],
) -> float:
    spacing = _median_nearest_spacing(points)
    eigenvalues = np.asarray(features.get("eigenvalues", np.empty((0, 3))), dtype=np.float64)
    if len(eigenvalues):
        smallest = eigenvalues[candidate_indices, 0]
        smallest = smallest[np.isfinite(smallest) & (smallest >= 0)]
        sigma_noise = float(np.median(np.sqrt(smallest))) if len(smallest) else 0.0
    else:
        sigma_noise = 0.0
    minimum = float(config.get("distance_base_m", config.get("min_distance_m", 0.02)))
    maximum = float(config.get("distance_max_m", config.get("max_distance_m", 0.05)))
    spacing_term = 0.5 * spacing if np.isfinite(spacing) else minimum
    threshold = max(minimum, spacing_term, 2.5 * sigma_noise)
    return float(np.clip(threshold, minimum, max(maximum, minimum)))


def _dynamic_region_thresholds(
    planarity: np.ndarray,
    curvature: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-point angle, distance and curvature gates.

    High-planarity/low-curvature points use the strict base thresholds.  A
    noisier point can use the configured upper bound, but it can never bypass
    the candidate curvature gate.  This is the V2 dynamic region-growing
    rule and keeps a curved surface from being connected transitively into one
    plane.
    """

    base_angle = float(config.get("normal_angle_base_deg", config.get("normal_angle_deg", 8.0)))
    max_angle = float(config.get("normal_angle_max_deg", max(base_angle, config.get("normal_angle_deg", 8.0))))
    base_distance = float(config.get("distance_base_m", config.get("min_distance_m", 0.02)))
    max_distance = float(config.get("distance_max_m", config.get("max_distance_m", 0.05)))
    max_curvature = float(config.get("max_curvature", 0.08))
    safe_planarity = np.clip(np.nan_to_num(planarity, nan=0.0), 0.0, 1.0)
    safe_curvature = np.clip(np.nan_to_num(curvature, nan=max_curvature), 0.0, max(max_curvature, 1e-12))
    looseness = np.clip((1.0 - safe_planarity) + safe_curvature / max(max_curvature, 1e-12), 0.0, 1.0)
    angles = base_angle + (max_angle - base_angle) * looseness
    distances = base_distance + (max_distance - base_distance) * np.clip(
        safe_curvature / max(max_curvature, 1e-12), 0.0, 1.0
    )
    return angles, distances, safe_curvature


def planarity_seeded_region_growing(
    points: np.ndarray,
    normals: np.ndarray,
    planarity: np.ndarray,
    candidate_mask: np.ndarray,
    features: dict[str, np.ndarray],
    config: dict[str, Any],
    min_instance_points: int,
) -> SegmentationResult:
    """Segment finite planar patches using planarity-seeded region growing.

    A region grows only when a point agrees with the current TLS plane in both
    normal direction and point-to-plane distance.  The region plane is refit
    periodically, which prevents the fixed seed plane from absorbing curved
    or neighbouring surfaces.
    """

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    planarity = np.asarray(planarity, dtype=np.float64)
    candidate_indices = np.flatnonzero(candidate_mask).astype(np.int64)
    orientation_labels = np.full(len(points), -1, dtype=np.int32)
    spatial_labels = np.full(len(points), -1, dtype=np.int32)
    if len(candidate_indices) < 3:
        return SegmentationResult(candidate_indices, orientation_labels, spatial_labels, [])

    candidate_points = points[candidate_indices]
    candidate_normals = normals[candidate_indices]
    candidate_planarity = planarity[candidate_indices]
    candidate_curvature = np.asarray(
        features.get("curvature", features.get("surface_variation")),
        dtype=np.float64,
    )[candidate_indices]
    candidate_stable = np.asarray(
        features.get("normal_stable", np.ones(len(points), dtype=bool)),
        dtype=bool,
    )[candidate_indices]
    tree = KDTree(candidate_points)
    radius = float(config.get("radius_m", 0.15))
    angle_thresholds, distance_thresholds, candidate_curvature = _dynamic_region_thresholds(
        candidate_planarity,
        candidate_curvature,
        config,
    )
    graph_angle = float(np.max(angle_thresholds)) if len(angle_thresholds) else float(config.get("normal_angle_deg", 8.0))
    normal_cosine = float(np.cos(np.deg2rad(graph_angle)))
    distance_threshold = _noise_aware_distance_threshold(points, features, candidate_indices, config)
    refit_every = max(3, int(config.get("refit_every_points", 48)))
    max_seed_points = max(3, int(config.get("max_seed_points", 64)))
    min_points = max(3, int(min_instance_points))

    try:
        pairs = tree.query_pairs(radius, output_type="ndarray")
    except TypeError:
        pairs = np.asarray(list(tree.query_pairs(radius)), dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    max_pairs = int(config.get("max_pairs", 50_000_000))
    if len(pairs) > max_pairs:
        raise ValueError(
            f"区域生长邻接边 {len(pairs)} 条超过 max_pairs={max_pairs}；"
            "请缩小 ROI、提高 voxel_size 或减小 region_growing.radius_m"
        )
    if len(pairs):
        normal_pairs = np.abs(
            np.sum(candidate_normals[pairs[:, 0]] * candidate_normals[pairs[:, 1]], axis=1)
        ) >= normal_cosine
        pairs = pairs[normal_pairs]
    if len(pairs):
        rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
        cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
        adjacency = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(len(candidate_indices), len(candidate_indices)),
        ).tocsr()
    else:
        adjacency = coo_matrix(
            (len(candidate_indices), len(candidate_indices)), dtype=np.uint8
        ).tocsr()

    finite_planarity = np.where(np.isfinite(candidate_planarity), candidate_planarity, -np.inf)
    seed_quality = finite_planarity - np.nan_to_num(candidate_curvature, nan=1.0)
    seed_order = np.argsort(-seed_quality, kind="stable")
    assigned = np.zeros(len(candidate_indices), dtype=bool)
    instances: list[tuple[int, np.ndarray]] = []

    for seed in seed_order:
        seed = int(seed)
        if assigned[seed]:
            continue
        if bool(config.get("require_stable_seed", False)) and not candidate_stable[seed]:
            # A scale-unstable point can still be absorbed by a stable region,
            # but it must not start a new plane instance at a boundary.
            continue
        seed_start = int(adjacency.indptr[seed])
        seed_stop = int(adjacency.indptr[seed + 1])
        seed_neighbours = adjacency.indices[seed_start:seed_stop]
        if len(seed_neighbours) < 3:
            assigned[seed] = True
            continue
        seed_normal = candidate_normals[seed]
        seed_neighbours = seed_neighbours[
            np.argsort(-finite_planarity[seed_neighbours], kind="stable")[:max_seed_points]
        ]
        normal, d, _, _ = _fit_region_plane(candidate_points[seed_neighbours])
        region: list[int] = [seed]
        assigned[seed] = True
        queue: deque[int] = deque([seed])
        next_refit = refit_every

        while queue:
            current = queue.popleft()
            start = int(adjacency.indptr[current])
            stop = int(adjacency.indptr[current + 1])
            neighbours = adjacency.indices[start:stop]
            if not len(neighbours):
                continue
            neighbours = neighbours[~assigned[neighbours]]
            if not len(neighbours):
                continue
            normal_dots = np.abs(candidate_normals[neighbours] @ normal)
            normal_ok = normal_dots >= np.cos(np.deg2rad(angle_thresholds[neighbours]))
            distances = np.abs(candidate_points[neighbours] @ normal + d)
            curvature_ok = candidate_curvature[neighbours] <= float(config.get("max_curvature", 0.08))
            accepted = neighbours[
                normal_ok
                & curvature_ok
                & (distances <= np.minimum(distance_thresholds[neighbours], distance_threshold))
            ]
            if not len(accepted):
                continue
            assigned[accepted] = True
            region.extend(int(value) for value in accepted)
            queue.extend(int(value) for value in accepted)
            if len(region) >= next_refit:
                normal, d, _, _ = _fit_region_plane(candidate_points[np.asarray(region, dtype=np.int64)])
                next_refit = ((len(region) // refit_every) + 1) * refit_every

        if len(region) < min_points:
            continue
        region_indices = candidate_indices[np.asarray(region, dtype=np.int64)]
        label = len(instances)
        orientation_labels[region_indices] = label
        spatial_labels[region_indices] = label
        instances.append((label, region_indices))

    return SegmentationResult(
        candidate_indices,
        orientation_labels,
        spatial_labels,
        instances,
        distance_threshold,
    )


def _dbscan_labels(values: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    if len(values) == 0:
        return np.empty(0, dtype=np.int32)
    return DBSCAN(
        eps=float(eps),
        min_samples=int(min_samples),
        metric="euclidean",
        n_jobs=-1,
    ).fit_predict(values).astype(np.int32)


def _angular_grid_labels(
    normals: np.ndarray,
    angle_eps_deg: float,
    min_samples: int,
) -> np.ndarray:
    """Cluster unit normals without a point-by-point dense radius graph.

    Occupied angular bins are clustered by a small sparse graph. Each bin is
    weighted by its point count, so a direction spread across neighbouring bins
    can still satisfy the DBSCAN-like ``min_samples`` requirement.
    """

    normals = np.asarray(normals, dtype=np.float64)
    if len(normals) == 0:
        return np.empty(0, dtype=np.int32)
    chord = max(2.0 * np.sin(np.deg2rad(float(angle_eps_deg)) / 2.0), 1e-6)
    bin_size = chord / 3.0
    bin_keys = np.floor((normals + 1.0) / bin_size).astype(np.int32)
    unique_keys, inverse, counts = np.unique(
        bin_keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    centers = -1.0 + (unique_keys.astype(np.float64) + 0.5) * bin_size
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-15)

    if len(centers) == 1:
        return (
            np.zeros(len(normals), dtype=np.int32)
            if len(normals) >= min_samples
            else np.full(len(normals), -1, dtype=np.int32)
        )

    center_tree = KDTree(centers)
    # The bin diagonal is included so points close to an angular boundary are
    # not split merely because of quantisation.
    graph_radius = chord + np.sqrt(3.0) * bin_size
    try:
        bin_pairs = center_tree.query_pairs(graph_radius, output_type="ndarray")
    except TypeError:
        bin_pairs = np.asarray(list(center_tree.query_pairs(graph_radius)), dtype=np.int64)
    bin_pairs = np.asarray(bin_pairs, dtype=np.int64).reshape(-1, 2)

    local_counts = counts.astype(np.int64, copy=True)
    if len(bin_pairs):
        np.add.at(local_counts, bin_pairs[:, 0], counts[bin_pairs[:, 1]])
        np.add.at(local_counts, bin_pairs[:, 1], counts[bin_pairs[:, 0]])
    core_bins = local_counts >= int(min_samples)
    if not np.any(core_bins):
        return np.full(len(normals), -1, dtype=np.int32)

    core_pairs = bin_pairs[core_bins[bin_pairs[:, 0]] & core_bins[bin_pairs[:, 1]]]
    core_indices = np.flatnonzero(core_bins)
    core_lookup = np.full(len(centers), -1, dtype=np.int64)
    core_lookup[core_indices] = np.arange(len(core_indices), dtype=np.int64)
    if len(core_pairs):
        rows = core_lookup[core_pairs[:, 0]]
        cols = core_lookup[core_pairs[:, 1]]
        adjacency = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(len(core_indices), len(core_indices)),
        )
        adjacency = adjacency + adjacency.T
        _, component_labels = connected_components(adjacency, directed=False)
    else:
        component_labels = np.arange(len(core_indices), dtype=np.int32)

    bin_labels = np.full(len(centers), -1, dtype=np.int32)
    bin_labels[core_indices] = component_labels.astype(np.int32)
    # Assign non-core bins to the nearest core direction when it lies inside
    # the same angular support. This is the DBSCAN border-point rule at bin
    # resolution.
    core_tree = KDTree(centers[core_indices])
    non_core = np.flatnonzero(~core_bins)
    if len(non_core):
        distances, nearest = core_tree.query(centers[non_core], k=1)
        border = distances <= graph_radius
        bin_labels[non_core[border]] = component_labels[nearest[border]].astype(np.int32)
    return bin_labels[inverse]


def _sparse_spatial_dbscan(
    points: np.ndarray,
    normals: np.ndarray | None,
    eps: float,
    min_samples: int,
    max_pairs: int,
    normal_angle_eps_deg: float | None = None,
) -> np.ndarray:
    """DBSCAN-like spatial labels using a sparse core-point graph.

    A cheap neighbour-count query is performed before materialising pair edges,
    so a pathological ROI fails with a clear limit instead of exhausting RAM.
    """

    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    if count == 0:
        return np.empty(0, dtype=np.int32)
    if count < min_samples:
        return np.full(count, -1, dtype=np.int32)
    tree = KDTree(points)
    try:
        neighbour_counts = tree.query_ball_point(points, float(eps), return_length=True, workers=-1)
    except TypeError:
        try:
            neighbour_counts = tree.query_ball_point(points, float(eps), return_length=True)
        except TypeError:
            neighbour_counts = None
    if neighbour_counts is not None:
        estimated_pairs = max(0, (int(np.asarray(neighbour_counts, dtype=np.int64).sum()) - count) // 2)
        if estimated_pairs > int(max_pairs):
            raise ValueError(
                f"空间聚类邻接边预计 {estimated_pairs} 条，超过 max_pairs={max_pairs}；"
                "请缩小 ROI 或提高 voxel_size"
            )
    try:
        pairs = tree.query_pairs(float(eps), output_type="ndarray")
    except TypeError:
        pairs = np.asarray(list(tree.query_pairs(float(eps))), dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if normals is not None and len(pairs) and normal_angle_eps_deg is not None:
        normal_threshold = np.cos(np.deg2rad(float(normal_angle_eps_deg)))
        pairs = pairs[
            np.sum(normals[pairs[:, 0]] * normals[pairs[:, 1]], axis=1) >= normal_threshold
        ]
    degree = np.bincount(pairs.ravel(), minlength=count) if len(pairs) else np.zeros(count, dtype=np.int64)
    core = (degree + 1) >= int(min_samples)
    core_indices = np.flatnonzero(core)
    labels = np.full(count, -1, dtype=np.int32)
    if not len(core_indices):
        return labels

    core_lookup = np.full(count, -1, dtype=np.int64)
    core_lookup[core_indices] = np.arange(len(core_indices), dtype=np.int64)
    if len(pairs):
        core_pairs = pairs[core[pairs[:, 0]] & core[pairs[:, 1]]]
    else:
        core_pairs = pairs
    if len(core_pairs):
        rows = core_lookup[core_pairs[:, 0]]
        cols = core_lookup[core_pairs[:, 1]]
        adjacency = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(len(core_indices), len(core_indices)),
        )
        adjacency = adjacency + adjacency.T
        _, core_labels = connected_components(adjacency, directed=False)
    else:
        core_labels = np.arange(len(core_indices), dtype=np.int32)
    labels[core_indices] = core_labels.astype(np.int32)

    border_indices = np.flatnonzero(~core)
    if len(border_indices):
        core_tree = KDTree(points[core_indices])
        try:
            distances, nearest = core_tree.query(points[border_indices], k=1, workers=-1)
        except TypeError:
            distances, nearest = core_tree.query(points[border_indices], k=1)
        border = distances <= float(eps)
        labels[border_indices[border]] = core_labels[nearest[border]].astype(np.int32)
    return labels


def _orientation_labels(normals: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    method = str(config.get("method", "angular_grid"))
    if method == "dbscan":
        angle_eps = float(config["angle_eps_deg"])
        normal_eps = 2.0 * np.sin(np.deg2rad(angle_eps) / 2.0)
        return _dbscan_labels(normals, normal_eps, int(config["min_samples"]))
    return _angular_grid_labels(
        normals,
        float(config["angle_eps_deg"]),
        int(config["min_samples"]),
    )


def _spatial_labels(points: np.ndarray, normals: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    method = str(config.get("method", "sparse_dbscan"))
    if method == "dbscan":
        return _dbscan_labels(points, float(config["eps"]), int(config["min_samples"]))
    return _sparse_spatial_dbscan(
        points,
        normals,
        float(config["eps"]),
        int(config["min_samples"]),
        int(config.get("max_pairs", 50_000_000)),
        float(config["normal_angle_eps_deg"])
        if config.get("normal_angle_eps_deg") is not None
        else None,
    )


def segment_candidates(
    points: np.ndarray,
    normals: np.ndarray,
    candidate_mask: np.ndarray,
    features: dict[str, np.ndarray],
    orientation_config: dict[str, Any],
    spatial_config: dict[str, Any],
    min_instance_points: int,
    region_config: dict[str, Any] | None = None,
) -> SegmentationResult:
    region_config = region_config or {}
    if str(region_config.get("method", "planarity_seeded")) == "planarity_seeded":
        return planarity_seeded_region_growing(
            points,
            normals,
            features["planarity"],
            candidate_mask,
            features,
            region_config
            | {
                "radius_m": region_config.get("radius_m", spatial_config.get("eps", 0.15)),
                "max_pairs": region_config.get(
                    "max_pairs", spatial_config.get("max_pairs", 50_000_000)
                ),
            },
            min_instance_points,
        )

    candidate_indices = np.flatnonzero(candidate_mask).astype(np.int64)
    orientation_labels = np.full(len(points), -1, dtype=np.int32)
    spatial_labels = np.full(len(points), -1, dtype=np.int32)
    if len(candidate_indices) == 0:
        return SegmentationResult(candidate_indices, orientation_labels, spatial_labels, [])

    normal_labels = _orientation_labels(normals[candidate_indices], orientation_config)
    orientation_labels[candidate_indices] = normal_labels

    instances: list[tuple[int, np.ndarray]] = []
    next_spatial_label = 0
    for raw_set in sorted(set(int(label) for label in normal_labels if label >= 0)):
        members = candidate_indices[normal_labels == raw_set]
        if len(members) < min_instance_points:
            continue
        local_labels = _spatial_labels(points[members], normals[members], spatial_config)
        for local_label in sorted(set(int(label) for label in local_labels if label >= 0)):
            indices = members[local_labels == local_label]
            if len(indices) < min_instance_points:
                continue
            spatial_labels[indices] = next_spatial_label
            instances.append((raw_set, indices.astype(np.int64)))
            next_spatial_label += 1

    return SegmentationResult(candidate_indices, orientation_labels, spatial_labels, instances)
