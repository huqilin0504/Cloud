from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..core.fitting import fit_plane_instance, fit_tls, plane_residuals
from ..core.geometry import (
    footprint_gap_on_plane,
    footprint_rings_3d,
    normal_angle_deg as axial_normal_angle_deg,
    normalize_normal,
)
from ..core.models import PlaneInstance


def _normal(normal: np.ndarray) -> np.ndarray | None:
    return normalize_normal(normal)


def _spatial_gap_3d(plane_a: PlaneInstance, plane_b: PlaneInstance) -> float | None:
    lower_a = np.asarray(plane_a.bbox_min, dtype=np.float64)
    upper_a = np.asarray(plane_a.bbox_max, dtype=np.float64)
    lower_b = np.asarray(plane_b.bbox_min, dtype=np.float64)
    upper_b = np.asarray(plane_b.bbox_max, dtype=np.float64)
    if not np.all(np.isfinite(np.concatenate((lower_a, upper_a, lower_b, upper_b)))):
        return None
    gap = np.maximum(
        0.0,
        np.maximum(lower_a, lower_b) - np.minimum(upper_a, upper_b),
    )
    return float(np.linalg.norm(gap))


def planes_can_merge(
    plane_a: PlaneInstance,
    plane_b: PlaneInstance,
    *,
    normal_angle_deg: float,
    plane_offset_m: float,
    spatial_gap_m: float,
    max_rms_m: float | None = None,
    footprint_a: list[list[list[float]]] | None = None,
    footprint_b: list[list[list[float]]] | None = None,
) -> bool:
    """Check the three geometric conditions for two observed plane patches.

    Orientation alone is never sufficient. The patches must also describe
    the same fitted plane and have nearby three-dimensional footprints.
    """

    normal_a = _normal(plane_a.normal)
    normal_b = _normal(plane_b.normal)
    if normal_a is None or normal_b is None:
        return False
    angle = axial_normal_angle_deg(normal_a, normal_b)
    if angle is None or angle > float(normal_angle_deg):
        return False

    d_a = float(plane_a.d)
    d_b = float(plane_b.d)
    dot = float(np.dot(normal_a, normal_b))
    aligned_b = normal_b if dot >= 0.0 else -normal_b
    aligned_d_b = d_b if dot >= 0.0 else -d_b
    if abs(d_a - aligned_d_b) > float(plane_offset_m):
        return False

    # Use both plane equations at the opposite centroids.  This avoids
    # merging two almost-parallel planes only because their d values happen
    # to be close in a large ENU coordinate system.
    centroid_a = np.asarray(plane_a.centroid, dtype=np.float64)
    centroid_b = np.asarray(plane_b.centroid, dtype=np.float64)
    if abs(float(np.dot(normal_a, centroid_b) + d_a)) > float(plane_offset_m):
        return False
    if abs(float(np.dot(aligned_b, centroid_a) + aligned_d_b)) > float(plane_offset_m):
        return False

    if max_rms_m is not None:
        for rms in (plane_a.rms, plane_b.rms):
            if np.isfinite(rms) and float(rms) > float(max_rms_m):
                return False
    footprint_gap = footprint_gap_on_plane(
        footprint_a,
        footprint_b,
        0.5 * (centroid_a + centroid_b),
        normal_a,
    )
    if footprint_gap is not None:
        return footprint_gap <= float(spatial_gap_m)
    gap = _spatial_gap_3d(plane_a, plane_b)
    return gap is not None and gap <= float(spatial_gap_m)


def _combined_fit_is_valid(
    point_indices: np.ndarray,
    points: np.ndarray,
    member_planes: list[PlaneInstance],
    *,
    normal_angle_deg: float,
    max_rms_m: float,
) -> bool:
    if len(point_indices) < 3:
        return False
    fit_normal, fit_d, _ = fit_tls(points[point_indices])
    residuals = plane_residuals(points[point_indices], fit_normal, fit_d)
    rms = float(np.sqrt(np.mean(residuals**2)))
    if not np.isfinite(rms) or rms > float(max_rms_m):
        return False
    limit = float(np.cos(np.deg2rad(normal_angle_deg)))
    for plane in member_planes:
        normal = _normal(plane.normal)
        if normal is None or abs(float(np.dot(normal, fit_normal))) < limit:
            return False
    return True


def merge_plane_instances(
    planes: list[PlaneInstance],
    *,
    points: np.ndarray,
    normals: np.ndarray,
    planarity: np.ndarray,
    fit_config: dict[str, Any],
    boundary_config: dict[str, Any],
    target_config: dict[str, Any],
    quality_config: dict[str, Any],
    source_bounds: tuple[np.ndarray, np.ndarray] | None,
    merge_config: dict[str, Any],
) -> tuple[list[PlaneInstance], dict[int, int], dict[str, Any]]:
    """Merge geometrically adjacent plane instances and refit each union.

    The function is used for both an ROI and one whole-cloud tile.  A union
    is accepted only when the pair passes orientation/coplanarity/footprint
    gates and the combined point set still fits one plane.  If a component
    fails the final refit, its original instances are retained.
    """

    original_count = len(planes)
    identity = {index: index for index in range(original_count)}
    stats: dict[str, Any] = {
        "enabled": bool(merge_config.get("enabled", True)),
        "input_plane_instances": original_count,
        "candidate_pairs": 0,
        "accepted_pairs": 0,
        "merged_components": 0,
        "output_planes": original_count,
        "fallback_components": 0,
    }
    if original_count < 2 or not stats["enabled"]:
        return list(planes), identity, stats

    normal_angle_deg = float(merge_config.get("normal_angle_deg", 5.0))
    plane_offset_m = float(merge_config.get("plane_offset_m", 0.08))
    spatial_gap_m = float(merge_config.get("spatial_gap_m", 0.30))
    max_rms_m = float(
        merge_config.get("max_rms_m", fit_config.get("max_rms", 0.05))
    )
    parent = list(range(original_count))
    members: dict[int, list[int]] = {index: [index] for index in range(original_count)}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> int:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return left_root
        if len(members[left_root]) < len(members[right_root]):
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        members[left_root].extend(members.pop(right_root))
        return left_root

    point_arrays = [np.asarray(plane.point_indices, dtype=np.int64) for plane in planes]
    footprint_config = merge_config.get("footprint", boundary_config)
    footprints = [
        footprint_rings_3d(
            points[indices],
            plane.centroid,
            plane.normal,
            alpha=footprint_config.get("alpha", "auto"),
            max_points=int(footprint_config.get("max_points", 2_000)),
        )
        for plane, indices in zip(planes, point_arrays)
    ]

    def components_are_compatible(left: int, right: int) -> bool:
        if str(merge_config.get("linkage", "complete")).lower() != "complete":
            return True
        left_members = members[find(left)]
        right_members = members[find(right)]
        return all(
            planes_can_merge(
                planes[a],
                planes[b],
                normal_angle_deg=normal_angle_deg,
                plane_offset_m=plane_offset_m,
                spatial_gap_m=spatial_gap_m,
                max_rms_m=max_rms_m,
                footprint_a=footprints[a],
                footprint_b=footprints[b],
            )
            for a in left_members
            for b in right_members
        )

    growth_ratio = float(merge_config.get("max_rms_growth_ratio", 0.0))
    minimum_merged_rms = float(merge_config.get("min_merged_rms_m", 0.0))

    def allowed_combined_rms(component_planes: list[PlaneInstance]) -> float:
        if growth_ratio <= 0.0:
            return max_rms_m
        child_rms = [float(plane.rms) for plane in component_planes if np.isfinite(plane.rms)]
        if not child_rms:
            return max_rms_m
        return min(max_rms_m, max(minimum_merged_rms, max(child_rms) * growth_ratio))

    for left in range(original_count):
        for right in range(left + 1, original_count):
            if not planes_can_merge(
                planes[left],
                planes[right],
                normal_angle_deg=normal_angle_deg,
                plane_offset_m=plane_offset_m,
                spatial_gap_m=spatial_gap_m,
                max_rms_m=max_rms_m,
                footprint_a=footprints[left],
                footprint_b=footprints[right],
            ):
                continue
            stats["candidate_pairs"] += 1
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                continue
            if not components_are_compatible(left, right):
                continue
            combined_indices = np.unique(
                np.concatenate(
                    [
                        np.concatenate([point_arrays[index] for index in members[left_root]]),
                        np.concatenate([point_arrays[index] for index in members[right_root]]),
                    ]
                )
            )
            component_planes = [
                planes[index]
                for index in members[left_root] + members[right_root]
            ]
            if not _combined_fit_is_valid(
                combined_indices,
                points,
                component_planes,
                normal_angle_deg=normal_angle_deg,
                max_rms_m=allowed_combined_rms(component_planes),
            ):
                continue
            union(left_root, right_root)
            stats["accepted_pairs"] += 1

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(original_count):
        components[find(index)].append(index)

    merged_planes: list[PlaneInstance] = []
    old_to_new: dict[int, int] = {}
    for component in sorted(components.values(), key=lambda value: min(value)):
        if len(component) == 1:
            new_index = len(merged_planes)
            merged_planes.append(planes[component[0]])
            old_to_new[component[0]] = new_index
            continue

        combined_indices = np.unique(
            np.concatenate([point_arrays[index] for index in component])
        )
        merged_plane, _ = fit_plane_instance(
            points,
            normals,
            planarity,
            combined_indices,
            raw_set_label=min(component),
            config=fit_config,
            boundary_config=boundary_config,
            target_config=target_config,
            quality_config=quality_config,
            source_bounds=source_bounds,
        )
        if merged_plane is None:
            stats["fallback_components"] += 1
            for index in component:
                new_index = len(merged_planes)
                merged_planes.append(planes[index])
                old_to_new[index] = new_index
            continue

        new_index = len(merged_planes)
        merged_planes.append(merged_plane)
        for index in component:
            old_to_new[index] = new_index
        stats["merged_components"] += 1

    stats["output_planes"] = len(merged_planes)

    hierarchical = merge_config.get("hierarchical", {})
    if bool(hierarchical.get("enabled", False)) and len(merged_planes) > 1:
        second_config = dict(merge_config)
        second_config["hierarchical"] = {"enabled": False}
        second_config["linkage"] = hierarchical.get("linkage", "complete")
        second_config["normal_angle_deg"] = float(
            hierarchical.get("normal_angle_deg", normal_angle_deg)
        )
        second_config["plane_offset_m"] = float(
            hierarchical.get("plane_offset_m", plane_offset_m)
        )
        second_config["spatial_gap_m"] = float(
            hierarchical.get("spatial_gap_m", spatial_gap_m)
        )
        second_config["max_rms_m"] = float(
            hierarchical.get("max_rms_m", max_rms_m)
        )
        second_config["max_rms_growth_ratio"] = float(
            hierarchical.get("max_rms_growth_ratio", 1.25)
        )
        second_config["min_merged_rms_m"] = float(
            hierarchical.get("min_merged_rms_m", 0.02)
        )
        second_planes, second_mapping, second_stats = merge_plane_instances(
            merged_planes,
            points=points,
            normals=normals,
            planarity=planarity,
            fit_config=fit_config,
            boundary_config=boundary_config,
            target_config=target_config,
            quality_config=quality_config,
            source_bounds=source_bounds,
            merge_config=second_config,
        )
        composed_mapping = {
            old_index: second_mapping[new_index]
            for old_index, new_index in old_to_new.items()
        }
        stats["hierarchical_enabled"] = True
        stats["hierarchical_input_planes"] = len(merged_planes)
        stats["hierarchical_candidate_pairs"] = second_stats["candidate_pairs"]
        stats["hierarchical_accepted_pairs"] = second_stats["accepted_pairs"]
        stats["hierarchical_merged_components"] = second_stats["merged_components"]
        stats["hierarchical_output_planes"] = len(second_planes)
        stats["output_planes"] = len(second_planes)
        return second_planes, composed_mapping, stats

    stats["hierarchical_enabled"] = False
    stats["hierarchical_input_planes"] = len(merged_planes)
    stats["hierarchical_output_planes"] = len(merged_planes)
    return merged_planes, old_to_new, stats
