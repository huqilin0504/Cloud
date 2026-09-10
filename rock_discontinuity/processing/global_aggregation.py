"""Cross-tile plane merging, orientation grouping, and global statistics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from ..core.fitting import confidence_score, dip_and_dip_direction, quality_grade
from ..core.geometry import footprint_geometry_on_plane, normal_angle_deg, normal_angle_matrix
from ..core.models import PlaneInstance
from .aggregation import (
    planes_can_merge_rows,
    row_bbox,
    row_float,
    row_normal,
    tile_id_indices,
)


def merge_plane_rows(
    rows: list[dict[str, Any]],
    plan: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Merge plane instances touching neighbouring tile boundaries."""

    rows = [dict(row) for row in rows]
    if not rows:
        return rows, {}, {}
    whole_config = config.get("whole_cloud", {})
    merge_enabled = bool(whole_config.get("merge_enabled", True))
    normal_angle_deg = float(whole_config.get("merge_normal_angle_deg", 5.0))
    plane_offset_m = float(whole_config.get("merge_plane_offset_m", 0.08))
    xy_gap_m = float(whole_config.get("merge_xy_gap_m", 0.30))
    max_merged_rms_m = float(
        whole_config.get("merge_max_rms_m", config.get("plane", {}).get("max_rms", 0.05))
    )

    parent = list(range(len(rows)))
    component_members: dict[int, list[int]] = {index: [index] for index in range(len(rows))}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root
            component_members[left_root].extend(component_members.pop(right_root))

    def components_are_compatible(left: int, right: int) -> bool:
        left_members = component_members[find(left)]
        right_members = component_members[find(right)]
        return all(
            planes_can_merge_rows(
                rows[a],
                rows[b],
                normal_angle_deg=normal_angle_deg,
                plane_offset_m=plane_offset_m,
                xy_gap_m=max(xy_gap_m, 2.0 * float(plan["overlap_m"])),
                max_merged_rms_m=max_merged_rms_m,
            )
            for a in left_members
            for b in right_members
        )

    by_tile: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        tile_indices = tile_id_indices(str(row.get("tile_id", "")))
        if tile_indices is not None:
            by_tile[tile_indices].append(index)

    if merge_enabled:
        neighbour_offsets = ((1, -1), (1, 0), (1, 1), (0, 1))
        for (ix, iy), left_indices in by_tile.items():
            for dx, dy in neighbour_offsets:
                right_indices = by_tile.get((ix + dx, iy + dy), [])
                for left in left_indices:
                    for right in right_indices:
                        if planes_can_merge_rows(
                            rows[left],
                            rows[right],
                            normal_angle_deg=normal_angle_deg,
                            plane_offset_m=plane_offset_m,
                            xy_gap_m=max(xy_gap_m, 2.0 * float(plan["overlap_m"])),
                            max_merged_rms_m=max_merged_rms_m,
                        ) and components_are_compatible(left, right):
                            union(left, right)

    root_members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        root_members[find(index)].append(index)
    ordered_members = sorted(root_members.values(), key=lambda members: min(members))
    groups: dict[str, list[dict[str, Any]]] = {}
    old_to_global: dict[str, str] = {}
    for group_index, members in enumerate(ordered_members, start=1):
        global_id = f"GJ-{group_index:05d}"
        group_rows = [rows[index] for index in members]
        groups[global_id] = group_rows
        tile_count = len({str(row.get("tile_id", "")) for row in group_rows})
        for row in group_rows:
            old_id = str(row.get("global_plane_id", ""))
            old_to_global[old_id] = global_id
            row["global_plane_id"] = global_id
            row["global_instance_count"] = len(group_rows)
            row["global_tile_count"] = tile_count
            row.setdefault("global_set_id", None)
    return rows, groups, old_to_global


def global_orientation_sets(
    groups: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, str]:
    if not groups:
        return {}
    group_ids = sorted(groups)
    normals: list[np.ndarray] = []
    for group_id in group_ids:
        members = groups[group_id]
        representative = max(members, key=lambda row: row_float(row, "core_red_points") or 0.0)
        normal = row_normal(representative)
        normals.append(normal if normal is not None else np.array([0.0, 0.0, 1.0]))
    normal_array = np.asarray(normals, dtype=np.float64)
    angles = np.deg2rad(normal_angle_matrix(normal_array))
    joint_config = config.get("joint_sets", {})
    eps = np.deg2rad(float(joint_config.get("angular_eps_deg", 10.0)))
    labels = DBSCAN(eps=eps, min_samples=1, metric="precomputed").fit_predict(angles)
    max_deviation = np.deg2rad(float(joint_config.get("max_set_deviation_deg", 12.0)))
    index_groups: list[list[int]] = []
    for label in sorted(set(int(value) for value in labels)):
        remaining = set(int(index) for index in np.flatnonzero(labels == label))
        while remaining:
            candidates = np.asarray(sorted(remaining), dtype=np.int64)
            medoid = int(candidates[np.argmin(angles[np.ix_(candidates, candidates)].sum(axis=1))])
            selected = candidates[angles[medoid, candidates] <= max_deviation]
            if not len(selected):
                selected = np.asarray([medoid], dtype=np.int64)
            index_groups.append([int(index) for index in selected])
            remaining.difference_update(int(index) for index in selected)
    index_groups.sort(
        key=lambda indices: -sum(
            row_float(
                max(groups[group_ids[index]], key=lambda row: row_float(row, "core_red_points") or 0.0),
                "core_red_points",
            )
            or 0.0
            for index in indices
        )
    )
    result: dict[str, str] = {}
    for set_index, indices in enumerate(index_groups, start=1):
        set_id = f"J{set_index}"
        for index in indices:
            result[group_ids[index]] = set_id
    return result


def aggregate_global_planes(
    groups: dict[str, list[dict[str, Any]]],
    global_set_ids: dict[str, str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    global_rows: list[dict[str, Any]] = []
    for global_id in sorted(groups):
        members = groups[global_id]
        representative = max(members, key=lambda row: row_float(row, "core_red_points") or 0.0)
        weights = np.asarray(
            [max(1.0, row_float(row, "core_red_points") or 0.0) for row in members],
            dtype=np.float64,
        )
        normals: list[np.ndarray] = []
        d_values: list[float] = []
        reference_normal = row_normal(representative)
        if reference_normal is None:
            reference_normal = np.array([0.0, 0.0, 1.0])
        normal_weights: list[float] = []
        for row, weight in zip(members, weights):
            normal = row_normal(row)
            d_value = row_float(row, "plane_d")
            if normal is None or d_value is None:
                continue
            if np.dot(normal, reference_normal) < 0:
                normal = -normal
                d_value = -d_value
            normals.append(normal)
            d_values.append(d_value)
            normal_weights.append(float(weight))
        if normals:
            mean_normal = np.average(
                np.asarray(normals), axis=0, weights=np.asarray(normal_weights, dtype=np.float64)
            )
            mean_normal /= max(float(np.linalg.norm(mean_normal)), 1e-15)
            if mean_normal[2] < 0:
                mean_normal = -mean_normal
                d_values = [-value for value in d_values]
            mean_d = float(np.average(np.asarray(d_values), weights=np.asarray(normal_weights, dtype=np.float64)))
        else:
            mean_normal = reference_normal
            mean_d = row_float(representative, "plane_d") or 0.0
        centers = np.asarray(
            [
                [
                    row_float(row, "center_x") or 0.0,
                    row_float(row, "center_y") or 0.0,
                    row_float(row, "center_z") or 0.0,
                ]
                for row in members
            ],
            dtype=np.float64,
        )
        center = np.average(centers, axis=0, weights=weights)
        lower = np.full(3, np.inf, dtype=np.float64)
        upper = np.full(3, -np.inf, dtype=np.float64)
        for row in members:
            bbox = row_bbox(row)
            if bbox is None:
                continue
            lower = np.minimum(lower, bbox[0])
            upper = np.maximum(upper, bbox[1])
        if not np.all(np.isfinite(lower)):
            lower = center.copy()
            upper = center.copy()
        areas = np.asarray([row_float(row, "observed_area_m2") or 0.0 for row in members], dtype=np.float64)
        footprint_geometries = [
            footprint_geometry_on_plane(item.get("_footprint_xyz"), center, mean_normal)
            for item in members
        ]
        footprint_geometries = [geometry for geometry in footprint_geometries if not geometry.is_empty]
        if footprint_geometries:
            from shapely.ops import unary_union

            footprint = unary_union(footprint_geometries).buffer(0)
            footprint_area = float(footprint.area)
            hull_area = float(footprint.convex_hull.area)
            rectangle = footprint.minimum_rotated_rectangle
            rectangle_points = np.asarray(rectangle.exterior.coords, dtype=np.float64)
            side_lengths = np.linalg.norm(np.diff(rectangle_points, axis=0), axis=1)
            global_minor = float(np.min(side_lengths)) if len(side_lengths) else 0.0
            global_major = float(np.max(side_lengths)) if len(side_lengths) else 0.0
            global_completeness = footprint_area / hull_area if hull_area > 1e-12 else 0.0
        else:
            footprint_area = float(np.max(areas)) if len(areas) else 0.0
            global_minor = max((row_float(item, "minor_extent_m") or 0.0) for item in members)
            global_major = max((row_float(item, "major_extent_m") or 0.0) for item in members)
            global_completeness = min((row_float(item, "boundary_completeness") or 0.0) for item in members)
        point_weights = np.asarray([max(1.0, row_float(item, "point_count") or 0.0) for item in members])

        def weighted_field(field: str, fallback: float = 0.0) -> float:
            values = np.asarray(
                [row_float(item, field) if row_float(item, field) is not None else fallback for item in members]
            )
            return float(np.average(values, weights=point_weights))

        global_rms = float(
            np.sqrt(np.average(np.square([row_float(item, "rms_m") or 0.0 for item in members]), weights=point_weights))
        )
        global_inlier_ratio = weighted_field("inlier_ratio")
        global_planarity = weighted_field("planarity")
        global_normal_dispersion = max(
            max((row_float(item, "normal_dispersion_deg") or 0.0) for item in members),
            max(
                (normal_angle_deg(normal, mean_normal) or 0.0 for normal in normals),
                default=0.0,
            )
            if normals
            else 0.0,
        )
        score_config = dict(config.get("plane", {}))
        score_config["min_area"] = float(config.get("plane_gate", {}).get("min_area_m2", 0.25))
        score_config["min_points"] = int(config.get("plane_gate", {}).get("min_effective_points", 100))
        global_quality = confidence_score(
            point_count=int(np.sum(point_weights)),
            area=footprint_area,
            planarity=global_planarity,
            inlier_ratio=global_inlier_ratio,
            rms=global_rms,
            normal_dispersion=global_normal_dispersion,
            boundary_completeness=global_completeness,
            config=score_config,
        )
        dip_direction, dip = dip_and_dip_direction(mean_normal)
        row = dict(representative)
        global_set_id = global_set_ids[global_id]
        row.update(
            {
                "global_plane_id": global_id,
                "global_set_id": global_set_id,
                "tile_count": len({str(item.get("tile_id", "")) for item in members}),
                "instance_count": len(members),
                "global_tile_count": len({str(item.get("tile_id", "")) for item in members}),
                "global_instance_count": len(members),
                "tile_id": "merged",
                "plane_id": global_id,
                "set_id": global_set_id,
                "core_red_points": int(sum(int(row_float(item, "core_red_points") or 0) for item in members)),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
                "centroid_x": float(center[0]),
                "centroid_y": float(center[1]),
                "centroid_z": float(center[2]),
                "nx": float(mean_normal[0]),
                "ny": float(mean_normal[1]),
                "nz": float(mean_normal[2]),
                "plane_d": mean_d,
                "dip_direction_deg": dip_direction,
                "dip_deg": dip,
                "point_count": int(np.sum(point_weights)),
                "observed_area_m2": footprint_area,
                "observed_area_sum_m2": float(np.sum(areas)),
                "max_instance_area_m2": float(np.max(areas)) if len(areas) else None,
                "major_extent_m": global_major,
                "minor_extent_m": global_minor,
                "boundary_completeness": global_completeness,
                "rms_m": global_rms,
                "inlier_ratio": global_inlier_ratio,
                "planarity": global_planarity,
                "normal_dispersion_deg": global_normal_dispersion,
                "normal_scale_m": weighted_field("normal_scale_m"),
                "normal_stability_deg": max((row_float(item, "normal_stability_deg") or 0.0) for item in members),
                "normal_stable_fraction": weighted_field("normal_stable_fraction"),
                "normal_valid_scale_count": weighted_field("normal_valid_scale_count"),
                "normal_stable": weighted_field("normal_stable_fraction") >= 0.5,
                "edge_censored": all(str(item.get("edge_censored", "")).lower() == "true" for item in members),
                "nearest_spacing_m": None,
                "bbox_min_x": float(lower[0]),
                "bbox_min_y": float(lower[1]),
                "bbox_min_z": float(lower[2]),
                "bbox_max_x": float(upper[0]),
                "bbox_max_y": float(upper[1]),
                "bbox_max_z": float(upper[2]),
                "quality_score": global_quality,
                "confidence": global_quality,
                "quality_grade": quality_grade(float(global_quality), config.get("quality", {})),
            }
        )
        global_rows.append(row)
    return global_rows


def apply_global_candidate_gate(
    global_plane_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Apply the existing global candidate gate to aggregated plane rows."""

    from .detachment import classify_candidate_joint_planes

    global_plane_objects = [
        PlaneInstance(
            plane_id=str(item["global_plane_id"]),
            area=float(item.get("observed_area_m2") or 0.0),
            minor_extent=float(item.get("minor_extent_m") or 0.0),
            boundary_completeness=float(item.get("boundary_completeness") or 0.0),
            inlier_ratio=float(item.get("inlier_ratio") or 0.0),
            normal_dispersion=float(item.get("normal_dispersion_deg") or 90.0),
            confidence=float(item.get("confidence") or 0.0),
        )
        for item in global_plane_rows
    ]
    global_selected, global_selection_rows = classify_candidate_joint_planes(
        global_plane_objects,
        config.get("detachment", {}),
        context="global",
    )
    selected_global_ids = {
        str(row["global_plane_id"])
        for row, selected in zip(global_plane_rows, global_selected)
        if selected
    }
    selection_by_id = {
        str(row["plane_id"]): row for row in global_selection_rows
    }
    return selected_global_ids, selection_by_id


def global_spacing_and_sets(
    global_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in global_rows:
        by_set[str(row["global_set_id"])].append(row)
    spacing_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    min_planes = int(config.get("joint_sets", {}).get("min_planes", 3))
    for set_id in sorted(by_set):
        members = by_set[set_id]
        normals = []
        weights = []
        for row in members:
            normal = row_normal(row)
            if normal is not None:
                normals.append(normal)
                weights.append(max(1.0, row_float(row, "core_red_points") or 0.0))
        if not normals:
            continue
        mean_normal = np.average(np.asarray(normals), axis=0, weights=np.asarray(weights))
        mean_normal /= max(float(np.linalg.norm(mean_normal)), 1e-15)
        if mean_normal[2] < 0:
            mean_normal = -mean_normal
        deviations = np.asarray(
            [normal_angle_deg(normal, mean_normal) or 0.0 for normal in normals],
            dtype=np.float64,
        )
        mean_dip_direction, mean_dip = dip_and_dip_direction(mean_normal)
        local_rows: list[dict[str, Any]] = []
        spacings: list[float] = []
        x0 = np.mean(
            [
                [
                    row_float(row, "center_x") or 0.0,
                    row_float(row, "center_y") or 0.0,
                    row_float(row, "center_z") or 0.0,
                ]
                for row in members
            ],
            axis=0,
        )
        parameters: list[tuple[float, dict[str, Any]]] = []
        for row in members:
            normal = row_normal(row)
            d_value = row_float(row, "plane_d")
            if normal is None or d_value is None:
                continue
            denominator = float(np.dot(normal, mean_normal))
            if denominator < 0:
                normal = -normal
                d_value = -d_value
                denominator = -denominator
            parameters.append((-float(np.dot(normal, x0) + d_value) / denominator, row))
        parameters.sort(key=lambda item: item[0])
        for (t_a, row_a), (t_b, row_b) in zip(parameters, parameters[1:]):
            spacing = abs(float(t_b - t_a))
            if spacing <= 1e-9:
                continue
            spacings.append(spacing)
            local_rows.append(
                {
                    "tile_id": "merged",
                    "plane_id_a": row_a["global_plane_id"],
                    "plane_id_b": row_b["global_plane_id"],
                    "global_plane_id_a": row_a["global_plane_id"],
                    "global_plane_id_b": row_b["global_plane_id"],
                    "spacing_m": spacing,
                    "method": "spacing_virtual_scanline_merged_tiles",
                    "sample_count": 1,
                }
            )
        spacing_rows.extend(local_rows)
        set_spacing = np.asarray(spacings, dtype=np.float64)
        stats: dict[str, Any] = {
            "set_id": set_id,
            "plane_count": len(members),
            "mean_dip_direction_deg": mean_dip_direction,
            "mean_dip_deg": mean_dip,
            "mean_normal_x": float(mean_normal[0]),
            "mean_normal_y": float(mean_normal[1]),
            "mean_normal_z": float(mean_normal[2]),
            "angular_dispersion_deg": float(np.max(deviations)) if len(deviations) else None,
            "spacing_available": bool(len(set_spacing)),
            "spacing_reason": None
            if len(set_spacing)
            else ("fewer_than_min_planes_for_set_statistics" if len(members) < min_planes else "no_positive_spacing"),
            "mean_spacing_m": None,
            "median_spacing_m": None,
            "std_spacing_m": None,
            "min_spacing_m": None,
            "max_spacing_m": None,
            "p10_spacing_m": None,
            "p90_spacing_m": None,
            "spacing_3d_sample_count": 0,
            "spacing_virtual_scanline_sample_count": len(set_spacing),
            "mean_spacing_3d_m": None,
            "median_spacing_3d_m": None,
            "mean_spacing_virtual_scanline_m": None,
            "median_spacing_virtual_scanline_m": None,
            "fisher_k": None,
        }
        if len(set_spacing):
            stats.update(
                {
                    "mean_spacing_m": float(np.mean(set_spacing)),
                    "median_spacing_m": float(np.median(set_spacing)),
                    "std_spacing_m": float(np.std(set_spacing)),
                    "min_spacing_m": float(np.min(set_spacing)),
                    "max_spacing_m": float(np.max(set_spacing)),
                    "p10_spacing_m": float(np.quantile(set_spacing, 0.10)),
                    "p90_spacing_m": float(np.quantile(set_spacing, 0.90)),
                    "mean_spacing_virtual_scanline_m": float(np.mean(set_spacing)),
                    "median_spacing_virtual_scanline_m": float(np.median(set_spacing)),
                }
            )
        resultant = float(np.linalg.norm(np.sum(np.asarray(normals), axis=0)) / max(len(normals), 1))
        stats["fisher_k"] = (
            float((3.0 * resultant - resultant**3) / max(1.0 - resultant**2, 1e-12))
            if resultant < 0.999999
            else None
        )
        joint_rows.append(stats)
    return spacing_rows, joint_rows
