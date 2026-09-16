"""Cross-tile plane merging, orientation grouping, and global statistics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from ..core.fitting import confidence_score, dip_and_dip_direction, quality_grade
from ..core.geometry import (
    footprint_geometry_on_plane,
    normal_angle_deg,
    sparse_axial_dbscan_labels,
    split_axial_by_deviation,
)
from ..core.models import (
    GlobalAggregationResult,
    GlobalJointSetRecord,
    GlobalPlaneRecord,
    SelectionRecord,
    SpacingRecord,
    TilePlaneRecord,
    PlaneInstance,
)
from ..core.records import nearest_spacing_by_plane
from .aggregation import (
    XYBBoxIndex,
    planes_can_merge_rows,
    row_bbox,
    row_float,
    row_normal,
    row_xy_bounds_array,
    tile_id_indices,
)


ProgressCallback = Callable[[str, int, int, str], None]


def _notify(
    progress: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    detail: str = "",
) -> None:
    if progress is not None:
        progress(stage, current, total, detail)


def merge_plane_rows(
    rows: list[TilePlaneRecord],
    plan: dict[str, Any],
    config: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
) -> tuple[list[TilePlaneRecord], dict[str, list[TilePlaneRecord]], dict[str, str]]:
    """Merge plane instances touching neighbouring tile boundaries."""

    rows = [dict(row) for row in rows]
    if not rows:
        _notify(progress, "跨瓦片合并", 1, 1, "无平面记录")
        return rows, {}, {}
    whole_config = config.get("whole_cloud", {})
    merge_enabled = bool(whole_config.get("merge_enabled", True))
    normal_angle_deg = float(whole_config.get("merge_normal_angle_deg", 5.0))
    plane_offset_m = float(whole_config.get("merge_plane_offset_m", 0.08))
    xy_gap_m = float(whole_config.get("merge_xy_gap_m", 0.30))
    merge_gap_m = max(xy_gap_m, 2.0 * float(plan["overlap_m"]))
    max_merged_rms_m = float(
        whole_config.get("merge_max_rms_m", config.get("plane", {}).get("max_rms", 0.05))
    )

    row_xy_bounds = (
        row_xy_bounds_array(rows)
        if merge_enabled
        else np.empty((0, 4), dtype=np.float64)
    )
    tile_size_m = max(float(plan.get("tile_size_m", 25.0)), 1e-6)
    cell_size_m = max(tile_size_m / 4.0, merge_gap_m * 4.0, 1e-6)

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
            _rows_can_merge(a, b, merge_gap_m)
            for a in left_members
            for b in right_members
        )

    def _rows_can_merge(left: int, right: int, gap: float) -> bool:
        left_bounds = row_xy_bounds[left]
        right_bounds = row_xy_bounds[right]
        if not (
            np.all(np.isfinite(left_bounds))
            and np.all(np.isfinite(right_bounds))
            and left_bounds[0] <= right_bounds[1] + gap
            and left_bounds[1] >= right_bounds[0] - gap
            and left_bounds[2] <= right_bounds[3] + gap
            and left_bounds[3] >= right_bounds[2] - gap
        ):
            return False
        return planes_can_merge_rows(
            rows[left],
            rows[right],
            normal_angle_deg=normal_angle_deg,
            plane_offset_m=plane_offset_m,
            xy_gap_m=gap,
            max_merged_rms_m=max_merged_rms_m,
        )

    by_tile: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        tile_indices = tile_id_indices(str(row.get("tile_id", "")))
        if tile_indices is not None:
            by_tile[tile_indices].append(index)

    tile_items = list(by_tile.items())
    tile_total = max(1, len(tile_items))
    _notify(progress, "跨瓦片合并", 0, tile_total, f"候选瓦片 {len(tile_items)} 个")
    if merge_enabled:
        neighbour_offsets = ((1, -1), (1, 0), (1, 1), (0, 1))
        candidate_pair_count = 0
        accepted_pair_count = 0
        for tile_index, ((ix, iy), left_indices) in enumerate(tile_items, start=1):
            _notify(
                progress,
                "跨瓦片合并",
                tile_index - 1,
                tile_total,
                f"处理 {ix}_{iy}，空间候选 {candidate_pair_count} 对",
            )
            for dx, dy in neighbour_offsets:
                right_indices = by_tile.get((ix + dx, iy + dy), [])
                if not right_indices:
                    continue
                _notify(
                    progress,
                    "跨瓦片合并",
                    tile_index - 1,
                    tile_total,
                    f"处理 {ix}_{iy} 与 {ix + dx}_{iy + dy}",
                )
                right_index = XYBBoxIndex(
                    row_xy_bounds,
                    right_indices,
                    cell_size=cell_size_m,
                )
                for left in left_indices:
                    right_positions = right_index.query(row_xy_bounds[left], merge_gap_m)
                    candidate_pair_count += len(right_positions)
                    for right_position in right_positions:
                        right = right_indices[right_position]
                        if _rows_can_merge(left, right, merge_gap_m) and components_are_compatible(
                            left, right
                        ):
                            union(left, right)
                            accepted_pair_count += 1
            _notify(
                progress,
                "跨瓦片合并",
                tile_index,
                tile_total,
                f"完成 {ix}_{iy}，空间候选 {candidate_pair_count} 对，接受 {accepted_pair_count} 对",
            )
        if not tile_items:
            _notify(progress, "跨瓦片合并", tile_total, tile_total, "无有效瓦片索引")
    else:
        _notify(progress, "跨瓦片合并", tile_total, tile_total, "配置已关闭")

    root_members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        root_members[find(index)].append(index)
    ordered_members = sorted(root_members.values(), key=lambda members: min(members))
    groups: dict[str, list[TilePlaneRecord]] = {}
    old_to_global: dict[str, str] = {}
    group_total = max(1, len(ordered_members))
    _notify(progress, "建立全局平面组", 0, group_total, f"形成 {len(ordered_members)} 组")
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
        _notify(progress, "建立全局平面组", group_index, group_total, f"完成 {global_id}")
    return rows, groups, old_to_global


def global_orientation_sets(
    groups: dict[str, list[TilePlaneRecord]],
    config: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
    stage: str = "全局方向聚类",
) -> dict[str, str]:
    if not groups:
        _notify(progress, stage, 1, 1, "无平面组")
        return {}
    _notify(progress, stage, 0, 3, f"输入 {len(groups)} 个平面组")
    group_ids = sorted(groups)
    normal_array = np.empty((len(group_ids), 3), dtype=np.float64)
    for index, group_id in enumerate(group_ids):
        members = groups[group_id]
        representative = max(members, key=lambda row: row_float(row, "core_red_points") or 0.0)
        normal = row_normal(representative)
        normal_array[index] = normal if normal is not None else np.array([0.0, 0.0, 1.0])
    _notify(progress, stage, 1, 3, "代表法向量完成")
    joint_config = config.get("joint_sets", {})
    eps = np.deg2rad(float(joint_config.get("angular_eps_deg", 10.0)))
    labels = sparse_axial_dbscan_labels(normal_array, eps)
    _notify(progress, stage, 2, 3, f"初始方向簇 {len(np.unique(labels))} 个")
    max_deviation = np.deg2rad(float(joint_config.get("max_set_deviation_deg", 12.0)))
    index_groups: list[list[int]] = []
    for label in sorted(int(value) for value in np.unique(labels)):
        indices = np.flatnonzero(labels == label).astype(np.int64)
        index_groups.extend(
            [group.tolist() for group in split_axial_by_deviation(indices, normal_array, max_deviation)]
        )
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
    _notify(progress, stage, 3, 3, f"完成 {len(index_groups)} 个节理组")
    return result


def aggregate_global_planes(
    groups: dict[str, list[TilePlaneRecord]],
    global_set_ids: dict[str, str],
    config: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
    stage: str = "全局平面汇总",
) -> list[GlobalPlaneRecord]:
    global_rows: list[GlobalPlaneRecord] = []
    ordered_groups = sorted(groups)
    total = max(1, len(ordered_groups))
    _notify(progress, stage, 0, total, f"输入 {len(ordered_groups)} 个平面组")
    for group_index, global_id in enumerate(ordered_groups, start=1):
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
        _notify(progress, stage, group_index, total, f"完成 {global_id}")
    if not ordered_groups:
        _notify(progress, stage, total, total, "无平面组")
    return global_rows


def apply_global_candidate_gate(
    global_plane_rows: list[GlobalPlaneRecord],
    config: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
) -> tuple[set[str], dict[str, SelectionRecord]]:
    """Apply the existing global candidate gate to aggregated plane rows."""

    from .detachment import classify_candidate_joint_planes

    total = max(1, len(global_plane_rows))
    _notify(progress, "全局候选筛选", 0, total, f"输入 {len(global_plane_rows)} 个")
    global_plane_objects: list[PlaneInstance] = []
    for index, item in enumerate(global_plane_rows, start=1):
        global_plane_objects.append(
            PlaneInstance(
                plane_id=str(item["global_plane_id"]),
                area=float(item.get("observed_area_m2") or 0.0),
                minor_extent=float(item.get("minor_extent_m") or 0.0),
                boundary_completeness=float(item.get("boundary_completeness") or 0.0),
                inlier_ratio=float(item.get("inlier_ratio") or 0.0),
                normal_dispersion=float(item.get("normal_dispersion_deg") or 90.0),
                confidence=float(item.get("confidence") or 0.0),
            )
        )
        _notify(progress, "全局候选筛选", index, total, f"检查 {item['global_plane_id']}")
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
    selection_by_id: dict[str, SelectionRecord] = {
        str(row["plane_id"]): row for row in global_selection_rows
    }
    _notify(progress, "全局候选筛选", total, total, f"保留 {len(selected_global_ids)} 个")
    return selected_global_ids, selection_by_id


def global_spacing_and_sets(
    global_rows: list[GlobalPlaneRecord],
    config: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
) -> tuple[list[SpacingRecord], list[GlobalJointSetRecord]]:
    by_set: dict[str, list[GlobalPlaneRecord]] = defaultdict(list)
    for row in global_rows:
        by_set[str(row["global_set_id"])].append(row)
    spacing_rows: list[SpacingRecord] = []
    joint_rows: list[GlobalJointSetRecord] = []
    min_planes = int(config.get("joint_sets", {}).get("min_planes", 3))
    ordered_sets = sorted(by_set)
    total = max(1, len(ordered_sets))
    _notify(progress, "间距与节理组统计", 0, total, f"输入 {len(ordered_sets)} 个节理组")
    for set_index, set_id in enumerate(ordered_sets, start=1):
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
        local_rows: list[SpacingRecord] = []
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
        parameters: list[tuple[float, GlobalPlaneRecord]] = []
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
        stats: GlobalJointSetRecord = {
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
        _notify(progress, "间距与节理组统计", set_index, total, f"完成 {set_id}")
    if not ordered_sets:
        _notify(progress, "间距与节理组统计", total, total, "无节理组")
    return spacing_rows, joint_rows


def aggregate_global_results(
    all_plane_rows: list[TilePlaneRecord],
    plan: dict[str, Any],
    config: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
) -> GlobalAggregationResult:
    """Run the complete global aggregation stage without performing I/O.

    This is the typed boundary used by the whole-cloud orchestrator.  The
    lower-level functions remain public for compatibility and for focused
    geometry tests; this function only preserves their existing order and
    joins their outputs into one named result object.
    """

    merged_rows, plane_groups, old_to_global = merge_plane_rows(
        all_plane_rows,
        plan,
        config,
        progress=progress,
    )
    provisional_set_ids = global_orientation_sets(
        plane_groups,
        config,
        progress=progress,
        stage="全局方向聚类(预筛选)",
    )
    for row in merged_rows:
        row["global_set_id"] = provisional_set_ids.get(str(row["global_plane_id"]))
    all_global_rows = aggregate_global_planes(
        plane_groups,
        provisional_set_ids,
        config,
        progress=progress,
        stage="全局平面汇总(预筛选)",
    )
    selected_global_ids, selection_by_id = apply_global_candidate_gate(
        all_global_rows,
        config,
        progress=progress,
    )
    for row in merged_rows:
        selection = selection_by_id.get(str(row["global_plane_id"]), {})
        row["status"] = selection.get("status", "not_selected")
        row["selection_reason"] = selection.get("selection_reason", "global_gate_missing")

    selected_groups = {
        key: value for key, value in plane_groups.items() if key in selected_global_ids
    }
    global_set_ids = global_orientation_sets(
        selected_groups,
        config,
        progress=progress,
        stage="全局方向聚类(最终)",
    )
    global_rows = aggregate_global_planes(
        selected_groups,
        global_set_ids,
        config,
        progress=progress,
        stage="全局平面汇总(最终)",
    )
    spacing_rows, joint_set_rows = global_spacing_and_sets(
        global_rows,
        config,
        progress=progress,
    )
    nearest_spacing = nearest_spacing_by_plane(spacing_rows)
    for row in merged_rows:
        row["nearest_spacing_m"] = nearest_spacing.get(str(row["global_plane_id"]))
    for row in global_rows:
        row["nearest_spacing_m"] = nearest_spacing.get(str(row["global_plane_id"]))

    return GlobalAggregationResult(
        merged_plane_rows=cast(list[TilePlaneRecord], merged_rows),
        plane_groups=cast(dict[str, list[TilePlaneRecord]], plane_groups),
        old_to_global=old_to_global,
        provisional_set_ids=provisional_set_ids,
        all_global_plane_rows=cast(list[GlobalPlaneRecord], all_global_rows),
        selected_global_ids=selected_global_ids,
        selection_by_id=cast(dict[str, SelectionRecord], selection_by_id),
        global_set_ids=global_set_ids,
        global_plane_rows=cast(list[GlobalPlaneRecord], global_rows),
        spacing_rows=cast(list[SpacingRecord], spacing_rows),
        joint_set_rows=cast(list[GlobalJointSetRecord], joint_set_rows),
        nearest_spacing_by_plane=nearest_spacing,
    )
