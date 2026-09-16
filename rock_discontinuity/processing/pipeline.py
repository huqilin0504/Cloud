from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from ..core.fitting import fit_plane_instance
from ..core.geometry import sparse_axial_dbscan_labels, split_axial_by_deviation
from .audit import audit_points
from .density import assess_density, discontinuity_density_metrics, estimate_roi_density
from .detachment import classify_candidate_joint_planes
from ..io import LasReadResult, read_las_roi
from ..core.models import JointSet, PlaneInstance, ProcessingResult, RejectedRecord, SpacingRecord
from ..core.records import nearest_spacing_by_plane
from ..io.output import write_outputs
from .preprocess import estimate_local_features, select_planar_candidates, statistical_inlier_mask, voxel_downsample
from .segment import segment_candidates
from .merge import merge_plane_instances
from .spacing import compute_joint_set_spacing
from .records import prepare_detachment_rows
from .output_records import (
    prepare_aperture_rows,
    prepare_plane_boundaries,
    prepare_trace_rows,
)


def _split_by_angular_deviation(
    indices: np.ndarray,
    normals: np.ndarray,
    max_deviation_rad: float,
) -> list[np.ndarray]:
    """Split DBSCAN chains without creating a dense angle matrix."""

    return split_axial_by_deviation(indices, normals, max_deviation_rad)


def _cluster_plane_orientations(planes: list[PlaneInstance], config: dict[str, Any]) -> list[list[PlaneInstance]]:
    if not planes:
        return []
    normals = np.asarray([plane.normal for plane in planes], dtype=np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    angular_eps = np.deg2rad(float(config.get("angular_eps_deg", 10.0)))
    max_deviation = np.deg2rad(float(config.get("max_set_deviation_deg", 12.0)))
    labels = sparse_axial_dbscan_labels(normals, angular_eps)
    groups: list[list[PlaneInstance]] = []
    for label in sorted(int(value) for value in np.unique(labels)):
        indices = np.flatnonzero(labels == label).astype(np.int64)
        for split_indices in _split_by_angular_deviation(indices, normals, max_deviation):
            groups.append([planes[int(index)] for index in split_indices])
    return groups


def _assign_set_and_plane_ids(planes: list[PlaneInstance], config: dict[str, Any] | None = None) -> None:
    groups = _cluster_plane_orientations(planes, config or {})
    grouped: dict[int, list[PlaneInstance]] = {
        index: group for index, group in enumerate(groups)
    }
    ordered_groups = sorted(grouped.items(), key=lambda item: -sum(len(p.point_indices) for p in item[1]))
    set_names = {raw_label: f"J{index + 1}" for index, (raw_label, _) in enumerate(ordered_groups)}
    for raw_label, group in ordered_groups:
        for plane in group:
            plane.raw_set_label = raw_label
        group.sort(key=lambda plane: (-len(plane.point_indices), float(plane.centroid[0]), float(plane.centroid[1])))
        for index, plane in enumerate(group, start=1):
            plane.set_id = set_names[raw_label]
            plane.plane_id = f"{plane.set_id}-{index:03d}"


def _expand_labels(
    original_to_voxel: np.ndarray,
    retained_voxel_indices: np.ndarray,
    filtered_labels: np.ndarray,
) -> np.ndarray:
    original_labels = np.full(len(original_to_voxel), -1, dtype=np.int32)
    reverse = np.full(int(original_to_voxel.max()) + 1, -1, dtype=np.int64) if len(original_to_voxel) else np.empty(0, dtype=np.int64)
    reverse[retained_voxel_indices] = np.arange(len(retained_voxel_indices), dtype=np.int64)
    retained = reverse[original_to_voxel]
    valid = retained >= 0
    original_labels[valid] = filtered_labels[retained[valid]]
    return original_labels


def _labels_from_planes(point_count: int, planes: list[PlaneInstance]) -> np.ndarray:
    labels = np.full(int(point_count), -1, dtype=np.int32)
    for plane_index, plane in enumerate(planes):
        indices = np.asarray(plane.point_indices, dtype=np.int64)
        indices = indices[(indices >= 0) & (indices < len(labels))]
        labels[indices] = plane_index
    return labels


def process_points(
    points: np.ndarray,
    config: dict[str, Any],
    *,
    read_metadata: dict[str, Any] | None = None,
    processing_context: str = "roi",
) -> ProcessingResult:
    """Run the ROI algorithm in memory and return a typed result."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("点云必须是非空的 N×3 数组")
    world_points = points.copy()
    coordinate_origin = world_points.mean(axis=0)
    points = world_points - coordinate_origin
    preprocess_config = config["preprocess"]
    batch_size = int(config.get("performance", {}).get("feature_batch_size", 50_000))
    downsampled, original_to_voxel = voxel_downsample(points, float(preprocess_config["voxel_size"]))
    keep = np.ones(len(downsampled), dtype=bool)
    if preprocess_config.get("remove_outliers", True):
        keep = statistical_inlier_mask(
            downsampled,
            int(preprocess_config["outlier_neighbors"]),
            float(preprocess_config["outlier_std_ratio"]),
            batch_size=batch_size,
        )
    retained_voxel_indices = np.flatnonzero(keep).astype(np.int64)
    filtered_points = downsampled[keep]
    if len(filtered_points) < 3:
        raise ValueError("预处理后点数不足以拟合平面")

    normal_config = config["normal"]
    features = estimate_local_features(
        filtered_points,
        radius=float(normal_config["radius"]),
        min_neighbors=int(normal_config["min_neighbors"]),
        knn=int(normal_config.get("knn", 24)),
        max_neighbors=int(
            config.get("normal_multiscale", {}).get(
                "max_neighbors", normal_config.get("max_neighbors", 96)
            )
        ),
        batch_size=batch_size,
        multiscale_config=config.get("normal_multiscale", {}),
    )
    feature_config = dict(config["feature"])
    feature_config["min_neighbors"] = int(normal_config["min_neighbors"])
    candidate_mask = select_planar_candidates(features, feature_config)
    segmentation = segment_candidates(
        filtered_points,
        features["normals"],
        candidate_mask,
        features,
        config["orientation_cluster"],
        config["spatial_cluster"],
        int(config["plane"]["min_points"]),
        config.get("region_growing", {}),
    )

    planes: list[PlaneInstance] = []
    rejected: list[RejectedRecord] = []
    plane_config = dict(config["plane"])
    plane_gate = config.get("plane_gate", {})
    if plane_gate:
        plane_config["min_points"] = max(
            int(plane_config.get("min_points", 0)),
            int(plane_gate.get("min_effective_points", 0)),
        )
        plane_config["min_area"] = max(
            float(plane_config.get("min_area", 0.0)),
            float(plane_gate.get("min_area_m2", 0.0)),
        )
        plane_config["normal_dispersion_limit_deg"] = float(
            plane_config.get("normal_dispersion_limit_deg", 12.0)
        )
    plane_refine = config.get("plane_refine", {})
    if plane_refine.get("random_seed") is not None:
        plane_config["random_seed"] = int(plane_refine["random_seed"])
    target_config = dict(config.get("target", {}))
    target_config["min_area_m2"] = max(
        float(target_config.get("min_area_m2", 0.0)),
        float(plane_gate.get("min_area_m2", 0.0)),
    )
    target_config["min_minor_extent_m"] = max(
        float(target_config.get("min_minor_extent_m", 0.0)),
        float(plane_gate.get("min_minor_extent_m", 0.0)),
    )
    quality_config = dict(config.get("quality", {}))
    quality_config["min_inlier_ratio"] = max(
        float(quality_config.get("min_inlier_ratio", 0.0)),
        float(plane_gate.get("min_inlier_ratio", 0.0)),
    )
    if np.isfinite(segmentation.distance_threshold):
        plane_config["ransac_distance"] = min(
            float(plane_config["ransac_distance"]),
            float(segmentation.distance_threshold),
        )
    source_bounds = (points.min(axis=0), points.max(axis=0))
    for raw_set_label, indices in segmentation.instances:
        plane, reason = fit_plane_instance(
            filtered_points,
            features["normals"],
            features["planarity"],
            indices,
            raw_set_label=raw_set_label,
            config=plane_config,
            boundary_config=config["boundary"],
            target_config=target_config,
            quality_config=quality_config,
            source_bounds=source_bounds,
        )
        if plane is None:
            rejected.append({"raw_set_label": raw_set_label, "point_count": len(indices), "reason": reason})
        else:
            planes.append(plane)
    # Keep a copy of the local fitted instances for diagnosis and QA.  The
    # final output below is built from geometrically merged plane patches.
    plane_instances = [deepcopy(plane) for plane in planes]
    plane_instance_filtered_labels = _labels_from_planes(len(filtered_points), plane_instances)
    plane_instance_original_labels = _expand_labels(
        original_to_voxel,
        retained_voxel_indices,
        plane_instance_filtered_labels,
    )
    planes, _, plane_merge = merge_plane_instances(
        planes,
        points=filtered_points,
        normals=features["normals"],
        planarity=features["planarity"],
        fit_config=plane_config,
        boundary_config=config["boundary"],
        target_config=target_config,
        quality_config=quality_config,
        source_bounds=source_bounds,
        merge_config=config.get("plane_merge", {}),
    )

    def annotate_normal_stability(target_planes: list[PlaneInstance]) -> None:
        stability = np.asarray(features.get("normal_stability_deg", []), dtype=np.float64)
        scales = np.asarray(features.get("normal_scale_m", []), dtype=np.float64)
        stable = np.asarray(features.get("normal_stable", []), dtype=bool)
        valid_scales = np.asarray(features.get("normal_valid_scale_count", []), dtype=np.float64)
        for target_plane in target_planes:
            indices = np.asarray(target_plane.point_indices, dtype=np.int64)
            if not len(indices) or len(stability) <= int(np.max(indices)):
                continue
            local_stability = stability[indices]
            local_scales = scales[indices]
            local_stable = stable[indices]
            local_valid_scales = valid_scales[indices]
            finite_stability = local_stability[np.isfinite(local_stability)]
            finite_scales = local_scales[np.isfinite(local_scales)]
            finite_valid_scales = local_valid_scales[np.isfinite(local_valid_scales)]
            target_plane.normal_stability_deg = (
                float(np.quantile(finite_stability, 0.90)) if len(finite_stability) else float("nan")
            )
            target_plane.normal_scale_m = (
                float(np.median(finite_scales)) if len(finite_scales) else float("nan")
            )
            target_plane.normal_stable_fraction = float(np.mean(local_stable)) if len(local_stable) else 0.0
            target_plane.normal_valid_scale_count = (
                float(np.median(finite_valid_scales)) if len(finite_valid_scales) else 0.0
            )
            target_plane.normal_stable = bool(target_plane.normal_stable_fraction >= 0.5)

    annotate_normal_stability(plane_instances)
    annotate_normal_stability(planes)
    _assign_set_and_plane_ids(
        plane_instances,
        config.get("joint_sets", config.get("orientation_cluster", {})),
    )
    _assign_set_and_plane_ids(
        planes,
        config.get("joint_sets", config.get("orientation_cluster", {})),
    )
    for plane in plane_instances + planes:
        plane.centroid = plane.centroid + coordinate_origin
        plane.d = float(plane.d - np.dot(plane.normal, coordinate_origin))
        plane.bbox_min = plane.bbox_min + coordinate_origin
        plane.bbox_max = plane.bbox_max + coordinate_origin

    detachment_config = config.get("detachment", {})
    detachment_labels, detachment_rows = classify_candidate_joint_planes(
        planes,
        detachment_config,
        context=processing_context,
    )

    grouped: dict[str, list[PlaneInstance]] = defaultdict(list)
    for plane_index, plane in enumerate(planes):
        if plane_index >= len(detachment_labels) or not detachment_labels[plane_index]:
            continue
        assert plane.set_id is not None
        grouped[plane.set_id].append(plane)
    joint_sets: list[JointSet] = []
    spacing_rows: list[SpacingRecord] = []
    for set_id in sorted(grouped):
        joint_set, rows = compute_joint_set_spacing(
            set_id,
            grouped[set_id],
            config["spacing"],
            points=filtered_points + coordinate_origin,
        )
        joint_sets.append(joint_set)
        spacing_rows.extend(rows)

    filtered_labels = _labels_from_planes(len(filtered_points), planes)
    original_labels = _expand_labels(original_to_voxel, retained_voxel_indices, filtered_labels)
    return ProcessingResult(
        original_points=world_points,
        original_labels=original_labels,
        filtered_points=filtered_points + coordinate_origin,
        filtered_labels=filtered_labels,
        features=features,
        candidate_mask=candidate_mask,
        segmentation=segmentation,
        planes=planes,
        joint_sets=joint_sets,
        spacing_rows=spacing_rows,
        detachment_labels=detachment_labels,
        detachment_rows=detachment_rows,
        rejected=rejected,
        counts={
            "input_points": int(len(world_points)),
            "voxel_points": int(len(downsampled)),
            "retained_points": int(len(filtered_points)),
            "candidate_points": int(candidate_mask.sum()),
            "accepted_planes": int(len(planes)),
            "planar_facets": int(len(planes)),
            "candidate_joint_planes": int(detachment_labels.sum()),
            "plane_instances": int(len(plane_instances)),
            "rejected_instances": int(len(rejected)),
        },
        coordinate_origin=coordinate_origin,
        plane_instances=plane_instances,
        plane_instance_filtered_labels=plane_instance_filtered_labels,
        plane_instance_original_labels=plane_instance_original_labels,
        plane_merge=plane_merge,
    )


def run_points(
    points: np.ndarray,
    config: dict[str, Any],
    *,
    read_metadata: dict[str, Any] | None = None,
    processing_context: str = "roi",
) -> dict[str, Any]:
    """Backward-compatible mapping wrapper around :func:`process_points`."""

    return process_points(
        points,
        config,
        read_metadata=read_metadata,
        processing_context=processing_context,
    ).as_legacy_dict()


def run_las(
    input_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    bbox: dict[str, float] | None,
) -> dict[str, Path]:
    read_result: LasReadResult = read_las_roi(
        input_path,
        bbox,
        chunk_size=int(config["input"]["chunk_size"]),
        max_points_without_roi=int(config["input"]["max_points_without_roi"]),
    )
    result = process_points(
        read_result.xyz,
        config,
        read_metadata=read_result.metadata,
        processing_context="roi",
    )
    density_report = estimate_roi_density(
        read_result.xyz,
        sample_size=int(config.get("density", {}).get("sample_size", 20_000)),
        knn=int(config.get("density", {}).get("knn", 20)),
    )
    density_report = assess_density(density_report, config["density"])
    candidate_plane_count = int(result.detachment_labels.sum())
    candidate_point_mask = np.zeros(len(result.original_labels), dtype=bool)
    for plane_index, selected in enumerate(result.detachment_labels):
        if selected:
            candidate_point_mask |= result.original_labels == plane_index
    footprint_area = density_report.get("xy_bbox_density_points_m2")
    if footprint_area and footprint_area > 0:
        footprint_area = float(len(read_result.xyz) / footprint_area)
    else:
        footprint_area = None
    density_observables = discontinuity_density_metrics(
        point_count=len(read_result.xyz),
        footprint_area_m2=footprint_area,
        plane_count=candidate_plane_count,
    )
    data_audit = audit_points(
        read_result.xyz,
        metadata=read_result.metadata,
        density_report=density_report,
        tile_count=1,
        source_mode="roi_in_memory",
    )
    nearest_spacing = nearest_spacing_by_plane(result.spacing_rows)
    prepared_detachment_rows = prepare_detachment_rows(
        result.planes,
        result.detachment_rows,
        nearest_spacing,
    )
    report = {
        "version": "2.0.0",
        "algorithm_version": "v2.3-multiscale-stability-hierarchical-merge",
        "coordinate_assumption": config["coordinate"],
        "roi": bbox,
        "counts": result.counts,
        "plane_merge": result.plane_merge,
        "density": density_report,
        "density_observables": density_observables,
        "data_audit": data_audit,
        "detachment": {
            "status": "geometry_only_candidate",
            "method": "conservative_planar_discontinuity_candidate_gate",
            "candidate_plane_count": candidate_plane_count,
            "candidate_point_count": int(candidate_point_mask.sum()),
            "confirmed_unstable_count": None,
            "color_rgb": [230, 35, 35],
            "note": "红色是经尺度、边界完整度和拟合质量保守筛选的平面不连续候选；仍不能仅凭 XYZ 判定地质成因或失稳。",
        },
        "trace": {
            "available": False,
            "method": config.get("trace", {}).get("method", "normal_tensor_voting"),
            "verification": config.get("trace", {}).get("verification", "human_review"),
            "reason": "high_resolution_mesh_or_image_required",
        },
        "aperture": {
            "available": False,
            "source": config.get("aperture", {}).get("source", "highest_resolution_mesh_or_image"),
            "reason": "high_resolution_mesh_or_image_required",
        },
        "validation": config["validation"],
        "limitations": [
            "结果只代表 ROI 内点云可观测表面，不代表地下真实延伸。",
            "trace_length_m 在 V1 没有稳定暴露面迹线时输出 null。",
            "confidence 是几何质量启发式评分，不是统计概率。",
            "当前源 LAS 无原始 RGB；PLY 颜色是结构面标签颜色。",
            "P10 需要明确扫描线，P21 需要已验证迹线；当前分别输出 null 和原因，不用表观延伸替代迹线。",
            "张开度需要最高分辨率 Mesh/影像，当前 LAS 主链不估计 aperture。",
            "planes.csv 是同向、共面、空间相邻片段合并后的结果；plane_instances.csv 保留合并前的局部平面片段。",
            "planes.csv 保留所有通过拟合门槛的平面 facet；detachment_planes.csv 仅保留通过保守不连续候选门槛的子集。",
            "本版本不使用 CloudCompare 人工标注、现场罗盘或人工测线；外部真值标定不在验收范围。",
        ],
        "rejected_planes": result.rejected,
    }
    prepared_boundary_feature_collection = prepare_plane_boundaries(
        result.planes,
        result.filtered_points,
        alpha=config.get("boundary", {}).get("alpha", "auto"),
        max_points=int(config.get("boundary", {}).get("max_points", 5_000)),
    )
    prepared_trace_rows = prepare_trace_rows(result.planes, report["trace"])
    prepared_aperture_rows = prepare_aperture_rows(result.planes, report["aperture"])
    return write_outputs(
        output_dir,
        original_points=result.original_points,
        original_labels=result.original_labels,
        filtered_points=result.filtered_points,
        features=result.features,
        planes=result.planes,
        plane_instances=result.plane_instances,
        plane_instance_original_labels=result.plane_instance_original_labels,
        joint_sets=result.joint_sets,
        spacing_rows=result.spacing_rows,
        detachment_labels=result.detachment_labels,
        config=config,
        read_metadata=read_result.metadata,
        report=report,
        data_audit=data_audit,
        density_report=density_report,
        prepared_detachment_rows=prepared_detachment_rows,
        prepared_boundary_feature_collection=prepared_boundary_feature_collection,
        prepared_trace_rows=prepared_trace_rows,
        prepared_aperture_rows=prepared_aperture_rows,
    )
