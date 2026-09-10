from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypedDict

import numpy as np


FeatureMap: TypeAlias = dict[str, np.ndarray]
CountMap: TypeAlias = dict[str, int]


class PlaneMergeRecord(TypedDict, total=False):
    """Stable audit record for within-ROI plane merging."""

    input_plane_instances: int
    output_planes: int
    merged_components: int
    merge_events: int
    skipped_pairs: int
    reason: str


class SpacingRecord(TypedDict, total=False):
    set_id: str
    plane_id_a: str
    plane_id_b: str
    spacing_m: float
    method: str
    sample_count: int


class DetachmentRecord(TypedDict, total=False):
    plane_id: str
    set_id: str | None
    status: str
    selection_reason: str
    nearest_spacing_m: float | None
    center_x: float
    center_y: float
    center_z: float
    dip_direction_deg: float | None
    dip_deg: float | None
    trace_length_m: float | None
    apparent_persistence_m: float | None
    observed_area_m2: float | None
    major_extent_m: float | None
    minor_extent_m: float | None
    edge_censored: bool
    area_m2: float | None
    boundary_completeness: float | None
    inlier_ratio: float | None
    normal_dispersion_deg: float | None
    normal_scale_m: float | None
    normal_stability_deg: float | None
    normal_stable_fraction: float | None
    normal_valid_scale_count: float | None
    normal_stable: bool
    confidence: float | None
    quality_score: float | None
    quality_grade: str


class RejectedRecord(TypedDict, total=False):
    raw_set_label: int
    point_count: int
    reason: str


class PlaneRecord(TypedDict, total=False):
    """Stable row shape shared by tile and global plane tables."""

    plane_id: str
    set_id: str | None
    point_count: int
    candidate_point_count: int
    nx: float
    ny: float
    nz: float
    plane_d: float
    dip_direction_deg: float | None
    dip_deg: float | None
    centroid_x: float | None
    centroid_y: float | None
    centroid_z: float | None
    bbox_min_x: float | None
    bbox_min_y: float | None
    bbox_min_z: float | None
    bbox_max_x: float | None
    bbox_max_y: float | None
    bbox_max_z: float | None
    area_m2: float | None
    observed_area_m2: float | None
    apparent_persistence_m: float | None
    major_extent_m: float | None
    minor_extent_m: float | None
    trace_length_m: float | None
    rms_m: float | None
    mae_m: float | None
    p95_residual_m: float | None
    inlier_ratio: float | None
    planarity: float | None
    normal_dispersion_deg: float | None
    normal_scale_m: float | None
    normal_stability_deg: float | None
    normal_stable_fraction: float | None
    normal_valid_scale_count: float | None
    normal_stable: bool
    boundary_completeness: float | None
    edge_censored: bool
    confidence: float | None
    quality_score: float | None
    quality_grade: str
    near_vertical: bool
    data_gap_censored: bool
    trace_verified: bool
    aperture_available: bool
    aperture_reason: str


class TilePlaneRecord(PlaneRecord, total=False):
    tile_id: str
    global_plane_id: str
    global_set_id: str | None
    global_instance_count: int
    global_tile_count: int
    core_red_points: int
    tile_plane_index: int
    status: str
    selection_reason: str
    nearest_spacing_m: float | None
    center_x: float | None
    center_y: float | None
    center_z: float | None
    _footprint_xyz: list[list[list[float]]] | None


class GlobalPlaneRecord(PlaneRecord, total=False):
    global_plane_id: str
    global_set_id: str | None
    tile_count: int
    instance_count: int
    observed_area_sum_m2: float | None
    max_instance_area_m2: float | None
    core_red_points: int
    tile_id: str
    global_instance_count: int
    global_tile_count: int


class TraceRecord(TypedDict, total=False):
    plane_id: str | None
    available: bool
    trace_verified: bool
    method: str
    chord_length_m: float | None
    polyline_length_m: float | None
    reason: str


class ApertureRecord(TypedDict, total=False):
    plane_id: str | None
    available: bool
    source: str
    effective_resolution_m: float | None
    aperture_m: float | None
    reason: str


@dataclass
class SegmentationResult:
    """Neutral result shape shared by the segmenter and main pipeline."""

    candidate_indices: np.ndarray
    orientation_labels: np.ndarray
    spatial_labels: np.ndarray
    instances: list[tuple[int, np.ndarray]]
    distance_threshold: float = float("nan")


class TileReport(TypedDict, total=False):
    algorithm_version: str
    tile_id: str
    status: str
    counts: dict[str, int]
    density: dict[str, object]
    plane_merge: dict[str, object]
    source_tile: str
    overlay: str | None
    plane_rows: int
    spacing_rows: int
    ix: int
    iy: int
    core_bbox: dict[str, float]
    expanded_bbox: dict[str, float]


@dataclass
class PlaneInstance:
    """One observed planar point cluster."""

    plane_id: str = ""
    set_id: str | None = None
    raw_set_label: int = -1
    point_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    candidate_point_count: int = 0
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    d: float = float("nan")
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    bbox_min: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=float)
    )
    bbox_max: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=float)
    )
    dip_direction: float = float("nan")
    dip: float = float("nan")
    area: float = float("nan")
    apparent_persistence: float = float("nan")
    major_extent: float = float("nan")
    minor_extent: float = float("nan")
    trace_length: float | None = None
    rms: float = float("nan")
    mae: float = float("nan")
    p95_residual: float = float("nan")
    inlier_ratio: float = float("nan")
    planarity: float = float("nan")
    normal_dispersion: float = float("nan")
    normal_scale_m: float = float("nan")
    normal_stability_deg: float = float("nan")
    normal_stable_fraction: float = float("nan")
    normal_valid_scale_count: float = float("nan")
    normal_stable: bool = False
    boundary_completeness: float = float("nan")
    edge_censored: bool = False
    confidence: float = float("nan")
    quality_score: float = float("nan")
    quality_grade: str = ""
    near_vertical: bool = False
    data_gap_censored: bool = False
    trace_verified: bool = False
    aperture_available: bool = False
    aperture_reason: str = "high_resolution_mesh_or_image_required"


@dataclass
class JointSet:
    """One orientation family and its spacing statistics."""

    set_id: str
    plane_ids: list[str]
    mean_normal: np.ndarray
    mean_dip_direction: float
    mean_dip: float
    plane_count: int
    spacing_available: bool
    spacing_reason: str | None = None
    spacings: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    angular_dispersion: float = float("nan")
    spacing_3d: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    spacing_virtual_scanline: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    fisher_k: float = float("nan")


@dataclass
class ProcessingResult:
    """Typed result of one in-memory ROI or tile processing run.

    Arrays are intentionally held by reference.  The object is the internal
    boundary between the numerical pipeline and orchestration/output code;
    :meth:`as_legacy_dict` preserves the pre-refactor mapping API for callers
    that still consume the old result shape.
    """

    original_points: np.ndarray
    original_labels: np.ndarray
    filtered_points: np.ndarray
    filtered_labels: np.ndarray
    features: FeatureMap
    candidate_mask: np.ndarray
    segmentation: SegmentationResult
    planes: list[PlaneInstance]
    joint_sets: list[JointSet]
    spacing_rows: list[SpacingRecord]
    detachment_labels: np.ndarray
    detachment_rows: list[DetachmentRecord]
    rejected: list[RejectedRecord]
    counts: CountMap
    coordinate_origin: np.ndarray
    plane_instances: list[PlaneInstance]
    plane_instance_filtered_labels: np.ndarray
    plane_instance_original_labels: np.ndarray
    plane_merge: PlaneMergeRecord

    def as_legacy_dict(self) -> dict[str, Any]:
        return {
            "original_points": self.original_points,
            "original_labels": self.original_labels,
            "filtered_points": self.filtered_points,
            "filtered_labels": self.filtered_labels,
            "features": self.features,
            "candidate_mask": self.candidate_mask,
            "segmentation": self.segmentation,
            "planes": self.planes,
            "joint_sets": self.joint_sets,
            "spacing_rows": self.spacing_rows,
            "detachment_labels": self.detachment_labels,
            "detachment_rows": self.detachment_rows,
            "rejected": self.rejected,
            "counts": self.counts,
            "coordinate_origin": self.coordinate_origin,
            "plane_instances": self.plane_instances,
            "plane_instance_filtered_labels": self.plane_instance_filtered_labels,
            "plane_instance_original_labels": self.plane_instance_original_labels,
            "plane_merge": self.plane_merge,
        }


@dataclass
class TileResult:
    """Typed result emitted by one whole-cloud tile worker."""

    report: TileReport
    plane_rows: list[TilePlaneRecord]
    spacing_rows: list[SpacingRecord]

    def __iter__(self):
        """Allow legacy worker callers to unpack the historical 3-tuple."""

        yield self.report
        yield self.plane_rows
        yield self.spacing_rows


def finite_or_none(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def plane_to_row(plane: PlaneInstance) -> PlaneRecord:
    """Convert a plane to the stable CSV/JSON field set."""

    return {
        "plane_id": plane.plane_id,
        "set_id": plane.set_id,
        "point_count": int(len(plane.point_indices)),
        "candidate_point_count": int(plane.candidate_point_count),
        "nx": float(plane.normal[0]),
        "ny": float(plane.normal[1]),
        "nz": float(plane.normal[2]),
        "plane_d": float(plane.d),
        "dip_direction_deg": finite_or_none(float(plane.dip_direction)),
        "dip_deg": finite_or_none(float(plane.dip)),
        "centroid_x": float(plane.centroid[0]),
        "centroid_y": float(plane.centroid[1]),
        "centroid_z": float(plane.centroid[2]),
        "bbox_min_x": finite_or_none(float(plane.bbox_min[0])),
        "bbox_min_y": finite_or_none(float(plane.bbox_min[1])),
        "bbox_min_z": finite_or_none(float(plane.bbox_min[2])),
        "bbox_max_x": finite_or_none(float(plane.bbox_max[0])),
        "bbox_max_y": finite_or_none(float(plane.bbox_max[1])),
        "bbox_max_z": finite_or_none(float(plane.bbox_max[2])),
        "area_m2": finite_or_none(float(plane.area)),
        "observed_area_m2": finite_or_none(float(plane.area)),
        "apparent_persistence_m": finite_or_none(float(plane.apparent_persistence)),
        "major_extent_m": finite_or_none(float(plane.major_extent)),
        "minor_extent_m": finite_or_none(float(plane.minor_extent)),
        "trace_length_m": finite_or_none(plane.trace_length),
        "rms_m": finite_or_none(float(plane.rms)),
        "mae_m": finite_or_none(float(plane.mae)),
        "p95_residual_m": finite_or_none(float(plane.p95_residual)),
        "inlier_ratio": finite_or_none(float(plane.inlier_ratio)),
        "planarity": finite_or_none(float(plane.planarity)),
        "normal_dispersion_deg": finite_or_none(float(plane.normal_dispersion)),
        "normal_scale_m": finite_or_none(float(plane.normal_scale_m)),
        "normal_stability_deg": finite_or_none(float(plane.normal_stability_deg)),
        "normal_stable_fraction": finite_or_none(float(plane.normal_stable_fraction)),
        "normal_valid_scale_count": finite_or_none(float(plane.normal_valid_scale_count)),
        "normal_stable": bool(plane.normal_stable),
        "boundary_completeness": finite_or_none(float(plane.boundary_completeness)),
        "edge_censored": bool(plane.edge_censored),
        "confidence": finite_or_none(float(plane.confidence)),
        "quality_score": finite_or_none(float(plane.quality_score)),
        "quality_grade": plane.quality_grade,
        "near_vertical": bool(plane.near_vertical),
        "data_gap_censored": bool(plane.data_gap_censored),
        "trace_verified": bool(plane.trace_verified),
        "aperture_available": bool(plane.aperture_available),
        "aperture_reason": plane.aperture_reason,
    }


def joint_set_to_row(joint_set: JointSet) -> dict[str, Any]:
    spacing = joint_set.spacings
    spacing_3d = joint_set.spacing_3d
    spacing_scanline = joint_set.spacing_virtual_scanline
    stats: dict[str, Any] = {
        "set_id": joint_set.set_id,
        "plane_count": int(joint_set.plane_count),
        "mean_dip_direction_deg": finite_or_none(float(joint_set.mean_dip_direction)),
        "mean_dip_deg": finite_or_none(float(joint_set.mean_dip)),
        "mean_normal_x": float(joint_set.mean_normal[0]),
        "mean_normal_y": float(joint_set.mean_normal[1]),
        "mean_normal_z": float(joint_set.mean_normal[2]),
        "spacing_available": bool(joint_set.spacing_available),
        "spacing_reason": joint_set.spacing_reason,
        "angular_dispersion_deg": finite_or_none(float(joint_set.angular_dispersion)),
        "mean_spacing_m": None,
        "median_spacing_m": None,
        "std_spacing_m": None,
        "min_spacing_m": None,
        "max_spacing_m": None,
        "p10_spacing_m": None,
        "p90_spacing_m": None,
        "spacing_3d_sample_count": int(len(spacing_3d)),
        "spacing_virtual_scanline_sample_count": int(len(spacing_scanline)),
        "mean_spacing_3d_m": None,
        "median_spacing_3d_m": None,
        "mean_spacing_virtual_scanline_m": None,
        "median_spacing_virtual_scanline_m": None,
        "fisher_k": finite_or_none(float(joint_set.fisher_k)),
    }
    if len(spacing):
        stats.update(
            {
                "mean_spacing_m": float(np.mean(spacing)),
                "median_spacing_m": float(np.median(spacing)),
                "std_spacing_m": float(np.std(spacing)),
                "min_spacing_m": float(np.min(spacing)),
                "max_spacing_m": float(np.max(spacing)),
                "p10_spacing_m": float(np.quantile(spacing, 0.10)),
                "p90_spacing_m": float(np.quantile(spacing, 0.90)),
            }
        )
    if len(spacing_3d):
        stats.update(
            {
                "mean_spacing_3d_m": float(np.mean(spacing_3d)),
                "median_spacing_3d_m": float(np.median(spacing_3d)),
            }
        )
    if len(spacing_scanline):
        stats.update(
            {
                "mean_spacing_virtual_scanline_m": float(np.mean(spacing_scanline)),
                "median_spacing_virtual_scanline_m": float(np.median(spacing_scanline)),
            }
        )
    return stats
