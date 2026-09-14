from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import KDTree

from ..core.geometry import normal_angle_deg
from ..core.fitting import dip_and_dip_direction
from ..core.models import JointSet, PlaneInstance, SpacingRecord
from ..core.records import nearest_spacing_by_plane


def _finite_patch_spacing(
    plane_a: PlaneInstance,
    plane_b: PlaneInstance,
    points: np.ndarray | None,
    mean_normal: np.ndarray,
    max_samples: int,
) -> tuple[float | None, int]:
    """Estimate nearest finite-patch normal spacing between two plane instances."""

    if points is None:
        return None, 0
    points = np.asarray(points, dtype=np.float64)
    if len(plane_a.point_indices) < 1 or len(plane_b.point_indices) < 1:
        return None, 0
    a = points[plane_a.point_indices]
    b = points[plane_b.point_indices]
    if not len(a) or not len(b):
        return None, 0
    if len(a) > max_samples:
        a = a[np.linspace(0, len(a) - 1, max_samples, dtype=np.int64)]
    if len(b) > max_samples:
        b = b[np.linspace(0, len(b) - 1, max_samples, dtype=np.int64)]
    tree_b = KDTree(b)
    tree_a = KDTree(a)
    try:
        _, nearest_b = tree_b.query(a, k=1, workers=-1)
        _, nearest_a = tree_a.query(b, k=1, workers=-1)
    except TypeError:
        _, nearest_b = tree_b.query(a, k=1)
        _, nearest_a = tree_a.query(b, k=1)
    differences = np.concatenate(
        (
            b[np.asarray(nearest_b, dtype=np.int64)] - a,
            b - a[np.asarray(nearest_a, dtype=np.int64)],
        ),
        axis=0,
    )
    projected = np.abs(differences @ mean_normal)
    finite = projected[np.isfinite(projected) & (projected > 1e-9)]
    if not len(finite):
        return None, 0
    # The median is less sensitive than a minimum to a single accidental
    # cross-edge pair when finite patches only partially overlap.
    return float(np.median(finite)), int(len(finite))


def _fisher_k(normals: np.ndarray) -> float:
    if len(normals) < 2:
        return float("nan")
    resultant = float(np.linalg.norm(np.sum(normals, axis=0)) / len(normals))
    if resultant >= 0.999999:
        return float("inf")
    if resultant < 1e-6:
        return 0.0
    # A stable axial 3-D approximation suitable for a compact report.  The
    # full Fisher confidence interval is intentionally not inferred here.
    return float((3.0 * resultant - resultant**3) / max(1.0 - resultant**2, 1e-12))


def compute_joint_set_spacing(
    set_id: str,
    planes: list[PlaneInstance],
    config: dict[str, Any],
    *,
    points: np.ndarray | None = None,
) -> tuple[JointSet, list[SpacingRecord]]:
    """Compute finite-patch and virtual-scanline spacing for a joint set."""

    normals = []
    for plane in planes:
        normal = plane.normal.copy()
        if normals and np.dot(normal, normals[0]) < 0:
            normal = -normal
        normals.append(normal)
    normal_array = np.asarray(normals, dtype=np.float64)
    mean_normal = np.sum(normal_array, axis=0)
    mean_normal /= max(float(np.linalg.norm(mean_normal)), 1e-15)
    if mean_normal[2] < 0:
        mean_normal = -mean_normal
    mean_dip_direction, mean_dip = dip_and_dip_direction(mean_normal)
    deviations = np.asarray(
        [normal_angle_deg(normal, mean_normal) or 0.0 for normal in normal_array],
        dtype=np.float64,
    )
    angular_dispersion = float(np.max(deviations)) if len(deviations) else float("nan")
    plane_ids = [plane.plane_id for plane in planes]
    min_planes = int(config.get("min_planes", 2))
    if len(planes) == 0:
        return (
            JointSet(
                set_id,
                plane_ids,
                mean_normal,
                mean_dip_direction,
                mean_dip,
                0,
                False,
                "no_planes",
                np.empty(0, dtype=float),
                angular_dispersion,
            ),
            [],
        )

    alignment = float(np.cos(np.deg2rad(float(config.get("max_set_deviation_deg", 12.0)))))
    parameters: list[tuple[float, PlaneInstance]] = []
    x0 = np.mean([plane.centroid for plane in planes], axis=0)
    for plane in planes:
        denominator = float(np.dot(plane.normal, mean_normal))
        if denominator < alignment:
            return (
                JointSet(
                    set_id,
                    plane_ids,
                    mean_normal,
                    mean_dip_direction,
                    mean_dip,
                    len(planes),
                    False,
                    "normal_dispersion_too_large",
                    np.empty(0, dtype=float),
                    angular_dispersion,
                    fisher_k=_fisher_k(normal_array),
                ),
                [],
            )
        t = -float(np.dot(plane.normal, x0) + plane.d) / denominator
        parameters.append((t, plane))
    parameters.sort(key=lambda item: item[0])

    rows: list[SpacingRecord] = []
    finite_values: list[float] = []
    virtual_values: list[float] = []
    finite_enabled = bool(config.get("finite_3d_enabled", True))
    virtual_enabled = bool(config.get("virtual_scanline_enabled", True))
    max_samples = max(10, int(config.get("max_pair_samples", 2_000)))
    for (t_a, plane_a), (t_b, plane_b) in zip(parameters, parameters[1:]):
        if virtual_enabled:
            virtual = abs(float(t_b - t_a))
            if virtual > 1e-9:
                virtual_values.append(virtual)
                rows.append(
                    {
                        "set_id": set_id,
                        "plane_id_a": plane_a.plane_id,
                        "plane_id_b": plane_b.plane_id,
                        "spacing_m": virtual,
                        "method": "spacing_virtual_scanline",
                        "sample_count": 1,
                    }
                )
        if finite_enabled:
            finite, sample_count = _finite_patch_spacing(
                plane_a,
                plane_b,
                points,
                mean_normal,
                max_samples,
            )
            if finite is not None and finite > 1e-9:
                finite_values.append(finite)
                rows.append(
                    {
                        "set_id": set_id,
                        "plane_id_a": plane_a.plane_id,
                        "plane_id_b": plane_b.plane_id,
                        "spacing_m": finite,
                        "method": "spacing_3d_nonpersistent",
                        "sample_count": sample_count,
                    }
                )

    finite_array = np.asarray(finite_values, dtype=np.float64)
    virtual_array = np.asarray(virtual_values, dtype=np.float64)
    preferred = finite_array if len(finite_array) else virtual_array
    if len(planes) < min_planes:
        reason = "fewer_than_min_planes_for_set_statistics"
    elif not len(preferred):
        reason = "no_positive_spacing"
    else:
        reason = None
    joint_set = JointSet(
        set_id,
        plane_ids,
        mean_normal,
        mean_dip_direction,
        mean_dip,
        len(planes),
        bool(len(preferred)),
        reason,
        preferred,
        angular_dispersion,
        spacing_3d=finite_array,
        spacing_virtual_scanline=virtual_array,
        fisher_k=_fisher_k(normal_array),
    )
    return joint_set, rows
