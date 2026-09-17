from __future__ import annotations

from typing import Any

import numpy as np

from ..core.models import DetachmentRecord, PlaneInstance


def classify_candidate_joint_planes(
    planes: list[PlaneInstance],
    config: dict[str, Any] | None = None,
    *,
    context: str = "roi",
) -> tuple[np.ndarray, list[DetachmentRecord]]:
    """Mark planar facets that pass a conservative discontinuity gate.

    Plane fitting quality alone is insufficient: a smooth exposed rock face
    can be tiled into many excellent local planes. The additional extent and
    boundary gates keep those elementary facets available for QA while only a
    conservative subset is highlighted as planar discontinuity candidates.

    ROI boundary-censored facets are rejected when configured. Tile-local
    processing keeps them because overlap and cross-tile merging must be able
    to recover a plane that crosses a processing boundary.
    """

    config = config or {}
    min_confidence = float(config.get("min_confidence", 0.0))
    min_area = float(config.get("min_area_m2", 0.0))
    min_minor_extent = float(config.get("min_minor_extent_m", 0.0))
    min_major_extent = float(config.get("min_major_extent_m", 0.0))
    min_boundary_completeness = float(config.get("min_boundary_completeness", 0.0))
    min_inlier_ratio = float(config.get("min_inlier_ratio", 0.0))
    max_normal_dispersion = float(config.get("max_normal_dispersion_deg", float("inf")))
    reject_edge_censored = (
        bool(config.get("reject_edge_censored_roi", False)) and context == "roi"
    )
    selected = np.zeros(len(planes), dtype=bool)
    rows: list[DetachmentRecord] = []
    for index, plane in enumerate(planes):
        failures: list[str] = []
        if not np.isfinite(plane.confidence) or plane.confidence < min_confidence:
            failures.append("below_min_confidence")
        if not np.isfinite(plane.area) or plane.area < min_area:
            failures.append("below_min_area")
        if "min_minor_extent_m" in config and (
            not np.isfinite(plane.minor_extent) or plane.minor_extent < min_minor_extent
        ):
            failures.append("below_min_minor_extent")
        if "min_major_extent_m" in config and (
            not np.isfinite(plane.major_extent) or plane.major_extent < min_major_extent
        ):
            failures.append("below_min_major_extent")
        if "min_boundary_completeness" in config and (
            not np.isfinite(plane.boundary_completeness)
            or plane.boundary_completeness < min_boundary_completeness
        ):
            failures.append("below_min_boundary_completeness")
        if "min_inlier_ratio" in config and (
            not np.isfinite(plane.inlier_ratio) or plane.inlier_ratio < min_inlier_ratio
        ):
            failures.append("below_min_inlier_ratio")
        if "max_normal_dispersion_deg" in config and (
            not np.isfinite(plane.normal_dispersion)
            or plane.normal_dispersion > max_normal_dispersion
        ):
            failures.append("above_max_normal_dispersion")
        if reject_edge_censored and plane.edge_censored:
            failures.append("roi_edge_censored")
        quality_ok = not failures
        if quality_ok:
            selected[index] = True
        rows.append(
            {
                "plane_id": plane.plane_id,
                "set_id": plane.set_id,
                "status": "candidate_joint_plane" if quality_ok else "not_selected",
                "selection_reason": (
                    "accepted_conservative_planar_discontinuity_candidate"
                    if quality_ok
                    else ";".join(failures)
                ),
                "area_m2": float(plane.area),
                "major_extent_m": float(plane.major_extent),
                "minor_extent_m": float(plane.minor_extent),
                "boundary_completeness": float(plane.boundary_completeness),
                "edge_censored": bool(plane.edge_censored),
                "inlier_ratio": float(plane.inlier_ratio),
                "normal_dispersion_deg": float(plane.normal_dispersion),
                "confidence": float(plane.confidence),
            }
        )
    return selected, rows


# Compatibility name for existing imports and historical output filenames.
classify_candidate_detachment_planes = classify_candidate_joint_planes
