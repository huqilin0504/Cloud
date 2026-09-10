"""Single-tile numerical processing for whole-cloud jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..core.geometry import footprint_rings_3d
from ..core.models import TilePlaneRecord, TileResult, SpacingRecord, plane_to_row
from ..core.records import nearest_spacing_by_plane
from ..io import read_las_roi
from .density import assess_density, estimate_roi_density
from .pipeline import process_points
from .tiling import core_bbox, core_mask, expanded_bbox, parse_tile_name
from .whole_cloud_contract import WHOLE_ALGORITHM_VERSION, footprint_max_points


def process_tile(
    tile_path: Path,
    *,
    plan: dict[str, Any],
    config: dict[str, Any],
    source: dict[str, Any],
    source_crs: Any,
    overlay_dir: Path,
    write_red_overlay: Callable[..., None],
) -> TileResult:
    """Run the shared point pipeline and prepare one tile's output records."""

    ix, iy = parse_tile_name(tile_path)
    tile_id = f"{ix}_{iy}"
    core = core_bbox(plan, ix, iy)
    expanded = expanded_bbox(core, float(plan["overlap_m"]))
    read_result = read_las_roi(
        tile_path,
        expanded,
        chunk_size=int(config["input"]["chunk_size"]),
        max_points_without_roi=int(config["input"]["max_points_without_roi"]),
    )
    tile_points = read_result.xyz
    result = process_points(
        tile_points,
        config,
        read_metadata=read_result.metadata,
        processing_context="tile",
    )
    tile_density = assess_density(
        estimate_roi_density(
            tile_points,
            sample_size=min(int(config.get("density", {}).get("sample_size", 20_000)), 20_000),
            knn=int(config.get("density", {}).get("knn", 20)),
        ),
        config.get("density", {}),
    )
    labels = result.original_labels
    selected_planes = result.detachment_labels
    candidate_mask = np.zeros(len(labels), dtype=bool)
    for plane_index, selected in enumerate(selected_planes):
        if selected:
            candidate_mask |= labels == plane_index
    core_points_mask = core_mask(tile_points, core, ix, iy, plan)
    red_mask = core_points_mask & candidate_mask
    red_points = tile_points[red_mask]
    red_plane_indices = labels[red_mask] + 1
    overlay_path = overlay_dir / f"tile_{ix}_{iy}.laz"
    if np.any(red_plane_indices > np.iinfo(np.uint16).max):
        raise RuntimeError(f"瓦片 {tile_id} 的平面索引超过 LAS point_source_id 容量")
    write_red_overlay(
        overlay_path,
        red_points,
        source,
        source_crs,
        plane_indices=red_plane_indices,
    )

    spacing_by_plane = nearest_spacing_by_plane(result.spacing_rows)
    spacing_rows: list[SpacingRecord] = []
    for spacing_row in result.spacing_rows:
        plane_id_a = str(spacing_row["plane_id_a"])
        plane_id_b = str(spacing_row["plane_id_b"])
        spacing_rows.append(
            {
                "tile_id": tile_id,
                "plane_id_a": plane_id_a,
                "plane_id_b": plane_id_b,
                "global_plane_id_a": f"{tile_id}:{plane_id_a}",
                "global_plane_id_b": f"{tile_id}:{plane_id_b}",
                "spacing_m": spacing_row["spacing_m"],
                "method": spacing_row["method"],
            }
        )

    plane_rows: list[TilePlaneRecord] = []
    for plane_index, plane in enumerate(result.planes):
        if plane_index >= len(selected_planes) or not selected_planes[plane_index]:
            continue
        plane_mask = red_mask & (labels == plane_index)
        core_red_points = int(plane_mask.sum())
        if core_red_points <= 0:
            continue
        detachment_row = result.detachment_rows[plane_index]
        row = plane_to_row(plane)
        plane_points = tile_points[plane_mask]
        row.update(
            {
                "tile_id": tile_id,
                "global_plane_id": f"{tile_id}:{plane.plane_id}",
                "core_red_points": core_red_points,
                "tile_plane_index": plane_index + 1,
                "_footprint_xyz": footprint_rings_3d(
                    plane_points,
                    plane.centroid,
                    plane.normal,
                    alpha=config.get("boundary", {}).get("alpha", "auto"),
                    max_points=footprint_max_points(config),
                ),
                "status": detachment_row["status"],
                "selection_reason": detachment_row["selection_reason"],
                "nearest_spacing_m": spacing_by_plane.get(plane.plane_id),
                "center_x": float(plane.centroid[0]),
                "center_y": float(plane.centroid[1]),
                "center_z": float(plane.centroid[2]),
            }
        )
        plane_rows.append(row)

    tile_report = {
        "algorithm_version": WHOLE_ALGORITHM_VERSION,
        "tile_id": tile_id,
        "ix": ix,
        "iy": iy,
        "core_bbox": core,
        "expanded_bbox": expanded,
        "source_tile": str(tile_path),
        "overlay": str(overlay_path) if len(red_points) else None,
        "status": "done",
        "counts": {
            **result.counts,
            "core_points": int(core_points_mask.sum()),
            "candidate_red_points": int(len(red_points)),
            "candidate_plane_instances": len(plane_rows),
        },
        "plane_merge": result.plane_merge,
        "density": {
            "core_xy_bbox_points_m2": float(core_points_mask.sum() / (float(plan["tile_size_m"]) ** 2)),
            "local_surface_density": tile_density,
        },
        "plane_rows": len(plane_rows),
        "spacing_rows": len(spacing_rows),
    }
    return TileResult(tile_report, plane_rows, spacing_rows)
