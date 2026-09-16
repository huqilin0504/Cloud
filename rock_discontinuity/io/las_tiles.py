"""LAS/COPC primitives used by the whole-cloud tiling workflow.

The processing layer decides which tiles to request and how to assign points
to a core cell.  This module owns the format-specific reads and writes so the
orchestrator does not need to know LAS header details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import laspy
import numpy as np
from laspy.copc import CopcReader


def is_copc_file(path: Path) -> bool:
    """Detect COPC from its VLR instead of trusting a filename suffix."""

    try:
        with CopcReader.open(path):
            return True
    except Exception:
        return False


def source_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"LAS/LAZ 不存在：{path}")
    is_copc = is_copc_file(path)
    with laspy.open(path) as reader:
        header = reader.header
        parsed_crs = None
        try:
            crs = header.parse_crs()
            parsed_crs = crs.to_string() if crs else None
        except Exception:
            parsed_crs = None
        return {
            "path": str(path.resolve()),
            "total_points": int(header.point_count),
            "point_format": int(header.point_format.id),
            "scale": [float(value) for value in header.scales],
            "offset": [float(value) for value in header.offsets],
            "bounds": {
                "min": [float(value) for value in header.mins],
                "max": [float(value) for value in header.maxs],
            },
            "crs": parsed_crs,
            "is_copc": is_copc,
            "source_format": "COPC" if is_copc else "LAS/LAZ",
            "dimensions": sorted(str(name).lower() for name in header.point_format.dimension_names),
        }


def source_crs(path: Path) -> Any:
    """Read the source CRS without exposing a LAS reader to processing code."""

    with laspy.open(path) as reader:
        try:
            return reader.header.parse_crs()
        except Exception:
            return None


def make_laz_header(source: dict[str, Any], crs: Any = None) -> laspy.LasHeader:
    header = laspy.LasHeader(point_format=2, version="1.4")
    header.scales = source["scale"]
    header.offsets = source["offset"]
    if crs is not None:
        try:
            header.add_crs(crs)
        except Exception:
            pass
    return header


def write_copc_query(
    path: Path,
    records: Any,
    source: dict[str, Any],
    crs: Any = None,
) -> int:
    """Materialize one COPC spatial query as a compact XYZ-only LAZ tile."""

    point_count = int(len(records))
    if point_count <= 0:
        return 0
    header = make_laz_header(source, crs)
    temporary = path.with_name(path.name + ".part")
    with laspy.open(temporary, mode="w", header=header) as writer:
        for start in range(0, point_count, 500_000):
            stop = min(point_count, start + 500_000)
            record = laspy.ScaleAwarePointRecord.zeros(stop - start, header=header)
            record.x = np.asarray(records.x[start:stop], dtype=np.float64)
            record.y = np.asarray(records.y[start:stop], dtype=np.float64)
            record.z = np.asarray(records.z[start:stop], dtype=np.float64)
            writer.write_points(record)
    temporary.replace(path)
    return point_count


def write_red_overlay(
    path: Path,
    points: np.ndarray,
    source: dict[str, Any],
    crs: Any = None,
    plane_indices: np.ndarray | None = None,
) -> None:
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        if path.exists():
            path.unlink()
        return
    header = make_laz_header(source, crs)
    temporary = path.with_name(path.name + ".part")
    with laspy.open(temporary, mode="w", header=header) as writer:
        for start in range(0, len(points), 500_000):
            chunk = points[start : start + 500_000]
            record = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
            record.x, record.y, record.z = chunk.T
            # LAS RGB dimensions are 16-bit, unlike the 8-bit PLY colors.
            record.red = np.full(len(chunk), 230 * 257, dtype=np.uint16)
            record.green = np.full(len(chunk), 35 * 257, dtype=np.uint16)
            record.blue = np.full(len(chunk), 35 * 257, dtype=np.uint16)
            if plane_indices is not None:
                record.point_source_id = np.asarray(
                    plane_indices[start : start + len(chunk)],
                    dtype=np.uint16,
                )
            writer.write_points(record)
    temporary.replace(path)


def merge_overlays(
    overlay_paths: list[Path],
    output_path: Path,
    source: dict[str, Any],
    source_crs: Any,
    selected_plane_indices_by_tile: dict[str, set[int]] | None,
    *,
    parse_tile_name: Callable[[Path], tuple[int, int]],
    progress: Callable[[str, int, int, str], None] | None = None,
) -> int:
    """Merge candidate LAZ overlays while applying global plane selection."""

    header = make_laz_header(source, source_crs)
    temporary = output_path.with_name(output_path.name + ".part")
    total = 0
    overlay_total = max(1, len(overlay_paths))
    if progress is not None:
        progress("合并候选点云", 0, overlay_total, f"输入 {len(overlay_paths)} 个瓦片")
    with laspy.open(temporary, mode="w", header=header):
        pass
    with laspy.open(temporary, mode="a") as writer:
        for overlay_index, overlay in enumerate(overlay_paths, start=1):
            if not overlay.is_file():
                if progress is not None:
                    progress("合并候选点云", overlay_index, overlay_total, f"跳过 {overlay.name}")
                continue
            with laspy.open(overlay) as reader:
                for points in reader.chunk_iterator(500_000):
                    if selected_plane_indices_by_tile is not None:
                        ix, iy = parse_tile_name(overlay)
                        allowed = selected_plane_indices_by_tile.get(f"{ix}_{iy}", set())
                        mask = np.isin(np.asarray(points.point_source_id), list(allowed))
                        points = points[mask]
                        if not len(points):
                            continue
                    writer.append_points(points)
                    total += len(points)
            if progress is not None:
                progress("合并候选点云", overlay_index, overlay_total, f"完成 {overlay.name}，{total} 点")
    if not overlay_paths and progress is not None:
        progress("合并候选点云", overlay_total, overlay_total, "无候选瓦片")
    temporary.replace(output_path)
    return total
