"""Constant-memory LAS/LAZ cropping in an oblique ENU projection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import laspy
import numpy as np
from shapely import contains_xy, prepare
from shapely.geometry import Polygon
from tqdm.auto import tqdm


@dataclass(frozen=True)
class ProjectionPolygonROI:
    """A polygon expressed in an oblique orthographic projection of ENU XYZ."""

    origin_xyz: np.ndarray
    azimuth_deg: float
    elevation_deg: float
    polygon_uv: np.ndarray
    boundary_buffer_m: float = 0.0
    name: str = "projection_polygon_roi"


def projection_roi_from_mapping(value: Mapping[str, Any]) -> ProjectionPolygonROI:
    projection = value.get("projection")
    if not isinstance(projection, Mapping):
        raise ValueError("ROI 配置缺少 projection 对象")
    origin = np.asarray(projection.get("origin_xyz"), dtype=np.float64)
    polygon_uv = np.asarray(value.get("polygon_uv"), dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("projection.origin_xyz 必须是三个有限 ENU 坐标")
    if polygon_uv.ndim != 2 or polygon_uv.shape[1] != 2 or len(polygon_uv) < 3:
        raise ValueError("polygon_uv 至少需要三个二维顶点")
    if not np.isfinite(polygon_uv).all():
        raise ValueError("polygon_uv 不能包含 NaN 或 Inf")
    azimuth = float(projection.get("azimuth_deg", 0.0))
    elevation = float(projection.get("elevation_deg", 0.0))
    boundary_buffer = float(value.get("boundary_buffer_m", 0.0))
    if not np.isfinite([azimuth, elevation, boundary_buffer]).all():
        raise ValueError("投影角度和边界缓冲必须是有限数")
    if boundary_buffer < 0.0:
        raise ValueError("boundary_buffer_m 不能小于 0")
    polygon = Polygon(polygon_uv)
    if not polygon.is_valid or polygon.area <= 0.0:
        raise ValueError("polygon_uv 必须构成无自交且面积大于 0 的多边形")
    return ProjectionPolygonROI(
        origin_xyz=origin,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        polygon_uv=polygon_uv,
        boundary_buffer_m=boundary_buffer,
        name=str(value.get("name", "projection_polygon_roi")),
    )


def load_projection_roi(path: Path) -> ProjectionPolygonROI:
    if not path.is_file():
        raise FileNotFoundError(f"ROI 配置不存在：{path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError("ROI 配置必须是 JSON 对象")
    return projection_roi_from_mapping(value)


def project_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    roi: ProjectionPolygonROI,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ENU coordinates to the ROI's screen-plane coordinates."""

    azimuth = np.deg2rad(roi.azimuth_deg)
    elevation = np.deg2rad(roi.elevation_deg)
    dx = np.asarray(x, dtype=np.float64) - roi.origin_xyz[0]
    dy = np.asarray(y, dtype=np.float64) - roi.origin_xyz[1]
    dz = np.asarray(z, dtype=np.float64) - roi.origin_xyz[2]
    u = dx * np.cos(azimuth) + dy * np.sin(azimuth)
    horizontal_depth = -dx * np.sin(azimuth) + dy * np.cos(azimuth)
    v = dz * np.cos(elevation) - horizontal_depth * np.sin(elevation)
    return u, v


def projection_polygon_mask(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    roi: ProjectionPolygonROI,
    *,
    polygon: Polygon | None = None,
) -> np.ndarray:
    """Return a vectorized mask for finite points inside the projected polygon."""

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    z_array = np.asarray(z, dtype=np.float64)
    if not (x_array.shape == y_array.shape == z_array.shape):
        raise ValueError("X、Y、Z 数组形状必须一致")
    result = np.zeros(x_array.shape, dtype=bool)
    finite = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(z_array)
    if not np.any(finite):
        return result
    geometry = polygon
    if geometry is None:
        geometry = Polygon(roi.polygon_uv)
        if roi.boundary_buffer_m > 0.0:
            geometry = geometry.buffer(roi.boundary_buffer_m)
        prepare(geometry)
    u, v = project_xyz(x_array[finite], y_array[finite], z_array[finite], roi)
    min_u, min_v, max_u, max_v = geometry.bounds
    bbox = (u >= min_u) & (u <= max_u) & (v >= min_v) & (v <= max_v)
    if not np.any(bbox):
        return result
    finite_indices = np.flatnonzero(finite)
    candidate_indices = finite_indices[bbox]
    result[candidate_indices] = contains_xy(geometry, u[bbox], v[bbox])
    return result


def _bounds_payload(lower: np.ndarray, upper: np.ndarray) -> dict[str, list[float]] | None:
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        return None
    return {"min": lower.tolist(), "max": upper.tolist()}


def crop_las_to_projection_roi(
    input_path: Path,
    output_path: Path,
    roi: ProjectionPolygonROI,
    *,
    report_path: Path | None = None,
    chunk_size: int = 1_000_000,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Stream a LAS/LAZ file through a projection polygon without losing dimensions."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"LAS/LAZ 不存在：{input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("输出文件不能覆盖输入文件")
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"输出已存在；如需覆盖请使用 --overwrite：{output_path}")
    if report_path is not None and report_path.exists() and not overwrite:
        raise FileExistsError(f"报告已存在；如需覆盖请使用 --overwrite：{report_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".part")
    if temporary.exists():
        if not overwrite:
            raise FileExistsError(f"发现未完成临时文件：{temporary}")
        temporary.unlink()

    geometry = Polygon(roi.polygon_uv)
    if roi.boundary_buffer_m > 0.0:
        geometry = geometry.buffer(roi.boundary_buffer_m)
    prepare(geometry)
    selected_total = 0
    finite_total = 0
    selected_min = np.full(3, np.inf, dtype=np.float64)
    selected_max = np.full(3, -np.inf, dtype=np.float64)

    try:
        with laspy.open(input_path) as reader:
            source_header = reader.header
            source_total = int(source_header.point_count)
            source_dimensions = [str(name) for name in source_header.point_format.dimension_names]
            output_header = deepcopy(source_header)
            progress = tqdm(
                total=source_total,
                desc="裁剪上岸岩坡 ROI",
                unit="点",
                unit_scale=True,
                disable=not show_progress,
                dynamic_ncols=True,
            )
            with progress, laspy.open(
                temporary,
                mode="w",
                header=output_header,
                do_compress=output_path.suffix.lower() == ".laz",
            ) as writer:
                for points in reader.chunk_iterator(chunk_size):
                    x = np.asarray(points.x, dtype=np.float64)
                    y = np.asarray(points.y, dtype=np.float64)
                    z = np.asarray(points.z, dtype=np.float64)
                    finite_total += int((np.isfinite(x) & np.isfinite(y) & np.isfinite(z)).sum())
                    mask = projection_polygon_mask(x, y, z, roi, polygon=geometry)
                    if np.any(mask):
                        selected = points[mask]
                        writer.write_points(selected)
                        selected_xyz = np.column_stack((x[mask], y[mask], z[mask]))
                        selected_min = np.minimum(selected_min, selected_xyz.min(axis=0))
                        selected_max = np.maximum(selected_max, selected_xyz.max(axis=0))
                        selected_total += int(mask.sum())
                    progress.update(len(points))
                    progress.set_postfix_str(f"保留 {selected_total:,}")
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    report: dict[str, Any] = {
        "status": "complete",
        "method": "streaming_oblique_projection_polygon",
        "input": {
            "path": str(input_path.resolve()),
            "point_count": source_total,
            "dimensions": source_dimensions,
        },
        "output": {
            "path": str(output_path.resolve()),
            "point_count": selected_total,
            "selected_fraction": selected_total / source_total if source_total else 0.0,
            "bounds": _bounds_payload(selected_min, selected_max),
        },
        "finite_input_points": finite_total,
        "discarded_nonfinite_points": source_total - finite_total,
        "roi": {
            "name": roi.name,
            "coordinate_system": "ENU",
            "origin_xyz": roi.origin_xyz.tolist(),
            "azimuth_deg": roi.azimuth_deg,
            "elevation_deg": roi.elevation_deg,
            "boundary_buffer_m": roi.boundary_buffer_m,
            "polygon_uv": roi.polygon_uv.tolist(),
        },
        "preservation": {
            "point_format": int(output_header.point_format.id),
            "version": str(output_header.version),
            "scale": [float(value) for value in output_header.scales],
            "offset": [float(value) for value in output_header.offsets],
            "all_source_dimensions_preserved": True,
        },
        "chunk_size": int(chunk_size),
    }
    if report_path is not None:
        temporary_report = report_path.with_name(report_path.name + ".tmp")
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_report.replace(report_path)
    return report


__all__ = [
    "ProjectionPolygonROI",
    "crop_las_to_projection_roi",
    "load_projection_roi",
    "project_xyz",
    "projection_polygon_mask",
    "projection_roi_from_mapping",
]
