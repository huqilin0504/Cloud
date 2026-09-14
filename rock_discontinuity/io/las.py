from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np

@dataclass
class LasReadResult:
    xyz: np.ndarray
    metadata: dict[str, Any]


def _dimension_names(header: laspy.LasHeader) -> set[str]:
    return {str(name).lower() for name in header.point_format.dimension_names}


def read_las_roi(
    path: Path,
    bbox: dict[str, float] | None,
    *,
    chunk_size: int = 500_000,
    max_points_without_roi: int = 2_000_000,
) -> LasReadResult:
    """Read only points inside an optional axis-aligned ROI.

    Chunked reading is deliberate: the current source can exceed one billion
    points and this implementation only promises one small ROI at a time.
    """

    if not path.is_file():
        raise FileNotFoundError(f"LAS/LAZ 不存在：{path}")
    selected: list[np.ndarray] = []
    with laspy.open(path) as reader:
        header = reader.header
        names = _dimension_names(header)
        total = int(header.point_count)
        intensity_nonzero = False
        classification_nonzero = False
        if bbox is None and total > max_points_without_roi:
            raise ValueError(
                f"输入有 {total} 个点，必须提供 20–50m ROI；"
                f"无 ROI 时最多允许 {max_points_without_roi} 个点"
            )
        for points in reader.chunk_iterator(chunk_size):
            xyz_columns: list[np.ndarray] = [
                np.asarray(points.x, dtype=np.float64),
                np.asarray(points.y, dtype=np.float64),
                np.asarray(points.z, dtype=np.float64),
            ]
            xyz = np.column_stack(xyz_columns)
            if "intensity" in names:
                intensity_nonzero = intensity_nonzero or bool(np.any(np.asarray(points.intensity) != 0))
            if "classification" in names:
                classification_nonzero = classification_nonzero or bool(
                    np.any(np.asarray(points.classification) != 0)
                )
            mask = np.isfinite(xyz).all(axis=1)
            if bbox is not None:
                mask &= (xyz[:, 0] >= bbox["min_x"]) & (xyz[:, 0] <= bbox["max_x"])
                mask &= (xyz[:, 1] >= bbox["min_y"]) & (xyz[:, 1] <= bbox["max_y"])
                if "min_z" in bbox:
                    mask &= (xyz[:, 2] >= bbox["min_z"]) & (xyz[:, 2] <= bbox["max_z"])
            if np.any(mask):
                selected.append(xyz[mask])

        crs = None
        try:
            parsed_crs = header.parse_crs()
            crs = parsed_crs.to_string() if parsed_crs else None
        except Exception:
            crs = None

        metadata = {
            "path": str(path.resolve()),
            "total_points": total,
            "selected_points": int(sum(len(chunk) for chunk in selected)),
            "point_format": int(header.point_format.id),
            "dimensions": sorted(names),
            "has_rgb": {"red", "green", "blue"}.issubset(names),
            "has_intensity_dimension": "intensity" in names,
            "has_intensity": bool(intensity_nonzero),
            "has_classification_dimension": "classification" in names,
            "has_classification": bool(classification_nonzero),
            "scale": [float(value) for value in header.scales],
            "offset": [float(value) for value in header.offsets],
            "bounds": {
                "min": [float(value) for value in header.mins],
                "max": [float(value) for value in header.maxs],
            },
            "crs": crs,
        }

    xyz = np.concatenate(selected, axis=0) if selected else np.empty((0, 3), dtype=np.float64)
    if not len(xyz):
        raise ValueError("ROI 内没有有限 XYZ 点")
    metadata["selected_points"] = int(len(xyz))
    return LasReadResult(xyz=xyz, metadata=metadata)
