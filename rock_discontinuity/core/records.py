"""Small, dependency-free record transformations shared by processing and IO."""

from __future__ import annotations

from typing import Any

import numpy as np


def nearest_spacing_by_plane(spacing_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Return the nearest positive measured spacing for each plane.

    This is a pure record reduction, so it belongs below both the numerical
    processing and file-output layers.  It intentionally ignores malformed or
    non-positive values to preserve the historical CSV behavior.
    """

    nearest: dict[str, float] = {}
    for row in spacing_rows:
        try:
            spacing = float(row["spacing_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(spacing) or spacing <= 0:
            continue
        for field in ("plane_id_a", "plane_id_b"):
            plane_id = row.get(field)
            if not plane_id:
                continue
            current = nearest.get(str(plane_id))
            if current is None or spacing < current:
                nearest[str(plane_id)] = spacing
    return nearest

