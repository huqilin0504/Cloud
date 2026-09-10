from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, KDTree, distance
from shapely.geometry import MultiLineString, MultiPoint, Polygon
from shapely.ops import polygonize, unary_union


def normalize_normal(normal: np.ndarray) -> np.ndarray | None:
    """Return a unit normal or ``None`` for a degenerate vector."""

    value = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(norm) or norm <= 1e-15:
        return None
    return value / norm


def normal_angle_deg(normal_a: np.ndarray, normal_b: np.ndarray) -> float | None:
    """Axial angle between two normals in degrees (0..90)."""

    first = normalize_normal(normal_a)
    second = normalize_normal(normal_b)
    if first is None or second is None:
        return None
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(first, second))), 0.0, 1.0))))


def normal_angle_matrix(normals: np.ndarray) -> np.ndarray:
    """Return the axial angle matrix for a batch of normal vectors."""

    values = np.asarray(normals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("normals 必须是形状为 (n, 3) 的数组")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-15):
        raise ValueError("normals 不能包含零向量或非有限值")
    unit = values / norms
    return np.degrees(np.arccos(np.clip(np.abs(unit @ unit.T), 0.0, 1.0)))


def align_plane_equation(
    normal_a: np.ndarray,
    d_a: float,
    normal_b: np.ndarray,
    d_b: float,
) -> tuple[np.ndarray, float, np.ndarray, float] | None:
    """Normalize two plane equations and align their normal directions."""

    first = normalize_normal(normal_a)
    second = normalize_normal(normal_b)
    if first is None or second is None:
        return None
    first_d = float(d_a)
    second_d = float(d_b)
    if float(np.dot(first, second)) < 0.0:
        second = -second
        second_d = -second_d
    return first, first_d, second, second_d


def plane_offset_between(
    normal_a: np.ndarray,
    d_a: float,
    normal_b: np.ndarray,
    d_b: float,
) -> float | None:
    """Absolute offset between aligned plane equations."""

    aligned = align_plane_equation(normal_a, d_a, normal_b, d_b)
    if aligned is None:
        return None
    _, first_d, _, second_d = aligned
    return abs(first_d - second_d)


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, reference))) > 0.90:
        reference = np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, reference)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v


def project_to_plane(points: np.ndarray, centroid: np.ndarray, normal: np.ndarray) -> np.ndarray:
    u, v = plane_basis(normal)
    centered = np.asarray(points, dtype=np.float64) - np.asarray(centroid, dtype=np.float64)
    return np.column_stack((centered @ u, centered @ v))


def footprint_rings_3d(
    points: np.ndarray,
    centroid: np.ndarray,
    normal: np.ndarray,
    *,
    alpha: float | str = "auto",
    max_points: int = 2_000,
) -> list[list[list[float]]]:
    """Return compact global-XYZ rings for a planar patch footprint."""

    projected = project_to_plane(points, centroid, normal)
    geometry = alpha_shape_geometry(projected, alpha=alpha, max_points=max_points)
    if geometry.is_empty:
        return []
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(getattr(geometry, "geoms", []))
    u, v = plane_basis(normal)
    origin = np.asarray(centroid, dtype=np.float64)
    rings: list[list[list[float]]] = []
    for polygon in polygons:
        if polygon.geom_type != "Polygon" or polygon.is_empty:
            continue
        coordinates = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
        if len(coordinates) < 3:
            continue
        xyz = origin + coordinates[:, :1] * u + coordinates[:, 1:2] * v
        rings.append(xyz.tolist())
    return rings


def footprint_geometry_on_plane(
    rings: list[list[list[float]]] | None,
    origin: np.ndarray,
    normal: np.ndarray,
):
    """Project serialized 3-D footprint rings onto one comparison plane."""

    if not rings:
        return Polygon()
    u, v = plane_basis(normal)
    origin = np.asarray(origin, dtype=np.float64)
    polygons = []
    for ring in rings:
        xyz = np.asarray(ring, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3:
            continue
        centered = xyz - origin
        polygon = Polygon(np.column_stack((centered @ u, centered @ v))).buffer(0)
        if not polygon.is_empty:
            polygons.append(polygon)
    return unary_union(polygons).buffer(0) if polygons else Polygon()


def footprint_gap_on_plane(
    rings_a: list[list[list[float]]] | None,
    rings_b: list[list[list[float]]] | None,
    origin: np.ndarray,
    normal: np.ndarray,
) -> float | None:
    geometry_a = footprint_geometry_on_plane(rings_a, origin, normal)
    geometry_b = footprint_geometry_on_plane(rings_b, origin, normal)
    if geometry_a.is_empty or geometry_b.is_empty:
        return None
    return float(geometry_a.distance(geometry_b))


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def _convex_hull_geometry(points: np.ndarray):
    if len(points) < 3:
        return MultiPoint(points).convex_hull
    try:
        return MultiPoint(points).convex_hull
    except Exception:
        return Polygon()


def alpha_shape_geometry(points: np.ndarray, alpha: float | str = "auto", max_points: int = 5_000):
    """Build a 2-D alpha shape from projected plane points.

    Degenerate or overly sparse inputs fall back to a convex hull and are
    reported by the caller through the resulting boundary completeness.
    """

    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(points) < 4:
        return _convex_hull_geometry(points)
    points = _sample_points(points, max_points)
    try:
        if alpha == "auto":
            tree = KDTree(points)
            nearest, _ = tree.query(points, k=2)
            median_spacing = float(np.median(nearest[:, 1]))
            alpha_value = 1.0 / max(3.0 * median_spacing, 1e-12)
        else:
            alpha_value = float(alpha)
        if alpha_value <= 0:
            return _convex_hull_geometry(points)

        triangulation = Delaunay(points)
        edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
        threshold = 1.0 / alpha_value
        for triangle in triangulation.simplices:
            a, b, c = points[triangle]
            side_a = float(np.linalg.norm(b - c))
            side_b = float(np.linalg.norm(a - c))
            side_c = float(np.linalg.norm(a - b))
            vector_ab = b - a
            vector_ac = c - a
            twice_area = abs(float(vector_ab[0] * vector_ac[1] - vector_ab[1] * vector_ac[0]))
            if twice_area <= 1e-15:
                continue
            circumradius = side_a * side_b * side_c / (2.0 * twice_area)
            if circumradius > threshold:
                continue
            edges.extend(
                [
                    ((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))),
                    ((float(b[0]), float(b[1])), (float(c[0]), float(c[1]))),
                    ((float(c[0]), float(c[1])), (float(a[0]), float(a[1]))),
                ]
            )
        if not edges:
            return _convex_hull_geometry(points)
        polygons = list(polygonize(MultiLineString(edges)))
        if not polygons:
            return _convex_hull_geometry(points)
        geometry = unary_union(polygons)
        return geometry.buffer(0)
    except Exception:
        return _convex_hull_geometry(points)


def geometry_metrics(projected_points: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    projected_points = np.asarray(projected_points, dtype=np.float64)
    geometry = alpha_shape_geometry(
        projected_points,
        alpha=config.get("alpha", "auto"),
        max_points=int(config.get("max_points", 5_000)),
    )
    hull = _convex_hull_geometry(projected_points)
    area = float(getattr(geometry, "area", 0.0))
    hull_area = float(getattr(hull, "area", 0.0))
    completeness = area / hull_area if hull_area > 1e-12 else 0.0
    try:
        rectangle = geometry.minimum_rotated_rectangle
        rectangle_points = np.asarray(rectangle.exterior.coords, dtype=np.float64)
        side_lengths = np.linalg.norm(np.diff(rectangle_points, axis=0), axis=1)
        minor_extent = float(np.min(side_lengths)) if len(side_lengths) else 0.0
        major_extent = float(np.max(side_lengths)) if len(side_lengths) else 0.0
    except Exception:
        extents = np.ptp(projected_points, axis=0) if len(projected_points) else np.zeros(2)
        minor_extent = float(np.min(extents))
        major_extent = float(np.max(extents))

    try:
        hull_indices = ConvexHull(projected_points).vertices if len(projected_points) >= 3 else []
        hull_points = projected_points[hull_indices] if len(hull_indices) else projected_points
    except Exception:
        hull_points = projected_points
    if len(hull_points) >= 2:
        apparent_persistence = float(np.max(distance.pdist(hull_points)))
    else:
        apparent_persistence = 0.0
    return {
        "area": area,
        "minor_extent": minor_extent,
        "major_extent": major_extent,
        "apparent_persistence": apparent_persistence,
        "boundary_completeness": float(np.clip(completeness, 0.0, 1.0)),
        "geometry": geometry,
    }
