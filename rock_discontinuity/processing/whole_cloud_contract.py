"""Stable whole-cloud run contract constants."""

WHOLE_ALGORITHM_VERSION = "0.7.0-multiscale-stability-hierarchical-merge"
GLOBAL_AGGREGATION_VERSION = "0.8.0-adaptive-orientation-bisection-merge-diagnostics"


def _packaged_defaults() -> dict[str, object]:
    from ..config import load_config

    return load_config(None)


def footprint_max_points(config: dict[str, object]) -> int:
    """Resolve the footprint sampling limit from the YAML default source.

    A partial caller configuration is allowed for compatibility.  The
    fallback reloads the packaged YAML instead of duplicating its default in
    Python, so a missing user field still has the same runtime value.
    """

    whole_cloud = config.get("whole_cloud")
    if isinstance(whole_cloud, dict) and whole_cloud.get("footprint_max_points") is not None:
        return int(whole_cloud["footprint_max_points"])
    defaults = _packaged_defaults()
    return int(defaults["whole_cloud"]["footprint_max_points"])


def whole_cloud_workers(config: dict[str, object]) -> int:
    section = config.get("whole_cloud")
    if isinstance(section, dict) and section.get("workers") is not None:
        return int(section["workers"])
    defaults = _packaged_defaults()
    return int(defaults["whole_cloud"]["workers"])


def tiling_parameters(
    config: dict[str, object],
    tile_size: float | None = None,
    overlap: float | None = None,
) -> tuple[float, float]:
    """Resolve CLI/API overrides on top of the packaged YAML defaults."""

    section = config.get("tiling")
    if not isinstance(section, dict):
        section = _packaged_defaults()["tiling"]
    if tile_size is None:
        tile_size = float(section["tile_size_m"])
    if overlap is None:
        overlap = float(section["overlap_m"])
    return float(tile_size), float(overlap)
