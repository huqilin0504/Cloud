from __future__ import annotations

from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


def _read_default_config() -> dict[str, Any]:
    """Load the package-owned YAML defaults.

    Keeping defaults in one resource makes installed wheels behave exactly
    like a source checkout and prevents Python/YAML drift.
    """

    package = resources.files("rock_discontinuity.config")
    value = yaml.safe_load(package.joinpath("default.yaml").read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("rock_discontinuity/config/default.yaml 必须是 YAML 对象")
    return value


def _default_config() -> dict[str, Any]:
    return deepcopy(_read_default_config())


# Compatibility export for callers that imported the old constant.  The
# source of truth remains the YAML resource above; callers receive a fresh
# copy through ``load_config``.
DEFAULT_CONFIG = _read_default_config()


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None) -> dict[str, Any]:
    config = _default_config()
    if path is None:
        return config
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置必须是 YAML 对象：{path}")
    return deep_merge(config, data)


def parse_bbox(value: str | list[float] | tuple[float, ...] | None) -> dict[str, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        numbers = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        numbers = [float(item) for item in value]
    if len(numbers) not in (4, 6):
        raise ValueError("ROI bbox 需要 xmin,xmax,ymin,ymax，或再加 zmin,zmax")
    bbox = {
        "min_x": numbers[0],
        "max_x": numbers[1],
        "min_y": numbers[2],
        "max_y": numbers[3],
    }
    if len(numbers) == 6:
        bbox.update({"min_z": numbers[4], "max_z": numbers[5]})
    if bbox["min_x"] >= bbox["max_x"] or bbox["min_y"] >= bbox["max_y"]:
        raise ValueError("ROI bbox 的最小值必须小于最大值")
    if "min_z" in bbox and bbox["min_z"] >= bbox["max_z"]:
        raise ValueError("ROI bbox 的 z 最小值必须小于最大值")
    return bbox


def config_bbox(config: dict[str, Any]) -> dict[str, float] | None:
    roi = config.get("input", {}).get("roi")
    if roi is None:
        return None
    if isinstance(roi, dict):
        aliases = {
            "xmin": "min_x",
            "xmax": "max_x",
            "ymin": "min_y",
            "ymax": "max_y",
            "zmin": "min_z",
            "zmax": "max_z",
        }
        normalized = {aliases.get(key, key): value for key, value in roi.items()}
        values = [normalized[key] for key in ("min_x", "max_x", "min_y", "max_y")]
        if "min_z" in normalized or "max_z" in normalized:
            values += [normalized.get("min_z"), normalized.get("max_z")]
        return parse_bbox(values)
    return parse_bbox(roi)


def dump_config(config: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
