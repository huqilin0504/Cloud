"""Pure grid planning and tile-boundary operations for whole-cloud runs."""

from __future__ import annotations

import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..io import las_tiles
from .state import atomic_json, invalid_source_tiles, quarantine_source_tiles


TILE_NAME = re.compile(r"^tile_(-?\d+)_(-?\d+)\.laz$")


def grid_plan(source: dict[str, Any], tile_size: float, overlap: float) -> dict[str, Any]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size / 2.0:
        raise ValueError("tile_size 必须大于 0，overlap 必须满足 0 <= overlap < tile_size/2")
    minimum = np.asarray(source["bounds"]["min"], dtype=np.float64)
    maximum = np.asarray(source["bounds"]["max"], dtype=np.float64)
    origin_x = math.floor(float(minimum[0]) / tile_size) * tile_size
    origin_y = math.floor(float(minimum[1]) / tile_size) * tile_size
    max_ix = max(0, math.floor((float(maximum[0]) - origin_x) / tile_size))
    max_iy = max(0, math.floor((float(maximum[1]) - origin_y) / tile_size))
    return {
        "tile_size_m": float(tile_size),
        "overlap_m": float(overlap),
        "dimension_mode": "2d_xy_fallback",
        "dimension_mode_reason": "PDAL tile splitter exposes XY length; Z is retained in each local tile",
        "target_points_per_tile": 1_200_000,
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "max_ix": int(max_ix),
        "max_iy": int(max_iy),
        "nominal_columns": int(max_ix + 1),
        "nominal_rows": int(max_iy + 1),
    }


def parse_tile_name(path: Path) -> tuple[int, int]:
    match = TILE_NAME.match(path.name)
    if not match:
        raise ValueError(f"无法解析 PDAL 瓦片名：{path.name}")
    return int(match.group(1)), int(match.group(2))


def core_bbox(plan: dict[str, Any], ix: int, iy: int) -> dict[str, float]:
    size = float(plan["tile_size_m"])
    return {
        "min_x": float(plan["origin_x"] + ix * size),
        "max_x": float(plan["origin_x"] + (ix + 1) * size),
        "min_y": float(plan["origin_y"] + iy * size),
        "max_y": float(plan["origin_y"] + (iy + 1) * size),
    }


def tile_core_intersects_source(
    tile_path: Path,
    plan: dict[str, Any],
    source: dict[str, Any],
) -> bool:
    ix, iy = parse_tile_name(tile_path)
    core = core_bbox(plan, ix, iy)
    source_min = source["bounds"]["min"]
    source_max = source["bounds"]["max"]
    return not (
        core["max_x"] <= float(source_min[0])
        or core["min_x"] > float(source_max[0])
        or core["max_y"] <= float(source_min[1])
        or core["min_y"] > float(source_max[1])
    )


def expanded_bbox(core: dict[str, float], overlap: float) -> dict[str, float]:
    return {
        "min_x": core["min_x"] - overlap,
        "max_x": core["max_x"] + overlap,
        "min_y": core["min_y"] - overlap,
        "max_y": core["max_y"] + overlap,
    }


def core_mask(
    points: np.ndarray,
    core: dict[str, float],
    ix: int,
    iy: int,
    plan: dict[str, Any],
) -> np.ndarray:
    tolerance = 1e-8
    x_mask = points[:, 0] >= core["min_x"] - tolerance
    y_mask = points[:, 1] >= core["min_y"] - tolerance
    if ix == int(plan["max_ix"]):
        x_mask &= points[:, 0] <= core["max_x"] + tolerance
    else:
        x_mask &= points[:, 0] < core["max_x"]
    if iy == int(plan["max_iy"]):
        y_mask &= points[:, 1] <= core["max_y"] + tolerance
    else:
        y_mask &= points[:, 1] < core["max_y"]
    return x_mask & y_mask


def tile_report_path(report_dir: Path, ix: int, iy: int) -> Path:
    return report_dir / f"tile_{ix}_{iy}.json"


def run_pdal_split(
    command: list[str],
    tile_dir: Path,
    expected_tiles: int,
    *,
    progress: Callable[[str, int, int, str], None],
) -> None:
    """Run PDAL while reporting materialized output tiles."""

    process = subprocess.Popen(command)
    try:
        while process.poll() is None:
            generated = len(list(tile_dir.glob("tile_*.laz")))
            progress("PDAL切块", generated, expected_tiles, f"已生成 {generated} 个")
            time.sleep(1.0)
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    generated = len(list(tile_dir.glob("tile_*.laz")))
    progress(
        "PDAL切块",
        expected_tiles,
        expected_tiles,
        f"完成，生成 {generated} 个非空 LAZ 瓦片",
    )


def _split_copc_source(
    input_path: Path,
    tile_dir: Path,
    plan: dict[str, Any],
    source: dict[str, Any],
    *,
    progress: Callable[[str, int, int, str], None],
) -> list[Path]:
    """Use the COPC octree for each expanded tile instead of scanning LAS."""

    from laspy.copc import Bounds, CopcReader

    tiles: list[Path] = []
    expected_tiles = int(plan["nominal_columns"]) * int(plan["nominal_rows"])
    source_min = np.asarray(source["bounds"]["min"], dtype=np.float64)
    source_max = np.asarray(source["bounds"]["max"], dtype=np.float64)
    queried = 0
    with CopcReader.open(input_path) as reader:
        try:
            source_crs = reader.header.parse_crs()
        except Exception:
            source_crs = None
        for ix in range(int(plan["nominal_columns"])):
            for iy in range(int(plan["nominal_rows"])):
                core = core_bbox(plan, ix, iy)
                expanded = expanded_bbox(core, float(plan["overlap_m"]))
                minimum = np.array(
                    [expanded["min_x"], expanded["min_y"], source_min[2]],
                    dtype=np.float64,
                )
                maximum = np.array(
                    [expanded["max_x"], expanded["max_y"], source_max[2]],
                    dtype=np.float64,
                )
                if (
                    maximum[0] < source_min[0]
                    or minimum[0] > source_max[0]
                    or maximum[1] < source_min[1]
                    or minimum[1] > source_max[1]
                ):
                    queried += 1
                    continue
                records = reader.spatial_query(Bounds(minimum, maximum))
                if len(records):
                    path = tile_dir / f"tile_{ix}_{iy}.laz"
                    las_tiles.write_copc_query(path, records, source, source_crs)
                    tiles.append(path)
                queried += 1
                progress("COPC索引切块", queried, expected_tiles, f"非空 {len(tiles)} 个")
    if not tiles:
        progress("COPC索引切块", expected_tiles, expected_tiles, "完成，但没有非空瓦片")
        raise RuntimeError("COPC 空间索引查询没有生成任何非空 LAZ 瓦片")
    progress(
        "COPC索引切块",
        expected_tiles,
        expected_tiles,
        f"完成，生成 {len(tiles)} 个非空 LAZ 瓦片",
    )
    marker = {
        "input": source,
        "plan": plan,
        "tiles": [path.name for path in sorted(tiles, key=parse_tile_name)],
        "source_format": "COPC",
        "splitter": "laspy.copc.CopcReader.spatial_query",
    }
    atomic_json(tile_dir.parent / "split_state.json", marker)
    return sorted(tiles, key=parse_tile_name)


def split_source(
    input_path: Path,
    tile_dir: Path,
    plan: dict[str, Any],
    source: dict[str, Any],
    pdal_command: str,
    *,
    progress: Callable[[str, int, int, str], None],
) -> list[Path]:
    """Materialize source tiles and persist the split manifest."""

    tile_dir.mkdir(parents=True, exist_ok=True)
    existing = list(tile_dir.glob("tile_*.laz"))
    if existing:
        invalid = invalid_source_tiles(existing)
        state_path = tile_dir.parent / "processing_state.json"
        report_dir = tile_dir.parent / "tile_reports"
        has_processing_state = state_path.is_file() or any(report_dir.glob("tile_*.json"))
        if invalid and not has_processing_state:
            quarantined = quarantine_source_tiles(tile_dir)
            print(
                f"[whole-cloud] quarantined {len(invalid)} incomplete source tiles: {quarantined}",
                flush=True,
            )
            tile_dir.mkdir(parents=True, exist_ok=True)
        elif invalid:
            raise RuntimeError(
                f"发现 {len(invalid)}/{len(existing)} 个损坏源瓦片，且已有处理状态；"
                "请使用 --resume 重试，或换一个输出目录。"
            )
        else:
            raise RuntimeError(
                f"发现未登记的源瓦片 {len(existing)} 个：{tile_dir}；"
                "请使用已有 split_state.json 恢复，或换一个输出目录。"
            )
    if bool(source.get("is_copc")):
        return _split_copc_source(input_path, tile_dir, plan, source, progress=progress)

    output_template = tile_dir / "tile_#.laz"
    command = [
        pdal_command,
        "tile",
        "--input",
        str(input_path),
        "--output",
        str(output_template),
        "--length",
        str(plan["tile_size_m"]),
        f"--origin_x={plan['origin_x']}",
        f"--origin_y={plan['origin_y']}",
        "--buffer",
        str(plan["overlap_m"]),
    ]
    expected_tiles = int(plan["nominal_columns"]) * int(plan["nominal_rows"])
    run_pdal_split(command, tile_dir, expected_tiles, progress=progress)
    generated_tiles = sorted(tile_dir.glob("tile_*.laz"), key=parse_tile_name)
    tiles = [path for path in generated_tiles if tile_core_intersects_source(path, plan, source)]
    outside_tiles = [path for path in generated_tiles if path not in tiles]
    if outside_tiles:
        outside_dir = tile_dir.parent / "source_tiles.outside_extent"
        suffix = 1
        while outside_dir.exists():
            outside_dir = tile_dir.parent / f"source_tiles.outside_extent.{suffix}"
            suffix += 1
        outside_dir.mkdir(parents=True)
        for path in outside_tiles:
            path.rename(outside_dir / path.name)
        print(
            f"[whole-cloud] quarantined {len(outside_tiles)} buffer-only tiles: {outside_dir}",
            flush=True,
        )
    if not tiles:
        raise RuntimeError("PDAL 分块没有生成任何 LAZ 瓦片")
    marker = {
        "input": source,
        "plan": plan,
        "tiles": [path.name for path in tiles],
        "ignored_outside_extent_tiles": len(outside_tiles),
        "source_format": "LAS/LAZ",
        "splitter": "pdal tile",
    }
    atomic_json(tile_dir.parent / "split_state.json", marker)
    return tiles
