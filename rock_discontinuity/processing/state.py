"""Resumable whole-cloud state and source-tile integrity helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import laspy


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    raise TypeError(f"无法序列化类型：{type(value)!r}")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Write a state/report JSON atomically in the same directory."""

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def invalid_source_tiles(tile_paths: list[Path]) -> list[Path]:
    """Return source tiles whose LAS/LAZ headers cannot be read."""

    invalid: list[Path] = []
    for path in tile_paths:
        try:
            with laspy.open(path) as reader:
                if int(reader.header.point_count) <= 0:
                    invalid.append(path)
        except Exception:
            invalid.append(path)
    return invalid


def quarantine_source_tiles(tile_dir: Path) -> Path:
    """Move stale source tiles aside without deleting evidence."""

    candidate = tile_dir.with_name(f"{tile_dir.name}.incomplete")
    suffix = 1
    while candidate.exists():
        candidate = tile_dir.with_name(f"{tile_dir.name}.incomplete.{suffix}")
        suffix += 1
    tile_dir.rename(candidate)
    return candidate


def load_split_state(path: Path, source: dict[str, Any], plan: dict[str, Any]) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少分块状态文件：{path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("input", {}).get("path") != source["path"]:
        raise ValueError("已有分块状态对应另一份输入文件")
    old_plan = state.get("plan", {})
    for key in ("tile_size_m", "overlap_m", "origin_x", "origin_y"):
        if not math.isclose(float(old_plan[key]), float(plan[key]), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"已有分块状态的 {key} 与本次参数不一致")
    tile_dir = path.parent / "source_tiles"
    return [tile_dir / name for name in state.get("tiles", [])]


def new_processing_state(plan: dict[str, Any], algorithm_version: str) -> dict[str, Any]:
    """Create the in-memory state shape used by the scheduler."""

    return {
        "algorithm_version": algorithm_version,
        "plan": plan,
        "tiles": {},
    }


def load_processing_state(
    path: Path,
    *,
    plan: dict[str, Any],
    algorithm_version: str,
    resume: bool,
) -> dict[str, Any]:
    """Load a compatible state file or return a new state for a fresh run."""

    if not (path.is_file() and resume):
        return new_processing_state(plan, algorithm_version)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("algorithm_version") != algorithm_version:
        raise RuntimeError(
            "已有 whole-cloud 处理状态来自旧算法，不能与当前区域生长/跨块合并结果混用；"
            "请换一个输出目录重新处理，或先完成旧任务后再启动新目录。"
        )
    stored_plan = state.get("plan")
    if stored_plan is not None:
        for key in (
            "tile_size_m",
            "overlap_m",
            "origin_x",
            "origin_y",
            "max_ix",
            "max_iy",
            "nominal_columns",
            "nominal_rows",
        ):
            if key not in stored_plan or key not in plan:
                raise ValueError(f"processing_state.json 缺少分块计划字段：{key}")
            if isinstance(plan[key], (int, float)) and isinstance(stored_plan[key], (int, float)):
                if not math.isclose(
                    float(stored_plan[key]), float(plan[key]), rel_tol=0.0, abs_tol=1e-8
                ):
                    raise ValueError(f"已有处理状态的 {key} 与本次分块计划不一致")
            elif stored_plan[key] != plan[key]:
                raise ValueError(f"已有处理状态的 {key} 与本次分块计划不一致")
    if not isinstance(state.get("tiles"), dict):
        raise ValueError("processing_state.json 的 tiles 字段必须是对象")
    return state


def validate_tile_report_versions(
    report_dir: Path,
    *,
    algorithm_version: str,
    resume: bool,
) -> None:
    """Reject reports from another algorithm version before resuming."""

    if not resume:
        return
    stale_reports: list[str] = []
    for report_path in report_dir.glob("tile_*.json"):
        try:
            report_version = json.loads(report_path.read_text(encoding="utf-8")).get(
                "algorithm_version"
            )
        except (OSError, json.JSONDecodeError):
            report_version = None
        if report_version != algorithm_version:
            stale_reports.append(report_path.name)
    if stale_reports:
        raise RuntimeError(
            "已有 tile_reports 来自旧算法，不能与当前区域生长/跨块合并结果混用；"
            "请换一个输出目录重新处理。"
        )


def mark_tile_done(
    state: dict[str, Any],
    *,
    tile_id: str,
    report_name: str,
    candidate_red_points: int,
) -> dict[str, Any]:
    state.setdefault("tiles", {})[tile_id] = {
        "status": "done",
        "report": report_name,
        "candidate_red_points": int(candidate_red_points),
    }
    return state


def mark_tile_error(state: dict[str, Any], *, tile_id: str, error: str) -> dict[str, Any]:
    state.setdefault("tiles", {})[tile_id] = {"status": "error", "error": error}
    return state
