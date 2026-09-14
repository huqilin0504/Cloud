from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..config import dump_config
from ..core.models import (
    SpacingRecord,
    TilePlaneRecord,
    TileResult,
    spacing_record_from_json,
    tile_plane_record_from_json,
)
from ..io.las_tiles import (
    merge_overlays as _merge_overlays_impl,
    source_crs as _source_crs,
    source_info as _source_info,
    write_red_overlay as _write_red_overlay,
)
from ..io.whole_cloud_output import (
    PLANE_FIELDS,
    WHOLE_GLOBAL_PLANE_FIELDS,
    WHOLE_JOINT_SET_FIELDS,
    WHOLE_PLANE_FIELDS,
    WHOLE_SPACING_FIELDS,
    write_json_reports,
    write_whole_cloud_outputs,
)
from .tiling import (
    grid_plan as _grid_plan,
    parse_tile_name as _parse_tile_name,
    split_source as _split_source,
    tile_report_path as _tile_report_path,
)
from .tile_processing import process_tile as _process_tile_impl
from .whole_cloud_contract import WHOLE_ALGORITHM_VERSION, tiling_parameters, whole_cloud_workers
from .state import (
    atomic_json as _atomic_json,
    load_processing_state as _load_processing_state,
    load_split_state as _load_split_state,
    mark_tile_done as _mark_tile_done,
    mark_tile_error as _mark_tile_error,
    validate_tile_report_versions as _validate_tile_report_versions,
)
from .global_aggregation import aggregate_global_results
from .whole_cloud_reporting import (
    prepare_selected_plane_indices_by_tile,
    prepare_whole_cloud_report,
)


def _format_progress(stage: str, current: int, total: int, detail: str = "") -> str:
    """Build a dependency-free terminal progress line."""

    total = max(1, int(total))
    current = max(0, min(int(current), total))
    ratio = current / total
    width = 28
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" {detail}" if detail else ""
    return f"\r[{bar}] {stage} {current}/{total} {ratio * 100:5.1f}%{suffix}"


def _print_progress(stage: str, current: int, total: int, detail: str = "") -> None:
    print(_format_progress(stage, current, total, detail), end="", flush=True)


def _merge_overlays(
    overlay_paths: list[Path],
    output_path: Path,
    source: dict[str, Any],
    source_crs: Any,
    selected_plane_indices_by_tile: dict[str, set[int]] | None = None,
) -> int:
    """Compatibility adapter for callers of the old private helper."""

    return _merge_overlays_impl(
        overlay_paths,
        output_path,
        source,
        source_crs,
        selected_plane_indices_by_tile,
        parse_tile_name=_parse_tile_name,
    )


def _process_tile(
    tile_path: Path,
    *,
    plan: dict[str, Any],
    config: dict[str, Any],
    source: dict[str, Any],
    source_crs: Any,
    overlay_dir: Path,
) -> TileResult:
    """Compatibility adapter for the extracted single-tile worker."""

    return _process_tile_impl(
        tile_path,
        plan=plan,
        config=config,
        source=source,
        source_crs=source_crs,
        overlay_dir=overlay_dir,
        write_red_overlay=_write_red_overlay,
    )


def run_whole_cloud(
    input_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    *,
    tile_size: float | None = None,
    overlap: float | None = None,
    resume: bool = False,
    prepare_only: bool = False,
    keep_source_tiles: bool = False,
    max_tiles: int | None = None,
    workers: int | None = None,
    pdal_command: str = "pdal",
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_size, overlap = tiling_parameters(config, tile_size, overlap)
    source = _source_info(input_path)
    plan = _grid_plan(source, tile_size, overlap)
    plan["target_points_per_tile"] = int(
        config.get("tiling", {}).get("target_points_per_tile", plan["target_points_per_tile"])
    )
    split_state_path = output_dir / "split_state.json"
    source_tile_dir = output_dir / "source_tiles"
    overlay_dir = output_dir / "candidate_overlays"
    tile_report_dir = output_dir / "tile_reports"
    overlay_dir.mkdir(exist_ok=True)
    tile_report_dir.mkdir(exist_ok=True)
    if split_state_path.is_file():
        if not resume:
            raise FileExistsError(f"输出已存在；请使用 --resume 继续：{output_dir}")
        tile_paths = _load_split_state(split_state_path, source, plan)
    else:
        if output_dir.iterdir() and resume:
            raise FileNotFoundError("要求恢复，但输出目录没有 split_state.json")
        tile_paths = _split_source(
            input_path,
            source_tile_dir,
            plan,
            source,
            pdal_command,
            progress=_print_progress,
        )

    run_config = dict(config)
    source_mode = "copc_indexed" if bool(source.get("is_copc")) else "pdal_spatial_tiles"
    configured_workers = whole_cloud_workers(config)
    worker_count = max(1, int(workers if workers is not None else configured_workers))
    worker_count = min(worker_count, max(1, len(tile_paths)))
    whole_config = dict(config.get("whole_cloud", {}))
    whole_config.update(
        {
            "algorithm_version": WHOLE_ALGORITHM_VERSION,
            "method": source_mode,
            "tile_size_m": float(tile_size),
            "overlap_m": float(overlap),
            "source_tile_format": "LAZ",
            "core_deduplication": "half_open_xy_grid",
            "candidate_overlay_color_rgb": [230, 35, 35],
            "parallel_mode": "thread_pool",
            "workers": worker_count,
        }
    )
    run_config["whole_cloud"] = whole_config
    config_path = output_dir / "run_config.yaml"
    if not config_path.exists():
        dump_config(run_config, config_path)

    state_path = output_dir / "processing_state.json"
    processing_state = _load_processing_state(
        state_path,
        plan=plan,
        algorithm_version=WHOLE_ALGORITHM_VERSION,
        resume=resume,
    )
    _validate_tile_report_versions(
        tile_report_dir,
        algorithm_version=WHOLE_ALGORITHM_VERSION,
        resume=resume,
    )

    source_crs = _source_crs(input_path)

    selected_tiles = tile_paths[:max_tiles] if max_tiles is not None else tile_paths
    failures: list[dict[str, Any]] = []
    if prepare_only:
        report = {
            "version": WHOLE_ALGORITHM_VERSION,
            "algorithm_version": WHOLE_ALGORITHM_VERSION,
            "input": source,
            "coordinate_assumption": config["coordinate"],
            "tiling": {
                **plan,
                "source_mode": source_mode,
                "materialized_tiles": len(tile_paths),
                "selected_tiles": 0,
                "processed_tiles": 0,
                "failed_tiles": 0,
                "partial_run": True,
                "prepared_only": True,
                "parallel_mode": "thread_pool",
                "workers": worker_count,
            },
            "counts": {
                "source_points": int(source["total_points"]),
                "expanded_tile_points_processed": 0,
                "core_points_processed": 0,
                "candidate_red_points": 0,
                "merged_overlay_points": 0,
                "candidate_plane_instances": 0,
                "spacing_rows": 0,
            },
            "detachment": {
                "status": "not_processed",
                "method": source_mode,
                "confirmed_unstable_count": None,
                "overlay": None,
                "note": "仅完成源点云分块，尚未进行节理候选识别。",
            },
            "failures": [],
            "data_audit": audit_source_metadata(
                source,
                tile_count=len(tile_paths),
                source_mode=source_mode,
            ),
            "density_report": {
                "method": "tile_local_tangent_knn",
                "status": "not_assessed_prepare_only",
                "target_raw_density_points_m2": float(
                    config.get("density", {}).get("target_raw_density_points_m2", 600.0)
                ),
            },
        }
        write_json_reports(output_dir, report)
        return report
    if not prepare_only:
        completed_tiles = 0
        total_selected_tiles = len(selected_tiles)
        pending_tiles: list[tuple[Path, str, Path]] = []
        for tile_path in selected_tiles:
            ix, iy = _parse_tile_name(tile_path)
            tile_id = f"{ix}_{iy}"
            previous = processing_state["tiles"].get(tile_id, {})
            previous_report = _tile_report_path(tile_report_dir, ix, iy)
            if previous.get("status") == "done" and previous_report.is_file():
                completed_tiles += 1
                _print_progress("节理候选识别", completed_tiles, total_selected_tiles, f"跳过 {tile_id}")
                continue
            if not tile_path.is_file():
                error = f"未找到待处理源瓦片：{tile_path}"
                failures.append({"tile_id": tile_id, "error": error})
                _mark_tile_error(processing_state, tile_id=tile_id, error=error)
                _atomic_json(state_path, processing_state)
                completed_tiles += 1
                _print_progress("节理候选识别", completed_tiles, total_selected_tiles, f"失败 {tile_id}")
                continue
            pending_tiles.append((tile_path, tile_id, previous_report))

        def process_tile(tile_path: Path):
            return _process_tile(
                tile_path,
                plan=plan,
                config=config,
                source=source,
                source_crs=source_crs,
                overlay_dir=overlay_dir,
            )

        if pending_tiles:
            with ThreadPoolExecutor(
                max_workers=min(worker_count, len(pending_tiles)),
                thread_name_prefix="joint-tile",
            ) as executor:
                futures = {
                    executor.submit(process_tile, tile_path): (tile_path, tile_id, report_path)
                    for tile_path, tile_id, report_path in pending_tiles
                }
                for future in as_completed(futures):
                    tile_path, tile_id, previous_report = futures[future]
                    try:
                        tile_result = future.result()
                        tile_report = tile_result.report
                        plane_rows = tile_result.plane_rows
                        spacing_rows = tile_result.spacing_rows
                        _atomic_json(
                            previous_report,
                            tile_report | {
                                "plane_rows_data": plane_rows,
                                "spacing_rows_data": spacing_rows,
                            },
                        )
                        _mark_tile_done(
                            processing_state,
                            tile_id=tile_id,
                            report_name=previous_report.name,
                            candidate_red_points=tile_report["counts"]["candidate_red_points"],
                        )
                        _atomic_json(state_path, processing_state)
                        if not keep_source_tiles:
                            tile_path.unlink()
                        completed_tiles += 1
                        _print_progress("节理候选识别", completed_tiles, total_selected_tiles, f"完成 {tile_id}")
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        failures.append({"tile_id": tile_id, "error": error})
                        _mark_tile_error(processing_state, tile_id=tile_id, error=error)
                        _atomic_json(state_path, processing_state)
                        completed_tiles += 1
                        _print_progress("节理候选识别", completed_tiles, total_selected_tiles, f"失败 {tile_id}")
        print()

    done_reports: list[dict[str, Any]] = []
    all_plane_rows: list[TilePlaneRecord] = []
    all_spacing_rows: list[SpacingRecord] = []
    for report_path in sorted(tile_report_dir.glob("tile_*.json")):
        tile_report = json.loads(report_path.read_text(encoding="utf-8"))
        if tile_report.get("status") != "done":
            continue
        done_reports.append(tile_report)
        all_plane_rows.extend(
            tile_plane_record_from_json(row)
            for row in tile_report.pop("plane_rows_data", [])
        )
        all_spacing_rows.extend(
            spacing_record_from_json(row)
            for row in tile_report.pop("spacing_rows_data", [])
        )
    all_plane_rows.sort(key=lambda row: (row["tile_id"], row["plane_id"]))
    all_spacing_rows.sort(
        key=lambda row: (row["tile_id"], row["plane_id_a"], row["plane_id_b"])
    )
    aggregation = aggregate_global_results(all_plane_rows, plan, config)
    merged_plane_rows = aggregation.merged_plane_rows
    global_plane_rows = aggregation.global_plane_rows
    global_spacing_rows = aggregation.spacing_rows
    global_joint_set_rows = aggregation.joint_set_rows

    overlay_paths = sorted(overlay_dir.glob("tile_*.laz"), key=lambda path: _parse_tile_name(path))
    merged_overlay = output_dir / "candidate_detachment_points.laz"
    selected_plane_indices_by_tile = prepare_selected_plane_indices_by_tile(aggregation)
    merged_count = _merge_overlays(
        overlay_paths,
        merged_overlay,
        source,
        source_crs,
        selected_plane_indices_by_tile,
    )
    prepared_report = prepare_whole_cloud_report(
        source=source,
        plan=plan,
        source_mode=source_mode,
        tile_paths=tile_paths,
        selected_tiles=selected_tiles,
        done_reports=done_reports,
        failures=failures,
        all_spacing_rows=all_spacing_rows,
        aggregation=aggregation,
        selected_plane_indices_by_tile=selected_plane_indices_by_tile,
        output_dir=output_dir,
        merged_overlay=merged_overlay,
        merged_overlay_points=merged_count,
        worker_count=worker_count,
        max_tiles=max_tiles,
        config=config,
    )
    report = prepared_report.report
    trace_rows = prepared_report.trace_rows
    aperture_rows = prepared_report.aperture_rows
    write_whole_cloud_outputs(
        output_dir,
        report,
        plane_rows=merged_plane_rows,
        global_plane_rows=global_plane_rows,
        joint_set_rows=global_joint_set_rows,
        spacing_rows=global_spacing_rows,
        tile_spacing_rows=all_spacing_rows,
        trace_rows=trace_rows,
        aperture_rows=aperture_rows,
        plane_fields=WHOLE_PLANE_FIELDS,
        global_plane_fields=WHOLE_GLOBAL_PLANE_FIELDS,
        joint_set_fields=WHOLE_JOINT_SET_FIELDS,
        spacing_fields=WHOLE_SPACING_FIELDS,
    )
    return report
