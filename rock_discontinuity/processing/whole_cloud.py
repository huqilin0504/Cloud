from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..config import dump_config
from ..core.models import PlaneInstance, TileResult, plane_to_row
from ..core.records import nearest_spacing_by_plane
from ..io.las_tiles import (
    merge_overlays as _merge_overlays_impl,
    source_crs as _source_crs,
    source_info as _source_info,
    write_red_overlay as _write_red_overlay,
)
from ..io.whole_cloud_output import (
    write_json_reports,
    write_whole_cloud_outputs,
)
from .audit import audit_source_metadata
from .density import discontinuity_density_metrics
from .pipeline import process_points
from .tiling import (
    grid_plan as _grid_plan,
    parse_tile_name as _parse_tile_name,
    split_source as _split_source,
    tile_report_path as _tile_report_path,
)
from .tile_processing import process_tile as _process_tile_impl
from .whole_cloud_contract import WHOLE_ALGORITHM_VERSION, tiling_parameters, whole_cloud_workers
from .output_records import (
    prepare_whole_cloud_auxiliary_rows,
    summarize_tile_density,
)
from .state import (
    atomic_json as _atomic_json,
    load_processing_state as _load_processing_state,
    load_split_state as _load_split_state,
    mark_tile_done as _mark_tile_done,
    mark_tile_error as _mark_tile_error,
    validate_tile_report_versions as _validate_tile_report_versions,
)
from .global_aggregation import (
    apply_global_candidate_gate as _apply_global_candidate_gate,
    aggregate_global_planes as _aggregate_global_planes,
    global_orientation_sets as _global_orientation_sets,
    global_spacing_and_sets as _global_spacing_and_sets,
    merge_plane_rows as _merge_plane_rows,
)


PLANE_FIELDS = list(plane_to_row(PlaneInstance()).keys())
WHOLE_PLANE_FIELDS = [
    "tile_id",
    "global_plane_id",
    "global_set_id",
    "global_instance_count",
    "global_tile_count",
    "core_red_points",
    "tile_plane_index",
    "status",
    "selection_reason",
    "nearest_spacing_m",
    "center_x",
    "center_y",
    "center_z",
    *PLANE_FIELDS,
]
WHOLE_GLOBAL_PLANE_FIELDS = [
    "global_plane_id",
    "global_set_id",
    "tile_count",
    "instance_count",
    "core_red_points",
    "plane_id",
    "set_id",
    "dip_direction_deg",
    "dip_deg",
    "trace_length_m",
    "observed_area_m2",
    "observed_area_sum_m2",
    "max_instance_area_m2",
    "major_extent_m",
    "minor_extent_m",
    "edge_censored",
    "nearest_spacing_m",
    "center_x",
    "center_y",
    "center_z",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "nx",
    "ny",
    "nz",
    "plane_d",
    "bbox_min_x",
    "bbox_min_y",
    "bbox_min_z",
    "bbox_max_x",
    "bbox_max_y",
    "bbox_max_z",
    "rms_m",
    "mae_m",
    "p95_residual_m",
    "inlier_ratio",
    "planarity",
    "normal_dispersion_deg",
    "normal_scale_m",
    "normal_stability_deg",
    "normal_stable_fraction",
    "normal_valid_scale_count",
    "normal_stable",
    "boundary_completeness",
    "confidence",
    "quality_score",
    "quality_grade",
    "near_vertical",
    "data_gap_censored",
    "trace_verified",
    "aperture_available",
    "aperture_reason",
]
WHOLE_JOINT_SET_FIELDS = [
    "set_id",
    "plane_count",
    "mean_dip_direction_deg",
    "mean_dip_deg",
    "mean_normal_x",
    "mean_normal_y",
    "mean_normal_z",
    "angular_dispersion_deg",
    "spacing_available",
    "spacing_reason",
    "mean_spacing_m",
    "median_spacing_m",
    "std_spacing_m",
    "min_spacing_m",
    "max_spacing_m",
    "p10_spacing_m",
    "p90_spacing_m",
    "spacing_3d_sample_count",
    "spacing_virtual_scanline_sample_count",
    "mean_spacing_3d_m",
    "median_spacing_3d_m",
    "mean_spacing_virtual_scanline_m",
    "median_spacing_virtual_scanline_m",
    "fisher_k",
]
WHOLE_SPACING_FIELDS = [
    "tile_id",
    "plane_id_a",
    "plane_id_b",
    "global_plane_id_a",
    "global_plane_id_b",
    "spacing_m",
    "method",
]


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
    all_plane_rows: list[dict[str, Any]] = []
    all_spacing_rows: list[dict[str, Any]] = []
    for report_path in sorted(tile_report_dir.glob("tile_*.json")):
        tile_report = json.loads(report_path.read_text(encoding="utf-8"))
        if tile_report.get("status") != "done":
            continue
        done_reports.append(tile_report)
        all_plane_rows.extend(tile_report.pop("plane_rows_data", []))
        all_spacing_rows.extend(tile_report.pop("spacing_rows_data", []))
    all_plane_rows.sort(key=lambda row: (row["tile_id"], row["plane_id"]))
    all_spacing_rows.sort(
        key=lambda row: (row["tile_id"], row["plane_id_a"], row["plane_id_b"])
    )
    merged_plane_rows, plane_groups, old_to_global = _merge_plane_rows(
        all_plane_rows,
        plan,
        config,
    )
    provisional_set_ids = _global_orientation_sets(plane_groups, config)
    for row in merged_plane_rows:
        row["global_set_id"] = provisional_set_ids.get(str(row["global_plane_id"]))
    all_global_plane_rows = _aggregate_global_planes(plane_groups, provisional_set_ids, config)
    selected_global_ids, selection_by_id = _apply_global_candidate_gate(
        all_global_plane_rows,
        config,
    )
    for row in merged_plane_rows:
        selection = selection_by_id.get(str(row["global_plane_id"]), {})
        row["status"] = selection.get("status", "not_selected")
        row["selection_reason"] = selection.get("selection_reason", "global_gate_missing")
    selected_groups = {key: value for key, value in plane_groups.items() if key in selected_global_ids}
    global_set_ids = _global_orientation_sets(selected_groups, config)
    global_plane_rows = _aggregate_global_planes(selected_groups, global_set_ids, config)
    global_spacing_rows, global_joint_set_rows = _global_spacing_and_sets(
        global_plane_rows,
        config,
    )
    nearest_global_spacing = nearest_spacing_by_plane(global_spacing_rows)
    for row in merged_plane_rows:
        row["nearest_spacing_m"] = nearest_global_spacing.get(str(row["global_plane_id"]))
    for row in global_plane_rows:
        row["nearest_spacing_m"] = nearest_global_spacing.get(str(row["global_plane_id"]))

    overlay_paths = sorted(overlay_dir.glob("tile_*.laz"), key=lambda path: _parse_tile_name(path))
    merged_overlay = output_dir / "candidate_detachment_points.laz"
    selected_plane_indices_by_tile: dict[str, set[int]] = defaultdict(set)
    for row in merged_plane_rows:
        if str(row["global_plane_id"]) in selected_global_ids:
            selected_plane_indices_by_tile[str(row["tile_id"])].add(int(row["tile_plane_index"]))
    merged_count = _merge_overlays(
        overlay_paths,
        merged_overlay,
        source,
        source_crs,
        selected_plane_indices_by_tile,
    )
    done_tile_ids = {report["tile_id"] for report in done_reports}
    total_red = sum(int(report["counts"]["candidate_red_points"]) for report in done_reports)
    total_core = sum(int(report["counts"]["core_points"]) for report in done_reports)
    total_input = sum(int(report["counts"]["input_points"]) for report in done_reports)
    report = {
        "version": WHOLE_ALGORITHM_VERSION,
        "algorithm_version": WHOLE_ALGORITHM_VERSION,
        "input": source,
        "coordinate_assumption": config["coordinate"],
        "tiling": {
            **plan,
            "source_mode": source_mode,
            "materialized_tiles": len(tile_paths),
            "selected_tiles": len(selected_tiles),
            "processed_tiles": len(done_tile_ids),
            "failed_tiles": len(failures),
            "partial_run": max_tiles is not None and len(selected_tiles) < len(tile_paths),
            "parallel_mode": "thread_pool",
            "workers": worker_count,
        },
        "counts": {
            "source_points": int(source["total_points"]),
            "expanded_tile_points_processed": total_input,
            "core_points_processed": total_core,
            "tile_local_candidate_points": total_red,
            "candidate_red_points": int(merged_count),
            "merged_overlay_points": int(merged_count),
            "candidate_plane_instances": len(all_plane_rows),
            "global_plane_groups_before_gate": len(all_global_plane_rows),
            "cross_tile_duplicate_instances_merged": len(all_plane_rows) - len(all_global_plane_rows),
            "global_gate_rejected_planes": len(all_global_plane_rows) - len(global_plane_rows),
            "global_joint_plane_count": len(global_plane_rows),
            "global_joint_set_count": len(global_joint_set_rows),
            "spacing_rows": len(global_spacing_rows),
            "tile_spacing_rows": len(all_spacing_rows),
        },
        "detachment": {
            "status": "geometry_only_candidate",
            "method": "projected_footprint_merge_then_global_candidate_gate",
            "confirmed_unstable_count": None,
            "overlay": str(merged_overlay),
            "note": "红色仅表示观测节理面候选；相邻瓦片平面按法向、平面偏移和投影足迹完整链接合并，并经全局质量门槛复判；未进行块体拓扑或力学稳定性判定。",
        },
        "failures": sorted(failures, key=lambda item: str(item["tile_id"])),
        "data_audit": audit_source_metadata(
            source,
            tile_count=len(tile_paths),
            source_mode=source_mode,
        ),
        "density_report": summarize_tile_density(done_reports, config),
        "limitations": [
            "candidate_plane_instances 是瓦片内实例数；global_joint_plane_count 是当前几何门限下的跨瓦片合并结果。",
            "全局 observed_area_m2 来自各瓦片投影足迹的联合面积；observed_area_sum_m2 保留未去重面积和，仅用于审计。",
            "全局候选门槛可淘汰局部候选，但当前不会恢复在瓦片内部已因非边界质量门槛淘汰的片段。",
            "当前结果不等于已经失稳或可能失稳块体数量。",
            "整幅点云没有独立危岩体边界和非岩体分类字段时，非岩体平面也可能进入候选结果。",
            "普通 LAS/LAZ 当前使用 PDAL XY 分块并保留完整 Z；报告明确标记 2d_xy_fallback，不把它伪称为三维体素切块。",
            "整体输出的 P10/P21 需要扫描线/已验证迹线，未提供时保持不可用；全局平面合并后仅提供有限观测量和虚拟扫描线间距。",
        ],
        "outputs": {
            "tile_plane_instances": str(output_dir / "detachment_planes.csv"),
            "global_joint_planes": str(output_dir / "joint_planes.csv"),
            "global_joint_sets": str(output_dir / "joint_sets.csv"),
            "global_spacings": str(output_dir / "spacings.csv"),
            "tile_spacings": str(output_dir / "tile_spacings.csv"),
            "candidate_overlay": str(merged_overlay),
        },
    }
    footprint = report["data_audit"].get("footprint_bbox_area_m2")
    report["density_observables"] = discontinuity_density_metrics(
        point_count=int(source["total_points"]),
        footprint_area_m2=footprint,
        plane_count=len(global_plane_rows),
    )
    trace_rows, aperture_rows = prepare_whole_cloud_auxiliary_rows(global_plane_rows)
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
