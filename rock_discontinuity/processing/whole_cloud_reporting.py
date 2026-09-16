"""Pure preparation of whole-cloud reports and selected overlay mappings."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.models import (
    ApertureRecord,
    GlobalAggregationResult,
    SpacingRecord,
    TraceRecord,
)
from .audit import audit_source_metadata
from .density import discontinuity_density_metrics
from .output_records import prepare_whole_cloud_auxiliary_rows, summarize_tile_density
from .whole_cloud_contract import GLOBAL_AGGREGATION_VERSION, WHOLE_ALGORITHM_VERSION


@dataclass
class WholeCloudReportPreparation:
    """Computed report data handed to the I/O layer by the orchestrator."""

    report: dict[str, Any]
    selected_plane_indices_by_tile: dict[str, set[int]]
    trace_rows: list[TraceRecord]
    aperture_rows: list[ApertureRecord]


def prepare_whole_cloud_report(
    *,
    source: dict[str, Any],
    plan: dict[str, Any],
    source_mode: str,
    tile_paths: list[Path],
    selected_tiles: list[Path],
    done_reports: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    all_spacing_rows: list[SpacingRecord],
    aggregation: GlobalAggregationResult,
    selected_plane_indices_by_tile: dict[str, set[int]],
    output_dir: Path,
    merged_overlay: Path,
    merged_overlay_points: int,
    worker_count: int,
    max_tiles: int | None,
    config: dict[str, Any],
) -> WholeCloudReportPreparation:
    """Prepare all global report values without writing files or changing state."""

    total_red = sum(int(report["counts"]["candidate_red_points"]) for report in done_reports)
    total_core = sum(int(report["counts"]["core_points"]) for report in done_reports)
    total_input = sum(int(report["counts"]["input_points"]) for report in done_reports)
    report = {
        "version": WHOLE_ALGORITHM_VERSION,
        "algorithm_version": WHOLE_ALGORITHM_VERSION,
        "global_aggregation_version": GLOBAL_AGGREGATION_VERSION,
        "input": source,
        "coordinate_assumption": config["coordinate"],
        "tiling": {
            **plan,
            "source_mode": source_mode,
            "materialized_tiles": len(tile_paths),
            "selected_tiles": len(selected_tiles),
            "processed_tiles": len({report["tile_id"] for report in done_reports}),
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
            "candidate_red_points": int(merged_overlay_points),
            "merged_overlay_points": int(merged_overlay_points),
            "candidate_plane_instances": len(aggregation.merged_plane_rows),
            "global_plane_groups_before_gate": len(aggregation.all_global_plane_rows),
            "cross_tile_duplicate_instances_merged": len(aggregation.merged_plane_rows)
            - len(aggregation.all_global_plane_rows),
            "global_gate_rejected_planes": len(aggregation.all_global_plane_rows)
            - len(aggregation.global_plane_rows),
            "global_joint_plane_count": len(aggregation.global_plane_rows),
            "global_joint_set_count": len(aggregation.joint_set_rows),
            "spacing_rows": len(aggregation.spacing_rows),
            "tile_spacing_rows": len(all_spacing_rows),
        },
        "global_aggregation_diagnostics": {
            "merge": aggregation.merge_diagnostics,
            "orientation": aggregation.orientation_diagnostics,
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
        plane_count=len(aggregation.global_plane_rows),
    )
    trace_rows, aperture_rows = prepare_whole_cloud_auxiliary_rows(
        aggregation.global_plane_rows
    )
    return WholeCloudReportPreparation(
        report=report,
        selected_plane_indices_by_tile=dict(selected_plane_indices_by_tile),
        trace_rows=trace_rows,
        aperture_rows=aperture_rows,
    )


def prepare_selected_plane_indices_by_tile(
    aggregation: GlobalAggregationResult,
    *,
    progress: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, set[int]]:
    """Build the overlay selection map without touching the filesystem."""

    selected: dict[str, set[int]] = {}
    rows = aggregation.merged_plane_rows
    total = max(1, len(rows))
    if progress is not None:
        progress("准备候选点云筛选", 0, total, f"输入 {len(rows)} 个平面实例")
    for index, row in enumerate(rows, start=1):
        if str(row["global_plane_id"]) not in aggregation.selected_global_ids:
            if progress is not None:
                progress("准备候选点云筛选", index, total, "淘汰")
            continue
        selected.setdefault(str(row["tile_id"]), set()).add(int(row["tile_plane_index"]))
        if progress is not None:
            progress("准备候选点云筛选", index, total, "保留")
    if not rows and progress is not None:
        progress("准备候选点云筛选", total, total, "无平面实例")
    return selected


__all__ = [
    "WholeCloudReportPreparation",
    "prepare_selected_plane_indices_by_tile",
    "prepare_whole_cloud_report",
]
