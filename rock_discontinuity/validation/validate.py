from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path
from typing import Any

import laspy
import numpy as np


REQUIRED_FILES = (
    "planes.csv",
    "joint_sets.csv",
    "spacings.csv",
    "detachment_planes.csv",
    "detachment_candidates.ply",
    "rejected_planes.csv",
    "segmented_planes.ply",
    "features.npz",
    "report.json",
    "run.json",
    "data_audit.json",
    "density_report.json",
    "traces.csv",
    "aperture.csv",
    "plane_boundaries.geojson",
    "trace_lines.geojson",
    "run_config.yaml",
)

WHOLE_CLOUD_REQUIRED_FILES = (
    "detachment_planes.csv",
    "joint_planes.csv",
    "joint_sets.csv",
    "spacings.csv",
    "tile_spacings.csv",
    "traces.csv",
    "aperture.csv",
    "whole_cloud_report.json",
    "run.json",
    "data_audit.json",
    "density_report.json",
    "run_config.yaml",
    "processing_state.json",
    "split_state.json",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _ply_vertex_count(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header_lines: list[bytes] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header 缺少 end_header")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(header_lines).decode("ascii")
        vertex_line = next(line for line in header.splitlines() if line.startswith("element vertex "))
        count = int(vertex_line.split()[2])
        payload_offset = stream.tell()
    return count, payload_offset


def validate_output(output_dir: Path) -> dict[str, Any]:
    if (output_dir / "whole_cloud_report.json").is_file() and not (output_dir / "report.json").is_file():
        return validate_whole_cloud_output(output_dir)
    missing = [name for name in REQUIRED_FILES if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"缺少输出文件：{missing}")
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    planes = _read_csv(output_dir / "planes.csv")
    joint_sets = _read_csv(output_dir / "joint_sets.csv")
    spacings = _read_csv(output_dir / "spacings.csv")
    detachment_planes = _read_csv(output_dir / "detachment_planes.csv")
    if len(planes) != int(report["plane_count"]):
        raise ValueError("planes.csv 与 report.json 数量不一致")
    if len(joint_sets) != int(report["joint_set_count"]):
        raise ValueError("joint_sets.csv 与 report.json 数量不一致")
    detachment = report.get("detachment")
    if not isinstance(detachment, dict):
        raise ValueError("report.json 缺少 detachment 统计")
    selected_detachment_planes = [
        row
        for row in detachment_planes
        if row.get("status") in {"candidate_joint_plane", "candidate_detachment_plane"}
    ]
    if len(selected_detachment_planes) != int(detachment["candidate_plane_count"]):
        raise ValueError("detachment_planes.csv 与 report.json 候选数量不一致")
    if detachment.get("status") != "geometry_only_candidate":
        raise ValueError("detachment.status 不受支持")
    density = report.get("density")
    if not isinstance(density, dict):
        raise ValueError("report.json 缺少 density 统计")
    for field in (
        "p10_points_m2",
        "p25_points_m2",
        "median_points_m2",
        "mean_points_m2",
        "p75_points_m2",
        "p90_points_m2",
    ):
        value = density.get(field)
        if value is None or not np.isfinite(float(value)):
            raise ValueError(f"density 缺少有限字段：{field}")
    if density.get("standard_status") not in {
        "pass",
        "DENSITY_TARGET_NOT_MET",
        "DENSITY_NOT_ASSESSED",
        "below_600_standard",
        "not_assessed",
    }:
        raise ValueError("density.standard_status 不受支持")
    density_report = json.loads((output_dir / "density_report.json").read_text(encoding="utf-8"))
    if density_report.get("standard_status") != density.get("standard_status"):
        raise ValueError("density_report.json 与 report.json 状态不一致")
    data_audit = json.loads((output_dir / "data_audit.json").read_text(encoding="utf-8"))
    if int(data_audit.get("point_count", -1)) != int(report["counts"]["input_points"]):
        raise ValueError("data_audit.json 与 report.json input_points 不一致")
    trace_rows = _read_csv(output_dir / "traces.csv")
    aperture_rows = _read_csv(output_dir / "aperture.csv")
    if len(trace_rows) != len(planes) or len(aperture_rows) != len(planes):
        raise ValueError("traces.csv/aperture.csv 必须覆盖每个已接受平面")
    for row in planes:
        for field in ("nx", "ny", "nz", "rms_m", "inlier_ratio", "confidence"):
            value = row.get(field, "")
            if value and not np.isfinite(float(value)):
                raise ValueError(f"planes.csv 存在非有限字段：{field}")
    vertex_count, payload_offset = _ply_vertex_count(output_dir / "segmented_planes.ply")
    expected_record_size = struct.calcsize("<fffBBBii")
    actual_payload = (output_dir / "segmented_planes.ply").stat().st_size - payload_offset
    if actual_payload != vertex_count * expected_record_size:
        raise ValueError("segmented_planes.ply payload size 不匹配")
    detachment_vertex_count, detachment_payload_offset = _ply_vertex_count(
        output_dir / "detachment_candidates.ply"
    )
    detachment_payload = (
        output_dir / "detachment_candidates.ply"
    ).stat().st_size - detachment_payload_offset
    if detachment_vertex_count != vertex_count or detachment_payload != detachment_vertex_count * expected_record_size:
        raise ValueError("detachment_candidates.ply payload size 或点数不匹配")
    feature_data = np.load(output_dir / "features.npz")
    if "xyz" not in feature_data or len(feature_data["xyz"]) != int(report["counts"]["retained_points"]):
        raise ValueError("features.npz 与 report.json retained_points 不一致")
    return {
        "status": "PASS",
        "plane_rows": len(planes),
        "joint_set_rows": len(joint_sets),
        "spacing_rows": len(spacings),
        "detachment_plane_rows": len(selected_detachment_planes),
        "detachment_candidate_points": int(detachment["candidate_point_count"]),
        "ply_vertices": vertex_count,
        "retained_points": int(report["counts"]["retained_points"]),
        "density_standard_status": density["standard_status"],
        "density_p10_points_m2": float(density["p10_points_m2"]),
        "density_median_points_m2": float(density["median_points_m2"]),
        "validation_stage": report["validation"]["stage"],
        "field_calibration": report["validation"]["field_calibration"],
    }


def validate_whole_cloud_output(output_dir: Path) -> dict[str, Any]:
    """Validate the cross-tile tables and state produced by whole-cloud runs."""

    missing = [name for name in WHOLE_CLOUD_REQUIRED_FILES if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"缺少全点云输出文件：{missing}")
    report = json.loads((output_dir / "whole_cloud_report.json").read_text(encoding="utf-8"))
    counts = report.get("counts")
    tiling = report.get("tiling")
    if not isinstance(counts, dict) or not isinstance(tiling, dict):
        raise ValueError("whole_cloud_report.json 缺少 counts 或 tiling")
    detachment = report.get("detachment")
    if not isinstance(detachment, dict) or detachment.get("status") != "geometry_only_candidate":
        raise ValueError("全点云 detachment.status 不受支持")

    def read_rows(name: str) -> list[dict[str, str]]:
        return _read_csv(output_dir / name)

    tile_planes = read_rows("detachment_planes.csv")
    global_planes = read_rows("joint_planes.csv")
    joint_sets = read_rows("joint_sets.csv")
    spacings = read_rows("spacings.csv")
    tile_spacings = read_rows("tile_spacings.csv")
    traces = read_rows("traces.csv")
    aperture = read_rows("aperture.csv")
    expected_counts = {
        "candidate_plane_instances": len(tile_planes),
        "global_joint_plane_count": len(global_planes),
        "global_joint_set_count": len(joint_sets),
        "spacing_rows": len(spacings),
        "tile_spacing_rows": len(tile_spacings),
    }
    for field, actual in expected_counts.items():
        if int(counts.get(field, -1)) != actual:
            raise ValueError(f"{field} 与输出表行数不一致")
    if len(traces) != len(global_planes) or len(aperture) != len(global_planes):
        raise ValueError("traces.csv/aperture.csv 必须覆盖每个全局平面")
    global_ids = [row.get("global_plane_id", "") for row in global_planes]
    if len(global_ids) != len(set(global_ids)):
        raise ValueError("joint_planes.csv 存在重复 global_plane_id")
    if int(tiling.get("failed_tiles", -1)) < 0 or int(tiling.get("processed_tiles", -1)) < 0:
        raise ValueError("tiling 处理计数不能为负数")

    data_audit = json.loads((output_dir / "data_audit.json").read_text(encoding="utf-8"))
    if int(data_audit.get("point_count", -1)) != int(counts.get("source_points", -2)):
        raise ValueError("data_audit.json 与全点云 source_points 不一致")
    state = json.loads((output_dir / "processing_state.json").read_text(encoding="utf-8"))
    if state.get("algorithm_version") != report.get("algorithm_version"):
        raise ValueError("processing_state.json 与报告算法版本不一致")
    tiles_state = state.get("tiles")
    if not isinstance(tiles_state, dict):
        raise ValueError("processing_state.json 的 tiles 必须是对象")
    done_states = [value for value in tiles_state.values() if value.get("status") == "done"]
    if len(done_states) != int(tiling.get("processed_tiles", -1)):
        raise ValueError("processing_state.json 与 processed_tiles 不一致")

    overlay_value = detachment.get("overlay")
    overlay_path = output_dir / Path(str(overlay_value)).name if overlay_value else None
    merged_points = int(counts.get("merged_overlay_points", 0))
    if merged_points and (overlay_path is None or not overlay_path.is_file()):
        raise ValueError("报告声明有候选叠加点，但 candidate_detachment_points.laz 不存在")
    if overlay_path is not None and overlay_path.is_file():
        with laspy.open(overlay_path) as reader:
            if int(reader.header.point_count) != merged_points:
                raise ValueError("候选叠加层点数与报告不一致")
    return {
        "status": "PASS",
        "tile_plane_rows": len(tile_planes),
        "global_plane_rows": len(global_planes),
        "joint_set_rows": len(joint_sets),
        "spacing_rows": len(spacings),
        "tile_spacing_rows": len(tile_spacings),
        "processed_tiles": int(tiling["processed_tiles"]),
        "failed_tiles": int(tiling["failed_tiles"]),
        "candidate_overlay_points": merged_points,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验结构面分析输出目录")
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    result = validate_output(args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
