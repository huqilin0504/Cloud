from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path
from typing import Any

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验结构面分析输出目录")
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    result = validate_output(args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
