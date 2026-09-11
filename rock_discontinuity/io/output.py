from __future__ import annotations

import colorsys
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import dump_config
from ..core.models import (
    DetachmentRecord,
    FeatureMap,
    JointSet,
    PlaneInstance,
    SpacingRecord,
    ApertureRecord,
    TraceRecord,
    joint_set_to_row,
    plane_to_row,
)
PLANE_FIELDS = list(plane_to_row(PlaneInstance()).keys())
JOINT_SET_FIELDS = list(joint_set_to_row(JointSet("J0", [], np.zeros(3), 0.0, 0.0, 0, False)).keys())
SPACING_FIELDS = ["set_id", "plane_id_a", "plane_id_b", "spacing_m", "method", "sample_count"]
TRACE_FIELDS = [
    "plane_id",
    "available",
    "trace_verified",
    "method",
    "chord_length_m",
    "polyline_length_m",
    "reason",
]
APERTURE_FIELDS = ["plane_id", "available", "source", "effective_resolution_m", "aperture_m", "reason"]
DETACHMENT_FIELDS = [
    "plane_id",
    "set_id",
    "status",
    "selection_reason",
    "dip_direction_deg",
    "dip_deg",
    "trace_length_m",
    "apparent_persistence_m",
    "observed_area_m2",
    "major_extent_m",
    "minor_extent_m",
    "edge_censored",
    "nearest_spacing_m",
    "center_x",
    "center_y",
    "center_z",
    "area_m2",
    "boundary_completeness",
    "inlier_ratio",
    "normal_dispersion_deg",
    "normal_scale_m",
    "normal_stability_deg",
    "normal_stable_fraction",
    "normal_valid_scale_count",
    "normal_stable",
    "confidence",
    "quality_score",
    "quality_grade",
]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    raise TypeError(f"无法序列化类型：{type(value)!r}")


def _plane_color(index: int) -> tuple[int, int, int]:
    if index < 0:
        return 128, 128, 128
    red, green, blue = colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.75, 0.95)
    return round(red * 255), round(green * 255), round(blue * 255)


def write_segmented_ply(
    path: Path,
    points: np.ndarray,
    labels: np.ndarray,
    planes: list[PlaneInstance],
) -> None:
    """Write a compact binary PLY with synthetic segmentation colors."""

    plane_index = {plane.plane_id: index for index, plane in enumerate(planes)}
    set_ids = sorted({plane.set_id for plane in planes if plane.set_id is not None})
    set_index = {set_id: index for index, set_id in enumerate(set_ids)}
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("plane_index", "<i4"),
            ("set_index", "<i4"),
        ]
    )
    output = np.empty(len(points), dtype=dtype)
    output["x"], output["y"], output["z"] = np.asarray(points, dtype=np.float32).T
    output["plane_index"] = -1
    output["set_index"] = -1
    output["red"] = 128
    output["green"] = 128
    output["blue"] = 128
    for label in sorted(set(int(value) for value in labels if value >= 0)):
        if label >= len(planes):
            continue
        mask = labels == label
        color = _plane_color(label)
        output["plane_index"][mask] = label
        plane_set_id = planes[label].set_id
        output["set_index"][mask] = (
            set_index.get(plane_set_id, -1) if plane_set_id is not None else -1
        )
        output["red"][mask], output["green"][mask], output["blue"][mask] = color

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment colors are synthetic labels; source LAS has no RGB\n"
        f"element vertex {len(output)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property int plane_index\n"
        "property int set_index\n"
        "end_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        output.tofile(stream)


def write_detachment_ply(
    path: Path,
    points: np.ndarray,
    labels: np.ndarray,
    planes: list[PlaneInstance],
    selected_planes: np.ndarray,
) -> None:
    """Write a PLY highlighting geometry-only candidate joint planes.

    Candidate plane points are red; all other points are neutral gray.  The
    integer plane/set fields are kept identical to ``segmented_planes.ply`` so
    common point-cloud viewers can inspect both files consistently.
    """

    selected_planes = np.asarray(selected_planes, dtype=bool)
    if len(selected_planes) != len(planes):
        raise ValueError("selected_planes 长度必须与 planes 一致")
    set_ids = sorted({plane.set_id for plane in planes if plane.set_id is not None})
    set_index = {set_id: index for index, set_id in enumerate(set_ids)}
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("plane_index", "<i4"),
            ("set_index", "<i4"),
        ]
    )
    output = np.empty(len(points), dtype=dtype)
    output["x"], output["y"], output["z"] = np.asarray(points, dtype=np.float32).T
    output["plane_index"] = -1
    output["set_index"] = -1
    output["red"] = 128
    output["green"] = 128
    output["blue"] = 128
    for label in sorted(set(int(value) for value in labels if value >= 0)):
        if label >= len(planes):
            continue
        mask = labels == label
        output["plane_index"][mask] = label
        plane_set_id = planes[label].set_id
        output["set_index"][mask] = (
            set_index.get(plane_set_id, -1) if plane_set_id is not None else -1
        )
        if selected_planes[label]:
            output["red"][mask] = 230
            output["green"][mask] = 35
            output["blue"][mask] = 35

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment red points are geometry-only candidate joint planes\n"
        "comment candidate color does not confirm geological origin or instability\n"
        f"element vertex {len(output)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property int plane_index\n"
        "property int set_index\n"
        "end_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        output.tofile(stream)


def write_outputs(
    output_dir: Path,
    *,
    original_points: np.ndarray,
    original_labels: np.ndarray,
    filtered_points: np.ndarray,
    features: FeatureMap,
    planes: list[PlaneInstance],
    plane_instances: list[PlaneInstance] | None = None,
    plane_instance_original_labels: np.ndarray | None = None,
    joint_sets: list[JointSet],
    spacing_rows: list[SpacingRecord],
    detachment_labels: np.ndarray,
    config: dict[str, Any],
    read_metadata: dict[str, Any],
    report: dict[str, Any],
    data_audit: dict[str, Any],
    density_report: dict[str, Any],
    prepared_detachment_rows: list[DetachmentRecord],
    prepared_boundary_feature_collection: dict[str, Any],
    prepared_trace_rows: list[TraceRecord],
    prepared_aperture_rows: list[ApertureRecord],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plane_rows = [plane_to_row(plane) for plane in planes]
    joint_rows = [joint_set_to_row(joint_set) for joint_set in joint_sets]
    _write_csv(output_dir / "planes.csv", plane_rows, PLANE_FIELDS)
    if plane_instances is not None:
        _write_csv(
            output_dir / "plane_instances.csv",
            [plane_to_row(plane) for plane in plane_instances],
            PLANE_FIELDS,
        )
    _write_csv(output_dir / "joint_sets.csv", joint_rows, JOINT_SET_FIELDS)
    _write_csv(output_dir / "spacings.csv", spacing_rows, SPACING_FIELDS)
    _write_csv(output_dir / "detachment_planes.csv", prepared_detachment_rows, DETACHMENT_FIELDS)
    _write_csv(output_dir / "traces.csv", prepared_trace_rows, TRACE_FIELDS)
    _write_csv(output_dir / "aperture.csv", prepared_aperture_rows, APERTURE_FIELDS)
    rejected_rows = report.get("rejected_planes", [])
    _write_csv(output_dir / "rejected_planes.csv", rejected_rows, ["raw_set_label", "point_count", "reason"])
    write_segmented_ply(output_dir / "segmented_planes.ply", original_points, original_labels, planes)
    if plane_instances is not None and plane_instance_original_labels is not None:
        write_segmented_ply(
            output_dir / "plane_instances.ply",
            original_points,
            plane_instance_original_labels,
            plane_instances,
        )
    write_detachment_ply(
        output_dir / "detachment_candidates.ply",
        original_points,
        original_labels,
        planes,
        detachment_labels,
    )
    feature_payload: dict[str, Any] = {"xyz": filtered_points, **features}
    np.savez_compressed(output_dir / "features.npz", **feature_payload)
    (output_dir / "plane_boundaries.geojson").write_text(
        json.dumps(
            prepared_boundary_feature_collection,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output_dir / "trace_lines.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dump_config(config, output_dir / "run_config.yaml")

    serializable_report = dict(report)
    serializable_report["input"] = read_metadata
    serializable_report["plane_count"] = len(planes)
    serializable_report["plane_instance_count"] = len(plane_instances or [])
    serializable_report["joint_set_count"] = len(joint_sets)
    serializable_report["data_audit"] = data_audit
    serializable_report["density_report"] = density_report
    (output_dir / "report.json").write_text(
        json.dumps(serializable_report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "run.json").write_text(
        json.dumps(serializable_report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "data_audit.json").write_text(
        json.dumps(serializable_report["data_audit"], ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "density_report.json").write_text(
        json.dumps(serializable_report["density_report"], ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    output_paths = {
        "planes": output_dir / "planes.csv",
        "joint_sets": output_dir / "joint_sets.csv",
        "spacings": output_dir / "spacings.csv",
        "detachment_planes": output_dir / "detachment_planes.csv",
        "detachment_visualization": output_dir / "detachment_candidates.ply",
        "segmented": output_dir / "segmented_planes.ply",
        "features": output_dir / "features.npz",
        "report": output_dir / "report.json",
        "run": output_dir / "run.json",
        "data_audit": output_dir / "data_audit.json",
        "density_report": output_dir / "density_report.json",
        "traces": output_dir / "traces.csv",
        "aperture": output_dir / "aperture.csv",
        "plane_boundaries": output_dir / "plane_boundaries.geojson",
        "trace_lines": output_dir / "trace_lines.geojson",
    }
    if plane_instances is not None:
        output_paths["plane_instances"] = output_dir / "plane_instances.csv"
    return output_paths
