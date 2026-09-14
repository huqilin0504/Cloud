"""Stream a high-density XYZ LAS from the OSGB root and atomically replace cloud.las."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import laspy
import numpy as np

BASE = Path(__file__).resolve().parent
DEFAULT_ROOT = BASE.parent / "科研 现场资料/导流洞进口危岩体 航拍三维/危岩体/Block.osgb"
DEFAULT_TARGET = BASE.parent / "outputs/root_full_xyz_10/cloud.las"
PARAMS = BASE / "structure_params_600.json"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".600.part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    params = json.loads(PARAMS.read_text())
    default_density = float(params["density"]["target_raw_density_points_m2"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--density", type=float, default=default_density)
    parser.add_argument("--max-geometries", type=int, default=0)
    parser.add_argument("--origin", choices=["keep", "add"], default="add")
    args = parser.parse_args()
    if not math.isfinite(args.density) or args.density <= 0 or args.max_geometries < 0:
        parser.error("invalid density or geometry limit")

    root = args.root.resolve()
    target = args.target.resolve()
    if not root.is_file():
        parser.error(f"root OSGB does not exist: {root}")
    if not target.parent.is_dir():
        parser.error(f"target directory does not exist: {target.parent}")

    metadata = ET.parse(root.parent / "metadata.xml").getroot()
    origin = np.array([float(value) for value in metadata.findtext(".//SRSOrigin").split(",")])
    if origin.shape != (3,) or not np.isfinite(origin).all():
        parser.error("invalid SRSOrigin")
    shift = origin if args.origin == "add" else np.zeros(3)
    # The current project coordinates are within the int32 LAS range relative
    # to the metadata origin.  A millimetre scale is retained.
    offset = np.floor(shift)
    part = target.with_name(target.name + ".600.part")
    if part.exists():
        parser.error(f"partial output already exists; inspect or remove it first: {part}")

    command = [
        str(BASE / "build/extract_las"),
        str(root),
        str(part),
        str(args.density),
        str(args.max_geometries),
        *(str(value) for value in shift),
        "42",
        *(str(value) for value in offset),
    ]
    subprocess.run(command, check=True)

    with laspy.open(part) as reader:
        header = reader.header
        count = int(header.point_count)
        if count <= 0 or header.point_format.id != 0 or str(header.version) != "1.4":
            raise RuntimeError("streamed LAS header validation failed")
        expected_size = 375 + count * int(header.point_format.size)
        if part.stat().st_size != expected_size:
            raise RuntimeError(
                f"streamed LAS size validation failed: {part.stat().st_size} != {expected_size}"
            )
        sample_count = min(10000, count)
        sample = reader.read_points(sample_count)
        if len(sample) != sample_count:
            raise RuntimeError("streamed LAS sample readback failed")
        sample_xyz = np.column_stack((sample.x, sample.y, sample.z))
        if not np.isfinite(sample_xyz).all():
            raise RuntimeError("streamed LAS contains non-finite sample coordinates")
        minimum = np.asarray(header.mins, dtype=float)
        maximum = np.asarray(header.maxs, dtype=float)
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise RuntimeError("streamed LAS bounds validation failed")

    qc = {
        "root": str(root),
        "origin_mode": args.origin,
        "metadata_origin": origin.tolist(),
        "srs": metadata.findtext(".//SRS", "").strip(),
        "assigned_crs": None,
        "points": count,
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
        "scale": [0.001, 0.001, 0.001],
        "density": args.density,
        "geometry_limit": args.max_geometries,
        "texture": False,
        "rgb": False,
        "streamed_las": True,
        "axis_convention": {"x": "east", "y": "north", "z": "up", "unit": "m"},
        "standard": str(PARAMS),
    }
    os.replace(part, target)
    atomic_json(target.parent / "qc.json", qc)
    stats_path = target.parent / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats.update({"points": count, "density": args.density, "streamed_las": True})
    atomic_json(stats_path, stats)
    print(json.dumps(qc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
