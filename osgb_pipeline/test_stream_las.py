"""Regression test for direct LAS streaming and LAS 1.4 header validity."""
import tempfile
from pathlib import Path
import subprocess

import laspy
import numpy as np


BASE = Path(__file__).resolve().parent


with tempfile.TemporaryDirectory(prefix="osgb-las-regression-") as directory:
    folder = Path(directory)
    fixture = folder / "input"
    output = folder / "cloud.las.part"
    subprocess.run([str(BASE / "build/make_fixture"), str(fixture)], check=True)
    subprocess.run(
        [
            str(BASE / "build/extract_las"),
            str(fixture / "Block.osgb"),
            str(output),
            "100",
            "0",
            "100",
            "200",
            "300",
            "42",
            "0",
            "0",
            "0",
        ],
        check=True,
    )
    with laspy.open(output) as reader:
        assert str(reader.header.version) == "1.4"
        assert reader.header.point_format.id == 0
        assert reader.header.point_count == 200
        points = reader.read_points(200)
        xyz = np.column_stack((points.x, points.y, points.z))
        assert np.isfinite(xyz).all()
        assert np.all(xyz[:, :2] >= np.array([110.0, 220.0]) - 0.001)
    print("PASS: direct LAS streaming, LAS 1.4 header, and quantized XYZ readback")
