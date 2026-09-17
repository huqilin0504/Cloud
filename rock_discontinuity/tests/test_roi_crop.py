from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from rock_discontinuity.processing.roi_crop import (
    crop_las_to_projection_roi,
    project_xyz,
    projection_polygon_mask,
    projection_roi_from_mapping,
)


class ProjectionRoiCropTests(unittest.TestCase):
    def _roi(self):
        return projection_roi_from_mapping(
            {
                "name": "test",
                "boundary_buffer_m": 0.0,
                "projection": {
                    "origin_xyz": [0.0, 0.0, 0.0],
                    "azimuth_deg": 0.0,
                    "elevation_deg": 0.0,
                },
                "polygon_uv": [[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]],
            }
        )

    def test_projection_and_polygon_mask(self):
        roi = self._roi()
        x = np.array([2.0, 8.0, 10.0])
        y = np.array([50.0, -50.0, 0.0])
        z = np.array([2.0, 8.0, 5.0])
        u, v = project_xyz(x, y, z, roi)
        np.testing.assert_allclose(u, x)
        np.testing.assert_allclose(v, z)
        np.testing.assert_array_equal(
            projection_polygon_mask(x, y, z, roi), [True, True, False]
        )

    def test_streaming_crop_preserves_dimensions_and_header(self):
        with tempfile.TemporaryDirectory(prefix="projection-roi-") as directory:
            root = Path(directory)
            source = root / "source.las"
            output = root / "cropped.laz"
            report = root / "cropped.roi.json"
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.scales = np.array([0.001, 0.001, 0.001])
            header.offsets = np.array([1000.0, 2000.0, 3000.0])
            data = laspy.LasData(header)
            count = 1000
            local_x = np.linspace(0.2, 9.8, count)
            local_y = np.linspace(-20.0, 20.0, count)
            local_z = np.linspace(0.2, 9.8, count)
            data.x = 1000.0 + local_x
            data.y = 2000.0 + local_y
            data.z = 3000.0 + local_z
            data.intensity = np.arange(count, dtype=np.uint16)
            data.classification = np.arange(count, dtype=np.uint8) % 8
            data.red = np.arange(count, dtype=np.uint16)
            data.green = np.arange(count, dtype=np.uint16) + 1
            data.blue = np.arange(count, dtype=np.uint16) + 2
            data.write(source)
            roi = projection_roi_from_mapping(
                {
                    "projection": {
                        "origin_xyz": [1000.0, 2000.0, 3000.0],
                        "azimuth_deg": 0.0,
                        "elevation_deg": 0.0,
                    },
                    "polygon_uv": [[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]],
                }
            )
            expected = (local_x > 1.0) & (local_x < 9.0) & (local_z > 1.0) & (local_z < 9.0)
            result = crop_las_to_projection_roi(
                source,
                output,
                roi,
                report_path=report,
                chunk_size=137,
                show_progress=False,
            )
            self.assertEqual(result["output"]["point_count"], int(expected.sum()))
            self.assertTrue(report.is_file())
            with laspy.open(source) as source_reader, laspy.open(output) as output_reader:
                source_points = source_reader.read()
                output_points = output_reader.read()
                self.assertEqual(output_reader.header.point_format.id, source_reader.header.point_format.id)
                np.testing.assert_allclose(output_reader.header.scales, source_reader.header.scales)
                np.testing.assert_allclose(output_reader.header.offsets, source_reader.header.offsets)
                np.testing.assert_allclose(output_points.x, source_points.x[expected])
                np.testing.assert_allclose(output_points.y, source_points.y[expected])
                np.testing.assert_allclose(output_points.z, source_points.z[expected])
                np.testing.assert_array_equal(output_points.intensity, source_points.intensity[expected])
                np.testing.assert_array_equal(
                    output_points.classification, source_points.classification[expected]
                )
                np.testing.assert_array_equal(output_points.red, source_points.red[expected])

    def test_refuses_to_overwrite_input(self):
        with tempfile.TemporaryDirectory(prefix="projection-roi-same-") as directory:
            source = Path(directory) / "source.las"
            laspy.LasData(laspy.LasHeader(point_format=0, version="1.2")).write(source)
            with self.assertRaises(ValueError):
                crop_las_to_projection_roi(
                    source, source, self._roi(), show_progress=False
                )


if __name__ == "__main__":
    unittest.main()
