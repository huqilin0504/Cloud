from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rock_discontinuity.config import load_config
from rock_discontinuity.core.models import PlaneInstance
from rock_discontinuity.io.output import write_outputs
from rock_discontinuity.core.records import nearest_spacing_by_plane
from rock_discontinuity.processing.detachment import classify_candidate_detachment_planes
from rock_discontinuity.processing.output_records import (
    prepare_aperture_rows,
    prepare_plane_boundaries,
    prepare_trace_rows,
)
from rock_discontinuity.processing.records import prepare_detachment_rows
from rock_discontinuity.processing.whole_cloud import WHOLE_PLANE_FIELDS, WHOLE_SPACING_FIELDS


class OutputFieldTests(unittest.TestCase):
    def test_detachment_output_contains_requested_metrics(self):
        planes = [
            PlaneInstance(
                plane_id="J1-001",
                set_id="J1",
                point_indices=np.array([0]),
                candidate_point_count=1,
                normal=np.array([1.0, 0.0, 0.0]),
                d=-10.0,
                centroid=np.array([10.0, 20.0, 30.0]),
                dip_direction=90.0,
                dip=90.0,
                area=1.0,
                apparent_persistence=2.0,
                confidence=0.8,
            ),
            PlaneInstance(
                plane_id="J1-002",
                set_id="J1",
                point_indices=np.array([1]),
                candidate_point_count=1,
                normal=np.array([1.0, 0.0, 0.0]),
                d=-11.25,
                centroid=np.array([11.25, 20.0, 30.0]),
                dip_direction=90.0,
                dip=90.0,
                area=1.0,
                apparent_persistence=2.5,
                confidence=0.8,
            ),
        ]
        detachment_labels, detachment_rows = classify_candidate_detachment_planes(
            planes,
            {"min_confidence": 0.6, "min_area_m2": 0.25},
        )
        prepared_detachment_rows = prepare_detachment_rows(
            planes,
            detachment_rows,
            nearest_spacing_by_plane(
                [
                    {
                        "plane_id_a": "J1-001",
                        "plane_id_b": "J1-002",
                        "spacing_m": 1.25,
                    }
                ]
            ),
        )
        with tempfile.TemporaryDirectory(prefix="output-fields-") as directory:
            output_dir = Path(directory)
            report = {
                "rejected_planes": [],
                "trace": {"method": "normal_tensor_voting"},
                "aperture": {"source": "highest_resolution_mesh_or_image"},
            }
            write_outputs(
                output_dir,
                original_points=np.array([[10.0, 20.0, 30.0], [11.25, 20.0, 30.0]]),
                original_labels=np.array([0, 1], dtype=np.int32),
                filtered_points=np.array([[10.0, 20.0, 30.0], [11.25, 20.0, 30.0]]),
                features={"normals": np.zeros((2, 3), dtype=np.float64)},
                planes=planes,
                joint_sets=[],
                spacing_rows=[
                    {
                        "set_id": "J1",
                        "plane_id_a": "J1-001",
                        "plane_id_b": "J1-002",
                        "spacing_m": 1.25,
                        "method": "virtual_scanline",
                    }
                ],
                detachment_labels=detachment_labels,
                config=load_config(None),
                read_metadata={},
                report=report,
                data_audit={},
                density_report={},
                prepared_detachment_rows=prepared_detachment_rows,
                prepared_boundary_feature_collection=prepare_plane_boundaries(
                    planes,
                    np.array([[10.0, 20.0, 30.0], [11.25, 20.0, 30.0]]),
                ),
                prepared_trace_rows=prepare_trace_rows(planes, report["trace"]),
                prepared_aperture_rows=prepare_aperture_rows(planes, report["aperture"]),
            )
            with (output_dir / "detachment_planes.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        required = {
            "dip_direction_deg",
            "dip_deg",
            "trace_length_m",
            "apparent_persistence_m",
            "nearest_spacing_m",
            "center_x",
            "center_y",
            "center_z",
        }
        self.assertTrue(required.issubset(rows[0].keys()))
        self.assertEqual(rows[0]["dip_direction_deg"], "90.0")
        self.assertEqual(rows[0]["dip_deg"], "90.0")
        self.assertEqual(rows[0]["trace_length_m"], "")
        self.assertEqual(rows[0]["apparent_persistence_m"], "2.0")
        self.assertEqual(rows[0]["nearest_spacing_m"], "1.25")
        self.assertEqual(rows[0]["center_x"], "10.0")
        self.assertEqual(rows[0]["center_y"], "20.0")
        self.assertEqual(rows[0]["center_z"], "30.0")

    def test_whole_cloud_export_declares_spacing_and_center_fields(self):
        self.assertTrue(
            {
                "dip_direction_deg",
                "dip_deg",
                "trace_length_m",
                "apparent_persistence_m",
                "nearest_spacing_m",
                "center_x",
                "center_y",
                "center_z",
            }.issubset(WHOLE_PLANE_FIELDS)
        )
        self.assertEqual(
            set(WHOLE_SPACING_FIELDS),
            {
                "tile_id",
                "plane_id_a",
                "plane_id_b",
                "global_plane_id_a",
                "global_plane_id_b",
                "spacing_m",
                "method",
            },
        )


if __name__ == "__main__":
    unittest.main()
