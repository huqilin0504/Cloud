from __future__ import annotations

import unittest

import numpy as np

from rock_discontinuity.processing.detachment import classify_candidate_detachment_planes
from rock_discontinuity.core.models import PlaneInstance


class DetachmentTests(unittest.TestCase):
    def test_quality_accepted_planes_are_marked_as_candidates(self):
        planes = [
            PlaneInstance(
                plane_id="J1-001",
                set_id="J1",
                area=1.0,
                confidence=0.8,
            ),
            PlaneInstance(
                plane_id="J1-002",
                set_id="J1",
                area=0.1,
                confidence=0.8,
            ),
        ]
        selected, rows = classify_candidate_detachment_planes(
            planes,
            {"min_confidence": 0.6, "min_area_m2": 0.25},
        )
        np.testing.assert_array_equal(selected, [True, False])
        self.assertEqual(rows[0]["status"], "candidate_joint_plane")
        self.assertEqual(rows[1]["status"], "not_selected")

    def test_conservative_roi_gate_rejects_narrow_incomplete_and_edge_facets(self):
        def plane(plane_id, *, minor=0.8, completeness=0.8, edge=False):
            return PlaneInstance(
                plane_id=plane_id,
                set_id="J1",
                area=1.0,
                minor_extent=minor,
                boundary_completeness=completeness,
                edge_censored=edge,
                inlier_ratio=0.95,
                normal_dispersion=3.0,
                confidence=0.85,
            )

        config = {
            "min_confidence": 0.7,
            "min_area_m2": 0.25,
            "min_minor_extent_m": 0.5,
            "min_boundary_completeness": 0.7,
            "min_inlier_ratio": 0.8,
            "max_normal_dispersion_deg": 8.0,
            "reject_edge_censored_roi": True,
        }
        planes = [
            plane("accepted"),
            plane("narrow", minor=0.3),
            plane("incomplete", completeness=0.5),
            plane("edge", edge=True),
        ]
        selected, rows = classify_candidate_detachment_planes(
            planes,
            config,
            context="roi",
        )
        np.testing.assert_array_equal(selected, [True, False, False, False])
        self.assertIn("below_min_minor_extent", rows[1]["selection_reason"])
        self.assertIn("below_min_boundary_completeness", rows[2]["selection_reason"])
        self.assertIn("roi_edge_censored", rows[3]["selection_reason"])

        tile_selected, _ = classify_candidate_detachment_planes(
            [planes[-1]],
            config,
            context="tile",
        )
        np.testing.assert_array_equal(tile_selected, [True])

    def test_gate_rejects_below_min_major_extent(self):
        planes = [
            PlaneInstance(
                plane_id="accepted",
                set_id="J1",
                area=2.5,
                major_extent=3.5,
                minor_extent=0.8,
                boundary_completeness=0.8,
                inlier_ratio=0.9,
                normal_dispersion=3.0,
                confidence=0.85,
            ),
            PlaneInstance(
                plane_id="short_major",
                set_id="J1",
                area=2.5,
                major_extent=2.5,
                minor_extent=0.8,
                boundary_completeness=0.8,
                inlier_ratio=0.9,
                normal_dispersion=3.0,
                confidence=0.85,
            ),
        ]
        config = {
            "min_confidence": 0.70,
            "min_area_m2": 2.0,
            "min_major_extent_m": 3.0,
            "min_minor_extent_m": 0.5,
        }
        selected, rows = classify_candidate_detachment_planes(planes, config, context="global")
        np.testing.assert_array_equal(selected, [True, False])
        self.assertEqual(rows[0]["status"], "candidate_joint_plane")
        self.assertEqual(rows[1]["status"], "not_selected")
        self.assertIn("below_min_major_extent", rows[1]["selection_reason"])
        self.assertEqual(rows[0]["major_extent_m"], 3.5)
        self.assertEqual(rows[1]["major_extent_m"], 2.5)


if __name__ == "__main__":
    unittest.main()
