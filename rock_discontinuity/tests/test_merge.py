from __future__ import annotations

import unittest

import numpy as np

from rock_discontinuity.config import load_config
from rock_discontinuity.core.models import PlaneInstance
from rock_discontinuity.processing.merge import merge_plane_instances, planes_can_merge


class PlaneMergeTests(unittest.TestCase):
    def test_reversed_normals_parallel_offset_threshold_and_missing_footprint(self):
        # Use projected-coordinate-sized values from the real site so that the
        # fallback bbox path is checked without relying on an alpha footprint.
        common = {
            "d": -525005.0,
            "centroid": np.array([525005.0, 3132565.0, 2205.0]),
            "bbox_min": np.array([525005.0, 3132560.0, 2200.0]),
            "bbox_max": np.array([525005.0, 3132570.0, 2210.0]),
            "rms": 0.001,
        }
        left = PlaneInstance(normal=np.array([1.0, 0.0, 0.0]), **common)
        within = PlaneInstance(
            normal=np.array([-1.0, 0.0, 0.0]),
            d=525005.1,
            centroid=np.array([525005.1, 3132565.0, 2205.0]),
            bbox_min=np.array([525005.1, 3132560.0, 2200.0]),
            bbox_max=np.array([525005.1, 3132570.0, 2210.0]),
            rms=0.001,
        )
        beyond = PlaneInstance(
            normal=np.array([-1.0, 0.0, 0.0]),
            d=525005.11,
            centroid=np.array([525005.11, 3132565.0, 2205.0]),
            bbox_min=np.array([525005.11, 3132560.0, 2200.0]),
            bbox_max=np.array([525005.11, 3132570.0, 2210.0]),
            rms=0.001,
        )
        self.assertTrue(
            planes_can_merge(
                left,
                within,
                normal_angle_deg=1.0,
                plane_offset_m=0.1,
                spatial_gap_m=0.1000001,
            )
        )
        self.assertFalse(
            planes_can_merge(
                left,
                beyond,
                normal_angle_deg=1.0,
                plane_offset_m=0.1,
                spatial_gap_m=0.1,
            )
        )

    def test_projected_footprints_prevent_bbox_false_merge(self):
        common = {
            "normal": np.array([0.0, 0.0, 1.0]),
            "d": 0.0,
            "rms": 0.001,
        }
        left = PlaneInstance(
            centroid=np.array([0.4, 0.4, 0.0]),
            bbox_min=np.array([0.0, 0.0, 0.0]),
            bbox_max=np.array([1.0, 1.0, 0.0]),
            **common,
        )
        right = PlaneInstance(
            centroid=np.array([1.6, 1.6, 0.0]),
            bbox_min=np.array([1.0, 1.0, 0.0]),
            bbox_max=np.array([2.0, 2.0, 0.0]),
            **common,
        )
        left_ring = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        right_ring = [[[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]]]
        self.assertFalse(
            planes_can_merge(
                left,
                right,
                normal_angle_deg=5.0,
                plane_offset_m=0.05,
                spatial_gap_m=0.3,
                footprint_a=left_ring,
                footprint_b=right_ring,
            )
        )

    def test_adjacent_coplanar_instances_are_refit_as_one_plane(self):
        left_x = np.arange(-2.0, -0.09, 0.1)
        right_x = np.arange(0.0, 2.01, 0.1)
        y = np.arange(-1.0, 1.01, 0.1)
        left_xx, left_yy = np.meshgrid(left_x, y)
        right_xx, right_yy = np.meshgrid(right_x, y)
        points = np.vstack(
            [
                np.column_stack((left_xx.ravel(), left_yy.ravel(), np.zeros(left_xx.size))),
                np.column_stack((right_xx.ravel(), right_yy.ravel(), np.zeros(right_xx.size))),
            ]
        )
        normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(points), 1))
        planarity = np.ones(len(points), dtype=np.float64)
        left_count = left_xx.size
        common = {
            "normal": np.array([0.0, 0.0, 1.0]),
            "d": 0.0,
            "dip_direction": 0.0,
            "dip": 0.0,
            "area": 1.0,
            "rms": 0.001,
        }
        planes = [
            PlaneInstance(
                plane_id="J1-001",
                point_indices=np.arange(left_count),
                candidate_point_count=left_count,
                centroid=points[:left_count].mean(axis=0),
                bbox_min=points[:left_count].min(axis=0),
                bbox_max=points[:left_count].max(axis=0),
                **common,
            ),
            PlaneInstance(
                plane_id="J1-002",
                point_indices=np.arange(left_count, len(points)),
                candidate_point_count=len(points) - left_count,
                centroid=points[left_count:].mean(axis=0),
                bbox_min=points[left_count:].min(axis=0),
                bbox_max=points[left_count:].max(axis=0),
                **common,
            ),
        ]
        config = load_config(None)
        merged, mapping, stats = merge_plane_instances(
            planes,
            points=points,
            normals=normals,
            planarity=planarity,
            fit_config=config["plane"],
            boundary_config=config["boundary"],
            target_config=config["target"],
            quality_config=config["quality"],
            source_bounds=(points.min(axis=0), points.max(axis=0)),
            merge_config=config["plane_merge"],
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(mapping, {0: 0, 1: 0})
        self.assertEqual(stats["merged_components"], 1)
        self.assertGreaterEqual(len(merged[0].point_indices), 2 * left_count - 2)
        self.assertLess(merged[0].rms, 1e-6)


if __name__ == "__main__":
    unittest.main()
