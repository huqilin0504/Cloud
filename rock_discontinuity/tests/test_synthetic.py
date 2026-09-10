from __future__ import annotations

import unittest

import numpy as np

from rock_discontinuity.config import load_config
from rock_discontinuity.core.fitting import dip_and_dip_direction
from rock_discontinuity.processing.pipeline import process_points, run_points
from rock_discontinuity.processing.preprocess import estimate_local_features


def plane_normal(dip_direction: float, dip: float) -> np.ndarray:
    return np.array(
        [
            np.sin(np.deg2rad(dip)) * np.sin(np.deg2rad(dip_direction)),
            np.sin(np.deg2rad(dip)) * np.cos(np.deg2rad(dip_direction)),
            np.cos(np.deg2rad(dip)),
        ]
    )


def plane_patch(normal: np.ndarray, center: np.ndarray, width: float, height: float, step: float) -> np.ndarray:
    normal = normal / np.linalg.norm(normal)
    reference = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, reference)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    values_u = np.arange(-width / 2.0, width / 2.0 + step / 2.0, step)
    values_v = np.arange(-height / 2.0, height / 2.0 + step / 2.0, step)
    uu, vv = np.meshgrid(values_u, values_v)
    return center + uu.ravel()[:, None] * u + vv.ravel()[:, None] * v


class SyntheticPipelineTests(unittest.TestCase):
    def test_multiscale_normals_are_stable_on_two_clean_planes(self):
        normal = plane_normal(90.0, 45.0)
        points = np.vstack(
            [
                plane_patch(normal, np.array([0.0, 0.0, 0.0]), 4.0, 4.0, 0.25),
                plane_patch(normal, normal * 2.0, 4.0, 4.0, 0.25),
            ]
        )
        features = estimate_local_features(
            points,
            radius=0.45,
            min_neighbors=8,
            knn=16,
            max_neighbors=64,
            multiscale_config={
                "enabled": True,
                "radii_m": [0.30, 0.45, 0.70],
                "max_neighbors": 64,
                "stable_angle_deg": 5.0,
                "min_stable_scales": 2,
                "selection": "smallest_adjacent_stable",
            },
        )
        stable = features["normal_stable"]
        self.assertGreater(float(np.mean(stable)), 0.50)
        self.assertLess(float(np.nanmedian(features["normal_stability_deg"])), 5.0)
        self.assertLessEqual(float(np.nanmedian(features["normal_scale_m"])), 0.45)

    def test_known_plane_orientation(self):
        normal = plane_normal(128.0, 55.0)
        direction, dip = dip_and_dip_direction(normal)
        self.assertAlmostEqual(direction, 128.0, places=5)
        self.assertAlmostEqual(dip, 55.0, places=5)

    def test_pipeline_finds_two_parallel_patches_and_spacing(self):
        config = load_config(None)
        config["preprocess"].update({"voxel_size": 0.0, "remove_outliers": False})
        # This is the fixed-scale baseline case; the multiscale ablation is
        # covered separately with scales matched to this synthetic spacing.
        config["normal_multiscale"]["enabled"] = False
        config["normal"].update({"radius": 0.9, "min_neighbors": 8, "knn": 16})
        config["feature"].update({"min_planarity": 0.80, "max_surface_variation": 0.02})
        config["orientation_cluster"].update({"angle_eps_deg": 5.0, "min_samples": 10})
        config["spatial_cluster"].update({"eps": 0.45, "min_samples": 8})
        config["region_growing"].update({"radius_m": 0.45, "normal_angle_deg": 5.0})
        config["plane"].update({"ransac_distance": 0.03, "ransac_iterations": 200, "min_points": 30, "min_area": 2.0, "max_rms": 0.05})
        normal = plane_normal(90.0, 45.0)
        patch_a = plane_patch(normal, np.array([0.0, 0.0, 0.0]), 6.0, 5.0, 0.25)
        patch_b = plane_patch(normal, normal * 2.0, 6.0, 5.0, 0.25)
        result = run_points(np.vstack((patch_a, patch_b)), config)
        self.assertGreaterEqual(len(result["planes"]), 2)
        self.assertGreaterEqual(len(result["joint_sets"]), 1)
        spacing_values = [row["spacing_m"] for row in result["spacing_rows"]]
        self.assertTrue(any(abs(value - 2.0) < 0.15 for value in spacing_values), spacing_values)
        self.assertEqual(
            {row["method"] for row in result["spacing_rows"]},
            {"spacing_3d_nonpersistent", "spacing_virtual_scanline"},
        )
        typed = process_points(np.vstack((patch_a, patch_b)), config)
        self.assertEqual(len(typed.planes), len(result["planes"]))
        legacy = typed.as_legacy_dict()
        self.assertIs(typed.original_points, legacy["original_points"])
        self.assertIs(typed.original_labels, legacy["original_labels"])
        self.assertIs(typed.features, legacy["features"])


if __name__ == "__main__":
    unittest.main()
