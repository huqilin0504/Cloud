from __future__ import annotations

import unittest

import numpy as np

from rock_discontinuity.core.fitting import dip_and_dip_direction, fit_tls
from rock_discontinuity.core.geometry import (
    geometry_metrics,
    plane_basis,
    project_to_plane,
    sparse_axial_dbscan_labels,
    split_axial_by_deviation,
)


class GeometryTests(unittest.TestCase):
    def test_dip_direction_uses_enu_azimuth(self):
        dip_direction = 90.0
        dip = 45.0
        normal = np.array(
            [
                np.sin(np.deg2rad(dip)) * np.sin(np.deg2rad(dip_direction)),
                np.sin(np.deg2rad(dip)) * np.cos(np.deg2rad(dip_direction)),
                np.cos(np.deg2rad(dip)),
            ]
        )
        actual_direction, actual_dip = dip_and_dip_direction(normal)
        self.assertAlmostEqual(actual_direction, dip_direction, places=6)
        self.assertAlmostEqual(actual_dip, dip, places=6)

    def test_tls_and_projected_square(self):
        normal = np.array([0.2, 0.4, 0.8944271909999159])
        normal /= np.linalg.norm(normal)
        u, v = plane_basis(normal)
        center = np.array([10.0, -3.0, 7.0])
        uv = np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]], dtype=float)
        points = center + uv[:, 0, None] * u + uv[:, 1, None] * v
        fitted_normal, d, fitted_center = fit_tls(points)
        self.assertLess(np.degrees(np.arccos(np.clip(abs(np.dot(fitted_normal, normal)), 0, 1))), 1e-6)
        projected = project_to_plane(points, fitted_center, fitted_normal)
        metrics = geometry_metrics(projected, {"alpha": "auto", "max_points": 100})
        self.assertAlmostEqual(metrics["area"], 8.0, places=4)
        self.assertGreater(metrics["apparent_persistence"], 4.0)
        self.assertAlmostEqual(float(np.dot(fitted_normal, fitted_center)) + d, 0.0, places=6)

    def test_sparse_axial_labels_keep_reversed_normals_together(self):
        tilt = np.deg2rad(5.0)
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [np.sin(tilt), 0.0, np.cos(tilt)],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        labels = sparse_axial_dbscan_labels(normals, np.deg2rad(10.0))
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])
        self.assertNotEqual(labels[0], labels[3])
        zero_eps_labels = sparse_axial_dbscan_labels(normals[:2], 0.0)
        self.assertEqual(zero_eps_labels[0], zero_eps_labels[1])

    def test_sparse_axial_labels_scale_without_pairwise_matrix(self):
        normals = np.repeat(np.array([[0.0, 0.0, 1.0]]), 20_000, axis=0)
        labels = sparse_axial_dbscan_labels(normals, np.deg2rad(10.0))
        self.assertEqual(len(np.unique(labels)), 1)

    def test_adaptive_axial_split_retains_points_and_respects_deviation(self):
        def unit(vector):
            vector = np.asarray(vector, dtype=float)
            return vector / np.linalg.norm(vector)

        normals = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [np.sin(np.deg2rad(5.0)), 0.0, np.cos(np.deg2rad(5.0))],
                [-np.sin(np.deg2rad(4.0)), 0.0, np.cos(np.deg2rad(4.0))],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [np.cos(np.deg2rad(5.0)), np.sin(np.deg2rad(5.0)), 0.0],
                [np.cos(np.deg2rad(4.0)), -np.sin(np.deg2rad(4.0)), 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, np.cos(np.deg2rad(5.0)), np.sin(np.deg2rad(5.0))],
                [0.0, np.cos(np.deg2rad(4.0)), -np.sin(np.deg2rad(4.0))],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        normals = np.asarray(
            [value if index % 2 else -value for index, value in enumerate(normals)],
            dtype=float,
        )
        indices = np.array([8, 2, 10, 0, 7, 4, 11, 1, 5, 3, 9, 6], dtype=np.int64)
        events: list[tuple[int, int, int]] = []
        groups = split_axial_by_deviation(
            indices,
            normals,
            np.deg2rad(12.0),
            progress=lambda assigned, total, leaves: events.append(
                (assigned, total, leaves)
            ),
        )

        flattened = np.concatenate(groups)
        self.assertEqual(sorted(flattened.tolist()), sorted(indices.tolist()))
        self.assertEqual(len(np.unique(flattened)), len(indices))
        self.assertLess(len(groups), len(indices))
        for group in groups:
            scatter = normals[group].T @ normals[group]
            _, eigenvectors = np.linalg.eigh(scatter)
            center = unit(eigenvectors[:, -1])
            deviations = [
                np.degrees(np.arccos(np.clip(abs(float(np.dot(center, normals[index]))), 0.0, 1.0)))
                for index in group
            ]
            self.assertLessEqual(max(deviations), 12.0 + 1e-8)
        self.assertTrue(events)
        self.assertEqual(events[-1][:2], (len(indices), len(indices)))


if __name__ == "__main__":
    unittest.main()
