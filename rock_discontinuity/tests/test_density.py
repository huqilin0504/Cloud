from __future__ import annotations

import unittest

import numpy as np

from rock_discontinuity.processing.density import assess_density, estimate_roi_density


class DensityTests(unittest.TestCase):
    def test_density_proxy_and_600_band(self):
        values = np.arange(0.0, 2.01, 0.1)
        xx, yy = np.meshgrid(values, values)
        points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
        report = estimate_roi_density(points, sample_size=100)
        self.assertIn("p25_points_m2", report)
        self.assertIn("p75_points_m2", report)
        assessed = assess_density(
            report,
            {
                "target_raw_density_points_m2": 100.0,
                "expected_retention_ratio": 0.70,
                "min_local_density_p10_points_m2": 50.0,
                "min_local_density_median_points_m2": 50.0,
            },
        )
        self.assertEqual(assessed["standard_status"], "pass")
        self.assertGreater(assessed["median_points_m2"], 50.0)

    def test_density_target_not_met_is_explicit(self):
        values = np.arange(0.0, 2.01, 0.2)
        xx, yy = np.meshgrid(values, values)
        points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
        report = estimate_roi_density(points, sample_size=100)
        assessed = assess_density(
            report,
            {
                "target_raw_density_points_m2": 600.0,
                "expected_retention_ratio": 0.70,
                "min_local_density_p10_points_m2": 400.0,
                "min_local_density_median_points_m2": 600.0,
            },
        )
        self.assertEqual(assessed["standard_status"], "DENSITY_TARGET_NOT_MET")


if __name__ == "__main__":
    unittest.main()
