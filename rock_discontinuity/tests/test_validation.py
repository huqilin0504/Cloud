from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rock_discontinuity.validation.validate import validate_output


class WholeCloudValidationTests(unittest.TestCase):
    def test_validate_dispatch_accepts_a_completed_empty_whole_cloud_run(self):
        with tempfile.TemporaryDirectory(prefix="whole-cloud-validation-") as directory:
            output = Path(directory)
            for name in (
                "detachment_planes.csv",
                "joint_planes.csv",
                "joint_sets.csv",
                "spacings.csv",
                "tile_spacings.csv",
                "traces.csv",
                "aperture.csv",
            ):
                (output / name).write_text("id\n", encoding="utf-8")
            report = {
                "algorithm_version": "test-v1",
                "counts": {
                    "source_points": 0,
                    "candidate_plane_instances": 0,
                    "global_joint_plane_count": 0,
                    "global_joint_set_count": 0,
                    "spacing_rows": 0,
                    "tile_spacing_rows": 0,
                    "merged_overlay_points": 0,
                },
                "tiling": {"processed_tiles": 0, "failed_tiles": 0},
                "detachment": {"status": "geometry_only_candidate"},
            }
            (output / "whole_cloud_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            (output / "run.json").write_text("{}", encoding="utf-8")
            (output / "data_audit.json").write_text(
                json.dumps({"point_count": 0}), encoding="utf-8"
            )
            (output / "density_report.json").write_text("{}", encoding="utf-8")
            (output / "run_config.yaml").write_text("{}\n", encoding="utf-8")
            (output / "processing_state.json").write_text(
                json.dumps({"algorithm_version": "test-v1", "tiles": {}}),
                encoding="utf-8",
            )
            (output / "split_state.json").write_text("{}", encoding="utf-8")

            result = validate_output(output)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["processed_tiles"], 0)


if __name__ == "__main__":
    unittest.main()
