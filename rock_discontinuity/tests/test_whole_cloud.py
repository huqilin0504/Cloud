from __future__ import annotations

import unittest
import tempfile
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import laspy

from rock_discontinuity.core.models import TileResult
from rock_discontinuity.io.las_tiles import write_red_overlay as _write_red_overlay
from rock_discontinuity.io.las_tiles import merge_overlays as _merge_overlays_io
from rock_discontinuity.processing.global_aggregation import (
    aggregate_global_planes as _aggregate_global_planes,
    apply_global_candidate_gate as _apply_global_candidate_gate,
    global_orientation_sets as _global_orientation_sets,
    merge_plane_rows as _merge_plane_rows,
)
from rock_discontinuity.processing.state import (
    atomic_json as _atomic_json,
    invalid_source_tiles as _invalid_source_tiles,
    load_processing_state as _load_processing_state,
    mark_tile_done as _mark_tile_done,
    mark_tile_error as _mark_tile_error,
    new_processing_state as _new_processing_state,
    quarantine_source_tiles as _quarantine_source_tiles,
)
from rock_discontinuity.processing.tiling import (
    core_bbox as _core_bbox,
    core_mask as _core_mask,
    grid_plan as _grid_plan,
    parse_tile_name as _parse_tile_name,
)
from rock_discontinuity.processing.whole_cloud import _format_progress, run_whole_cloud
from rock_discontinuity.config import load_config


class WholeCloudTilingTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            "bounds": {
                "min": [10.2, 20.4, 0.0],
                "max": [89.9, 99.8, 10.0],
            }
        }
        self.plan = _grid_plan(self.source, 40.0, 1.0)

    def test_grid_uses_stable_origin_and_negative_names(self):
        self.assertEqual(self.plan["origin_x"], 0.0)
        self.assertEqual(self.plan["origin_y"], 0.0)
        self.assertEqual(_parse_tile_name(Path("tile_-2_7.laz")), (-2, 7))

    def test_interior_core_is_half_open_and_has_no_overlap_duplicate(self):
        points = np.array(
            [
                [40.0, 20.0, 1.0],
                [39.999999, 20.0, 1.0],
                [80.0, 20.0, 1.0],
            ]
        )
        left = _core_mask(points, _core_bbox(self.plan, 0, 0), 0, 0, self.plan)
        middle = _core_mask(points, _core_bbox(self.plan, 1, 0), 1, 0, self.plan)
        self.assertFalse(left[0])
        self.assertTrue(middle[0])
        self.assertTrue(left[1])

    def test_incomplete_source_tiles_are_detected_and_quarantined(self):
        with tempfile.TemporaryDirectory(prefix="whole-cloud-stale-") as directory:
            tile_dir = Path(directory) / "source_tiles"
            tile_dir.mkdir()
            stale_tile = tile_dir / "tile_0_0.laz"
            stale_tile.write_bytes(b"truncated laz")
            self.assertEqual(_invalid_source_tiles([stale_tile]), [stale_tile])
            quarantined = _quarantine_source_tiles(tile_dir)
            self.assertFalse(tile_dir.exists())
            self.assertTrue((quarantined / stale_tile.name).is_file())

    def test_empty_las_tile_is_treated_as_invalid(self):
        with tempfile.TemporaryDirectory(prefix="whole-cloud-empty-") as directory:
            tile = Path(directory) / "tile_0_0.laz"
            header = laspy.LasHeader(point_format=3, version="1.2")
            laspy.LasData(header).write(tile)
            self.assertEqual(_invalid_source_tiles([tile]), [tile])

    def test_processing_state_write_is_atomic_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory(prefix="whole-cloud-atomic-") as directory:
            path = Path(directory) / "processing_state.json"
            _atomic_json(path, {"algorithm_version": "test-v1", "tiles": {}})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["tiles"], {})
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_old_processing_state_without_plan_remains_readable(self):
        plan = {
            "tile_size_m": 25.0,
            "overlap_m": 1.0,
            "origin_x": 0.0,
            "origin_y": 0.0,
            "max_ix": 1,
            "max_iy": 1,
            "nominal_columns": 2,
            "nominal_rows": 2,
        }
        with tempfile.TemporaryDirectory(prefix="whole-cloud-old-state-") as directory:
            path = Path(directory) / "processing_state.json"
            path.write_text(json.dumps({"algorithm_version": "test-v1", "tiles": {}}), encoding="utf-8")
            state = _load_processing_state(
                path,
                plan=plan,
                algorithm_version="test-v1",
                resume=True,
            )
            self.assertEqual(state["tiles"], {})

    def test_progress_line_reports_fraction_and_detail(self):
        line = _format_progress("节理候选识别", 5, 10, "完成 tile_0_0")
        self.assertIn("5/10", line)
        self.assertIn("50.0%", line)
        self.assertIn("完成 tile_0_0", line)

    def test_processing_state_preserves_markers_and_rejects_incompatible_plan(self):
        plan = {
            "tile_size_m": 25.0,
            "overlap_m": 1.0,
            "origin_x": 0.0,
            "origin_y": 0.0,
            "max_ix": 1,
            "max_iy": 1,
            "nominal_columns": 2,
            "nominal_rows": 2,
        }
        state = _new_processing_state(plan, "test-v1")
        _mark_tile_done(
            state,
            tile_id="0_0",
            report_name="tile_0_0.json",
            candidate_red_points=12,
        )
        _mark_tile_error(state, tile_id="1_0", error="read failed")
        self.assertEqual(state["tiles"]["0_0"]["candidate_red_points"], 12)
        self.assertEqual(state["tiles"]["1_0"]["status"], "error")
        with tempfile.TemporaryDirectory(prefix="whole-cloud-state-") as directory:
            path = Path(directory) / "processing_state.json"
            path.write_text(json.dumps(state), encoding="utf-8")
            loaded = _load_processing_state(
                path,
                plan=plan,
                algorithm_version="test-v1",
                resume=True,
            )
            self.assertEqual(loaded["tiles"], state["tiles"])
            with self.assertRaises(ValueError):
                _load_processing_state(
                    path,
                    plan={**plan, "tile_size_m": 20.0},
                    algorithm_version="test-v1",
                    resume=True,
                )
            with self.assertRaises(RuntimeError):
                _load_processing_state(
                    path,
                    plan=plan,
                    algorithm_version="test-v2",
                    resume=True,
                )

    def test_adjacent_tile_plane_instances_are_merged(self):
        def make_row(tile_id, plane_id, xmin, xmax, plane_d):
            return {
                "tile_id": tile_id,
                "plane_id": plane_id,
                "global_plane_id": f"{tile_id}:{plane_id}",
                "core_red_points": 100,
                "nx": 0.0,
                "ny": 0.0,
                "nz": 1.0,
                "plane_d": plane_d,
                "center_x": (xmin + xmax) / 2.0,
                "center_y": 0.5,
                "center_z": 10.0,
                "bbox_min_x": xmin,
                "bbox_min_y": 0.0,
                "bbox_min_z": 10.0,
                "bbox_max_x": xmax,
                "bbox_max_y": 1.0,
                "bbox_max_z": 10.0,
                "observed_area_m2": 1.0,
                "major_extent_m": 1.0,
                "minor_extent_m": 1.0,
                "quality_score": 0.8,
                "quality_grade": "B",
                "edge_censored": False,
            }

        rows = [
            make_row("0_0", "J1-001", 0.0, 40.0, 0.0),
            make_row("1_0", "J1-001", 39.0, 80.0, 0.0),
            make_row("2_0", "J1-002", 80.0, 120.0, 1.0),
        ]
        merged, groups, _ = _merge_plane_rows(rows, {"overlap_m": 1.0}, load_config(None))
        self.assertEqual(len(groups), 2)
        self.assertEqual(merged[0]["global_plane_id"], merged[1]["global_plane_id"])
        self.assertNotEqual(merged[1]["global_plane_id"], merged[2]["global_plane_id"])
        set_ids = _global_orientation_sets(groups, load_config(None))
        global_rows = _aggregate_global_planes(groups, set_ids, load_config(None))
        self.assertEqual(len(global_rows), 2)

    def test_global_candidate_gate_uses_aggregated_quality_rows(self):
        rows = [
            {
                "global_plane_id": "GJ-00001",
                "observed_area_m2": 1.0,
                "minor_extent_m": 0.8,
                "boundary_completeness": 0.85,
                "inlier_ratio": 0.95,
                "normal_dispersion_deg": 3.0,
                "confidence": 0.9,
            },
            {
                "global_plane_id": "GJ-00002",
                "observed_area_m2": 0.1,
                "minor_extent_m": 0.8,
                "boundary_completeness": 0.85,
                "inlier_ratio": 0.95,
                "normal_dispersion_deg": 3.0,
                "confidence": 0.9,
            },
        ]
        selected, selection = _apply_global_candidate_gate(
            rows,
            {
                "detachment": {
                    "min_confidence": 0.7,
                    "min_area_m2": 0.25,
                    "min_minor_extent_m": 0.5,
                    "min_boundary_completeness": 0.7,
                    "min_inlier_ratio": 0.8,
                    "max_normal_dispersion_deg": 8.0,
                }
            },
        )
        self.assertEqual(selected, {"GJ-00001"})
        self.assertEqual(selection["GJ-00002"]["status"], "not_selected")

    def test_cross_tile_merge_does_not_bridge_an_incompatible_chain(self):
        def row(tile_id, plane_id, xmin, xmax):
            ring = [[
                [xmin, 0.0, 10.0],
                [xmax, 0.0, 10.0],
                [xmax, 1.0, 10.0],
                [xmin, 1.0, 10.0],
            ]]
            return {
                "tile_id": tile_id,
                "plane_id": plane_id,
                "global_plane_id": f"{tile_id}:{plane_id}",
                "core_red_points": 100,
                "nx": 0.0,
                "ny": 0.0,
                "nz": 1.0,
                "plane_d": -10.0,
                "center_x": (xmin + xmax) / 2.0,
                "center_y": 0.5,
                "center_z": 10.0,
                "bbox_min_x": xmin,
                "bbox_min_y": 0.0,
                "bbox_min_z": 10.0,
                "bbox_max_x": xmax,
                "bbox_max_y": 1.0,
                "bbox_max_z": 10.0,
                "rms_m": 0.001,
                "_footprint_xyz": ring,
            }

        rows = [
            row("0_0", "P1", 0.0, 1.0),
            row("1_0", "P2", 1.1, 2.1),
            row("2_0", "P3", 2.2, 3.2),
        ]
        config = load_config(None)
        config["whole_cloud"]["merge_xy_gap_m"] = 0.15
        _, groups, _ = _merge_plane_rows(rows, {"overlap_m": 0.05}, config)
        self.assertEqual(sorted(len(group) for group in groups.values()), [1, 2])

    def test_cross_tile_merge_uses_bbox_fallback_when_footprint_is_missing(self):
        def row(tile_id, xmin, xmax):
            return {
                "tile_id": tile_id,
                "plane_id": f"{tile_id}:P1",
                "global_plane_id": f"{tile_id}:P1",
                "core_red_points": 100,
                "nx": 0.0,
                "ny": 0.0,
                "nz": 1.0,
                "plane_d": -2205.0,
                "center_x": (xmin + xmax) / 2.0,
                "center_y": 3132565.0,
                "center_z": 2205.0,
                "bbox_min_x": xmin,
                "bbox_min_y": 3132560.0,
                "bbox_min_z": 2205.0,
                "bbox_max_x": xmax,
                "bbox_max_y": 3132570.0,
                "bbox_max_z": 2205.0,
                "rms_m": 0.001,
            }

        config = load_config(None)
        config["whole_cloud"].update(
            {"merge_xy_gap_m": 0.30, "merge_plane_offset_m": 0.08}
        )
        merged, groups, _ = _merge_plane_rows(
            [row("0_0", 525000.0, 525005.0), row("1_0", 525005.2, 525010.0)],
            {"overlap_m": 0.05},
            config,
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(merged[0]["global_plane_id"], merged[1]["global_plane_id"])

    def test_final_overlay_keeps_only_globally_selected_plane_indices(self):
        source = {
            "scale": [0.01, 0.01, 0.01],
            "offset": [0.0, 0.0, 0.0],
        }
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        with tempfile.TemporaryDirectory(prefix="whole-cloud-overlay-") as directory:
            root = Path(directory)
            tile = root / "tile_0_0.laz"
            output = root / "selected.laz"
            _write_red_overlay(
                tile,
                points,
                source,
                plane_indices=np.array([1, 2, 1]),
            )
            count = _merge_overlays_io(
                [tile],
                output,
                source,
                None,
                {"0_0": {2}},
                parse_tile_name=_parse_tile_name,
            )
            self.assertEqual(count, 1)
            with laspy.open(output) as reader:
                selected = reader.read()
            self.assertEqual(len(selected), 1)
            self.assertAlmostEqual(float(selected.x[0]), 1.0)

    def test_pending_tiles_are_processed_in_parallel_threads(self):
        source = {
            "path": "",
            "total_points": 100,
            "point_format": 2,
            "scale": [0.01, 0.01, 0.01],
            "offset": [0.0, 0.0, 0.0],
            "bounds": {"min": [0.0, 0.0, 0.0], "max": [49.0, 24.0, 1.0]},
            "crs": None,
            "is_copc": False,
            "source_format": "LAS/LAZ",
            "dimensions": ["x", "y", "z"],
        }
        with tempfile.TemporaryDirectory(prefix="whole-cloud-workers-") as directory:
            root = Path(directory)
            input_path = root / "source.las"
            input_path.write_bytes(b"mock input")
            source["path"] = str(input_path.resolve())
            output_dir = root / "output"
            source_tile_dir = output_dir / "source_tiles"
            source_tile_dir.mkdir(parents=True)
            tile_paths = [source_tile_dir / "tile_0_0.laz", source_tile_dir / "tile_1_0.laz"]
            for tile_path in tile_paths:
                tile_path.write_bytes(b"mock tile")
            plan = _grid_plan(source, 25.0, 1.0)
            (output_dir / "split_state.json").write_text(
                json.dumps({"input": source, "plan": plan, "tiles": [path.name for path in tile_paths]}),
                encoding="utf-8",
            )

            lock = threading.Lock()
            active = 0
            max_active = 0
            thread_names: set[str] = set()

            def fake_process_tile(tile_path, **kwargs):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                    thread_names.add(threading.current_thread().name)
                time.sleep(0.05)
                with lock:
                    active -= 1
                ix, iy = _parse_tile_name(tile_path)
                return TileResult(
                    {
                        "tile_id": f"{ix}_{iy}",
                        "status": "done",
                        "counts": {
                            "candidate_red_points": 0,
                            "core_points": 0,
                            "input_points": 0,
                        },
                        "density": {"local_surface_density": {"status": "not_assessed"}},
                    },
                    [],
                    [],
                )

            with patch("rock_discontinuity.processing.whole_cloud._source_info", return_value=source), patch(
                "rock_discontinuity.processing.whole_cloud._process_tile", side_effect=fake_process_tile
            ), patch(
                "rock_discontinuity.processing.whole_cloud._merge_overlays", return_value=0
            ), patch("rock_discontinuity.processing.whole_cloud._source_crs", return_value=None):
                report = run_whole_cloud(
                    input_path,
                    output_dir,
                    load_config(None),
                    resume=True,
                    keep_source_tiles=True,
                    workers=2,
                )

            self.assertGreaterEqual(max_active, 2)
            self.assertEqual(len(thread_names), 2)
            self.assertEqual(report["tiling"]["parallel_mode"], "thread_pool")
            self.assertEqual(report["tiling"]["workers"], 2)
            self.assertEqual(report["tiling"]["processed_tiles"], 2)

    def test_failed_tile_is_retried_and_marked_done_on_resume(self):
        source = {
            "path": "",
            "total_points": 10,
            "point_format": 2,
            "scale": [0.01, 0.01, 0.01],
            "offset": [0.0, 0.0, 0.0],
            "bounds": {"min": [0.0, 0.0, 0.0], "max": [24.0, 24.0, 1.0]},
            "crs": None,
            "is_copc": False,
            "source_format": "LAS/LAZ",
            "dimensions": ["x", "y", "z"],
        }
        with tempfile.TemporaryDirectory(prefix="whole-cloud-retry-") as directory:
            root = Path(directory)
            input_path = root / "source.las"
            input_path.write_bytes(b"mock input")
            source["path"] = str(input_path.resolve())
            output_dir = root / "output"
            source_tile_dir = output_dir / "source_tiles"
            source_tile_dir.mkdir(parents=True)
            tile_path = source_tile_dir / "tile_0_0.laz"
            tile_path.write_bytes(b"mock tile")
            plan = _grid_plan(source, 25.0, 1.0)
            (output_dir / "split_state.json").write_text(
                json.dumps({"input": source, "plan": plan, "tiles": [tile_path.name]}),
                encoding="utf-8",
            )
            attempts = {"count": 0}

            def flaky_process_tile(tile_path, **kwargs):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("simulated tile failure")
                ix, iy = _parse_tile_name(tile_path)
                return TileResult(
                    {
                        "algorithm_version": "0.7.0-multiscale-stability-hierarchical-merge",
                        "tile_id": f"{ix}_{iy}",
                        "status": "done",
                        "counts": {
                            "candidate_red_points": 0,
                            "core_points": 0,
                            "input_points": 0,
                        },
                        "density": {"local_surface_density": {"status": "not_assessed"}},
                    },
                    [],
                    [],
                )

            patches = (
                patch("rock_discontinuity.processing.whole_cloud._source_info", return_value=source),
                patch("rock_discontinuity.processing.whole_cloud._process_tile", side_effect=flaky_process_tile),
                patch("rock_discontinuity.processing.whole_cloud._merge_overlays", return_value=0),
                patch("rock_discontinuity.processing.whole_cloud._source_crs", return_value=None),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                first = run_whole_cloud(
                    input_path,
                    output_dir,
                    load_config(None),
                    resume=True,
                    keep_source_tiles=True,
                    workers=1,
                )
                second = run_whole_cloud(
                    input_path,
                    output_dir,
                    load_config(None),
                    resume=True,
                    keep_source_tiles=True,
                    workers=1,
                )

            self.assertEqual(first["tiling"]["failed_tiles"], 1)
            self.assertEqual(second["tiling"]["failed_tiles"], 0)
            self.assertEqual(attempts["count"], 2)
            state = json.loads((output_dir / "processing_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["tiles"]["0_0"]["status"], "done")


if __name__ == "__main__":
    unittest.main()
