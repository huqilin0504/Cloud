from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from rock_discontinuity.config import load_config
from rock_discontinuity.core.geometry import align_plane_equation, normal_angle_deg, normal_angle_matrix
from rock_discontinuity.core.models import (
    ProcessingResult,
    SegmentationResult,
    TileResult,
    spacing_record_from_json,
    tile_plane_record_from_json,
)
from rock_discontinuity.processing.tiling import core_mask, grid_plan
from rock_discontinuity.processing.whole_cloud_contract import (
    footprint_max_points,
    tiling_parameters,
    whole_cloud_workers,
)


class RefactorContractTests(unittest.TestCase):
    def test_yaml_is_the_single_default_source_and_isolated_per_load(self):
        first = load_config(None)
        first["whole_cloud"]["footprint_max_points"] = 1
        second = load_config(None)
        self.assertEqual(second["whole_cloud"]["footprint_max_points"], 2000)
        self.assertIn("whole_cloud", second)

    def test_footprint_limit_uses_yaml_for_missing_field_and_honors_override(self):
        self.assertEqual(footprint_max_points(load_config(None)), 2000)
        self.assertEqual(footprint_max_points({}), 2000)
        self.assertEqual(footprint_max_points({"whole_cloud": {"footprint_max_points": 37}}), 37)

    def test_cli_defaults_are_resolved_from_yaml(self):
        config = load_config(None)
        self.assertEqual(tiling_parameters(config), (25.0, 1.0))
        self.assertEqual(tiling_parameters({}), (25.0, 1.0))
        self.assertEqual(tiling_parameters(config, 4.0, 0.5), (4.0, 0.5))
        self.assertEqual(whole_cloud_workers(config), 2)
        self.assertEqual(whole_cloud_workers({}), 2)

    def test_typed_result_contracts_exist(self):
        self.assertTrue(hasattr(ProcessingResult, "as_legacy_dict"))
        self.assertEqual(list(TileResult({"status": "done"}, [], [])), [{"status": "done"}, [], []])

    def test_recovered_rows_keep_missing_fields_and_reject_non_objects(self):
        tile = tile_plane_record_from_json({"tile_id": "0_0", "plane_id": "J1"})
        spacing = spacing_record_from_json({"spacing_m": None})
        self.assertEqual(tile, {"tile_id": "0_0", "plane_id": "J1"})
        self.assertEqual(spacing, {"spacing_m": None})
        with self.assertRaises(TypeError):
            tile_plane_record_from_json([])  # type: ignore[arg-type]

    def test_legacy_result_wrapper_keeps_array_references(self):
        points = np.zeros((3, 3), dtype=float)
        labels = np.zeros(3, dtype=int)
        features = {"normals": np.zeros((3, 3), dtype=float)}
        result = ProcessingResult(
            original_points=points,
            original_labels=labels,
            filtered_points=points,
            filtered_labels=labels,
            features=features,
            candidate_mask=labels.astype(bool),
            segmentation=SegmentationResult(
                np.empty(0, dtype=int),
                np.full(3, -1, dtype=int),
                np.full(3, -1, dtype=int),
                [],
            ),
            planes=[],
            joint_sets=[],
            spacing_rows=[],
            detachment_labels=np.empty(0, dtype=bool),
            detachment_rows=[],
            rejected=[],
            counts={},
            coordinate_origin=np.zeros(3),
            plane_instances=[],
            plane_instance_filtered_labels=np.empty(0, dtype=int),
            plane_instance_original_labels=np.empty(0, dtype=int),
            plane_merge={},
        )
        legacy = result.as_legacy_dict()
        self.assertIs(legacy["original_points"], points)
        self.assertIs(legacy["filtered_points"], points)
        self.assertIs(legacy["features"], features)

    def test_shared_geometry_handles_reversed_normals(self):
        self.assertAlmostEqual(normal_angle_deg([1, 0, 0], [-1, 0, 0]), 0.0)
        matrix = normal_angle_matrix(__import__("numpy").array([[1, 0, 0], [0, 1, 0]], dtype=float))
        self.assertAlmostEqual(float(matrix[0, 1]), 90.0)
        aligned = align_plane_equation([1, 0, 0], -10, [-1, 0, 0], 10)
        self.assertIsNotNone(aligned)
        assert aligned is not None
        self.assertEqual(aligned[2].tolist(), [1.0, -0.0, -0.0])
        self.assertAlmostEqual(aligned[3], -10.0)

    def test_tiling_boundary_policy_remains_half_open(self):
        source = {"bounds": {"min": [0.1, 0.1, 0], "max": [49.9, 24.9, 1]}}
        plan = grid_plan(source, 25.0, 1.0)
        points = __import__("numpy").array([[25.0, 2.0, 0.0], [24.999999, 2.0, 0.0]])
        left = core_mask(points, {"min_x": 0, "max_x": 25, "min_y": 0, "max_y": 25}, 0, 0, plan)
        right = core_mask(points, {"min_x": 25, "max_x": 50, "min_y": 0, "max_y": 25}, 1, 0, plan)
        self.assertFalse(left[0])
        self.assertTrue(right[0])
        self.assertTrue(left[1])

    @staticmethod
    def _module_name(path: Path) -> str:
        package_root = Path(__file__).parents[1]
        relative = path.relative_to(package_root)
        parts = ("rock_discontinuity", *relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    @classmethod
    def _resolved_imports_from_tree(
        cls,
        tree: ast.AST,
        module_name: str,
        module_paths: set[str],
        *,
        module_level_only: bool = True,
    ):
        """Resolve absolute and relative imports to existing package modules."""

        if module_name.rsplit(".", 1)[-1] == "__init__":
            package_parts = module_name.rsplit(".", 1)[0].split(".")
        else:
            package_parts = module_name.rsplit(".", 1)[0].split(".")
        nodes = tree.body if module_level_only else ast.walk(tree)
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in module_paths:
                        yield alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    base = node.module or ""
                else:
                    base_parts = package_parts[: -(node.level - 1)] if node.level > 1 else package_parts
                    base = ".".join([*base_parts, *(node.module.split(".") if node.module else [])])
                if base in module_paths:
                    yield base
                for alias in node.names:
                    candidate = f"{base}.{alias.name}" if base else alias.name
                    if candidate in module_paths:
                        yield candidate

    @classmethod
    def _resolved_imports(cls, path: Path, module_paths: set[str] | None = None):
        package_root = Path(__file__).parents[1]
        if module_paths is None:
            module_paths = {
                cls._module_name(item)
                for item in package_root.rglob("*.py")
                if "tests" not in item.parts and "__pycache__" not in item.parts
            }
        module_name = cls._module_name(path)
        if path.name == "__init__.py":
            module_name = f"{module_name}.__init__"
        yield from cls._resolved_imports_from_tree(
            ast.parse(path.read_text(encoding="utf-8")),
            module_name,
            module_paths,
        )

    def test_core_and_io_layer_boundaries_include_relative_imports(self):
        package_root = Path(__file__).parents[1]
        for path in (package_root / "core").glob("*.py"):
            imports = set(self._resolved_imports(path))
            self.assertFalse(
                any(
                    name.startswith("rock_discontinuity.processing")
                    or name.startswith("rock_discontinuity.io")
                    or name.startswith("rock_discontinuity.app")
                    for name in imports
                ),
                path.name,
            )
        for path in (package_root / "io").glob("*.py"):
            imports = set(self._resolved_imports(path))
            self.assertFalse(
                any(
                    name.startswith("rock_discontinuity.processing")
                    or name.startswith("rock_discontinuity.app")
                    for name in imports
                ),
                path.name,
            )

    def test_project_import_graph_has_no_cycle(self):
        package_root = Path(__file__).parents[1]
        module_paths = {}
        for path in package_root.rglob("*.py"):
            if "tests" in path.parts or "__pycache__" in path.parts:
                continue
            module = self._module_name(path)
            module_paths[module] = path
        graph = {
            module: {
                imported
                for imported in self._resolved_imports(path, set(module_paths))
                if imported in module_paths
            }
            for module, path in module_paths.items()
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(module: str) -> None:
            if module in visiting:
                self.fail(f"项目内循环导入：{module}")
            if module in visited:
                return
            visiting.add(module)
            for imported in graph[module]:
                visit(imported)
            visiting.remove(module)
            visited.add(module)

        for module in graph:
            visit(module)

    def test_import_resolver_handles_package_modules_and_aliases(self):
        module_paths = {
            "rock_discontinuity.pkg",
            "rock_discontinuity.pkg.child",
            "rock_discontinuity.other",
        }
        imports = set(
            self._resolved_imports_from_tree(
                ast.parse(
                    "from . import child\n"
                    "from ..other import value\n"
                    "def load():\n"
                    "    from . import delayed\n"
                ),
                "rock_discontinuity.pkg.worker",
                module_paths,
            )
        )
        self.assertEqual(
            imports,
            {"rock_discontinuity.pkg", "rock_discontinuity.pkg.child", "rock_discontinuity.other"},
        )
        all_imports = set(
            self._resolved_imports_from_tree(
                ast.parse("def load():\n    from . import delayed\n"),
                "rock_discontinuity.pkg.worker",
                {"rock_discontinuity.pkg", "rock_discontinuity.pkg.delayed"},
                module_level_only=False,
            )
        )
        self.assertIn("rock_discontinuity.pkg.delayed", all_imports)

    def test_io_modules_only_serialize_prepared_records(self):
        io_root = Path(__file__).parents[1] / "io"
        output_source = (io_root / "output.py").read_text(encoding="utf-8")
        whole_source = (io_root / "whole_cloud_output.py").read_text(encoding="utf-8")
        self.assertNotIn("alpha_shape_geometry", output_source)
        self.assertNotIn("project_to_plane", output_source)
        self.assertNotIn("summarize_tile_density", whole_source)


if __name__ == "__main__":
    unittest.main()
