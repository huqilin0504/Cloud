"""Compatibility imports for the former processing output module.

New code should import whole-cloud writers from ``rock_discontinuity.io``.
"""

from ..io.las_tiles import merge_overlays
from ..io.whole_cloud_output import (
    PLANE_FIELDS,
    WHOLE_GLOBAL_PLANE_FIELDS,
    WHOLE_JOINT_SET_FIELDS,
    WHOLE_PLANE_FIELDS,
    WHOLE_SPACING_FIELDS,
    write_json_reports,
    write_rows,
    write_whole_cloud_outputs,
)
from .output_records import (
    prepare_whole_cloud_auxiliary_rows,
    summarize_tile_density,
)

__all__ = [
    "PLANE_FIELDS",
    "WHOLE_GLOBAL_PLANE_FIELDS",
    "WHOLE_JOINT_SET_FIELDS",
    "WHOLE_PLANE_FIELDS",
    "WHOLE_SPACING_FIELDS",
    "merge_overlays",
    "prepare_whole_cloud_auxiliary_rows",
    "summarize_tile_density",
    "write_json_reports",
    "write_rows",
    "write_whole_cloud_outputs",
]
