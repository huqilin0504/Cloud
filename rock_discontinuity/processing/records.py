"""Prepare stable output records from typed processing results."""

from __future__ import annotations

from ..core.models import DetachmentRecord, PlaneInstance, plane_to_row


def prepare_detachment_rows(
    planes: list[PlaneInstance],
    detachment_rows: list[DetachmentRecord],
    nearest_spacing: dict[str, float],
) -> list[DetachmentRecord]:
    """Attach stable plane fields and derived spacing before serialization."""

    rows: list[DetachmentRecord] = []
    for index, plane in enumerate(planes):
        row = dict(detachment_rows[index])
        row.update(plane_to_row(plane))
        row.update(
            {
                "nearest_spacing_m": nearest_spacing.get(plane.plane_id),
                "center_x": float(plane.centroid[0]),
                "center_y": float(plane.centroid[1]),
                "center_z": float(plane.centroid[2]),
            }
        )
        rows.append(row)
    return rows
