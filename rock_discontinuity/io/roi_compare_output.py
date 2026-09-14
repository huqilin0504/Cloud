"""Serialization helpers for ROI comparison experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_json_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_comparison_summary(
    output_root: Path,
    rows: list[dict[str, Any]],
    input_path: Path,
    *,
    rois: dict[str, dict[str, float]],
    roi_inputs: dict[str, str],
    variants: dict[str, dict[str, bool]],
) -> dict[str, Any]:
    """Write the stable CSV/JSON summary and return its JSON representation."""

    fields = list(rows[0].keys()) if rows else []
    summary_csv = output_root / "roi_comparison.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "suite": "three_roi_multiscale_hierarchical_ablation",
        "input": str(input_path),
        "rois": rois,
        "roi_inputs": roi_inputs,
        "variants": variants,
        "rows": rows,
        "summary_csv": str(summary_csv),
    }
    (output_root / "roi_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def write_experiment_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "read_csv_rows",
    "read_json_report",
    "write_comparison_summary",
    "write_experiment_report",
]
