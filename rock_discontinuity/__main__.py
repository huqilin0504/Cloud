"""Historical ROI command entry point.

The repository-level ``python3 main.py`` and installed ``rock-discontinuity``
commands use the unified dispatcher; this module remains ROI-compatible.
"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
