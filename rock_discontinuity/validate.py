"""Backward-compatible command wrapper for output validation."""

from .validation.validate import *  # noqa: F401,F403
from .validation.validate import main


if __name__ == "__main__":
    raise SystemExit(main())
