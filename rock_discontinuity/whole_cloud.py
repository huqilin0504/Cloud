"""Backward-compatible command wrapper for whole-cloud processing."""

from .processing.whole_cloud import *  # noqa: F401,F403
from .app.whole_cloud import build_parser, main


if __name__ == "__main__":
    raise SystemExit(main())
