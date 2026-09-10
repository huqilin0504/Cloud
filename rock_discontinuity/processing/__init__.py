"""Point-cloud algorithms and orchestration.

The numerical pipeline returns typed domain results.  Whole-cloud tiling and
resume helpers live in :mod:`tiling` and :mod:`state` so the main orchestrator
does not own their low-level policies.
"""
