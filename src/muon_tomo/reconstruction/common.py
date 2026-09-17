"""Compatibility exports for shared path geometry and momentum policy."""

from ..physics.policies import MOMENTUM_POLICIES, reconstruction_momenta
from .paths import (
    _merge_traced_segments,
    closest_approach,
    downstream_path_gaps,
    path_occupancy,
    physical_grid_intersections,
    physical_track_order,
    prepare_paths,
    prepare_poca_paths,
)

__all__ = [
    "MOMENTUM_POLICIES",
    "_merge_traced_segments",
    "closest_approach",
    "downstream_path_gaps",
    "path_occupancy",
    "physical_grid_intersections",
    "physical_track_order",
    "prepare_paths",
    "prepare_poca_paths",
    "reconstruction_momenta",
]
