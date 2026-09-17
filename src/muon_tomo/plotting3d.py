"""Backward-compatible imports for interactive voxel visualization."""

from .visualization.viewer import (
    CUBE_OFFSETS,
    CUBE_TRIANGLES,
    _density_opacity_groups,
    _region_edges,
    _select_voxels,
    _surface_mask,
    _voxel_cube_mesh,
    plot_comparison_3d,
)

__all__ = [
    "CUBE_OFFSETS",
    "CUBE_TRIANGLES",
    "_density_opacity_groups",
    "_region_edges",
    "_select_voxels",
    "_surface_mask",
    "_voxel_cube_mesh",
    "plot_comparison_3d",
]
