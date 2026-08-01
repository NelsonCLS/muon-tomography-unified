import numpy as np

from muon_tomo.plotting3d import (
    _density_opacity_groups,
    _select_voxels,
    _surface_mask,
    _voxel_cube_mesh,
)


def test_single_voxel_builds_a_physical_cube():
    mesh = _voxel_cube_mesh(
        np.asarray([[0, 0, 0]]),
        np.asarray([1.25]),
        (-50.0, -50.0, -50.0),
        (1.5625, 1.5625, 1.5625),
    )

    assert len(mesh["x"]) == 8
    assert len(mesh["i"]) == 12
    assert np.min(mesh["x"]) == -50.0
    assert np.max(mesh["x"]) == -48.4375
    np.testing.assert_allclose(mesh["intensity"], 1.25)


def test_surface_mask_removes_solid_interior():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True

    surface = _surface_mask(mask)

    assert surface.sum() == 26
    assert not surface[2, 2, 2]


def test_voxel_selection_is_thresholded_bounded_and_deterministic():
    volume = np.zeros((8, 8, 8), dtype=float)
    volume[1:7, 1:7, 1:7] = 0.5

    first, values, candidates = _select_voxels(
        volume, 0.1, 20, surface_only=True, seed=4
    )
    repeated, repeated_values, repeated_candidates = _select_voxels(
        volume, 0.1, 20, surface_only=True, seed=4
    )

    assert candidates == 152
    assert repeated_candidates == candidates
    assert len(first) == 20
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(values, repeated_values)
    np.testing.assert_allclose(values, 0.5)


def test_density_opacity_is_monotonic_and_capped_at_sixty_percent():
    values = np.asarray([0.1, 0.5, 1.5, 3.1583333])

    groups = _density_opacity_groups(values, 3.1583333, 0.6)
    opacities = [opacity for _, opacity in groups]

    assert np.all(np.diff(opacities) > 0.0)
    assert opacities[-1] == 0.6
    assert opacities[0] < opacities[-1]
