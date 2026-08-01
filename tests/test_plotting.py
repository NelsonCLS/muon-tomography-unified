import numpy as np

from muon_tomo.geometry import GridGeometry
from muon_tomo.plotting import _axis_projection, _z_projection, _z_slab


def test_z_slab_is_a_64_by_64_layer_average():
    volume = np.zeros(GridGeometry().shape, dtype=float)
    for z in range(64):
        volume[:, :, z] = z
    slab, first, last = _z_slab(volume, center_index=32, slab_layers=7)
    assert slab.shape == (64, 64)
    assert (first, last) == (29, 35)
    np.testing.assert_allclose(slab, 32.0)


def test_z_slab_requires_odd_in_bounds_layer_count():
    volume = np.zeros((64, 64, 64), dtype=float)
    for layers in (0, 2):
        try:
            _z_slab(volume, center_index=32, slab_layers=layers)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid layer count was accepted")


def test_full_depth_projection_averages_all_64_layers():
    volume = np.zeros((64, 64, 64), dtype=float)
    for z in range(64):
        volume[:, :, z] = z
    projection, first, last = _z_projection(
        volume, center_index=3, slab_layers=5, full_depth=True
    )
    assert projection.shape == (64, 64)
    assert (first, last) == (0, 63)
    np.testing.assert_allclose(projection, 31.5)


def test_full_depth_projection_supports_x_y_and_z():
    coordinates = np.indices((64, 64, 64), dtype=float)
    volume = coordinates[0] + 10.0 * coordinates[1] + 100.0 * coordinates[2]
    expected = {
        "x": 31.5 + 10.0 * coordinates[1, 0] + 100.0 * coordinates[2, 0],
        "y": coordinates[0, :, 0] + 315.0 + 100.0 * coordinates[2, :, 0],
        "z": coordinates[0, :, :, 0] + 10.0 * coordinates[1, :, :, 0] + 3150.0,
    }
    for axis in ("x", "y", "z"):
        projection, first, last = _axis_projection(
            volume, center_index=32, slab_layers=7, full_depth=True,
            projection_axis=axis,
        )
        assert projection.shape == (64, 64)
        assert (first, last) == (0, 63)
        np.testing.assert_allclose(projection, expected[axis])


def test_projection_axis_is_validated():
    volume = np.zeros((64, 64, 64), dtype=float)
    try:
        _axis_projection(volume, 32, 7, True, "time")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid projection axis was accepted")
