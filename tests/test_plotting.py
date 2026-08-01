import numpy as np

from muon_tomo.geometry import GridGeometry
from muon_tomo.plotting import _z_slab


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
