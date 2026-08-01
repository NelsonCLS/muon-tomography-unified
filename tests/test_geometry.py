import numpy as np
import pytest

from muon_tomo.geometry import GridGeometry


def test_only_64_cubed_is_allowed():
    with pytest.raises(ValueError, match="64"):
        GridGeometry(shape=(32, 32, 32))


def test_trace_segment_conserves_clipped_length():
    geometry = GridGeometry()
    start = np.array([-99.0, 0.0, 99.0])
    end = np.array([99.0, 0.0, -99.0])
    indices, lengths = geometry.trace_segment(start, end)
    assert indices.shape == (64, 3)
    assert np.isclose(lengths.sum(), np.linalg.norm(end - start))
    assert np.all(indices >= 0)
    assert np.all(indices < 64)


def test_geometry_metadata_fixes_axis_contract():
    metadata = GridGeometry().metadata()
    assert metadata["shape"] == [64, 64, 64]
    assert metadata["axis_order"] == "xyz"
    assert metadata["array_indexing"] == "volume[x,y,z]"
