import numpy as np
import pytest

from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction.common import (
    physical_grid_intersections,
    prepare_poca_paths,
)
from muon_tomo.types import TrackTable


def test_only_64_cubed_is_allowed():
    with pytest.raises(ValueError, match="64"):
        GridGeometry(shape=(32, 32, 32))


def test_only_frozen_munet_physical_bounds_are_allowed():
    with pytest.raises(ValueError, match="canonical bounds"):
        GridGeometry(lower_cm=(-100.0, -100.0, -100.0), upper_cm=(100.0, 100.0, 100.0))


def test_trace_segment_conserves_clipped_length():
    geometry = GridGeometry()
    start = np.array([-49.0, 0.0, 49.0])
    end = np.array([49.0, 0.0, -49.0])
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
    assert metadata["lower_cm"] == [-50.0, -50.0, -50.0]
    assert metadata["upper_cm"] == [50.0, 50.0, 50.0]
    assert metadata["voxel_size_cm"] == [1.5625, 1.5625, 1.5625]


def test_tfrecord_order_is_extrapolated_to_shared_grid_faces():
    geometry = GridGeometry()
    tracks = TrackTable(
        entry_position_cm=np.array([[10.0, 0.0, -100.0]]),
        entry_direction=np.array([[0.1, 0.0, -1.0]]),
        exit_position_cm=np.array([[0.0, 0.0, 100.0]]),
        exit_direction=np.array([[0.0, 0.0, -1.0]]),
        momentum_mev=np.array([3000.0]),
    )

    entry, incoming, exit_, outgoing = physical_grid_intersections(tracks, geometry)

    np.testing.assert_allclose(entry[0], [0.0, 0.0, 50.0])
    np.testing.assert_allclose(exit_[0], [5.0, 0.0, -50.0])
    assert incoming[0, 2] < 0.0
    assert outgoing[0, 2] < 0.0


def test_poca_path_and_covariance_baseline_share_grid_faces():
    geometry = GridGeometry()
    tracks = TrackTable(
        entry_position_cm=np.array([[10.0, 0.0, -100.0]]),
        entry_direction=np.array([[0.1, 0.0, -1.0]]),
        exit_position_cm=np.array([[0.0, 0.0, 100.0]]),
        exit_direction=np.array([[0.0, 0.0, -1.0]]),
        momentum_mev=np.array([3000.0]),
    )

    paths, lengths, occupancy, kinked = prepare_poca_paths(tracks, geometry)

    assert kinked == 1
    assert len(paths[0]) > 0
    assert np.isclose(lengths[0].sum(), 50.0 + np.hypot(50.0, 5.0))
    assert occupancy.sum() == len(paths[0])
