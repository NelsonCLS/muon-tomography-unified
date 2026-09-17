import numpy as np
import pytest

from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction.common import (
    physical_grid_intersections,
    prepare_poca_paths,
)
from muon_tomo.types import TrackTable


def test_general_grid_is_independent_of_frozen_model_capabilities():
    from muon_tomo.reconstruction import (
        LikelihoodReconstructor,
        MuNetReconstructor,
        PoCAReconstructor,
    )

    grid = GridGeometry(shape=(16, 32, 48), lower_cm=(-10, -20, -30), upper_cm=(10, 20, 30))
    assert grid.n_voxels == 16 * 32 * 48
    ids, lengths = grid.trace_segment(np.array([0, 0, -40]), np.array([0, 0, 40]))
    assert len(ids) == 48
    assert np.isclose(lengths.sum(), 60)
    for cls in (MuNetReconstructor, PoCAReconstructor, LikelihoodReconstructor):
        with pytest.raises(ValueError, match="canonical"):
            cls(grid)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": (0, 64, 64)},
        {"shape": (32.5, 32, 32)},
        {"lower_cm": (float("nan"), -50, -50)},
        {"upper_cm": (float("inf"), 50, 50)},
    ],
)
def test_invalid_grid_is_rejected(kwargs):
    with pytest.raises(ValueError):
        GridGeometry(**kwargs)


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


def test_side_exit_keeps_downstream_distance_to_observation_plane():
    from muon_tomo.reconstruction.common import downstream_path_gaps

    tracks = TrackTable(
        np.array([[200.0, 0.0, -100.0]]),
        np.array([[2.0, 0.0, -1.0]]),
        np.array([[0.0, 0.0, 100.0]]),
        np.array([[0.0, 0.0, -1.0]]),
        np.array([3000.0]),
    )
    grid = GridGeometry()
    # Kink at origin, leave through x=50 at z=-25, observe at z=-50.
    expected_gap = np.hypot(50.0, 25.0)
    np.testing.assert_allclose(downstream_path_gaps(tracks, grid), [expected_gap])
    _, lengths, _, _ = prepare_poca_paths(tracks, grid)
    np.testing.assert_allclose(lengths[0].sum(), 50.0 + expected_gap)
