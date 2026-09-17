import numpy as np

from muon_tomo.cli import _track_fingerprint
from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction.common import closest_approach
from muon_tomo.reconstruction.munet import MuNetReconstructor
from muon_tomo.types import TrackTable


def _one_track(entry, exit_):
    entry = np.asarray(entry, dtype=float)[None, :]
    exit_ = np.asarray(exit_, dtype=float)[None, :]
    direction = exit_ - entry
    return TrackTable(entry, direction, exit_, direction, np.asarray([3000.0]))


def test_munet_normalization_and_grid_index_are_identical():
    geometry = GridGeometry()
    reconstructor = MuNetReconstructor(geometry, dosage=512)
    tracks = _one_track((-40.0, -20.0, 0.0), (40.0, 20.0, 10.0))
    features = reconstructor._features(tracks)

    model_entry_index = np.floor(features[0, :3] * 64).astype(int)
    model_exit_index = np.floor(features[0, 6:9] * 64).astype(int)

    assert np.array_equal(model_entry_index, geometry.point_to_index(tracks.entry_position_cm[0]))
    assert np.array_equal(model_exit_index, geometry.point_to_index(tracks.exit_position_cm[0]))
    assert np.allclose(geometry.lower / 100.0 + 0.5, 0.0)
    assert np.allclose(geometry.upper / 100.0 + 0.5, 1.0)
    assert features[0, 12] == np.float32((1743.4601815051958 - 5585.2666) / 13839.263)


def test_xyz_axes_increase_without_swap_or_reflection():
    geometry = GridGeometry()
    origin = geometry.point_to_index((0.0, 0.0, 0.0))
    plus_x = geometry.point_to_index((25.0, 0.0, 0.0))
    plus_y = geometry.point_to_index((0.0, 25.0, 0.0))
    plus_z = geometry.point_to_index((0.0, 0.0, 25.0))

    assert np.array_equal(origin, (32, 32, 32))
    assert np.array_equal(plus_x - origin, (16, 0, 0))
    assert np.array_equal(plus_y - origin, (0, 16, 0))
    assert np.array_equal(plus_z - origin, (0, 0, 16))
    assert np.allclose(geometry.index_to_center(origin), (0.78125, 0.78125, 0.78125))


def test_closest_approach_landmark_maps_to_expected_xyz_voxel():
    geometry = GridGeometry()
    landmark = np.asarray((10.0, -20.0, 5.0))
    d0 = np.asarray((0.1, 0.05, 1.0))
    d0 /= np.linalg.norm(d0)
    d1 = np.asarray((-0.05, 0.1, 1.0))
    d1 /= np.linalg.norm(d1)
    point = closest_approach(landmark - 80.0 * d0, d0, landmark + 80.0 * d1, d1)

    assert np.allclose(point, landmark, atol=1e-10)
    assert np.array_equal(geometry.point_to_index(point), (38, 19, 35))


def test_track_fingerprint_proves_exact_ordered_selection(synthetic_tracks):
    first = synthetic_tracks.take_first(64)
    same = synthetic_tracks.take_first(64)
    shorter = synthetic_tracks.take_first(63)
    reverse = np.arange(63, -1, -1)
    reordered = TrackTable(
        first.entry_position_cm[reverse],
        first.entry_direction[reverse],
        first.exit_position_cm[reverse],
        first.exit_direction[reverse],
        first.momentum_mev[reverse],
        first.event_id[reverse],
    )

    assert _track_fingerprint(first) == _track_fingerprint(same)
    assert _track_fingerprint(first) != _track_fingerprint(shorter)
    assert _track_fingerprint(first) != _track_fingerprint(reordered)
    assert np.array_equal(first.event_id, synthetic_tracks.event_id[:64])
