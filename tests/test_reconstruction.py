import numpy as np

from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction import LikelihoodReconstructor, PoCAReconstructor
from muon_tomo.types import TrackTable


def _assert_contract(result):
    assert result.inverse_radiation_length_cm_inv.shape == (64, 64, 64)
    assert result.raw_estimator.shape == (64, 64, 64)
    assert result.occupancy_count.shape == (64, 64, 64)
    assert result.valid_mask.shape == (64, 64, 64)
    assert np.isfinite(result.inverse_radiation_length_cm_inv).all()
    assert (result.inverse_radiation_length_cm_inv >= 0).all()


def test_poca_contract(synthetic_tracks):
    result = PoCAReconstructor(GridGeometry()).reconstruct(synthetic_tracks)
    _assert_contract(result)
    assert result.method == "poca"
    assert result.metadata["occupancy_definition"].startswith("events localized")


def test_poca_normalizes_by_all_path_exposure(monkeypatch, synthetic_tracks):
    import muon_tomo.reconstruction.poca as module

    tracks = synthetic_tracks.subset(2)
    geometry = GridGeometry()
    flat = np.ravel_multi_index((32, 32, 32), geometry.shape)
    monkeypatch.setattr(module, "prepare_poca_paths", lambda *args: (
        [np.array([flat]), np.array([flat])],
        [np.array([2.0]), np.array([6.0])], None, 1,
    ))
    points = iter([np.array([0.1, 0.1, 0.1]), None])
    monkeypatch.setattr(module, "closest_approach", lambda *args: next(points))
    result = PoCAReconstructor(pooling_sigma_cm=0).reconstruct(tracks)
    tx, ty = tracks.projected_scattering_angles()
    expected = 0.5 * (tx[0] ** 2 + ty[0] ** 2) * (3000 / 15) ** 2 / 8
    np.testing.assert_allclose(result.raw_estimator[32:34, 32:34, 32:34], expected)
    assert np.count_nonzero(result.raw_estimator) == 8


def test_poca_pooling_preserves_constant_signal_per_exposure(monkeypatch, synthetic_tracks):
    import muon_tomo.reconstruction.poca as module

    tracks = synthetic_tracks.subset(2)
    geometry = GridGeometry()
    flat = np.ravel_multi_index((32, 32, 32), geometry.shape)
    monkeypatch.setattr(module, "prepare_poca_paths", lambda *args: (
        [np.array([flat])] * 2, [np.array([4.0])] * 2, None, 2,
    ))
    monkeypatch.setattr(module, "closest_approach", lambda *args: np.array([0.1, 0.1, 0.1]))
    result = PoCAReconstructor().reconstruct(tracks)
    tx, ty = tracks.projected_scattering_angles()
    expected = np.sum(0.5 * (tx ** 2 + ty ** 2) * (3000 / 15) ** 2) / 8
    np.testing.assert_allclose(result.raw_estimator[result.valid_mask], expected)


def test_angular_likelihood_contract(synthetic_tracks):
    result = LikelihoodReconstructor(GridGeometry(), iterations=2, mode="angular").reconstruct(
        synthetic_tracks
    )
    _assert_contract(result)
    assert result.method == "likelihood"
    assert len(result.metadata["convergence_history"]) == 2


def test_full_covariance_likelihood_contract(synthetic_tracks):
    result = LikelihoodReconstructor(GridGeometry(), iterations=1, mode="full").reconstruct(
        synthetic_tracks.subset(16)
    )
    _assert_contract(result)
    assert result.metadata["mode"] == "full"


def _with_momenta(tracks, momenta):
    return TrackTable(
        tracks.entry_position_cm,
        tracks.entry_direction,
        tracks.exit_position_cm,
        tracks.exit_direction,
        momenta,
        tracks.event_id,
        tracks.source,
        tracks.metadata,
    )


def test_mean_momentum_poca_ignores_per_muon_momentum(synthetic_tracks):
    momenta = np.linspace(1000.0, 9000.0, len(synthetic_tracks))
    first = _with_momenta(synthetic_tracks, momenta)
    changed = np.linspace(2000.0, 10000.0, len(synthetic_tracks))[::-1]
    permuted = _with_momenta(synthetic_tracks, changed)
    reconstructor = PoCAReconstructor(GridGeometry(), momentum_policy="mean")
    result = reconstructor.reconstruct(first)
    repeated = reconstructor.reconstruct(permuted)
    np.testing.assert_allclose(
        result.inverse_radiation_length_cm_inv,
        repeated.inverse_radiation_length_cm_inv,
    )
    assert result.metadata["momentum_policy"] == "mean"
    assert result.metadata["per_muon_momentum_used"] is False
    assert result.metadata["assumed_momentum_mev"] == 3000.0
    assert result.metadata["average_momentum_source"] == "configured population prior"


def test_mean_momentum_angular_likelihood_ignores_per_muon_momentum(synthetic_tracks):
    momenta = np.linspace(1000.0, 9000.0, len(synthetic_tracks))
    first = _with_momenta(synthetic_tracks, momenta)
    changed = np.linspace(2000.0, 10000.0, len(synthetic_tracks))[::-1]
    permuted = _with_momenta(synthetic_tracks, changed)
    reconstructor = LikelihoodReconstructor(
        GridGeometry(), iterations=2, mode="angular", momentum_policy="mean"
    )
    result = reconstructor.reconstruct(first)
    repeated = reconstructor.reconstruct(permuted)
    np.testing.assert_allclose(
        result.inverse_radiation_length_cm_inv,
        repeated.inverse_radiation_length_cm_inv,
    )
    assert result.metadata["momentum_policy"] == "mean"
    assert result.metadata["per_muon_momentum_used"] is False
    assert result.metadata["assumed_momentum_mev"] == 3000.0


def test_mean_momentum_full_likelihood_ignores_per_muon_momentum(synthetic_tracks):
    tracks = synthetic_tracks.subset(16)
    momenta = np.linspace(1000.0, 9000.0, len(tracks))
    first = _with_momenta(tracks, momenta)
    changed = np.linspace(2000.0, 10000.0, len(tracks))[::-1]
    permuted = _with_momenta(tracks, changed)
    reconstructor = LikelihoodReconstructor(
        GridGeometry(), iterations=1, mode="full", momentum_policy="mean"
    )
    result = reconstructor.reconstruct(first)
    repeated = reconstructor.reconstruct(permuted)
    np.testing.assert_allclose(
        result.inverse_radiation_length_cm_inv,
        repeated.inverse_radiation_length_cm_inv,
    )
