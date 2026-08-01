import numpy as np

from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction import LikelihoodReconstructor, PoCAReconstructor


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


def test_angular_likelihood_contract(synthetic_tracks):
    result = LikelihoodReconstructor(
        GridGeometry(), iterations=2, mode="angular"
    ).reconstruct(synthetic_tracks)
    _assert_contract(result)
    assert result.method == "likelihood"
    assert len(result.metadata["convergence_history"]) == 2


def test_full_covariance_likelihood_contract(synthetic_tracks):
    result = LikelihoodReconstructor(
        GridGeometry(), iterations=1, mode="full"
    ).reconstruct(synthetic_tracks.subset(16))
    _assert_contract(result)
    assert result.metadata["mode"] == "full"
