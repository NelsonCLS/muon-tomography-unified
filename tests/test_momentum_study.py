"""Scientific invariants for the opt-in momentum benchmark."""

import numpy as np
import pytest

from muon_tomo.reconstruction.munet import MuNetReconstructor
from muon_tomo.reconstruction.poca import PoCAReconstructor
from muon_tomo.research.momentum_study import (
    HISTORICAL_MEV,
    CachedPoCA,
    MomentumPrior,
    assigned_policies,
    incoming_zenith,
    initial_momentum_from_record,
    momentum_features,
    momentum_statistics,
)
from muon_tomo.types import TrackTable


def example_tracks(n=10):
    # Legacy lower first, upper second; distinct incoming/outgoing zeniths.
    low = np.tile([0.0, 0.0, -100.0], (n, 1))
    up = np.tile([0.0, 0.0, 100.0], (n, 1))
    out = np.tile([0.01, 0.02, -1.0], (n, 1))
    inc = np.tile([-0.01, -0.02, -1.0], (n, 1))
    return TrackTable(low, out, up, inc, np.linspace(1000, 10000, n))


def test_reference_independence_and_statistics():
    p = np.array([1000.0, 4000.0])
    s = momentum_statistics(p)
    assert s["percentile10"] == 1300
    assert s["arithmetic"] == 2500
    assert s["geometric"] == pytest.approx(2000)
    assert s["inverse_square"] == pytest.approx(np.sqrt(2 / (1e-6 + 1 / 16e6)))
    reference = MomentumPrior.fit(p, [0, 10], minimum_count=2)
    assert reference.assign("arithmetic", [0, 10, 90], True).tolist() == [2500] * 3
    a, _ = assigned_policies(reference, [0, 10], p, p)
    b, _ = assigned_policies(reference, [0, 10], p * 2, p * 3)
    for policy in a:
        if policy.startswith("ref_") or policy == "historical":
            np.testing.assert_array_equal(a[policy], b[policy])
    assert not np.array_equal(a["object_arithmetic"], b["object_arithmetic"])
    with pytest.raises(ValueError):
        momentum_statistics([0, 1])


def test_physical_zenith_and_feature_isolation():
    tracks = example_tracks()
    tracks.exit_direction[:] = [0, 0, -1]
    np.testing.assert_array_equal(incoming_zenith(tracks), np.zeros(len(tracks)))
    base = MuNetReconstructor()._features(tracks)
    fixed = momentum_features(tracks, np.full(len(tracks), HISTORICAL_MEV))
    np.testing.assert_array_equal(base, fixed)
    varying = momentum_features(tracks, tracks.momentum_mev)
    np.testing.assert_array_equal(np.delete(varying, 12, axis=1), np.delete(base, 12, axis=1))
    assert np.ptp(varying[:, 12]) > 0


def test_poca_cache_production_parity_and_quadratic_scaling():
    tracks = example_tracks()
    cache = CachedPoCA(tracks)
    for policy, p in [
        ("mean", np.full(len(tracks), HISTORICAL_MEV)),
        ("per-track", tracks.momentum_mev),
    ]:
        production = PoCAReconstructor(
            momentum_policy=policy, average_momentum_mev=HISTORICAL_MEV
        ).reconstruct(tracks)
        np.testing.assert_allclose(
            cache.predict(p), production.inverse_radiation_length_cm_inv, rtol=1e-6, atol=1e-8
        )
        np.testing.assert_array_equal(cache.counts, production.occupancy_count)
        np.testing.assert_array_equal(cache.valid_mask, production.valid_mask)
    np.testing.assert_allclose(
        cache.predict(tracks.momentum_mev * 2), cache.predict(tracks.momentum_mev) * 4, rtol=1e-6
    )


def test_initial_momentum_uses_kinetic_energy_not_unit_direction():
    x = np.zeros((2, 13))
    x[:, 9:12] = [0, 0, -1]
    x[:, 12] = [1000, 5000]
    p = initial_momentum_from_record(x)
    np.testing.assert_allclose(p**2, x[:, 12] * (x[:, 12] + 2 * 105.6583755))
    assert np.all(p > x[:, 12])
    assert np.all(p > 1000)
    with pytest.raises(ValueError):
        initial_momentum_from_record(x[:, :12])


def test_missing_initial_energy_omits_only_two_optional_oracles():
    prior = MomentumPrior.fit([1000, 2000], [0, 10])
    policies, _ = assigned_policies(prior, [0, 10], [1000, 2000], None)
    assert len(policies) == 14
    assert "truth_outgoing" in policies
    assert "truth_incoming" not in policies
    assert "truth_endpoint_geometric" not in policies
