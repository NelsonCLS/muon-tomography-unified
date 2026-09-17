import numpy as np
import pytest

from muon_tomo.physics import ExternalSpectrumPrior, FixedEffectiveMomentum


def test_inverse_square_prior_matches_population_variance():
    p = np.array([1000.0, 3000.0])
    effective = np.mean(p**-2) ** -0.5
    prior = FixedEffectiveMomentum(effective, "synthetic control")
    np.testing.assert_allclose(prior.inverse_square_coefficient(), np.mean((15 / p) ** 2))
    assert prior.metadata()["moment_definition"] == "<p^-2>^-1/2"
    assert not np.isclose(prior.inverse_square_coefficient(), (15 / p.mean()) ** 2)


def test_external_prior_is_finite_and_direction_conditioned():
    prior = ExternalSpectrumPrior()
    coeff = prior.inverse_square_coefficient(np.array([1.0, 0.5]))
    assert coeff.shape == (2,)
    assert np.isfinite(coeff).all() and (coeff > 0).all()
    assert coeff[0] != coeff[1]
    with pytest.raises(ValueError):
        FixedEffectiveMomentum(0)


def test_external_prior_scalar_and_invalid_angles():
    prior = ExternalSpectrumPrior()
    assert isinstance(prior.inverse_square_coefficient(1.0), float)
    assert prior.inverse_square_coefficient(1.0) == prior.inverse_square_coefficient([1.0])[0]
    for invalid in [0, -0.1, 1.1, np.nan, np.inf]:
        with pytest.raises(ValueError, match="cos_zenith"):
            prior.inverse_square_coefficient(invalid)
