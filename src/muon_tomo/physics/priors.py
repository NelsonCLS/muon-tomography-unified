"""Population priors for the simplified beta=1 scattering model.

A fixed effective momentum is <p^-2>^-1/2, never an arithmetic mean.
The source kinetic-energy distribution belongs to simulation.SourceConfig.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedEffectiveMomentum:
    effective_momentum_mev_c: float
    provenance: str = "user supplied; origin unspecified"

    def __post_init__(self):
        if not np.isfinite(self.effective_momentum_mev_c) or self.effective_momentum_mev_c <= 0:
            raise ValueError("effective momentum must be finite and positive MeV/c")

    def inverse_square_coefficient(self, cos_zenith=None, scattering_constant_mev=15.0):
        value = (scattering_constant_mev / self.effective_momentum_mev_c) ** 2
        return value if cos_zenith is None else np.full_like(np.asarray(cos_zenith, float), value)

    def metadata(self):
        return {
            "kind": "fixed_effective",
            "moment_definition": "<p^-2>^-1/2",
            "effective_momentum_mev_c": self.effective_momentum_mev_c,
            "provenance": self.provenance,
            "per_track_momentum_used": False,
        }


@dataclass(frozen=True)
class ExternalSpectrumPrior:
    """The existing research Reyna prior, explicitly versioned and opt-in.

    This is an incident spectrum, not an object-conditioned accepted spectrum.
    Numerical quadrature delegates to the preserved research implementation.
    """

    model: str = "reyna2006"

    def __post_init__(self):
        if self.model != "reyna2006":
            raise ValueError("only the preserved reyna2006 external prior is available")

    def inverse_square_coefficient(self, cos_zenith):
        from ..research.detector_only_likelihood.spectrum_method import spectrum

        angles = np.asarray(cos_zenith, dtype=float)
        if not np.isfinite(angles).all() or np.any((angles <= 0) | (angles > 1)):
            raise ValueError("cos_zenith must be finite and within (0, 1]")
        coefficients, log_weights = spectrum(angles.reshape(-1))
        values = np.sum(np.exp(log_weights) * coefficients, axis=1).reshape(angles.shape)
        return float(values) if angles.ndim == 0 else values

    def metadata(self):
        return {
            "kind": "external_spectrum",
            "model": self.model,
            "conditioning": "incoming zenith",
            "per_track_momentum_used": False,
            "support": "1 to 2000/cos(zenith) GeV/c",
            "quadrature_nodes": 64,
            "population": "incident; no object attenuation correction",
        }
