"""Experimental all-method comparison under one external momentum spectrum.

This module does not alter production reconstruction. PoCA and likelihood use
the fixed Reyna spectrum coefficient directly. Mu-Net receives the resulting
direction-conditioned effective momentum as an inference-time feature; that
case is explicitly a model distribution-shift experiment, not retraining.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from muon_tomo.research.detector_only_likelihood.method import DetectorTracks, SETTINGS
from muon_tomo.research.detector_only_likelihood.spectrum_method import spectrum
from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction.common import (
    closest_approach,
    physical_grid_intersections,
    prepare_poca_paths,
)
from muon_tomo.reconstruction.munet import (
    MuNetReconstructor,
    MUNET_TRAINING_MOMENTUM_MEAN_MEV,
    MUNET_TRAINING_MOMENTUM_STD_MEV,
)


def _effective_momentum(tracks: DetectorTracks) -> np.ndarray:
    """Return p_eff(theta)=<1/p²|theta>^-1/2 from the fixed external prior."""
    _, incoming, _, _ = physical_grid_intersections(tracks, GridGeometry())
    coefficient, log_weights = spectrum(np.abs(incoming[:, 2]))
    inverse_p2 = np.sum(np.exp(log_weights) * coefficient, axis=1)
    return 15.0 / np.sqrt(inverse_p2)


def poca_external(tracks: DetectorTracks, *, pooling_sigma_cm: float = 6.25) -> tuple[np.ndarray, dict]:
    """Exposure-normalized PoCA with an external direction-conditioned prior."""
    geometry = GridGeometry()
    factor = 2
    coarse_shape = (32, 32, 32)
    signal = np.zeros(coarse_shape, dtype=float)
    exposure = np.zeros(coarse_shape, dtype=float)
    paths, lengths, _, _ = prepare_poca_paths(tracks, geometry)
    starts, incoming, ends, outgoing = physical_grid_intersections(tracks, geometry)
    dtx, dty = tracks.projected_scattering_angles()
    p_eff = _effective_momentum(tracks)
    for i, (ids, lens) in enumerate(zip(paths, lengths, strict=True)):
        if not len(ids):
            continue
        coarse = np.asarray(np.unravel_index(ids, geometry.shape)).T // factor
        np.add.at(exposure, tuple(coarse.T), lens)
        point = closest_approach(starts[i], incoming[i], ends[i], outgoing[i])
        if point is None or not bool(geometry.contains(point)):
            continue
        index = tuple(geometry.point_to_index(point) // factor)
        signal[index] += 0.5 * (dtx[i] ** 2 + dty[i] ** 2) * (p_eff[i] / 15.0) ** 2
    sigma = pooling_sigma_cm / 3.125
    numerator = gaussian_filter(signal, sigma, mode="constant")
    denominator = gaussian_filter(exposure, sigma, mode="constant")
    coarse = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    result = coarse.repeat(2, 0).repeat(2, 1).repeat(2, 2)
    return result, {
        "algorithm": "external-spectrum exposure-normalized PoCA",
        "unit": "cm^-1",
        "pooling_sigma_cm": pooling_sigma_cm,
        "truth_used": False,
        "measured_momentum_used": False,
        "external_prior": "Reyna 2006 fixed spectrum; see spectrum_method.PRIOR",
    }


class ExternalSpectrumMuNet(MuNetReconstructor):
    """Inference-only Mu-Net adapter with a direction-conditioned p feature."""

    def _features(self, tracks):
        incoming = tracks.entry_direction.astype(np.float32)
        outgoing = tracks.exit_direction.astype(np.float32)
        p_eff = _effective_momentum(tracks)
        momentum_feature = ((p_eff - MUNET_TRAINING_MOMENTUM_MEAN_MEV) / MUNET_TRAINING_MOMENTUM_STD_MEV).astype(np.float32)[:, None]
        return np.concatenate(
            [
                tracks.entry_position_cm.astype(np.float32) / self.position_scale_cm + self.position_offset,
                incoming,
                tracks.exit_position_cm.astype(np.float32) / self.position_scale_cm + self.position_offset,
                outgoing,
                momentum_feature,
                np.linalg.norm(outgoing - incoming, axis=1, keepdims=True).astype(np.float32),
            ], axis=1,
        )

    def reconstruct(self, tracks):
        result = super().reconstruct(tracks)
        result.metadata.update(
            algorithm="frozen Mu-Net inference with external-spectrum direction-conditioned p feature",
            momentum_feature_policy="external population p_eff(theta), inference-time distribution shift",
            truth_used=False,
            measured_momentum_used=False,
            retraining_performed=False,
            caution="weights were not trained with this per-direction feature policy",
        )
        return result


def validate_detector_only(tracks: DetectorTracks) -> None:
    """Fail if a caller accidentally supplies simulation momentum or labels."""
    forbidden = {"momentum_mev", "material_labels", "truth", "labels"}
    if forbidden.intersection(tracks.__dataclass_fields__):
        raise ValueError("experimental adapters accept only four detector arrays")
