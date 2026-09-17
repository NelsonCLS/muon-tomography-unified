"""Momentum-normalized point-of-closest-approach reconstruction."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from ..constants import HIGHLAND_CONSTANT_MEV, REFERENCE_MOMENTUM_MEV
from ..geometry import GridGeometry
from ..types import ReconstructionResult, TrackTable
from .common import (
    closest_approach,
    physical_grid_intersections,
    prepare_poca_paths,
    reconstruction_momenta,
)


class PoCAReconstructor:
    """Divide spatially pooled PoCA scattering by pooled track path exposure."""

    def __init__(
        self,
        geometry: GridGeometry | None = None,
        *,
        highland_constant_mev: float = HIGHLAND_CONSTANT_MEV,
        minimum_coverage: int = 1,
        momentum_policy: str = "mean",
        average_momentum_mev: float = REFERENCE_MOMENTUM_MEV,
        coarse_grid_size: int = 32,
        pooling_sigma_cm: float = 6.25,
    ) -> None:
        self.geometry = geometry or GridGeometry()
        self.highland_constant_mev = float(highland_constant_mev)
        self.minimum_coverage = int(minimum_coverage)
        self.momentum_policy = momentum_policy
        self.average_momentum_mev = float(average_momentum_mev)
        if coarse_grid_size not in (32, 64):
            raise ValueError("coarse_grid_size must be 32 or 64")
        self.coarse_grid_size = int(coarse_grid_size)
        if not np.isfinite(pooling_sigma_cm) or pooling_sigma_cm < 0:
            raise ValueError("pooling_sigma_cm must be finite and nonnegative")
        self.pooling_sigma_cm = float(pooling_sigma_cm)

    def reconstruct(self, tracks: TrackTable) -> ReconstructionResult:
        geometry = self.geometry
        factor = geometry.shape[0] // self.coarse_grid_size
        coarse_shape = (self.coarse_grid_size,) * 3
        n_voxels = int(np.prod(coarse_shape))
        sums = np.zeros(n_voxels, dtype=np.float64)
        exposure = np.zeros(n_voxels, dtype=np.float64)
        counts = np.zeros(n_voxels, dtype=np.int64)
        paths, path_lengths, _, _ = prepare_poca_paths(tracks, geometry)
        incoming, incoming_dir, outgoing, outgoing_dir = physical_grid_intersections(tracks, geometry)
        dtx, dty = tracks.projected_scattering_angles()
        momenta_mev, momentum_metadata = reconstruction_momenta(
            tracks, self.momentum_policy, self.average_momentum_mev
        )
        accepted = 0
        for i, (indices, lengths) in enumerate(zip(paths, path_lengths)):
            if not len(indices):
                continue
            coarse_indices = np.asarray(np.unravel_index(indices, geometry.shape)).T // factor
            flat_indices = np.ravel_multi_index(coarse_indices.T, coarse_shape)
            # Include exposure even when the event has no in-volume PoCA.
            np.add.at(exposure, flat_indices, lengths)
            point = closest_approach(incoming[i], incoming_dir[i], outgoing[i], outgoing_dir[i])
            if point is None or not bool(geometry.contains(point)):
                continue
            flat = int(np.ravel_multi_index(geometry.point_to_index(point) // factor, coarse_shape))
            sums[flat] += 0.5 * (dtx[i] ** 2 + dty[i] ** 2) * (
                momenta_mev[i] / self.highland_constant_mev
            ) ** 2
            counts[flat] += 1
            accepted += 1

        numerator = sums.reshape(coarse_shape)
        denominator = exposure.reshape(coarse_shape)
        if self.pooling_sigma_cm:
            sigma = self.pooling_sigma_cm / (np.asarray(geometry.voxel_size_cm) * factor)
            # Pool sufficient statistics, not already-divided voxel estimates.
            numerator = gaussian_filter(numerator, sigma, mode="constant")
            denominator = gaussian_filter(denominator, sigma, mode="constant")
        raw = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        count_volume = counts.reshape(coarse_shape)
        valid = denominator > 0
        if self.minimum_coverage > 1:
            valid &= count_volume >= self.minimum_coverage

        def expand(volume):
            return volume.repeat(factor, axis=0).repeat(factor, axis=1).repeat(factor, axis=2)

        raw, count_volume, valid = map(expand, (raw, count_volume, valid))
        return ReconstructionResult(
            method="poca",
            inverse_radiation_length_cm_inv=raw,
            raw_estimator=raw,
            occupancy_count=count_volume,
            valid_mask=valid,
            uncertainty=None,
            geometry=geometry,
            metadata={
                "algorithm": "path-exposure-normalized PoCA with spatial pooling",
                "raw_estimator_unit": "cm^-1",
                "highland_constant_mev": self.highland_constant_mev,
                "tracks_input": len(tracks),
                "tracks_with_valid_poca": accepted,
                "occupancy_definition": "events localized to the PoCA voxel",
                "empty_voxel_fill": "pooled signal divided by pooled path exposure; zero if no exposure",
                "pooling_sigma_cm": self.pooling_sigma_cm,
                "normalization": "G(sum(0.5*(theta_x^2+theta_y^2)*(p/highland_constant)^2)) / G(total_path_cm)",
                "poca_grid_size": list(coarse_shape),
                "poca_upsampling": f"each coarse voxel repeated {factor}x{factor}x{factor} to canonical 64^3",
                **momentum_metadata,
            },
        )
