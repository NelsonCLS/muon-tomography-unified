"""Momentum-normalized point-of-closest-approach reconstruction."""

from __future__ import annotations

import numpy as np

from ..constants import HIGHLAND_CONSTANT_MEV
from ..geometry import GridGeometry
from ..postprocess import fill_empty_with_six_neighbor_mean
from ..types import ReconstructionResult, TrackTable
from .common import closest_approach


class PoCAReconstructor:
    """Localize each event's scattering estimate at its PoCA voxel."""

    def __init__(
        self,
        geometry: GridGeometry | None = None,
        *,
        highland_constant_mev: float = HIGHLAND_CONSTANT_MEV,
        minimum_coverage: int = 1,
    ) -> None:
        self.geometry = geometry or GridGeometry()
        self.highland_constant_mev = float(highland_constant_mev)
        self.minimum_coverage = int(minimum_coverage)

    def reconstruct(self, tracks: TrackTable) -> ReconstructionResult:
        geometry = self.geometry
        sums = np.zeros(geometry.n_voxels, dtype=np.float64)
        sums_sq = np.zeros(geometry.n_voxels, dtype=np.float64)
        counts = np.zeros(geometry.n_voxels, dtype=np.int64)
        dtx, dty = tracks.projected_scattering_angles()
        accepted = 0
        for i in range(len(tracks)):
            point = closest_approach(
                tracks.entry_position_cm[i], tracks.entry_direction[i],
                tracks.exit_position_cm[i], tracks.exit_direction[i],
            )
            if point is None or not bool(geometry.contains(point)):
                continue
            indices, lengths = geometry.trace_segment(
                tracks.entry_position_cm[i], tracks.exit_position_cm[i]
            )
            if not len(indices):
                continue
            poca_index = geometry.point_to_index(point)
            match = np.flatnonzero(np.all(indices == poca_index, axis=1))
            path_cm = float(lengths[match[0]]) if len(match) else float(np.mean(geometry.voxel_size_cm))
            theta2 = 0.5 * (dtx[i] ** 2 + dty[i] ** 2)
            rho_event = theta2 * (tracks.momentum_mev[i] / self.highland_constant_mev) ** 2 / max(path_cm, 1e-9)
            flat = int(geometry.flat_index(poca_index[None, :])[0])
            sums[flat] += rho_event
            sums_sq[flat] += rho_event * rho_event
            counts[flat] += 1
            accepted += 1

        raw = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        variance = np.divide(sums_sq, counts, out=np.zeros_like(sums_sq), where=counts > 0) - raw * raw
        uncertainty = np.sqrt(np.maximum(variance, 0.0) / np.maximum(counts, 1))
        raw = raw.reshape(geometry.shape)
        count_volume = counts.reshape(geometry.shape)
        valid = count_volume >= self.minimum_coverage
        prediction = fill_empty_with_six_neighbor_mean(raw, count_volume == 0)
        return ReconstructionResult(
            method="poca",
            inverse_radiation_length_cm_inv=np.clip(prediction, 0.0, None),
            raw_estimator=raw,
            occupancy_count=count_volume,
            valid_mask=valid,
            uncertainty=uncertainty.reshape(geometry.shape),
            geometry=geometry,
            metadata={
                "algorithm": "momentum-normalized PoCA",
                "raw_estimator_unit": "cm^-1",
                "highland_constant_mev": self.highland_constant_mev,
                "tracks_input": len(tracks),
                "tracks_with_valid_poca": accepted,
                "occupancy_definition": "events localized to the PoCA voxel",
                "empty_voxel_fill": "single-pass mean of in-bounds 6-face neighbors",
            },
        )
