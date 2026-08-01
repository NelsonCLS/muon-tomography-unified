"""Sparse maximum-likelihood expectation-maximization reconstruction."""

from __future__ import annotations

import numpy as np

from ..constants import HIGHLAND_CONSTANT_MEV, REFERENCE_MOMENTUM_MEV
from ..geometry import GridGeometry
from ..types import ReconstructionResult, TrackTable
from .common import prepare_paths


class LikelihoodReconstructor:
    """Port of the muonTruck 2x2 covariance MLEM update on sparse ray paths.

    Internal lambda is reference-momentum scattering density in rad^2/cm.
    The public primary volume is always inverse radiation length in cm^-1.
    """

    def __init__(
        self,
        geometry: GridGeometry | None = None,
        *,
        iterations: int = 10,
        reference_momentum_mev: float = REFERENCE_MOMENTUM_MEV,
        highland_constant_mev: float = HIGHLAND_CONSTANT_MEV,
        initial_inverse_radiation_length_cm_inv: float = 0.001,
        minimum_coverage: int = 1,
        epsilon: float = 1e-12,
        mode: str = "full",
    ) -> None:
        self.geometry = geometry or GridGeometry()
        self.iterations = int(iterations)
        self.reference_momentum_mev = float(reference_momentum_mev)
        self.highland_constant_mev = float(highland_constant_mev)
        self.initial_rho = float(initial_inverse_radiation_length_cm_inv)
        self.minimum_coverage = int(minimum_coverage)
        self.epsilon = float(epsilon)
        if mode not in {"full", "angular"}:
            raise ValueError("likelihood mode must be 'full' or 'angular'")
        self.mode = mode

    def _angular_mlem(
        self,
        tracks: TrackTable,
        paths: list[np.ndarray],
        lengths: list[np.ndarray],
    ) -> tuple[np.ndarray, list[float]]:
        rho = np.full(self.geometry.n_voxels, self.initial_rho, dtype=np.float64)
        sensitivity = np.zeros_like(rho)
        for ids, lens in zip(paths, lengths):
            if len(ids):
                np.add.at(sensitivity, ids, lens)
        dtx, dty = tracks.projected_scattering_angles()
        observations = 0.5 * (dtx * dtx + dty * dty) * (
            tracks.momentum_mev / self.highland_constant_mev
        ) ** 2
        history: list[float] = []
        for _ in range(self.iterations):
            numerator = np.zeros_like(rho)
            residual = []
            for obs, ids, lens in zip(observations, paths, lengths):
                if not len(ids):
                    continue
                predicted = max(float(np.dot(rho[ids], lens)), self.epsilon)
                ratio = float(obs) / predicted
                np.add.at(numerator, ids, lens * ratio)
                residual.append(abs(float(obs) - predicted))
            update = np.divide(numerator, sensitivity, out=np.ones_like(rho), where=sensitivity > 0)
            rho = np.clip(rho * update, 0.0, 1.0e3)
            history.append(float(np.mean(residual)) if residual else 0.0)
        return rho, history

    def _full_covariance_mlem(
        self,
        tracks: TrackTable,
        paths: list[np.ndarray],
        lengths: list[np.ndarray],
    ) -> tuple[np.ndarray, list[float]]:
        scale = (self.highland_constant_mev / self.reference_momentum_mev) ** 2
        lambda_ref = np.full(self.geometry.n_voxels, self.initial_rho * scale, dtype=np.float64)
        dtx, dty = tracks.projected_scattering_angles()
        dz = tracks.exit_position_cm[:, 2] - tracks.entry_position_cm[:, 2]
        safe_dz = np.where(np.abs(tracks.entry_direction[:, 2]) > 1e-12, tracks.entry_direction[:, 2], np.nan)
        travel = dz / safe_dz
        predicted_exit = tracks.entry_position_cm + travel[:, None] * tracks.entry_direction
        dx = tracks.exit_position_cm[:, 0] - predicted_exit[:, 0]
        dy = tracks.exit_position_cm[:, 1] - predicted_exit[:, 1]
        p_relative2 = (self.reference_momentum_mev / tracks.momentum_mev) ** 2
        history: list[float] = []

        prepared_w: list[np.ndarray] = []
        for lens in lengths:
            remaining = np.cumsum(lens[::-1])[::-1] - lens if len(lens) else lens
            w = np.empty((len(lens), 2, 2), dtype=np.float64)
            if len(lens):
                w[:, 0, 0] = lens
                w[:, 0, 1] = w[:, 1, 0] = lens * lens / 2.0 + lens * remaining
                w[:, 1, 1] = lens**3 / 3.0 + lens * lens * remaining + lens * remaining**2
            prepared_w.append(w)

        for _ in range(self.iterations):
            accumulated = np.zeros_like(lambda_ref)
            counts = np.zeros(self.geometry.n_voxels, dtype=np.int64)
            changes: list[float] = []
            for i, (ids, w) in enumerate(zip(paths, prepared_w)):
                if not len(ids) or not np.isfinite(dx[i]) or not np.isfinite(dy[i]):
                    continue
                covariance = p_relative2[i] * np.tensordot(lambda_ref[ids], w, axes=(0, 0))
                covariance.flat[::3] += self.epsilon
                try:
                    inverse = np.linalg.inv(covariance)
                except np.linalg.LinAlgError:
                    continue
                m = np.einsum("ab,nbc,cd->nad", inverse, w, inverse)
                obs_x = np.asarray([dtx[i], dx[i]])
                obs_y = np.asarray([dty[i], dy[i]])
                quadratic = 0.5 * (
                    np.einsum("i,nij,j->n", obs_x, m, obs_x)
                    + np.einsum("i,nij,j->n", obs_y, m, obs_y)
                )
                trace = np.trace(np.einsum("ab,nbc->nac", inverse, w), axis1=1, axis2=2)
                old = lambda_ref[ids]
                score = 2.0 * old + (quadratic - trace) * p_relative2[i] * old * old
                np.add.at(accumulated, ids, score)
                np.add.at(counts, ids, 1)
            updated = np.divide(accumulated, 2.0 * counts, out=lambda_ref.copy(), where=counts > 0)
            updated = np.clip(updated, self.epsilon, 1.0)
            changed = counts > 0
            changes.append(float(np.mean(np.abs(updated[changed] - lambda_ref[changed]))) if np.any(changed) else 0.0)
            lambda_ref = updated
            history.append(changes[-1])
        return lambda_ref / scale, history

    def reconstruct(self, tracks: TrackTable) -> ReconstructionResult:
        paths, lengths, occupancy = prepare_paths(tracks, self.geometry)
        if self.mode == "full":
            rho, history = self._full_covariance_mlem(tracks, paths, lengths)
        else:
            rho, history = self._angular_mlem(tracks, paths, lengths)
        volume = rho.reshape(self.geometry.shape)
        valid = occupancy >= self.minimum_coverage
        volume = np.where(valid, volume, 0.0)
        return ReconstructionResult(
            method="likelihood",
            inverse_radiation_length_cm_inv=np.clip(volume, 0.0, None),
            raw_estimator=volume,
            occupancy_count=occupancy,
            valid_mask=valid,
            geometry=self.geometry,
            metadata={
                "algorithm": "muonTruck covariance MLEM" if self.mode == "full" else "angular-path MLEM",
                "mode": self.mode,
                "raw_estimator_unit": "cm^-1",
                "iterations": self.iterations,
                "convergence_history": history,
                "reference_momentum_mev": self.reference_momentum_mev,
                "highland_constant_mev": self.highland_constant_mev,
                "tracks_input": len(tracks),
                "tracks_with_grid_path": int(sum(bool(len(x)) for x in paths)),
                "occupancy_definition": "number of reconstructed chords crossing each voxel",
            },
        )
