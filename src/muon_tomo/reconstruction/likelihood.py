"""Sparse maximum-likelihood expectation-maximization reconstruction."""

from __future__ import annotations

import numpy as np

from ..constants import HIGHLAND_CONSTANT_MEV, REFERENCE_MOMENTUM_MEV
from ..geometry import GridGeometry
from ..types import ReconstructionResult, TrackTable
from .common import (
    physical_grid_intersections,
    prepare_paths,
    prepare_poca_paths,
    reconstruction_momenta,
)


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
        initial_inverse_radiation_length_cm_inv: float = 4.0e-5,
        minimum_coverage: int = 1,
        epsilon: float = 1e-12,
        mode: str = "full",
        momentum_policy: str = "mean",
        average_momentum_mev: float = REFERENCE_MOMENTUM_MEV,
        update_statistic: str = "median",
        angular_resolution_rad: float = 1.0e-6,
        position_resolution_cm: float = 1.0e-3,
        maximum_inverse_radiation_length_cm_inv: float = 20.0,
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
        self.momentum_policy = momentum_policy
        self.average_momentum_mev = float(average_momentum_mev)
        if update_statistic not in {"mean", "median"}:
            raise ValueError("update_statistic must be 'mean' or 'median'")
        self.update_statistic = update_statistic
        self.angular_resolution_rad = float(angular_resolution_rad)
        self.position_resolution_cm = float(position_resolution_cm)
        self.maximum_rho = float(maximum_inverse_radiation_length_cm_inv)

    def _angular_mlem(
        self,
        tracks: TrackTable,
        momenta_mev: np.ndarray,
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
            momenta_mev / self.highland_constant_mev
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
        momenta_mev: np.ndarray,
        paths: list[np.ndarray],
        lengths: list[np.ndarray],
    ) -> tuple[np.ndarray, list[float], int]:
        scale = (self.highland_constant_mev / self.reference_momentum_mev) ** 2
        lambda_ref = np.full(self.geometry.n_voxels, self.initial_rho * scale, dtype=np.float64)
        starts, incoming, ends, outgoing = physical_grid_intersections(
            tracks, self.geometry
        )
        theta_x_in = np.arctan(incoming[:, 0] / np.abs(incoming[:, 2]))
        theta_y_in = np.arctan(incoming[:, 1] / np.abs(incoming[:, 2]))
        theta_x_out = np.arctan(outgoing[:, 0] / np.abs(outgoing[:, 2]))
        theta_y_out = np.arctan(outgoing[:, 1] / np.abs(outgoing[:, 2]))
        dtx = theta_x_out - theta_x_in
        dty = theta_y_out - theta_y_in
        dz = ends[:, 2] - starts[:, 2]
        travel = np.divide(
            dz,
            incoming[:, 2],
            out=np.full_like(dz, np.nan),
            where=np.abs(incoming[:, 2]) > 1.0e-12,
        )
        projected = starts + travel[:, None] * incoming
        f_xy = 1.0 + np.tan(theta_x_in) ** 2 + np.tan(theta_y_in) ** 2
        dx = (
            (ends[:, 0] - projected[:, 0])
            * np.cos(theta_x_in)
            * np.sqrt(f_xy)
            * np.cos(dtx + theta_x_in)
            / np.cos(dtx)
        )
        dy = (
            (ends[:, 1] - projected[:, 1])
            * np.cos(theta_y_in)
            * np.sqrt(f_xy)
            * np.cos(dty + theta_y_in)
            / np.cos(dty)
        )
        p_relative2 = (self.reference_momentum_mev / momenta_mev) ** 2
        history: list[float] = []
        clipped_total = 0
        error_covariance = np.diag(
            [self.angular_resolution_rad**2, self.position_resolution_cm**2]
        )

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
            score_ids: list[np.ndarray] = []
            score_values: list[np.ndarray] = []
            for i, (ids, w) in enumerate(zip(paths, prepared_w)):
                if not len(ids) or not np.isfinite(dx[i]) or not np.isfinite(dy[i]):
                    continue
                covariance = (
                    p_relative2[i] * np.tensordot(lambda_ref[ids], w, axes=(0, 0))
                    + error_covariance
                )
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
                finite = np.isfinite(score)
                if np.any(finite):
                    score_ids.append(ids[finite])
                    score_values.append(score[finite])
            if not score_ids:
                history.append(0.0)
                continue
            all_ids = np.concatenate(score_ids)
            all_scores = np.concatenate(score_values)
            counts = np.bincount(all_ids, minlength=self.geometry.n_voxels)
            sums = np.bincount(
                all_ids, weights=all_scores, minlength=self.geometry.n_voxels
            )
            statistic = np.divide(
                sums, counts, out=np.zeros_like(lambda_ref), where=counts > 0
            )
            if self.update_statistic == "median":
                order = np.argsort(all_ids, kind="stable")
                sorted_ids = all_ids[order]
                sorted_scores = all_scores[order]
                unique, starts_at, group_counts = np.unique(
                    sorted_ids, return_index=True, return_counts=True
                )
                for voxel, first, count in zip(unique, starts_at, group_counts):
                    if count >= 3:
                        statistic[voxel] = np.median(
                            sorted_scores[first : first + count]
                        )
            updated = lambda_ref.copy()
            changed = counts > 0
            updated[changed] = 0.5 * statistic[changed]
            upper = self.maximum_rho * scale
            clipped_total += int(np.count_nonzero(updated > upper))
            updated = np.clip(updated, self.epsilon, upper)
            history.append(
                float(np.mean(np.abs(updated[changed] - lambda_ref[changed])))
                if np.any(changed)
                else 0.0
            )
            lambda_ref = updated
        return lambda_ref / scale, history, clipped_total

    def reconstruct(self, tracks: TrackTable) -> ReconstructionResult:
        momenta_mev, momentum_metadata = reconstruction_momenta(
            tracks, self.momentum_policy, self.average_momentum_mev
        )
        if self.mode == "full":
            paths, lengths, occupancy, kinked = prepare_poca_paths(
                tracks, self.geometry
            )
            rho, history, clipped = self._full_covariance_mlem(
                tracks, momenta_mev, paths, lengths
            )
            path_model = "physical entry-PoCA-exit; straight fallback"
        else:
            paths, lengths, occupancy = prepare_paths(tracks, self.geometry)
            rho, history = self._angular_mlem(tracks, momenta_mev, paths, lengths)
            kinked = 0
            clipped = 0
            path_model = "legacy straight chord"
        volume = rho.reshape(self.geometry.shape)
        valid = occupancy >= self.minimum_coverage
        volume = np.where(valid, volume, 0.0)
        prediction = volume
        return ReconstructionResult(
            method="likelihood",
            inverse_radiation_length_cm_inv=np.clip(prediction, 0.0, None),
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
                "update_statistic": self.update_statistic,
                "path_model": path_model,
                "observation_baseline": (
                    "grid faces z=+50 cm to z=-50 cm"
                    if self.mode == "full"
                    else "detector-state angular difference"
                ),
                "tracks_with_kinked_poca_path": kinked,
                "angular_resolution_rad": self.angular_resolution_rad,
                "position_resolution_cm": self.position_resolution_cm,
                "maximum_inverse_radiation_length_cm_inv": self.maximum_rho,
                "upper_clip_applications": clipped,
                "reference_momentum_mev": self.reference_momentum_mev,
                "highland_constant_mev": self.highland_constant_mev,
                "tracks_input": len(tracks),
                "tracks_with_grid_path": int(sum(bool(len(x)) for x in paths)),
                "occupancy_definition": "number of reconstructed chords crossing each voxel",
                "empty_voxel_fill": "none; zero-occupancy voxels remain zero",
                **momentum_metadata,
            },
        )
