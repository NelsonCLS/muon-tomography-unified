"""Comparable physical-space metrics for aligned results."""

from __future__ import annotations

import numpy as np

from ..types import ReconstructionResult, TruthVolume


def compare_to_truth(
    result: ReconstructionResult, truth: TruthVolume
) -> dict[str, float | int | str | None]:
    if result.geometry != truth.geometry:
        raise ValueError("result and truth geometries differ")
    full_prediction = result.inverse_radiation_length_cm_inv.astype(float)
    full_target = truth.inverse_radiation_length_cm_inv.astype(float)
    if not np.all(np.isfinite(full_target)):
        raise ValueError("truth contains non-finite values")
    full_error = full_prediction - full_target
    mask = result.valid_mask & np.isfinite(truth.inverse_radiation_length_cm_inv)
    # Full-grid scoring remains defined even when coverage is empty. Covered-only
    # metrics are unavailable, never zero or a NaN that leaks into JSON.
    has_coverage = bool(np.any(mask))
    prediction = result.inverse_radiation_length_cm_inv[mask].astype(float)
    target = truth.inverse_radiation_length_cm_inv[mask].astype(float)
    error = prediction - target
    return {
        "method": result.method,
        "full_grid_voxels": int(full_error.size),
        "mse_full_grid_cm_inv_squared": float(np.mean(full_error * full_error)),
        "valid_voxels": int(mask.sum()),
        "coverage_fraction": float(mask.mean()),
        "mae_cm_inv": float(np.mean(np.abs(error))) if has_coverage else None,
        "rmse_cm_inv": float(np.sqrt(np.mean(error * error))) if has_coverage else None,
        "bias_cm_inv": float(np.mean(error)) if has_coverage else None,
        "prediction_min_cm_inv": float(prediction.min()) if has_coverage else None,
        "prediction_max_cm_inv": float(prediction.max()) if has_coverage else None,
        "truth_min_cm_inv": float(target.min()) if has_coverage else None,
        "truth_max_cm_inv": float(target.max()) if has_coverage else None,
    }
