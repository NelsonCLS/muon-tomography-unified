"""Versioned post-processing calibration against material truth in 1/X0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_erosion

from .constants import MATERIAL_NAMES
from .postprocess import fill_empty_with_six_neighbor_mean
from .types import ReconstructionResult, TruthVolume


@dataclass(frozen=True)
class CalibrationModel:
    method: str
    slope: float
    intercept: float
    calibration_id: str = "funny-v1"
    target: str = "inverse_radiation_length_cm_inv"

    def __post_init__(self) -> None:
        if not np.isfinite(self.slope) or not np.isfinite(self.intercept):
            raise ValueError("calibration coefficients must be finite")
        if self.slope < 0.0:
            raise ValueError("physical calibration slope must be nonnegative")

    def apply(self, raw: np.ndarray) -> np.ndarray:
        return np.clip(self.slope * np.asarray(raw) + self.intercept, 0.0, None)


def material_statistics(
    result: ReconstructionResult,
    truth: TruthVolume,
    *,
    erosion_voxels: int = 2,
) -> list[dict[str, float | int | str]]:
    if result.geometry != truth.geometry:
        raise ValueError("result and truth geometries differ")
    rows = []
    for label in np.unique(truth.material_labels):
        mask = truth.material_labels == label
        if label != 0 and erosion_voxels:
            eroded = binary_erosion(mask, iterations=erosion_voxels)
            if np.any(eroded):
                mask = eroded
        mask &= result.valid_mask
        if not np.any(mask):
            continue
        raw = result.raw_estimator[mask]
        rows.append(
            {
                "label": int(label),
                "material": MATERIAL_NAMES[int(label)],
                "voxels": int(mask.sum()),
                "truth_cm_inv": float(np.median(truth.inverse_radiation_length_cm_inv[mask])),
                "raw_median": float(np.median(raw)),
                "raw_mean": float(np.mean(raw)),
            }
        )
    return rows


def fit_calibration(
    method: str,
    pairs: Iterable[tuple[ReconstructionResult, TruthVolume]],
    *,
    erosion_voxels: int = 2,
    fit_intercept: bool = True,
    calibration_id: str = "funny-v1",
) -> tuple[CalibrationModel, list[dict[str, float | int | str]]]:
    rows: list[dict[str, float | int | str]] = []
    for result, truth in pairs:
        if result.method != method:
            raise ValueError(f"expected {method} result, got {result.method}")
        rows.extend(material_statistics(result, truth, erosion_voxels=erosion_voxels))
    if len(rows) < 2:
        raise ValueError("at least two covered material regions are needed for calibration")
    x = np.asarray([float(row["raw_median"]) for row in rows])
    y = np.asarray([float(row["truth_cm_inv"]) for row in rows])
    w = np.sqrt(np.asarray([int(row["voxels"]) for row in rows], dtype=float))
    if fit_intercept:
        design = np.column_stack([x, np.ones_like(x)])
        slope, intercept = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)[0]
    else:
        slope = float(np.dot(w * x, w * y) / max(np.dot(w * x, w * x), 1e-20))
        intercept = 0.0
    # Angular MLEM already reports its native estimator in cm^-1. Sparse or
    # weakly sampled material medians can make an unconstrained affine fit
    # negative, which would invert material and air. Preserve the native unit
    # instead of accepting an unphysical response reversal.
    if slope < 0.0:
        if method == "likelihood":
            slope, intercept = 1.0, 0.0
        else:
            raise ValueError(f"{method} calibration produced an unphysical negative slope")
    return CalibrationModel(method, float(slope), float(intercept), calibration_id), rows


def apply_calibration(result: ReconstructionResult, model: CalibrationModel) -> ReconstructionResult:
    if result.method != model.method:
        raise ValueError(f"calibration is for {model.method}, result is {result.method}")
    calibrated = model.apply(result.raw_estimator)
    calibrated = np.where(result.valid_mask, calibrated, 0.0)
    if result.method in {"poca", "likelihood"}:
        calibrated = fill_empty_with_six_neighbor_mean(
            calibrated, result.occupancy_count == 0
        )
    return ReconstructionResult(
        result.method,
        calibrated,
        result.raw_estimator,
        result.occupancy_count,
        result.valid_mask,
        result.geometry,
        {**result.metadata, "calibration": asdict(model)},
        result.uncertainty,
    )


def save_calibrations(path: str | Path, models: Iterable[CalibrationModel], report: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"models": [asdict(x) for x in models], "report": report}, indent=2) + "\n")
    return target


def load_calibrations(path: str | Path) -> dict[str, CalibrationModel]:
    payload = json.loads(Path(path).read_text())
    return {item["method"]: CalibrationModel(**item) for item in payload["models"]}
