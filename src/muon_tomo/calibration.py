"""Versioned post-processing calibration against material truth in 1/X0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_erosion

from .constants import MATERIAL_NAMES
from .types import ReconstructionResult, TruthVolume


@dataclass(frozen=True)
class CalibrationModel:
    method: str
    slope: float
    intercept: float
    calibration_id: str = "funny-v4-robust-mlsd"
    target: str = "inverse_radiation_length_cm_inv"
    fit_statistic: str = "raw_median"
    maximum_cm_inv: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.slope) or not np.isfinite(self.intercept):
            raise ValueError("calibration coefficients must be finite")
        if self.slope < 0.0:
            raise ValueError("physical calibration slope must be nonnegative")
        if self.fit_statistic not in {"raw_median", "raw_mean"}:
            raise ValueError("fit_statistic must be raw_median or raw_mean")
        if self.maximum_cm_inv is not None and (
            not np.isfinite(self.maximum_cm_inv) or self.maximum_cm_inv <= 0.0
        ):
            raise ValueError("maximum_cm_inv must be finite and positive")

    def apply(self, raw: np.ndarray) -> np.ndarray:
        calibrated = np.clip(self.slope * np.asarray(raw) + self.intercept, 0.0, None)
        if self.maximum_cm_inv is not None:
            calibrated = np.minimum(calibrated, self.maximum_cm_inv)
        return calibrated


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
    calibration_id: str = "funny-v4-robust-mlsd",
) -> tuple[CalibrationModel, list[dict[str, float | int | str]]]:
    rows: list[dict[str, float | int | str]] = []
    for result, truth in pairs:
        if result.method != method:
            raise ValueError(f"expected {method} result, got {result.method}")
        rows.extend(material_statistics(result, truth, erosion_voxels=erosion_voxels))
    if len(rows) < 2:
        raise ValueError("at least two covered material regions are needed for calibration")
    fit_statistic = "raw_mean" if method == "likelihood" else "raw_median"
    x = np.asarray([float(row[fit_statistic]) for row in rows])
    y = np.asarray([float(row["truth_cm_inv"]) for row in rows])
    w = np.sqrt(np.asarray([int(row["voxels"]) for row in rows], dtype=float))
    if fit_intercept:
        design = np.column_stack([x, np.ones_like(x)])
        slope, intercept = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)[0]
    else:
        slope = float(np.dot(w * x, w * y) / max(np.dot(w * x, w * x), 1e-20))
        intercept = 0.0
    if slope < 0.0:
        raise ValueError(f"{method} calibration produced an unphysical negative slope")
    return CalibrationModel(
        method,
        float(slope),
        float(intercept),
        calibration_id,
        fit_statistic=fit_statistic,
        maximum_cm_inv=float(np.max(y)),
    ), rows


def apply_calibration(result: ReconstructionResult, model: CalibrationModel) -> ReconstructionResult:
    if result.method != model.method:
        raise ValueError(f"calibration is for {model.method}, result is {result.method}")
    calibrated = model.apply(result.raw_estimator)
    calibrated = np.where(result.valid_mask, calibrated, 0.0)
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
