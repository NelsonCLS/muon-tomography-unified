import numpy as np
import pytest

from muon_tomo.calibration import CalibrationModel, apply_calibration, fit_calibration
from muon_tomo.geometry import GridGeometry
from muon_tomo.types import ReconstructionResult, TruthVolume


def test_affine_calibration_targets_inverse_radiation_length():
    geometry = GridGeometry()
    labels = np.zeros((64, 64, 64), dtype=np.int32)
    labels[:32] = 17
    truth_values = np.zeros_like(labels, dtype=np.float32)
    truth_values[:32] = 2.0
    raw = 0.5 * truth_values + 0.25
    result = ReconstructionResult(
        "poca", raw, raw, np.ones_like(labels), np.ones_like(labels, dtype=bool), geometry
    )
    truth = TruthVolume(labels, truth_values, geometry)
    model, rows = fit_calibration("poca", [(result, truth)], erosion_voxels=0)
    calibrated = apply_calibration(result, model)
    assert len(rows) == 2
    assert np.allclose(calibrated.inverse_radiation_length_cm_inv, truth_values, atol=1e-5)
    assert calibrated.metadata["calibration"]["target"] == "inverse_radiation_length_cm_inv"


def test_negative_physical_calibration_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        CalibrationModel("likelihood", slope=-1.0, intercept=1.0)


def test_likelihood_inverted_fit_is_rejected():
    geometry = GridGeometry()
    labels = np.zeros((64, 64, 64), dtype=np.int32)
    labels[:32] = 17
    truth_values = np.zeros_like(labels, dtype=np.float32)
    truth_values[:32] = 2.0
    raw = 1.0 - 0.25 * truth_values
    result = ReconstructionResult(
        "likelihood", raw, raw, np.ones_like(labels), np.ones_like(labels, dtype=bool), geometry
    )
    truth = TruthVolume(labels, truth_values, geometry)

    with pytest.raises(ValueError, match="negative slope"):
        fit_calibration("likelihood", [(result, truth)], erosion_voxels=0)


def test_likelihood_positive_affine_fit_is_applied():
    geometry = GridGeometry()
    labels = np.zeros((64, 64, 64), dtype=np.int32)
    labels[:32] = 17
    truth_values = np.zeros_like(labels, dtype=np.float32)
    truth_values[:32] = 2.0
    raw = 0.25 * truth_values + 0.1
    result = ReconstructionResult(
        "likelihood", raw, raw, np.ones_like(labels), np.ones_like(labels, dtype=bool), geometry
    )
    truth = TruthVolume(labels, truth_values, geometry)

    model, _ = fit_calibration("likelihood", [(result, truth)], erosion_voxels=0)

    assert np.isclose(model.slope, 4.0)
    assert np.isclose(model.intercept, -0.4)
    assert model.fit_statistic == "raw_mean"
    assert model.maximum_cm_inv == 2.0
    assert model.apply(np.asarray([100.0]))[0] == 2.0


@pytest.mark.parametrize("method", ["poca", "likelihood"])
def test_sparse_calibration_preserves_zero_for_invalid_voxels(method):
    geometry = GridGeometry()
    raw = np.zeros((64, 64, 64), dtype=np.float32)
    counts = np.zeros_like(raw, dtype=np.int64)
    raw[1, 2, 3] = 2.0
    counts[1, 2, 3] = 1
    result = ReconstructionResult(
        method, raw, raw, counts, counts > 0, geometry
    )
    model = CalibrationModel(method, slope=2.0, intercept=0.5)

    calibrated = apply_calibration(result, model)

    assert calibrated.inverse_radiation_length_cm_inv[0, 0, 0] == 0.0
    assert calibrated.inverse_radiation_length_cm_inv[1, 2, 3] == 4.5
    assert calibrated.inverse_radiation_length_cm_inv[2, 2, 3] == 0.75
