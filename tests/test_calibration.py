import numpy as np

from muon_tomo.calibration import apply_calibration, fit_calibration
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
