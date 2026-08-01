import numpy as np

from muon_tomo.compare import compare_to_truth
from muon_tomo.geometry import GridGeometry
from muon_tomo.types import ReconstructionResult, TruthVolume


def test_comparison_always_reports_full_64_cubed_mse():
    geometry = GridGeometry()
    shape = geometry.shape
    prediction = np.zeros(shape, dtype=np.float32)
    truth_values = np.ones(shape, dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    prediction[0, 0, 0] = 1.0
    valid[0, 0, 0] = True
    result = ReconstructionResult(
        "likelihood", prediction, prediction, valid.astype(np.int64), valid, geometry
    )
    truth = TruthVolume(np.zeros(shape, dtype=np.int32), truth_values, geometry)

    metrics = compare_to_truth(result, truth)

    assert metrics["full_grid_voxels"] == 64**3
    assert np.isclose(metrics["mse_full_grid_cm_inv_squared"], (64**3 - 1) / 64**3)
    assert metrics["rmse_cm_inv"] == 0.0
