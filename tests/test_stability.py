import numpy as np

from muon_tomo.stability import perturb_muon_columns


def test_detector_errors_preserve_z_momentum_and_direction_norms():
    rng = np.random.default_rng(12)
    columns = np.zeros((20000, 13), dtype=np.float64)
    columns[:, 0:3] = rng.normal(size=(len(columns), 3)) * 1000.0
    columns[:, 6:9] = rng.normal(size=(len(columns), 3)) * 1000.0
    entry = rng.normal(size=(len(columns), 3))
    entry /= np.linalg.norm(entry, axis=1, keepdims=True)
    columns[:, 3:6] = entry * rng.uniform(500.0, 5000.0, size=(len(columns), 1))
    exit_ = rng.normal(size=(len(columns), 3))
    exit_ /= np.linalg.norm(exit_, axis=1, keepdims=True)
    columns[:, 9:12] = exit_

    errored, report = perturb_muon_columns(columns, seed=99)

    assert np.array_equal(errored[:, 2], columns[:, 2])
    assert np.array_equal(errored[:, 8], columns[:, 8])
    np.testing.assert_allclose(
        np.linalg.norm(errored[:, 3:6], axis=1),
        np.linalg.norm(columns[:, 3:6], axis=1),
    )
    np.testing.assert_allclose(np.linalg.norm(errored[:, 9:12], axis=1), 1.0)
    assert abs(report["entry_x_error_std_cm"] - 0.29) < 0.01
    assert abs(report["exit_y_error_std_cm"] - 0.29) < 0.01
    assert abs(report["entry_angular_rms_deg"] - 0.5) < 0.01
    assert abs(report["exit_angular_rms_deg"] - 0.5) < 0.01


def test_detector_error_realization_is_fixed_by_seed():
    columns = np.zeros((20, 13), dtype=np.float64)
    columns[:, 3:6] = [0.0, 0.0, -1000.0]
    columns[:, 9:12] = [0.0, 0.0, -1.0]

    first, _ = perturb_muon_columns(columns, seed=7)
    second, _ = perturb_muon_columns(columns, seed=7)
    changed, _ = perturb_muon_columns(columns, seed=8)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, changed)
