import numpy as np

from muon_tomo.constants import (
    INVERSE_RADIATION_LENGTH_CM_INV,
    inverse_to_reference_scattering_density,
    labels_to_inverse_radiation_length,
    reference_scattering_density_to_inverse,
)


def test_reference_scattering_factor_is_25_for_existing_convention():
    rho = np.asarray([0.0, 1.0, 3.0])
    scattering = inverse_to_reference_scattering_density(rho)
    assert np.allclose(scattering, [0.0, 25.0, 75.0])
    assert np.allclose(reference_scattering_density_to_inverse(scattering), rho)


def test_truth_material_mapping():
    labels = np.zeros((64, 64, 64), dtype=np.int32)
    labels[0, 0, 0] = 17
    labels[1, 0, 0] = 19
    labels[2, 0, 0] = 21
    truth = labels_to_inverse_radiation_length(labels)
    assert truth.shape == (64, 64, 64)
    assert truth[3, 0, 0] == 0.0
    assert np.isclose(truth[0, 0, 0], INVERSE_RADIATION_LENGTH_CM_INV[17])
    assert np.isclose(truth[1, 0, 0], INVERSE_RADIATION_LENGTH_CM_INV[19])
    assert np.isclose(truth[2, 0, 0], INVERSE_RADIATION_LENGTH_CM_INV[21])
