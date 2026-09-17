import json

import numpy as np

from muon_tomo.geometry import GridGeometry
from muon_tomo.types import ReconstructionResult


def test_result_round_trip(tmp_path):
    geometry = GridGeometry()
    rho = np.ones((64, 64, 64), dtype=np.float32)
    result = ReconstructionResult(
        "test",
        rho,
        rho.copy(),
        np.ones_like(rho, dtype=np.int64),
        np.ones_like(rho, dtype=bool),
        geometry,
        {"purpose": "unit-test"},
    )
    path = result.save(tmp_path / "result.npz")
    loaded = ReconstructionResult.load(path)
    assert loaded.geometry == geometry
    assert np.array_equal(loaded.inverse_radiation_length_cm_inv, rho)
    assert np.allclose(loaded.scattering_density_ref_mrad2_per_cm, 25.0)
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        assert metadata["primary_unit"] == "cm^-1"
        assert metadata["geometry"]["shape"] == [64, 64, 64]
