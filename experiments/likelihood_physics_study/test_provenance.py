"""Small cache tests: code/config/selection changes must never reuse old fits."""
import json

import numpy as np
import pytest

from muon_tomo.research.provenance import (
    SCHEMA_VERSION, ensure_protocol, file_sha256, input_identity, verify_fit,
)


def inputs(path, ids=(3, 8)):
    arrays = {name: np.ones((2, 3)) for name in (
        "entry_position_cm", "entry_direction", "exit_position_cm", "exit_direction")}
    np.savez(path, **arrays, event_ids=np.asarray(ids))


def test_protocol_refuses_legacy_and_configuration_changes(tmp_path):
    path = tmp_path / "new" / "protocol.json"
    first = {"cache_schema_version": SCHEMA_VERSION, "sigma": 3.125}
    digest = ensure_protocol(path, first)
    assert ensure_protocol(path, first) == digest
    with pytest.raises(ValueError, match="changed"):
        ensure_protocol(path, {**first, "sigma": 6.25})
    path.write_text(json.dumps({"sigma": 3.125}))
    with pytest.raises(ValueError, match="Legacy"):
        ensure_protocol(path, first)


def test_fit_cache_binds_input_selection_and_artifacts(tmp_path):
    detector = tmp_path / "detector_inputs.npz"
    inputs(detector)
    protocol = {"cache_schema_version": SCHEMA_VERSION, "method": "fixed"}
    identity = input_identity(detector, protocol)
    artifact = tmp_path / "full_maps.npz"
    np.savez(artifact, rho=np.ones((2, 2, 2)))
    result = {"cache_identity": identity, "artifact_sha256": {artifact.name: file_sha256(artifact)}}
    (tmp_path / "fit_complete.json").write_text(json.dumps(result))
    assert verify_fit(tmp_path, identity) == result
    inputs(detector, ids=(8, 3))
    with pytest.raises(ValueError, match="does not match"):
        verify_fit(tmp_path, input_identity(detector, protocol))
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact missing or changed"):
        verify_fit(tmp_path, identity)


def test_input_identity_never_fabricates_missing_selection_ids(tmp_path):
    path = tmp_path / "unknown.npz"
    np.savez(path, **{name: np.ones((2, 3)) for name in (
        "entry_position_cm", "entry_direction", "exit_position_cm", "exit_direction")})
    with pytest.raises(ValueError, match="original selection IDs"):
        input_identity(path, {"cache_schema_version": SCHEMA_VERSION})
