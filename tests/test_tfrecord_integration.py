import os
from pathlib import Path

import numpy as np
import pytest

from muon_tomo.io.tfrecord import inspect_tfrecord, load_tfrecord_record


FUNNY = Path(os.environ.get("FUNNY_TFRECORD", "/nonexistent/funny.tfrecord"))


@pytest.mark.skipif(not FUNNY.is_file(), reason="set FUNNY_TFRECORD for integration test")
def test_funny_schema_and_truth():
    report = inspect_tfrecord(FUNNY)
    assert len(report["records"]) == 2
    tracks, truth = load_tfrecord_record(FUNNY, 0, limit=1024)
    assert len(tracks) == 1024
    assert truth.material_labels.shape == (64, 64, 64)
    assert set(np.unique(truth.material_labels)) == {0, 17, 19, 21}
    assert np.isfinite(truth.inverse_radiation_length_cm_inv).all()
    assert truth.geometry.lower_cm == (-50.0, -50.0, -50.0)
    assert truth.geometry.upper_cm == (50.0, 50.0, 50.0)
    normalized_lower = truth.geometry.lower / 100.0 + 0.5
    normalized_upper = truth.geometry.upper / 100.0 + 0.5
    assert np.allclose(normalized_lower, 0.0)
    assert np.allclose(normalized_upper, 1.0)
