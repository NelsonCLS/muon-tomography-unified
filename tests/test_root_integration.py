import os
from pathlib import Path

import numpy as np
import pytest

from muon_tomo.io.root import load_root_tracks


ROOT_SAMPLE = Path(os.environ.get("GEANT4_ROOT_SAMPLE", "/nonexistent/sample.root"))


@pytest.mark.skipif(not ROOT_SAMPLE.is_file(), reason="set GEANT4_ROOT_SAMPLE for integration test")
def test_root_adapter():
    tracks = load_root_tracks(ROOT_SAMPLE, limit=128)
    assert len(tracks) == 128
    assert tracks.metadata["format"] == "muontruck_root_hits"
    assert tracks.momentum_mev.min() > 0
    assert np.allclose(tracks.entry_position_cm[:, 2], -100.0)
    assert np.allclose(tracks.exit_position_cm[:, 2], 100.0)
    assert np.all(tracks.entry_direction[:, 2] < 0.0)
    assert np.all(tracks.exit_direction[:, 2] < 0.0)
    assert tracks.metadata["track_point_order"] == "lower_then_upper_matching_tfrecord"
