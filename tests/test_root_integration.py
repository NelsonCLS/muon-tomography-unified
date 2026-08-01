import os
from pathlib import Path

import pytest

from muon_tomo.io.root import load_root_tracks


ROOT_SAMPLE = Path(os.environ.get("GEANT4_ROOT_SAMPLE", "/nonexistent/sample.root"))


@pytest.mark.skipif(not ROOT_SAMPLE.is_file(), reason="set GEANT4_ROOT_SAMPLE for integration test")
def test_root_adapter():
    tracks = load_root_tracks(ROOT_SAMPLE, limit=128)
    assert len(tracks) == 128
    assert tracks.metadata["format"] == "muontruck_root_hits"
    assert tracks.momentum_mev.min() > 0
