import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from muon_tomo.io.tfrecord import load_tfrecord_record
from muon_tomo.reconstruction import MuNetReconstructor


FUNNY = Path(os.environ.get("FUNNY_TFRECORD", "/nonexistent/funny.tfrecord"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_weight_hashes():
    root = Path(__file__).parents[1]
    weights = root / "src" / "muon_tomo" / "_vendor" / "munet" / "weights"
    assert _sha256(weights / "model_512.weights.h5") == "6fc8764ddb899c969329f5c44f649da8872fc9dd669a8a5108485e7954ce67ec"
    assert _sha256(weights / "model_1024.weights.h5") == "d8383322563156697d93dacf1ffaa5caf0588a6c69096462908d9a6aba6e28d5"


@pytest.mark.skipif(not FUNNY.is_file(), reason="set FUNNY_TFRECORD for Mu-Net integration")
def test_frozen_munet_funny_inference():
    tracks, _ = load_tfrecord_record(FUNNY, 0, limit=512, seed=0)
    result = MuNetReconstructor(dosage=512, max_chunks=1).reconstruct(tracks)
    assert result.inverse_radiation_length_cm_inv.shape == (64, 64, 64)
    assert np.isfinite(result.inverse_radiation_length_cm_inv).all()
    assert result.metadata["training_performed"] is False
    assert result.metadata["weights_modified"] is False
