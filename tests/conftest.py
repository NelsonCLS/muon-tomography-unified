import numpy as np
import pytest

from muon_tomo.types import TrackTable


@pytest.fixture
def synthetic_tracks():
    rng = np.random.default_rng(7)
    n = 96
    entry = np.column_stack([rng.uniform(-75, 75, (n, 2)), np.full(n, 100.0)])
    exit_ = np.column_stack([rng.uniform(-75, 75, (n, 2)), np.full(n, -100.0)])
    incoming = exit_ - entry
    incoming /= np.linalg.norm(incoming, axis=1, keepdims=True)
    outgoing = incoming + rng.normal(0.0, 8e-4, incoming.shape)
    outgoing /= np.linalg.norm(outgoing, axis=1, keepdims=True)
    return TrackTable(entry, incoming, exit_, outgoing, np.full(n, 3000.0))
