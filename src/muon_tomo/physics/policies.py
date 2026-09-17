"""Compatibility policies for historical scalar and simulation-momentum runs.

Normal detector-only workflows pass a declared fixed effective prior. The
per-track branch exists only for explicit simulation diagnostics.
"""

from __future__ import annotations

import numpy as np

from ..types import TrackTable

MOMENTUM_POLICIES = ("mean", "per-track")


def reconstruction_momenta(
    tracks: TrackTable, policy: str, average_momentum_mev: float
) -> tuple[np.ndarray, dict[str, object]]:
    """Return momenta allowed by a declared classical-method information policy.

    ``mean`` deliberately discards all event-by-event momentum information and
    gives every event the same configured inverse-square effective prior.  It does
    not derive that scalar from hidden simulation truth.  The ``per-track``
    option is retained only to reproduce earlier studies.
    """
    if policy not in MOMENTUM_POLICIES:
        raise ValueError(f"momentum policy must be one of {MOMENTUM_POLICIES}, got {policy!r}")
    if not len(tracks):
        raise ValueError("reconstruction requires at least one track")
    average_momentum_mev = float(average_momentum_mev)
    if not np.isfinite(average_momentum_mev) or average_momentum_mev <= 0:
        raise ValueError("average_momentum_mev must be finite and positive")
    if policy == "mean":
        values = np.full(len(tracks), average_momentum_mev, dtype=np.float64)
    else:
        values = tracks.momentum_mev
    return values, {
        "momentum_policy": policy,
        "per_muon_momentum_used": policy == "per-track",
        "assumed_momentum_mev": average_momentum_mev if policy == "mean" else None,
        "average_momentum_source": ("configured population prior" if policy == "mean" else None),
    }
