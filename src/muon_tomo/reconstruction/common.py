"""Geometry operations shared by classical and learned methods."""

from __future__ import annotations

import numpy as np

from ..geometry import GridGeometry
from ..types import TrackTable


MOMENTUM_POLICIES = ("mean", "per-track")


def reconstruction_momenta(
    tracks: TrackTable, policy: str, average_momentum_mev: float
) -> tuple[np.ndarray, dict[str, object]]:
    """Return momenta allowed by a declared classical-method information policy.

    ``mean`` deliberately discards all event-by-event momentum information and
    gives every event the same configured population-average prior.  It does
    not derive that scalar from hidden simulation truth.  The ``per-track``
    option is retained only to reproduce earlier studies.
    """
    if policy not in MOMENTUM_POLICIES:
        raise ValueError(
            f"momentum policy must be one of {MOMENTUM_POLICIES}, got {policy!r}"
        )
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
        "average_momentum_source": (
            "configured population prior" if policy == "mean" else None
        ),
    }


def closest_approach(
    p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray, epsilon: float = 1e-12
) -> np.ndarray | None:
    """Midpoint of the shortest segment between two infinite 3-D lines."""
    w0 = p0 - p1
    a = float(np.dot(d0, d0))
    b = float(np.dot(d0, d1))
    c = float(np.dot(d1, d1))
    d = float(np.dot(d0, w0))
    e = float(np.dot(d1, w0))
    denominator = a * c - b * b
    if abs(denominator) < epsilon:
        return None
    s = (b * e - c * d) / denominator
    t = (a * e - b * d) / denominator
    return 0.5 * ((p0 + s * d0) + (p1 + t * d1))


def path_occupancy(tracks: TrackTable, geometry: GridGeometry) -> np.ndarray:
    flat = np.zeros(geometry.n_voxels, dtype=np.int64)
    for start, end in zip(tracks.entry_position_cm, tracks.exit_position_cm):
        indices, _ = geometry.trace_segment(start, end)
        if len(indices):
            np.add.at(flat, geometry.flat_index(indices), 1)
    return flat.reshape(geometry.shape)


def prepare_paths(
    tracks: TrackTable, geometry: GridGeometry
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Trace all chords and return flat ids, lengths, and path occupancy."""
    paths: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    occupancy = np.zeros(geometry.n_voxels, dtype=np.int64)
    for start, end in zip(tracks.entry_position_cm, tracks.exit_position_cm):
        idx, lens = geometry.trace_segment(start, end)
        flat = geometry.flat_index(idx) if len(idx) else np.empty(0, dtype=np.int64)
        paths.append(flat)
        lengths.append(lens)
        if len(flat):
            np.add.at(occupancy, flat, 1)
    return paths, lengths, occupancy.reshape(geometry.shape)
