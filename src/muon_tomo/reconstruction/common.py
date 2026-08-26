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


def physical_track_order(
    tracks: TrackTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return upper/incoming then lower/outgoing track state.

    Mu-Net TFRecords store the lower/outgoing state first and upper/incoming
    state second, while ROOT and synthetic adapters may use either order.  The
    physical MLSD covariance is directional, so establish temporal order from
    z and orient both fitted directions along the muon's flight direction.
    """
    first_is_upper = tracks.entry_position_cm[:, 2] >= tracks.exit_position_cm[:, 2]
    start = np.where(
        first_is_upper[:, None], tracks.entry_position_cm, tracks.exit_position_cm
    ).copy()
    incoming = np.where(
        first_is_upper[:, None], tracks.entry_direction, tracks.exit_direction
    ).copy()
    end = np.where(
        first_is_upper[:, None], tracks.exit_position_cm, tracks.entry_position_cm
    ).copy()
    outgoing = np.where(
        first_is_upper[:, None], tracks.exit_direction, tracks.entry_direction
    ).copy()
    flight = end - start
    incoming[np.sum(incoming * flight, axis=1) < 0] *= -1.0
    outgoing[np.sum(outgoing * flight, axis=1) < 0] *= -1.0
    return start, incoming, end, outgoing


def physical_grid_intersections(
    tracks: TrackTable, geometry: GridGeometry
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return physical track states at the upper and lower grid faces.

    Detector fits are defined outside the reconstruction volume. MLSD's
    displacement covariance and its W matrices must use the same baseline, so
    extrapolate the incoming and outgoing fits to the two z faces of the grid.
    """
    starts, incoming, ends, outgoing = physical_track_order(tracks)
    upper_z = float(geometry.upper_cm[2])
    lower_z = float(geometry.lower_cm[2])
    incoming_dz = incoming[:, 2]
    outgoing_dz = outgoing[:, 2]
    valid = (np.abs(incoming_dz) > 1.0e-12) & (np.abs(outgoing_dz) > 1.0e-12)
    entry_t = np.divide(
        upper_z - starts[:, 2],
        incoming_dz,
        out=np.full(len(tracks), np.nan),
        where=np.abs(incoming_dz) > 1.0e-12,
    )
    exit_t = np.divide(
        lower_z - ends[:, 2],
        outgoing_dz,
        out=np.full(len(tracks), np.nan),
        where=np.abs(outgoing_dz) > 1.0e-12,
    )
    entries = starts + entry_t[:, None] * incoming
    exits = ends + exit_t[:, None] * outgoing
    valid &= np.isfinite(entries).all(axis=1) & np.isfinite(exits).all(axis=1)
    entries[~valid] = np.nan
    exits[~valid] = np.nan
    return entries, incoming, exits, outgoing


def _merge_traced_segments(
    first_idx: np.ndarray,
    first_lengths: np.ndarray,
    second_idx: np.ndarray,
    second_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(first_idx):
        return second_idx, second_lengths
    if not len(second_idx):
        return first_idx, first_lengths
    if np.array_equal(first_idx[-1], second_idx[0]):
        indices = np.concatenate((first_idx, second_idx[1:]), axis=0)
        lengths = np.concatenate(
            (
                first_lengths[:-1],
                [first_lengths[-1] + second_lengths[0]],
                second_lengths[1:],
            )
        )
        return indices, lengths
    return (
        np.concatenate((first_idx, second_idx), axis=0),
        np.concatenate((first_lengths, second_lengths)),
    )


def prepare_poca_paths(
    tracks: TrackTable, geometry: GridGeometry
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, int]:
    """Trace physical entry-PoCA-exit paths for MLSD covariance weights."""
    starts, incoming, ends, outgoing = physical_grid_intersections(tracks, geometry)
    paths: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    occupancy = np.zeros(geometry.n_voxels, dtype=np.int64)
    kinked = 0
    for start, direction_in, end, direction_out in zip(
        starts, incoming, ends, outgoing, strict=True
    ):
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            paths.append(np.empty(0, dtype=np.int64))
            lengths.append(np.empty(0, dtype=np.float64))
            continue
        point = closest_approach(start, direction_in, end, direction_out)
        if point is not None and bool(geometry.contains(point)):
            idx0, lens0 = geometry.trace_segment(start, point)
            idx1, lens1 = geometry.trace_segment(point, end)
            idx, lens = _merge_traced_segments(idx0, lens0, idx1, lens1)
            kinked += 1
        else:
            idx, lens = geometry.trace_segment(start, end)
        flat = geometry.flat_index(idx) if len(idx) else np.empty(0, dtype=np.int64)
        paths.append(flat)
        lengths.append(lens)
        if len(flat):
            np.add.at(occupancy, flat, 1)
    return paths, lengths, occupancy.reshape(geometry.shape), kinked


def path_occupancy(tracks: TrackTable, geometry: GridGeometry) -> np.ndarray:
    flat = np.zeros(geometry.n_voxels, dtype=np.int64)
    for start, end in zip(tracks.entry_position_cm, tracks.exit_position_cm, strict=True):
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
    for start, end in zip(tracks.entry_position_cm, tracks.exit_position_cm, strict=True):
        idx, lens = geometry.trace_segment(start, end)
        flat = geometry.flat_index(idx) if len(idx) else np.empty(0, dtype=np.int64)
        paths.append(flat)
        lengths.append(lens)
        if len(flat):
            np.add.at(occupancy, flat, 1)
    return paths, lengths, occupancy.reshape(geometry.shape)
