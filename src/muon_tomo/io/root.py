"""Adapter for the muonTruck Geant4 ROOT hit tree."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..constants import TRACK_PLANE_LOWER_CM, TRACK_PLANE_UPPER_CM
from ..geometry import GridGeometry
from ..types import TrackTable

BRANCHES = (
    "eventNumber", "hitPositionX_truth", "hitPositionY_truth", "hitPositionZ_truth",
    "hitMomentumX_truth", "hitMomentumY_truth", "hitMomentumZ_truth",
)


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    direction = vt[0]
    if direction[2] > 0:
        direction = -direction
    return centroid, direction / np.linalg.norm(direction)


def _plane_intersection(point: np.ndarray, direction: np.ndarray, z_cm: float) -> np.ndarray | None:
    if abs(direction[2]) < 1e-12:
        return None
    return point + ((z_cm - point[2]) / direction[2]) * direction


def load_root_tracks(
    path: str | Path,
    *,
    geometry: GridGeometry | None = None,
    tree_name: str = "hits",
    position_scale_cm_per_stored_unit: float = 0.1,
    momentum_scale_mev_per_stored_unit: float = 1000.0,
    track_plane_lower_cm: float = TRACK_PLANE_LOWER_CM,
    track_plane_upper_cm: float = TRACK_PLANE_UPPER_CM,
    limit: int | None = None,
    seed: int = 0,
) -> TrackTable:
    """Fit upper/lower hit lines and return the common track representation."""
    try:
        import uproot
    except ImportError as exc:
        raise RuntimeError("ROOT support requires uproot") from exc
    grid = geometry or GridGeometry()
    source = str(Path(path).resolve())
    with uproot.open(source) as root_file:
        tree = root_file[tree_name]
        missing = sorted(set(BRANCHES) - set(tree.keys()))
        if missing:
            raise ValueError(f"ROOT tree {tree_name!r} is missing branches: {missing}")
        arr = tree.arrays(BRANCHES, library="np")

    event = np.asarray(arr["eventNumber"]).reshape(-1)
    positions = np.column_stack([arr[f"hitPosition{axis}_truth"] for axis in "XYZ"]).astype(float)
    positions *= position_scale_cm_per_stored_unit
    momenta = np.column_stack([arr[f"hitMomentum{axis}_truth"] for axis in "XYZ"]).astype(float)
    order = np.lexsort((positions[:, 2], event))
    event, positions, momenta = event[order], positions[order], momenta[order]
    starts = np.r_[0, np.flatnonzero(event[1:] != event[:-1]) + 1]
    ends = np.r_[starts[1:], len(event)]

    p0s: list[np.ndarray] = []
    d0s: list[np.ndarray] = []
    p1s: list[np.ndarray] = []
    d1s: list[np.ndarray] = []
    ps: list[float] = []
    ids: list[int] = []
    for start, end in zip(starts, ends):
        hits = positions[start:end]
        upper = hits[hits[:, 2] > 0]
        lower = hits[hits[:, 2] < 0]
        if len(upper) < 2 or len(lower) < 2:
            continue
        upper_point, upper_dir = _fit_line(upper)
        lower_point, lower_dir = _fit_line(lower)
        first = _plane_intersection(lower_point, lower_dir, track_plane_lower_cm)
        second = _plane_intersection(upper_point, upper_dir, track_plane_upper_cm)
        if first is None or second is None:
            continue
        momentum = float(np.linalg.norm(momenta[start])) * momentum_scale_mev_per_stored_unit
        if not np.isfinite(momentum) or momentum <= 0:
            continue
        # Match the frozen Mu-Net TFRecord ordering: lower detector point and
        # direction first, upper detector point and direction second.
        p0s.append(first); d0s.append(lower_dir); p1s.append(second); d1s.append(upper_dir)
        ps.append(momentum); ids.append(int(event[start]))

    tracks = TrackTable(
        np.asarray(p0s), np.asarray(d0s), np.asarray(p1s), np.asarray(d1s),
        np.asarray(ps), np.asarray(ids), source,
        {
            "format": "muontruck_root_hits",
            "tree": tree_name,
            "stored_position_to_cm": position_scale_cm_per_stored_unit,
            "stored_momentum_to_mev": momentum_scale_mev_per_stored_unit,
            "track_plane_lower_cm": track_plane_lower_cm,
            "track_plane_upper_cm": track_plane_upper_cm,
            "track_point_order": "lower_then_upper_matching_tfrecord",
            "events_seen": int(len(starts)),
            "events_reconstructed": int(len(ps)),
        },
    )
    return tracks.subset(limit, seed)
