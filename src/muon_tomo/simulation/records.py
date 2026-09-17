"""Strict simulation CSV input and the existing GZIP x/y TFRecord contract."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..constants import labels_to_inverse_radiation_length
from ..io.tfrecord import _tensorflow

CSV_COLUMNS = (
    "out_x_mm",
    "out_y_mm",
    "out_z_mm",
    "out_px_mev",
    "out_py_mev",
    "out_pz_mev",
    "in_x_mm",
    "in_y_mm",
    "in_z_mm",
    "in_dx",
    "in_dy",
    "in_dz",
    "initial_kinetic_energy_mev",
)


def validate_tracks(values: np.ndarray) -> np.ndarray:
    tracks = np.asarray(values, dtype=np.float64)
    if tracks.ndim != 2 or tracks.shape[1] not in (12, 13) or len(tracks) == 0:
        raise ValueError("Tracks must be a nonempty (N,12) or (N,13) array")
    if not np.isfinite(tracks).all():
        raise ValueError("Track values must be finite")
    if np.any(np.linalg.norm(tracks[:, 3:6], axis=1) <= 0):
        raise ValueError("Outgoing momenta must be nonzero")
    if not np.allclose(np.linalg.norm(tracks[:, 9:12], axis=1), 1, rtol=1e-5, atol=1e-6):
        raise ValueError("Incoming directions must be unit vectors")
    if np.any(tracks[:, 2] >= tracks[:, 8]):
        raise ValueError("Store lower/outgoing endpoint first, upper/incoming endpoint second")
    if np.any(tracks[:, 5] >= 0) or np.any(tracks[:, 11] >= 0):
        raise ValueError("Expected downward incoming and outgoing muon tracks")
    if tracks.shape[1] == 13 and np.any(tracks[:, 12] <= 0):
        raise ValueError("Initial kinetic energies must be positive MeV")
    return tracks


def read_simulation_csv(path: str | Path) -> np.ndarray:
    """Accept our named header or legacy numeric-only CSV without dropping row 1."""
    path = Path(path)
    with path.open(newline="") as handle:
        first = next(csv.reader(handle), None)
    if first is None:
        raise ValueError(f"Empty simulation CSV: {path}")
    skip = int(tuple(first) == CSV_COLUMNS)
    if not skip:
        try:
            [float(token) for token in first]
        except ValueError as exc:
            raise ValueError("CSV header does not match the 13-column simulation schema") from exc
    data = np.loadtxt(path, delimiter=",", skiprows=skip, ndmin=2)
    return validate_tracks(data)


def write_records(path: str | Path, records: Iterable[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """Write one object per Example: x=float64 (N,12/13), y=int32 (64,64,64).

    Write to a temporary sibling, then publish only a completely valid dataset.
    Existing datasets are never overwritten.
    """
    tf = _tensorflow()
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.NamedTemporaryFile(
        dir=target.parent, suffix=".tfrecord.tmp", delete=False
    ) as tmp:
        temporary = Path(tmp.name)
    summaries = []
    try:
        with tf.io.TFRecordWriter(str(temporary), options="GZIP") as writer:
            for tracks, labels in records:
                tracks = validate_tracks(tracks)
                labels_to_inverse_radiation_length(labels)
                labels = np.asarray(labels, dtype=np.int32)
                features = {
                    key: tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
                    )
                    for key, value in (("x", tracks), ("y", labels))
                }
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
                ids, counts = np.unique(labels, return_counts=True)
                summaries.append(
                    {
                        "record_index": len(summaries),
                        "tracks": len(tracks),
                        "x_shape": list(tracks.shape),
                        "y_shape": list(labels.shape),
                        "material_voxels": dict(
                            zip(map(str, ids.tolist()), counts.tolist(), strict=True)
                        ),
                    }
                )
        if not summaries:
            raise ValueError("At least one object record is required")
        # Hard-link creation fails atomically if another process already made target.
        import os

        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return summaries
