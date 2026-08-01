"""Read the Mu-Net GZIP TFRecord x/y schema without modifying the source."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..constants import GRID_SHAPE, labels_to_inverse_radiation_length
from ..geometry import GridGeometry
from ..types import TrackTable, TruthVolume


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TFRecord support requires the 'munet' extra (TensorFlow)") from exc
    return tf


def _decode(raw: object) -> tuple[np.ndarray, np.ndarray]:
    tf = _tensorflow()
    spec = {"x": tf.io.FixedLenFeature([], tf.string), "y": tf.io.FixedLenFeature([], tf.string)}
    parsed = tf.io.parse_single_example(raw, spec)
    x = tf.io.parse_tensor(parsed["x"], out_type=tf.double).numpy()
    y = tf.io.parse_tensor(parsed["y"], out_type=tf.int32).numpy()
    return x, y


def load_tfrecord_record(
    path: str | Path,
    record_index: int = 0,
    *,
    geometry: GridGeometry | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> tuple[TrackTable, TruthVolume]:
    """Load one GZIP record. Positions are stored in mm and momenta in MeV."""
    tf = _tensorflow()
    source = str(Path(path).resolve())
    dataset = tf.data.TFRecordDataset(source, compression_type="GZIP").skip(record_index).take(1)
    try:
        raw = next(iter(dataset))
    except StopIteration as exc:
        raise IndexError(f"TFRecord has no record {record_index}") from exc
    x, labels = _decode(raw)
    if x.ndim != 2 or x.shape[1] < 12:
        raise ValueError(f"x must have shape (n,>=12), got {x.shape}")
    if labels.shape != GRID_SHAPE:
        raise ValueError(f"y must have shape {GRID_SHAPE}, got {labels.shape}")
    momentum = np.linalg.norm(x[:, 3:6], axis=1)
    tracks = TrackTable(
        entry_position_cm=x[:, 0:3] / 10.0,
        entry_direction=x[:, 3:6],
        exit_position_cm=x[:, 6:9] / 10.0,
        exit_direction=x[:, 9:12],
        momentum_mev=momentum,
        event_id=np.arange(len(x)),
        source=source,
        metadata={
            "format": "tfrecord_xy_gzip",
            "record_index": record_index,
            "stored_track_columns": int(x.shape[1]),
            "stored_position_unit": "mm",
            "stored_momentum_unit": "MeV",
        },
    ).subset(limit, seed)
    grid = geometry or GridGeometry()
    truth = TruthVolume(
        labels,
        labels_to_inverse_radiation_length(labels),
        grid,
        {"source": source, "record_index": record_index},
    )
    return tracks, truth


def inspect_tfrecord(path: str | Path) -> dict[str, object]:
    tf = _tensorflow()
    records = []
    for i, raw in enumerate(tf.data.TFRecordDataset(str(path), compression_type="GZIP")):
        x, y = _decode(raw)
        labels, counts = np.unique(y, return_counts=True)
        records.append(
            {
                "index": i,
                "tracks": int(len(x)),
                "track_shape": list(x.shape),
                "truth_shape": list(y.shape),
                "material_voxels": {str(int(k)): int(v) for k, v in zip(labels, counts)},
                "momentum_mev_percentiles": np.percentile(np.linalg.norm(x[:, 3:6], axis=1), [0, 1, 50, 99, 100]).tolist(),
            }
        )
    return {"path": str(Path(path).resolve()), "compression": "GZIP", "records": records}
