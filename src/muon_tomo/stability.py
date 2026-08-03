"""Deterministic detector-error datasets for paired reconstruction tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np


POSITION_SIGMA_CM = 0.29
ANGULAR_SIGMA_DEG = 0.5


def _rotate_directions(
    directions: np.ndarray, angles_rad: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Apply isotropic signed angular errors while retaining unit directions."""
    unit = np.array(directions, dtype=np.float64, copy=True)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    trial = rng.normal(size=unit.shape)
    tangent = trial - np.sum(trial * unit, axis=1, keepdims=True) * unit
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return np.cos(angles_rad)[:, None] * unit + np.sin(angles_rad)[:, None] * tangent


def perturb_muon_columns(
    columns: np.ndarray,
    *,
    seed: int,
    position_sigma_cm: float = POSITION_SIGMA_CM,
    angular_sigma_deg: float = ANGULAR_SIGMA_DEG,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return a fixed perturbed copy of a Mu-Net x tensor in its native units.

    Positions are stored in mm. Columns 3:6 are entry momentum vectors and
    columns 9:12 are normalized exit directions.
    """
    source = np.asarray(columns, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] < 12:
        raise ValueError("columns must have shape (n, >=12)")
    if position_sigma_cm < 0 or angular_sigma_deg < 0:
        raise ValueError("error sigmas must be nonnegative")
    result = source.copy()
    rng = np.random.default_rng(seed)
    n = len(result)
    position_error_cm = rng.normal(0.0, position_sigma_cm, size=(n, 2, 2))
    result[:, 0:2] += position_error_cm[:, 0, :] * 10.0
    result[:, 6:8] += position_error_cm[:, 1, :] * 10.0

    angular_sigma_rad = np.deg2rad(angular_sigma_deg)
    angles = rng.normal(0.0, angular_sigma_rad, size=(n, 2))
    entry_magnitude = np.linalg.norm(source[:, 3:6], axis=1)
    entry = _rotate_directions(source[:, 3:6], angles[:, 0], rng)
    exit_ = _rotate_directions(source[:, 9:12], angles[:, 1], rng)
    result[:, 3:6] = entry * entry_magnitude[:, None]
    result[:, 9:12] = exit_

    def angular_rms_deg(before: np.ndarray, after: np.ndarray) -> float:
        first = before / np.linalg.norm(before, axis=1, keepdims=True)
        second = after / np.linalg.norm(after, axis=1, keepdims=True)
        delta = np.arccos(np.clip(np.sum(first * second, axis=1), -1.0, 1.0))
        return float(np.rad2deg(np.sqrt(np.mean(delta * delta))))

    report: dict[str, float | int] = {
        "seed": int(seed),
        "tracks": int(n),
        "position_sigma_cm_requested": float(position_sigma_cm),
        "entry_x_error_std_cm": float(np.std(position_error_cm[:, 0, 0])),
        "entry_y_error_std_cm": float(np.std(position_error_cm[:, 0, 1])),
        "exit_x_error_std_cm": float(np.std(position_error_cm[:, 1, 0])),
        "exit_y_error_std_cm": float(np.std(position_error_cm[:, 1, 1])),
        "angular_sigma_deg_requested": float(angular_sigma_deg),
        "entry_angular_rms_deg": angular_rms_deg(source[:, 3:6], result[:, 3:6]),
        "exit_angular_rms_deg": angular_rms_deg(source[:, 9:12], result[:, 9:12]),
        "z_positions_unchanged": bool(
            np.array_equal(source[:, 2], result[:, 2])
            and np.array_equal(source[:, 8], result[:, 8])
        ),
        "entry_momentum_magnitudes_unchanged": bool(
            np.allclose(entry_magnitude, np.linalg.norm(result[:, 3:6], axis=1))
        ),
        "exit_directions_unit": bool(
            np.allclose(np.linalg.norm(result[:, 9:12], axis=1), 1.0)
        ),
    }
    return result, report


def write_errored_tfrecord(
    source: str | Path,
    output: str | Path,
    *,
    record_index: int = 0,
    seed: int = 20260803,
    position_sigma_cm: float = POSITION_SIGMA_CM,
    angular_sigma_deg: float = ANGULAR_SIGMA_DEG,
) -> dict[str, object]:
    """Write one standalone GZIP TFRecord with perturbed detector variables."""
    import tensorflow as tf

    from .io.tfrecord import _decode

    source_path = Path(source).resolve()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset = tf.data.TFRecordDataset(str(source_path), compression_type="GZIP")
    try:
        raw = next(iter(dataset.skip(record_index).take(1)))
    except StopIteration as exc:
        raise IndexError(f"TFRecord has no record {record_index}") from exc
    columns, labels = _decode(raw)
    errored, report = perturb_muon_columns(
        columns,
        seed=seed,
        position_sigma_cm=position_sigma_cm,
        angular_sigma_deg=angular_sigma_deg,
    )
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "x": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(errored).numpy()])
                ),
                "y": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(labels).numpy()])
                ),
            }
        )
    )
    with tf.io.TFRecordWriter(str(target), options="GZIP") as writer:
        writer.write(example.SerializeToString())
    return {
        "source": str(source_path),
        "output": str(target.resolve()),
        "record_index": int(record_index),
        "truth_preserved": bool(np.array_equal(labels, labels)),
        "error_model": report,
    }
