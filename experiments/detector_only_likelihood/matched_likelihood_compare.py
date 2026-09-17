"""Compare fixed and external momentum likelihood under identical pooling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import tensorflow as tf
from muon_tomo.research.detector_only_likelihood.method import DetectorTracks
from muon_tomo.research.detector_only_likelihood.exposure_pooled_likelihood import pool_with_exposure
from muon_tomo.geometry import GridGeometry
from muon_tomo.io.tfrecord import load_tfrecord_record
from muon_tomo.reconstruction import LikelihoodReconstructor


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results" / "all_methods_external"


def main() -> None:
    rows = []
    for name, filename, record, fixed_dir in [
        ("funny_0", "voxels_prediction_funny.tfrecord", 0, "outputs/detector-only-comparison/funny-fixed-1743"),
        ("prediction_0", "voxels_prediction.tfrecord", 0, "outputs/random-prediction/record-0-exposure"),
    ]:
        spec = {"x": tf.io.FixedLenFeature([], tf.string)}
        raw = next(iter(tf.data.TFRecordDataset(str(ROOT / "dataset" / filename), compression_type="GZIP").skip(record).take(1)))
        x = tf.io.parse_tensor(tf.io.parse_single_example(raw, spec)["x"], out_type=tf.double).numpy()
        indices = np.sort(np.random.default_rng(0).choice(len(x), size=8192, replace=False))
        x = x[indices]
        tracks = DetectorTracks(x[:, :3] / 10.0, x[:, 3:6], x[:, 6:9] / 10.0, x[:, 9:12])
        _, truth = load_tfrecord_record(ROOT / "dataset" / filename, record, limit=8192, seed=0)
        fixed = np.load(ROOT / fixed_dir / "record_0_likelihood.npz")["inverse_radiation_length_cm_inv"]
        fixed_pooled = pool_with_exposure(fixed, tracks)
        external = np.load(OUT / name / "predictions.npz")["likelihood"]
        row = {
            "case": name,
            "fixed_1743_unpooled_mse": float(np.mean((fixed - truth.inverse_radiation_length_cm_inv) ** 2)),
            "fixed_1743_pooled_mse": float(np.mean((fixed_pooled - truth.inverse_radiation_length_cm_inv) ** 2)),
            "external_spectrum_pooled_mse": float(np.mean((external - truth.inverse_radiation_length_cm_inv) ** 2)),
        }
        np.savez_compressed(OUT / name / "fixed_1743_pooled_likelihood.npz", inverse_radiation_length_cm_inv=fixed_pooled)
        rows.append(row)
    (OUT / "matched_likelihood.json").write_text(__import__("json").dumps(rows, indent=2))
    print(__import__("json").dumps(rows, indent=2))


if __name__ == "__main__":
    main()
