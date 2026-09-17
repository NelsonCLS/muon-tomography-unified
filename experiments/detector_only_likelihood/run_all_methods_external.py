"""Run the all-method external-spectrum comparison on fixed examples.

Truth is loaded only after all three prediction files have been written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from muon_tomo.research.detector_only_likelihood.all_methods_external import ExternalSpectrumMuNet, poca_external
from muon_tomo.research.detector_only_likelihood.method import DetectorTracks


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results" / "all_methods_external"
OUT.mkdir(parents=True, exist_ok=True)
CASES = [("funny", "voxels_prediction_funny.tfrecord", 0), ("prediction", "voxels_prediction.tfrecord", 0)]


def load_x(path: Path, record_index: int) -> DetectorTracks:
    spec = {"x": tf.io.FixedLenFeature([], tf.string)}
    raw = next(iter(tf.data.TFRecordDataset(str(path), compression_type="GZIP").skip(record_index).take(1)))
    x = tf.io.parse_tensor(tf.io.parse_single_example(raw, spec)["x"], out_type=tf.double).numpy()
    indices = np.sort(np.random.default_rng(0).choice(len(x), size=8192, replace=False))
    x = x[indices]
    return DetectorTracks(x[:, :3] / 10.0, x[:, 3:6], x[:, 6:9] / 10.0, x[:, 9:12])


def main() -> None:
    manifest = dict(
        cases=[f"{name}_{record}" for name, _, record in CASES],
        tracks=8192,
        selection="seed-0 random subset; fixed before evaluation",
        methods={
            "poca": "external spectrum coefficient + existing exposure normalization and 6.25 cm pooling",
            "likelihood": "saved exposure-pooled external-prior likelihood result",
            "munet": "external-spectrum p_eff(theta) supplied at inference; weights unchanged",
        },
        production_likelihood_sha256=hashlib.sha256(
            (ROOT / "src/muon_tomo/reconstruction/likelihood.py").read_bytes()
        ).hexdigest(),
    )
    (OUT / "protocol.json").write_text(json.dumps(manifest, indent=2))
    for name, filename, record in CASES:
        tag = f"{name}_{record}"
        tracks = load_x(ROOT / "dataset" / filename, record)
        p = OUT / tag
        p.mkdir(exist_ok=True)
        poca, poca_meta = poca_external(tracks)
        munet = ExternalSpectrumMuNet(dosage=1024, max_chunks=8).reconstruct(tracks)
        likelihood = np.load(
            Path(__file__).resolve().parent / "results" / f"{tag}" / "pooled_likelihood.npz"
        )["inverse_radiation_length_cm_inv"]
        np.savez_compressed(p / "predictions.npz", poca=poca, munet=munet.inverse_radiation_length_cm_inv, likelihood=likelihood)
        (p / "metadata.json").write_text(json.dumps({"poca": poca_meta, "munet": munet.metadata}, indent=2))
        print(f"saved predictions for {tag}", flush=True)

    # Evaluation is deliberately a separate phase from fitting/saving.
    spec = {"y": tf.io.FixedLenFeature([], tf.string)}
    scores = []
    for name, filename, record in CASES:
        raw = next(iter(tf.data.TFRecordDataset(str(ROOT / "dataset" / filename), compression_type="GZIP").skip(record).take(1)))
        labels = tf.io.parse_tensor(tf.io.parse_single_example(raw, spec)["y"], out_type=tf.int32).numpy()
        from muon_tomo.constants import labels_to_inverse_radiation_length

        truth = labels_to_inverse_radiation_length(labels)
        p = OUT / f"{name}_{record}"
        predictions = np.load(p / "predictions.npz")
        row = {"case": f"{name}_{record}", "empty_mse": float(np.mean(truth.astype(float) ** 2)), "models": {}}
        for method in ("poca", "likelihood", "munet"):
            value = predictions[method]
            row["models"][method] = {
                "mse_cm_inv_squared": float(np.mean((value.astype(float) - truth) ** 2)),
                "maximum_cm_inv": float(value.max()),
            }
        np.savez_compressed(p / "truth_evaluation_only.npz", inverse_radiation_length_cm_inv=truth)
        scores.append(row)
        print(json.dumps(row), flush=True)
    (OUT / "scores.json").write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
