"""Run the frozen momentum protocol; use a new output directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/muon-momentum-mpl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction.munet import MuNetReconstructor
from muon_tomo.reconstruction.paths import path_occupancy
from muon_tomo.reconstruction.poca import PoCAReconstructor
from muon_tomo.research.momentum_study import (
    HISTORICAL_MEV,
    POLICIES,
    CachedMuNet,
    CachedPoCA,
    MomentumPrior,
    assigned_policies,
    evaluate,
    incoming_zenith,
    initial_momentum_from_record,
    momentum_features,
    momentum_statistics,
)
from muon_tomo.types import ReconstructionResult, TrackTable


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tracks_from_x(x):
    return TrackTable(
        x[:, :3] / 10, x[:, 3:6], x[:, 6:9] / 10, x[:, 9:12], np.linalg.norm(x[:, 3:6], axis=1)
    )


def selected_records(source, indices, labels=True):
    wanted = set(indices)
    spec = {"x": tf.io.FixedLenFeature([], tf.string)}
    if labels:
        spec["y"] = tf.io.FixedLenFeature([], tf.string)
    for index, raw in enumerate(tf.data.TFRecordDataset(str(source), compression_type="GZIP")):
        if index in wanted:
            parsed = tf.io.parse_single_example(raw, spec)
            x = tf.io.parse_tensor(parsed["x"], out_type=tf.double).numpy()
            y = tf.io.parse_tensor(parsed["y"], out_type=tf.int32).numpy() if labels else None
            yield index, x, y, hashlib.sha256(raw.numpy()).hexdigest()
            wanted.remove(index)
            if not wanted:
                break
    if wanted:
        raise IndexError(f"Missing records {wanted} in {source}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(0)
    start = time.monotonic()
    case_specs = [
        (
            "lead25",
            "dataset/generated_custom/lead_block_25cm_filtered_8192.tfrecord",
            0,
            "new_source",
        ),
        ("funny0", "dataset_target_crossing/voxels_prediction_funny.tfrecord", 0, "legacy"),
        ("funny1", "dataset_target_crossing/voxels_prediction_funny.tfrecord", 1, "legacy"),
        ("custom0", "dataset_target_crossing/custom_object.tfrecord", 0, "legacy"),
        *[
            (f"val{i}", "dataset_target_crossing/voxels_prediction_val.tfrecord", i, "legacy")
            for i in [17, 53, 101, 257, 503, 997]
        ],
        *[
            (f"test{i}", "dataset_target_crossing/voxels_prediction_test.tfrecord", i, "legacy")
            for i in [37, 211]
        ],
    ]
    reference_path = root / "dataset_target_crossing/voxels_prediction.tfrecord"
    files = [
        Path(__file__),
        root / "src/muon_tomo/research/momentum_study.py",
        root / "src/muon_tomo/reconstruction/munet.py",
        root / "src/muon_tomo/reconstruction/poca.py",
        root / "src/muon_tomo/reconstruction/paths.py",
        Path(__file__).with_name("README.md"),
    ]
    weights = MuNetReconstructor().weights_path
    sources = sorted({s[1] for s in case_specs} | {str(reference_path.relative_to(root))})
    manifest = {
        "protocol_date": "2026-10-20",
        "tracks": 8192,
        "selection_seed": 0,
        "inference_seed": 0,
        "rng_reset": "after model construction, before each policy",
        "cases": case_specs,
        "policies": POLICIES,
        "code_sha256": {str(p.relative_to(root)): sha(p) for p in files},
        "weights": str(weights.relative_to(root)),
        "weights_sha256": sha(weights),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "devices": [str(d) for d in tf.config.list_physical_devices()],
        "source_stat_not_hash": {
            s: {"bytes": (root / s).stat().st_size, "mtime_ns": (root / s).stat().st_mtime_ns}
            for s in sources
        },
    }
    dump(output / "manifest.json", manifest)
    (output / "protocol.md").write_text(Path(__file__).with_name("README.md").read_text())
    # No y decoding in the reference-prior fit.
    ref_p, ref_angles, ref_identity = [], [], []
    for index, x, _, digest in selected_records(reference_path, range(32), labels=False):
        tracks = tracks_from_x(x)
        ref_p.append(tracks.momentum_mev)
        ref_angles.append(incoming_zenith(tracks))
        ref_identity.append({"record": index, "tracks": len(tracks), "serialized_sha256": digest})
    ref_p, ref_angles = np.concatenate(ref_p), np.concatenate(ref_angles)
    reference = MomentumPrior.fit(ref_p, ref_angles)
    dump(
        output / "reference.json",
        {
            **reference.metadata(),
            "records": ref_identity,
            "source": str(reference_path.relative_to(root)),
        },
    )
    np.savez_compressed(output / "reference_tracks.npz", p_out_mev=ref_p, zenith_degrees=ref_angles)
    print("Reference fitted:", len(ref_p), reference.global_stats, flush=True)
    # Each compressed source is scanned once. Retain complete selected inputs.
    cases = {}
    for source in dict.fromkeys(spec[1] for spec in case_specs):
        specs = {s[2]: s for s in case_specs if s[1] == source}
        for index, x, labels, digest in selected_records(root / source, specs):
            name, _, _, group = specs[index]
            if len(x) < 8192:
                raise ValueError(f"{name} has only {len(x)} tracks")
            rows = (
                np.arange(len(x))
                if len(x) == 8192
                else np.sort(np.random.default_rng(0).choice(len(x), 8192, replace=False))
            )
            selected = x[rows]
            # Assert this is the documented legacy outgoing-then-incoming schema.
            if not np.all(selected[:, 2] < selected[:, 8]):
                raise ValueError("Unexpected TFRecord endpoint order")
            case_dir = output / name
            case_dir.mkdir()
            np.savez_compressed(
                case_dir / "input.npz", x=selected, labels=labels, filtered_row_ids=rows
            )
            tracks = tracks_from_x(selected)
            p_in = initial_momentum_from_record(selected) if selected.shape[1] >= 13 else None
            angles = incoming_zenith(tracks)
            assigned, local = assigned_policies(reference, angles, tracks.momentum_mev, p_in)
            np.savez_compressed(case_dir / "assigned_momenta.npz", **assigned)
            meta = {
                "name": name,
                "group": group,
                "source": source,
                "record": index,
                "source_tracks": len(x),
                "selected_tracks": len(selected),
                "serialized_record_sha256": digest,
                "selected_x_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
                "outgoing": local.metadata(),
                "initial_source": momentum_statistics(p_in) if p_in is not None else None,
                "unavailable_policies": [p for p in POLICIES if p not in assigned],
                "zenith_degrees_min_max": [float(angles.min()), float(angles.max())],
            }
            dump(case_dir / "input.json", meta)
            cases[name] = (tracks, labels, assigned, group)
            print("Cached", name, len(x), "rows; selected 8192", flush=True)
    print("Building frozen MuNet", flush=True)
    model = CachedMuNet()
    parity = {}
    results, materials = [], []
    for case_index, (name, *_rest) in enumerate(case_specs):
        tracks, labels, assigned, group = cases[name]
        case_dir = output / name
        case_start = time.monotonic()
        poca = CachedPoCA(tracks)
        occupancy = path_occupancy(tracks, GridGeometry())
        if case_index == 0:
            for policy, assigned_name in [("mean", "historical"), ("per-track", "truth_outgoing")]:
                production = PoCAReconstructor(
                    momentum_policy=policy, average_momentum_mev=HISTORICAL_MEV
                ).reconstruct(tracks)
                cached = poca.predict(assigned[assigned_name])
                np.testing.assert_allclose(
                    cached, production.inverse_radiation_length_cm_inv, rtol=2e-6, atol=1e-8
                )
                np.testing.assert_array_equal(poca.counts, production.occupancy_count)
                np.testing.assert_array_equal(poca.valid_mask, production.valid_mask)
                parity[f"poca_{policy}_max_abs"] = float(
                    np.max(np.abs(cached - production.inverse_radiation_length_cm_inv))
                )
            np.testing.assert_array_equal(
                momentum_features(tracks, assigned["historical"]),
                MuNetReconstructor()._features(tracks),
            )
            parity["constant_feature_bitwise_equal"] = True
            # Fresh independently loaded model, same post-build RNG convention.
            chunk = tracks.take_first(1024)
            _pred1, raw1 = model.predict(chunk, assigned["historical"][:1024])
            fresh = CachedMuNet()
            _pred2, raw2 = fresh.predict(chunk, assigned["historical"][:1024])
            np.testing.assert_allclose(raw1, raw2, rtol=1e-6, atol=1e-7)
            parity["munet_cached_fresh_1024_max_abs"] = float(np.max(np.abs(raw1 - raw2)))
            del fresh
            # Policy order must not alter random fallback placements.
            model.predict(chunk, assigned["truth_outgoing"][:1024])
            _pred3, raw3 = model.predict(chunk, assigned["historical"][:1024])
            np.testing.assert_array_equal(raw1, raw3)
            parity["munet_policy_order_bitwise_equal"] = True
            dump(output / "parity.json", parity)
            print("Parity checks passed", parity, flush=True)
        for policy, p in assigned.items():
            for method in ("poca", "munet"):
                t0 = time.monotonic()
                if method == "poca":
                    prediction = raw = poca.predict(p)
                    count, valid = poca.counts, poca.valid_mask
                else:
                    prediction, raw = model.predict(tracks, p)
                    count, valid = occupancy, np.ones((64,) * 3, dtype=bool)
                elapsed = time.monotonic() - t0
                meta = {
                    "study": "momentum-study-20261020",
                    "case": name,
                    "policy": policy,
                    "policy_label": POLICIES[policy][0],
                    "information": POLICIES[policy][1],
                    "tracks": 8192,
                    "p_summary_mev": momentum_statistics(p),
                    "no_truth_calibration": True,
                    "weights_sha256": manifest["weights_sha256"] if method == "munet" else None,
                    "settings": {
                        "poca_grid": 32,
                        "pooling_sigma_cm": 6.25,
                        "munet_chunks": 8,
                        "dosage": 1024,
                    },
                    "seed": 0,
                    "elapsed_prediction_seconds": elapsed,
                }
                ReconstructionResult(
                    method, prediction, raw, count, valid, GridGeometry(), meta
                ).save(case_dir / f"{method}__{policy}.npz")
                metrics, per_material = evaluate(prediction, labels)
                results.append(
                    {
                        "case": name,
                        "group": group,
                        "method": method,
                        "policy": policy,
                        "information": POLICIES[policy][1],
                        "seconds": elapsed,
                        **metrics,
                    }
                )
                materials.extend(
                    {"case": name, "method": method, "policy": policy, **m} for m in per_material
                )
            dump(output / "metrics.json", results)
            dump(output / "materials.json", materials)
            print(
                f"{name} {policy}: PoCA MSE={results[-2]['mse_full']:.6g}, MuNet MSE={results[-1]['mse_full']:.6g}; elapsed {time.monotonic() - start:.0f}s",
                flush=True,
            )
        print(f"Completed {name} in {time.monotonic() - case_start:.1f}s", flush=True)
    manifest["completed"] = True
    manifest["elapsed_seconds"] = time.monotonic() - start
    manifest["outputs"] = len(results)
    if sha(weights) != manifest["weights_sha256"]:
        raise RuntimeError("Weights changed during study")
    dump(output / "manifest.json", manifest)
    print(
        "DONE", len(results), "reconstructions", manifest["elapsed_seconds"], "seconds", flush=True
    )


if __name__ == "__main__":
    main()
