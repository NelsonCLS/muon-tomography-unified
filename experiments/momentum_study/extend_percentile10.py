"""Add an independently fitted 10th-percentile policy to saved study inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/muon-momentum-mpl")

import numpy as np
import tensorflow as tf

from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction.paths import path_occupancy
from muon_tomo.research.momentum_study import (
    POLICIES,
    CachedMuNet,
    CachedPoCA,
    MomentumPrior,
    evaluate,
    momentum_statistics,
)
from muon_tomo.types import ReconstructionResult, TrackTable


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base_run.resolve(), args.output.resolve()
    manifest = json.loads((base / "manifest.json").read_text())
    if not manifest.get("completed"):
        raise ValueError("Base run must be complete")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(0)
    ref = np.load(base / "reference_tracks.npz")
    prior = MomentumPrior.fit(ref["p_out_mev"], ref["zenith_degrees"])
    p10 = prior.global_stats["percentile10"]
    policy = "ref_percentile10"
    reference = json.loads((base / "reference.json").read_text())
    reference.update(prior.metadata())
    dump(output / "reference.json", reference)
    shutil.copy2(base / "reference_tracks.npz", output)
    rows = json.loads((base / "metrics.json").read_text())
    materials = json.loads((base / "materials.json").read_text())
    manifest.update(
        completed=False,
        parent_run=str(base),
        added_policy=policy,
        reference_percentile10_mev=p10,
        percentile_method="linear",
        policies=POLICIES,
        same_saved_track_rows=True,
    )
    for source in (Path(__file__), Path("src/muon_tomo/research/momentum_study.py")):
        manifest["code_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    dump(output / "manifest.json", manifest)
    model = CachedMuNet()
    if hashlib.sha256(model.weights_path.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Model weights differ from base study")
    print("Reference 10th percentile:", p10, "MeV/c", flush=True)
    for case, *_ in manifest["cases"]:
        dest = output / case
        dest.mkdir()
        # Copy saved inputs/results; preserve every parent file unchanged.
        for source in (base / case).glob("*.npz"):
            shutil.copy2(source, dest / source.name)
        shutil.copy2(base / case / "input.json", dest)
        saved = np.load(dest / "input.npz")
        x, labels = saved["x"], saved["labels"]
        meta = json.loads((dest / "input.json").read_text())
        if len(x) != 8192 or hashlib.sha256(x.tobytes()).hexdigest() != meta["selected_x_sha256"]:
            raise ValueError("Saved input identity mismatch")
        tracks = TrackTable(
            x[:, :3] / 10, x[:, 3:6], x[:, 6:9] / 10, x[:, 9:12], np.linalg.norm(x[:, 3:6], axis=1)
        )
        assigned = dict(np.load(dest / "assigned_momenta.npz"))
        p = np.full(len(x), p10)
        assigned[policy] = p
        np.savez_compressed(dest / "assigned_momenta.npz", **assigned)
        poca = CachedPoCA(tracks)
        for method in ("poca", "munet"):
            t0 = time.monotonic()
            if method == "poca":
                pred = raw = poca.predict(p)
                count, valid = poca.counts, poca.valid_mask
            else:
                pred, raw = model.predict(tracks, p, seed=0)
                count = path_occupancy(tracks, GridGeometry())
                valid = np.ones((64,) * 3, dtype=bool)
            elapsed = time.monotonic() - t0
            result_meta = dict(
                study="momentum-study-percentile10",
                case=case,
                policy=policy,
                policy_label=POLICIES[policy][0],
                information=POLICIES[policy][1],
                tracks=len(x),
                p_summary_mev=momentum_statistics(p),
                seed=0,
                no_truth_calibration=True,
                weights_sha256=manifest["weights_sha256"] if method == "munet" else None,
                settings=dict(poca_grid=32, pooling_sigma_cm=6.25, munet_chunks=8, dosage=1024),
                elapsed_prediction_seconds=elapsed,
            )
            ReconstructionResult(method, pred, raw, count, valid, GridGeometry(), result_meta).save(
                dest / f"{method}__{policy}.npz"
            )
            metrics, per_material = evaluate(pred, labels)
            rows.append(
                dict(
                    case=case,
                    group=meta["group"],
                    method=method,
                    policy=policy,
                    information=POLICIES[policy][1],
                    seconds=elapsed,
                    **metrics,
                )
            )
            materials.extend(
                dict(case=case, method=method, policy=policy, **m) for m in per_material
            )
            print(case, method, metrics["mse_full"], flush=True)
        dump(output / "metrics.json", rows)
        dump(output / "materials.json", materials)
    manifest.update(
        completed=True,
        outputs=len(rows),
        added_reconstructions=24,
        extension_elapsed_seconds=time.monotonic() - start,
    )
    dump(output / "manifest.json", manifest)
    (output / "protocol.md").write_text(
        (base / "protocol.md").read_text()
        + "\n2026-10-23 extension: reference 10th percentile (NumPy linear interpolation),\n"
        "fitted on the same independent reference outgoing tracks; all saved evaluation rows,\n"
        "seeds, weights and settings unchanged. Earlier 356 reconstructions reused; 24 added.\n"
    )


if __name__ == "__main__":
    main()
