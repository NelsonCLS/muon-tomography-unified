"""Reproducible paired Funny 0 reconstruction comparison.

It holds all production reconstruction settings fixed and compares equal stored
track counts from the original and target-crossing-only TFRecord copies.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from muon_tomo.cli import _save_truth
from muon_tomo.compare import compare_to_truth
from muon_tomo.geometry import GridGeometry
from muon_tomo.io import load_tfrecord_record
from muon_tomo.plotting import plot_comparison_slab
from muon_tomo.plotting3d import plot_comparison_3d
from muon_tomo.reconstruction import LikelihoodReconstructor, MuNetReconstructor, PoCAReconstructor
from muon_tomo.reconstruction.common import prepare_poca_paths


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT / "src/muon_tomo/configs/default.yaml").read_text())
G = GridGeometry(
    tuple(CONFIG["geometry"]["lower_cm"]),
    tuple(CONFIG["geometry"]["upper_cm"]),
    tuple(CONFIG["geometry"]["shape"]),
    CONFIG["geometry"]["axis_order"],
)
FIXED_SEED = 20260912
LIMIT = 8192
SOURCES = {
    "old_stored": ROOT / "dataset/voxels_prediction_funny.tfrecord",
    "target_crossing": ROOT / "dataset_target_crossing/voxels_prediction_funny.tfrecord",
}


def fingerprint(tracks) -> str:
    digest = hashlib.sha256()
    for values, dtype in (
        (tracks.event_id, "<i8"),
        (tracks.entry_position_cm, "<f8"),
        (tracks.entry_direction, "<f8"),
        (tracks.exit_position_cm, "<f8"),
        (tracks.exit_direction, "<f8"),
        (tracks.momentum_mev, "<f8"),
    ):
        digest.update(np.asarray(values, dtype=dtype).tobytes(order="C"))
    return digest.hexdigest()


def path_stats(tracks) -> dict[str, int | float]:
    paths, lengths, _, kinked = prepare_poca_paths(tracks, G)
    positive = np.asarray([float(np.sum(item)) > 0 for item in lengths], dtype=bool)
    return {
        "tracks_with_positive_target_path": int(positive.sum()),
        "tracks_without_target_path": int((~positive).sum()),
        "tracks_with_kinked_poca_path": int(kinked),
        "total_target_path_cm": float(sum(float(np.sum(item)) for item in lengths)),
        "mean_target_path_cm_all_tracks": float(np.mean([float(np.sum(item)) for item in lengths])),
    }


def reconstruct(tracks):
    physics = CONFIG["physics"]
    mu = CONFIG["munet"]
    po = CONFIG["poca"]
    li = CONFIG["likelihood"]
    # MuNet's unscattered fallback has a stochastic position.  Resetting this
    # seed makes both conditions exactly reproducible without changing weights.
    tf.random.set_seed(FIXED_SEED)
    munet = MuNetReconstructor(
        G,
        dosage=int(mu["dosage"]),
        position_scale_cm=float(mu["position_scale_cm"]),
        position_offset=float(mu["position_offset"]),
        momentum_estimate_mev=float(mu["momentum_estimate_mev"]),
    ).reconstruct(tracks)
    poca = PoCAReconstructor(
        G,
        highland_constant_mev=float(physics["highland_constant_mev"]),
        minimum_coverage=int(po["minimum_coverage"]),
        momentum_policy=physics["classical_momentum_policy"],
        average_momentum_mev=float(physics["average_momentum_mev"]),
        coarse_grid_size=int(po["grid_size"]),
        pooling_sigma_cm=float(po["pooling_sigma_cm"]),
    ).reconstruct(tracks)
    likelihood = LikelihoodReconstructor(
        G,
        iterations=int(li["iterations"]),
        reference_momentum_mev=float(physics["reference_momentum_mev"]),
        highland_constant_mev=float(physics["highland_constant_mev"]),
        initial_inverse_radiation_length_cm_inv=float(li["initial_inverse_radiation_length_cm_inv"]),
        minimum_coverage=int(li["minimum_coverage"]),
        epsilon=float(li["epsilon"]),
        mode="full",
        momentum_policy=physics["classical_momentum_policy"],
        average_momentum_mev=float(physics["average_momentum_mev"]),
        update_statistic=li["update_statistic"],
        angular_resolution_rad=float(li["angular_resolution_rad"]),
        position_resolution_cm=float(li["position_resolution_cm"]),
        maximum_inverse_radiation_length_cm_inv=float(li["maximum_inverse_radiation_length_cm_inv"]),
    ).reconstruct(tracks)
    return [munet, poca, likelihood]


def main() -> None:
    conditions = {}
    for name, source in SOURCES.items():
        condition_out = OUT / name
        condition_out.mkdir(parents=True, exist_ok=True)
        tracks, truth = load_tfrecord_record(source, 0, geometry=G, limit=LIMIT, seed=0)
        # 8192 is exactly eight MuNet dosage chunks; all methods receive this
        # same sampled TrackTable in each condition.
        tracks = tracks.take_first(LIMIT)
        results = reconstruct(tracks)
        common = {
            "input_source": str(source.resolve()),
            "record_index": 0,
            "selection": "load_tfrecord_record(limit=8192, seed=0); sorted random rows",
            "track_count": len(tracks),
            "track_fingerprint_sha256": fingerprint(tracks),
            "comparison_random_seed": FIXED_SEED,
            "calibration": "none",
            "classical_momentum_policy": "fixed shared effective momentum",
            "assumed_momentum_mev": float(CONFIG["physics"]["average_momentum_mev"]),
        }
        for result in results:
            result.metadata.update(common)
            result.save(condition_out / f"record_0_{result.method}.npz")
        _save_truth(truth, condition_out / "record_0_truth.npz")
        plot = plot_comparison_slab(
            truth, results, output=condition_out / "cross_section.png", center_index=32, slab_layers=7
        )
        viewer = plot_comparison_3d(
            truth,
            results,
            output=condition_out / "viewer.html",
            threshold_cm_inv=float(np.nextafter(np.float32(0.8), np.float32(np.inf))),
            max_cubes=262144,
            opacity=0.6,
            surface_only=False,
        )
        metrics = {result.method: compare_to_truth(result, truth) for result in results}
        conditions[name] = {**common, "target_path_statistics": path_stats(tracks), "metrics": metrics, "plot": plot, "viewer": viewer}
        (condition_out / "metrics.json").write_text(json.dumps(conditions[name], indent=2, sort_keys=True) + "\n")

    comparison = {
        "design": {
            "record": "Funny 0 (record index 0)",
            "each_condition_tracks": LIMIT,
            "sampling": "independent seed=0 random sample in each file; 8192 stored tracks per condition",
            "only_changed_factor": "source TFRecord population: original storage versus target-crossing-only copy",
            "all_reconstructor_settings": "production default.yaml; MuNet random fallback seed fixed at 20260912",
            "calibration": "none",
            "important_interpretation": "The filtered condition has 8192 target-crossing tracks. The original condition has a mixture of crossing and miss tracks, so the comparison measures useful target exposure at equal stored-track dosage, not the same physical muon rows.",
        },
        "conditions": conditions,
        "mse_change_target_crossing_minus_old": {
            method: float(
                conditions["target_crossing"]["metrics"][method]["mse_full_grid_cm_inv_squared"]
                - conditions["old_stored"]["metrics"][method]["mse_full_grid_cm_inv_squared"]
            )
            for method in ("munet", "poca", "likelihood")
        },
    }
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparison["mse_change_target_crossing_minus_old"], indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
