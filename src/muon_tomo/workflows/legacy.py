"""Compatibility workflows for established reconstruction and research commands.

Numerical algorithms live in reconstruction/. The ordinary prediction workflow
is in predict.py; historical truth calibration stays explicit in this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..calibration import apply_calibration, fit_calibration, load_calibrations, save_calibrations
from ..compare import compare_to_truth
from ..config import geometry_from_config as _geometry
from ..config import load_config as _config
from ..geometry import GridGeometry
from ..io import inspect_tfrecord, load_root_tracks, load_tfrecord_record
from ..plotting import plot_comparison_slab
from ..plotting3d import plot_comparison_3d
from ..reconstruction.registry import build_reconstructor
from ..stability import write_errored_tfrecord
from ..types import ReconstructionResult, TruthVolume
from .reconstruction import select_shared_tracks


def _format(path: str, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = Path(path).suffix.lower()
    if suffix == ".tfrecord":
        return "tfrecord"
    if suffix == ".root":
        return "root"
    raise ValueError("cannot infer format; use --format tfrecord or --format root")


def _load_input(args: argparse.Namespace, geometry: GridGeometry):
    fmt = _format(args.input, args.format)
    if fmt == "tfrecord":
        return load_tfrecord_record(
            args.input, args.record_index, geometry=geometry, limit=args.limit, seed=args.seed
        )
    return load_root_tracks(args.input, geometry=geometry, limit=args.limit, seed=args.seed), None


def _write_json(payload: object, path: str | None = None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    else:
        print(text, end="")


def _track_fingerprint(tracks) -> str:
    from ..artifacts.results import legacy_track_fingerprint

    return legacy_track_fingerprint(tracks)


def _save_truth(truth: TruthVolume, path: str | Path) -> Path:
    from ..artifacts.results import save_truth

    return save_truth(truth, path)


def _load_truth(path: str | Path) -> TruthVolume:
    from ..artifacts.results import load_truth

    return load_truth(path)


def command_inspect(args: argparse.Namespace) -> None:
    fmt = _format(args.input, args.format)
    if fmt == "tfrecord":
        report = inspect_tfrecord(args.input)
    else:
        cfg = _config(args.config)
        tracks = load_root_tracks(
            args.input, geometry=_geometry(cfg), limit=args.limit, seed=args.seed
        )
        report = {
            "path": str(Path(args.input).resolve()),
            "format": "root",
            "tracks": len(tracks),
            "metadata": tracks.metadata,
        }
    _write_json(report, args.output)


def command_truth(args: argparse.Namespace) -> None:
    geometry = _geometry(_config(args.config))
    if _format(args.input, args.format) != "tfrecord":
        raise ValueError("material truth is currently stored only by the x/y TFRecord format")
    _, truth = load_tfrecord_record(args.input, args.record_index, geometry=geometry, limit=0)
    print(_save_truth(truth, args.output))


def command_reconstruct(args: argparse.Namespace) -> None:
    cfg = _config(args.config)
    geometry = _geometry(cfg)
    tracks, truth = _load_input(args, geometry)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    is_tfrecord = _format(args.input, args.format) == "tfrecord"
    stem = f"record_{args.record_index}" if is_tfrecord else Path(args.input).stem
    if truth is not None:
        _save_truth(truth, output / f"{stem}_truth.npz")
    methods = ["munet", "poca", "likelihood"] if args.method == "all" else [args.method]
    tracks = select_shared_tracks(tracks, methods, cfg, overrides=vars(args))
    track_fingerprint = _track_fingerprint(tracks)
    calibration = load_calibrations(args.calibration) if args.calibration else {}
    momentum_policy = args.classical_momentum_policy or cfg["physics"].get(
        "classical_momentum_policy", "mean"
    )
    average_momentum_mev = (
        args.average_momentum_mev
        if args.average_momentum_mev is not None
        else float(cfg["physics"].get("average_momentum_mev", 3000.0))
    )
    created = []
    for method in methods:
        reconstructor = build_reconstructor(
            method,
            cfg,
            geometry=geometry,
            overrides=vars(args),
        )
        result = reconstructor.reconstruct(tracks)
        result.metadata.update(
            {
                "input_source": str(Path(args.input).resolve()),
                "input_format": _format(args.input, args.format),
                "record_index": args.record_index,
                "track_selection": tracks.metadata,
                "track_count": len(tracks),
                "track_fingerprint_sha256": track_fingerprint,
            }
        )
        if method in calibration:
            result = apply_calibration(result, calibration[method])
        target = result.save(output / f"{stem}_{method}.npz")
        created.append(str(target))
        print(target)
    _write_json(
        {
            "created": created,
            "geometry": geometry.metadata(),
            "tracks": len(tracks),
            "track_fingerprint_sha256": track_fingerprint,
            "classical_momentum_policy": momentum_policy,
            "average_momentum_mev": average_momentum_mev,
        },
        str(output / f"{stem}_manifest.json"),
    )


def command_calibrate(args: argparse.Namespace) -> None:
    cfg = _config(args.config)
    geometry = _geometry(cfg)
    fit_records = args.fit_records
    all_records = sorted(set(fit_records + args.validate_records))
    truths = {}
    for record in all_records:
        _, truths[record] = load_tfrecord_record(args.dataset, record, geometry=geometry, limit=0)
    loaded = {
        (record, method): ReconstructionResult.load(
            Path(args.results_dir) / f"record_{record}_{method}.npz"
        )
        for record in all_records
        for method in ("munet", "poca", "likelihood")
    }
    policies = {
        loaded[(record, method)].metadata.get("momentum_policy")
        for record in all_records
        for method in ("poca", "likelihood")
    }
    assumed_momenta = {
        loaded[(record, method)].metadata.get("assumed_momentum_mev")
        for record in all_records
        for method in ("poca", "likelihood")
    }
    if policies != {"mean"} or None in assumed_momenta or len(assumed_momenta) != 1:
        raise ValueError("calibration inputs must share one fixed mean-momentum policy and value")
    likelihood_modes = {
        loaded[(record, "likelihood")].metadata.get("mode") for record in all_records
    }
    if len(likelihood_modes) != 1:
        raise ValueError("calibration inputs mix likelihood estimator modes")
    models = []
    report: dict[str, object] = {
        "dataset": str(Path(args.dataset).resolve()),
        "fit_records": fit_records,
        "validate_records": args.validate_records,
        "classical_momentum_policy": "mean",
        "assumed_momentum_mev": float(assumed_momenta.pop()),
        "likelihood_mode": likelihood_modes.pop(),
        "methods": {},
    }
    for method in ("munet", "poca", "likelihood"):
        pairs = []
        for record in fit_records:
            result = loaded[(record, method)]
            pairs.append((result, truths[record]))
        model, rows = fit_calibration(
            method,
            pairs,
            erosion_voxels=int(cfg["calibration"]["erosion_voxels"]),
            fit_intercept=bool(cfg["calibration"]["fit_intercept"]),
            calibration_id=args.calibration_id,
        )
        models.append(model)
        validations = []
        for record in args.validate_records:
            original = loaded[(record, method)]
            calibrated = apply_calibration(original, model)
            validations.append({"record": record, **compare_to_truth(calibrated, truths[record])})
        report["methods"][method] = {
            "slope": model.slope,
            "intercept": model.intercept,
            "fit_statistic": model.fit_statistic,
            "maximum_cm_inv": model.maximum_cm_inv,
            "fit_regions": rows,
            "validation": validations,
        }
    print(save_calibrations(args.output, models, report))


def command_compare(args: argparse.Namespace) -> None:
    truth = _load_truth(args.truth)
    rows = [compare_to_truth(ReconstructionResult.load(path), truth) for path in args.results]
    _write_json(rows, args.output)


def command_stability(args: argparse.Namespace) -> None:
    """Create one fixed detector-error dataset and run paired reconstructions."""
    cfg = _config(args.config)
    geometry = _geometry(cfg)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    errored_input = output / "record_0_detector_error.tfrecord"
    error_report = write_errored_tfrecord(
        args.dataset,
        errored_input,
        record_index=args.record_index,
        seed=args.error_seed,
        position_sigma_cm=args.position_sigma_cm,
        angular_sigma_deg=args.angular_sigma_deg,
    )
    clean_tracks, truth = load_tfrecord_record(
        args.dataset, args.record_index, geometry=geometry, limit=args.limit, seed=args.seed
    )
    errored_tracks, errored_truth = load_tfrecord_record(
        errored_input, 0, geometry=geometry, limit=args.limit, seed=args.seed
    )
    if not np.array_equal(truth.material_labels, errored_truth.material_labels):
        raise ValueError("errored dataset truth differs from clean truth")
    dosage = int(cfg["munet"]["dosage"])
    count = min(len(clean_tracks), len(errored_tracks)) // dosage * dosage
    if count < dosage:
        raise ValueError(f"need at least {dosage} tracks for Mu-Net")
    clean_tracks = clean_tracks.take_first(count)
    errored_tracks = errored_tracks.take_first(count)
    calibration = load_calibrations(args.calibration) if args.calibration else {}
    average_momentum_mev = float(cfg["physics"]["average_momentum_mev"])

    def reconstruct(tracks, label: str):
        import tensorflow as tf

        tf.random.set_seed(args.munet_seed)
        reconstructors = [
            build_reconstructor(
                method,
                cfg,
                geometry=geometry,
                overrides={
                    "dosage": dosage,
                    "max_chunks": count // dosage,
                    "iterations": args.iterations,
                    "likelihood_mode": "full",
                    "classical_momentum_policy": "mean",
                    "average_momentum_mev": average_momentum_mev,
                },
            )
            for method in ("munet", "poca", "likelihood")
        ]
        fingerprint = _track_fingerprint(tracks)
        rows = {}
        for reconstructor in reconstructors:
            result = reconstructor.reconstruct(tracks)
            result.metadata.update(
                {
                    "stability_scenario": label,
                    "track_count": len(tracks),
                    "track_fingerprint_sha256": fingerprint,
                    "error_dataset": error_report if label == "errored" else None,
                }
            )
            if result.method in calibration:
                result = apply_calibration(result, calibration[result.method])
            result.save(output / f"record_{args.record_index}_{label}_{result.method}.npz")
            rows[result.method] = compare_to_truth(result, truth)
        return fingerprint, rows

    _save_truth(truth, output / f"record_{args.record_index}_truth.npz")
    clean_fingerprint, clean = reconstruct(clean_tracks, "clean")
    errored_fingerprint, errored = reconstruct(errored_tracks, "errored")
    comparison = []
    for method in ("munet", "poca", "likelihood"):
        baseline = clean[method]
        perturbed = errored[method]
        clean_mse = float(baseline["mse_full_grid_cm_inv_squared"])
        errored_mse = float(perturbed["mse_full_grid_cm_inv_squared"])
        comparison.append(
            {
                "method": method,
                "clean_mse_full_grid_cm_inv_squared": clean_mse,
                "errored_mse_full_grid_cm_inv_squared": errored_mse,
                "delta_mse_cm_inv_squared": errored_mse - clean_mse,
                "mse_ratio_errored_to_clean": errored_mse / clean_mse if clean_mse else None,
            }
        )
    _write_json(
        {
            "status": "success",
            "tracks_per_scenario": count,
            "calibration": str(Path(args.calibration).resolve()) if args.calibration else None,
            "clean_track_fingerprint_sha256": clean_fingerprint,
            "errored_track_fingerprint_sha256": errored_fingerprint,
            "error_dataset": error_report,
            "comparison": comparison,
        },
        output / "stability_mse_comparison.json",
    )
    print(output / "stability_mse_comparison.json")


def command_plot(args: argparse.Namespace) -> None:
    truth = _load_truth(args.truth) if args.truth else None
    results = [ReconstructionResult.load(path) for path in args.results]
    if args.calibration:
        models = load_calibrations(args.calibration)
        results = [apply_calibration(result, models[result.method]) for result in results]
    report = plot_comparison_slab(
        truth,
        results,
        output=args.output,
        center_index=args.center_index,
        slab_layers=args.slab_layers,
        full_depth=args.full_depth,
        projection_axis=args.projection_axis,
    )
    if args.metrics_output:
        _write_json(report, args.metrics_output)
    print(args.output)


def command_plot3d(args: argparse.Namespace) -> None:
    truth = _load_truth(args.truth) if args.truth else None
    results = [ReconstructionResult.load(path) for path in args.results]
    if args.calibration:
        models = load_calibrations(args.calibration)
        results = [apply_calibration(result, models[result.method]) for result in results]
    report = plot_comparison_3d(
        truth,
        results,
        output=args.output,
        threshold_cm_inv=args.threshold_cm_inv,
        max_cubes=args.max_cubes,
        opacity=args.opacity,
        surface_only=args.surface_only,
    )
    if args.metrics_output:
        _write_json(report, args.metrics_output)
    print(args.output)


def command_validate(args: argparse.Namespace) -> None:
    """Validate spatial/selection identity for any nonempty result collection.

    Detector and legacy fingerprints have different meanings. All files must
    share one fingerprint scheme; hashes from different schemes are never mixed.
    Momentum policies are reported per method rather than assumed identical.
    """
    results = [ReconstructionResult.load(path) for path in args.results]
    if not results:
        raise ValueError("at least one reconstruction result is required")
    methods = {result.method for result in results}
    if len(methods) != len(results):
        raise ValueError("results contain duplicate method IDs")
    first = results[0].geometry
    truth = _load_truth(args.truth) if args.truth else None
    if truth is not None and truth.geometry != first:
        raise ValueError("truth and reconstruction results do not share identical geometry")
    detector_key, legacy_key = "detector_fingerprint_sha256", "track_fingerprint_sha256"
    if all(result.metadata.get(detector_key) for result in results):
        fingerprint_key = detector_key
    elif all(result.metadata.get(legacy_key) for result in results):
        fingerprint_key = legacy_key
    else:
        raise ValueError("results lack a shared detector or legacy fingerprint scheme")
    fingerprints = {result.metadata[fingerprint_key] for result in results}
    if len(fingerprints) != 1:
        raise ValueError("results do not record the same exact selected-track fingerprint")
    policies = {}
    for result in results:
        if result.geometry != first or result.inverse_radiation_length_cm_inv.shape != first.shape:
            raise ValueError("results do not share identical grid geometry")
        if not np.all(np.isfinite(result.inverse_radiation_length_cm_inv)):
            raise ValueError(f"{result.method} contains non-finite output")
        policies[result.method] = {
            "momentum_policy": result.metadata.get(
                "momentum_policy", result.metadata.get("momentum_feature_policy")
            ),
            "assumed_momentum_mev": result.metadata.get(
                "assumed_momentum_mev", result.metadata.get("momentum_estimate_mev")
            ),
            "truth_calibration_applied": result.metadata.get("truth_calibration_applied"),
        }
    _write_json(
        {
            "status": "success",
            "methods": sorted(methods),
            "geometry": first.metadata(),
            "fingerprint_scheme": fingerprint_key,
            fingerprint_key: fingerprints.pop(),
            "truth_geometry_checked": truth is not None,
            "truth_geometry_matches": True if truth is not None else None,
            "information_policy_by_method": policies,
            "scope": "geometry and selected observations; differing priors are reported, not equated",
            "spatial_contract": {
                "bounds_cm": [list(first.lower_cm), list(first.upper_cm)],
                "array_indexing": "volume[x,y,z]",
                "orientation": "+x,+y,+z increase with array indices",
            },
        },
        args.output,
    )
