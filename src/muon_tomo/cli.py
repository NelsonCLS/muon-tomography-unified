"""Command-line workflow for inspecting, reconstructing, calibrating, and comparing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from .calibration import apply_calibration, fit_calibration, load_calibrations, save_calibrations
from .compare import compare_to_truth
from .geometry import GridGeometry
from .io import inspect_tfrecord, load_root_tracks, load_tfrecord_record
from .plotting import plot_comparison_slab
from .plotting3d import plot_comparison_3d
from .reconstruction import LikelihoodReconstructor, MuNetReconstructor, PoCAReconstructor
from .types import ReconstructionResult, TruthVolume

DEFAULT_CONFIG = Path(__file__).parents[2] / "configs" / "default.yaml"


def _config(path: str | None) -> dict:
    target = Path(path) if path else DEFAULT_CONFIG
    if not target.is_file():
        raise FileNotFoundError(f"configuration not found: {target}")
    return yaml.safe_load(target.read_text())


def _geometry(cfg: dict) -> GridGeometry:
    values = cfg["geometry"]
    return GridGeometry(
        tuple(values["lower_cm"]), tuple(values["upper_cm"]),
        tuple(values["shape"]), values["axis_order"],
    )


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


def _save_truth(truth: TruthVolume, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **truth.metadata,
        "schema_version": 1,
        "primary_quantity": "inverse_radiation_length",
        "primary_unit": "cm^-1",
        "geometry": truth.geometry.metadata(),
    }
    np.savez_compressed(
        target,
        material_labels=truth.material_labels,
        inverse_radiation_length_cm_inv=truth.inverse_radiation_length_cm_inv,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return target


def _load_truth(path: str | Path) -> TruthVolume:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        gm = metadata["geometry"]
        geometry = GridGeometry(
            tuple(gm["lower_cm"]), tuple(gm["upper_cm"]),
            tuple(gm["shape"]), gm["axis_order"],
        )
        return TruthVolume(
            data["material_labels"], data["inverse_radiation_length_cm_inv"], geometry, metadata
        )


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
    _, truth = load_tfrecord_record(
        args.input, args.record_index, geometry=geometry, limit=0
    )
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
    if "munet" in methods:
        dosage = args.dosage or int(cfg["munet"]["dosage"])
        chunks = len(tracks) // dosage
        if args.max_chunks is not None:
            chunks = min(chunks, args.max_chunks)
        if chunks < 1:
            raise ValueError(f"Mu-Net {dosage} needs at least {dosage} tracks")
        tracks = tracks.take_first(chunks * dosage)
    track_fingerprint = _track_fingerprint(tracks)
    calibration = load_calibrations(args.calibration) if args.calibration else {}
    momentum_policy = (
        args.classical_momentum_policy
        or cfg["physics"].get("classical_momentum_policy", "mean")
    )
    average_momentum_mev = (
        args.average_momentum_mev
        if args.average_momentum_mev is not None
        else float(cfg["physics"].get("average_momentum_mev", 3000.0))
    )
    created = []
    for method in methods:
        if method == "munet":
            values = cfg["munet"]
            reconstructor = MuNetReconstructor(
                geometry,
                dosage=args.dosage or int(values["dosage"]),
                weights_path=args.weights,
                position_scale_cm=float(values["position_scale_cm"]),
                position_offset=float(values["position_offset"]),
                max_chunks=args.max_chunks,
            )
        elif method == "poca":
            reconstructor = PoCAReconstructor(
                geometry,
                highland_constant_mev=float(cfg["physics"]["highland_constant_mev"]),
                minimum_coverage=int(cfg["poca"]["minimum_coverage"]),
                momentum_policy=momentum_policy,
                average_momentum_mev=average_momentum_mev,
            )
        else:
            values = cfg["likelihood"]
            reconstructor = LikelihoodReconstructor(
                geometry,
                iterations=args.iterations or int(values["iterations"]),
                reference_momentum_mev=float(cfg["physics"]["reference_momentum_mev"]),
                highland_constant_mev=float(cfg["physics"]["highland_constant_mev"]),
                initial_inverse_radiation_length_cm_inv=float(
                    values["initial_inverse_radiation_length_cm_inv"]
                ),
                minimum_coverage=int(values["minimum_coverage"]),
                epsilon=float(values["epsilon"]),
                mode=args.likelihood_mode,
                momentum_policy=momentum_policy,
                average_momentum_mev=average_momentum_mev,
                update_statistic=(
                    args.likelihood_update or values.get("update_statistic", "median")
                ),
                angular_resolution_rad=float(values["angular_resolution_rad"]),
                position_resolution_cm=float(values["position_resolution_cm"]),
                maximum_inverse_radiation_length_cm_inv=float(
                    values["maximum_inverse_radiation_length_cm_inv"]
                ),
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
        _, truths[record] = load_tfrecord_record(
            args.dataset, record, geometry=geometry, limit=0
        )
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
        raise ValueError(
            "calibration inputs must share one fixed mean-momentum policy and value"
        )
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
            validations.append(
                {"record": record, **compare_to_truth(calibrated, truths[record])}
            )
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
    rows = [
        compare_to_truth(ReconstructionResult.load(path), truth) for path in args.results
    ]
    _write_json(rows, args.output)


def command_plot(args: argparse.Namespace) -> None:
    truth = _load_truth(args.truth)
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
    truth = _load_truth(args.truth)
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
        surface_only=not args.include_interior,
    )
    if args.metrics_output:
        _write_json(report, args.metrics_output)
    print(args.output)


def command_validate(args: argparse.Namespace) -> None:
    results = [ReconstructionResult.load(path) for path in args.results]
    methods = {x.method for x in results}
    expected = {"munet", "poca", "likelihood"}
    if methods != expected:
        raise ValueError(f"expected exactly {sorted(expected)}, got {sorted(methods)}")
    first = results[0].geometry
    truth = _load_truth(args.truth) if args.truth else None
    if truth is not None and truth.geometry != first:
        raise ValueError("truth and reconstruction results do not share identical geometry")
    fingerprints = {result.metadata.get("track_fingerprint_sha256") for result in results}
    if None in fingerprints or len(fingerprints) != 1:
        raise ValueError("results do not record the same exact selected-track fingerprint")
    for result in results:
        if result.geometry != first or result.inverse_radiation_length_cm_inv.shape != (64, 64, 64):
            raise ValueError("results do not share the mandatory 64^3 geometry")
        if not np.all(np.isfinite(result.inverse_radiation_length_cm_inv)):
            raise ValueError(f"{result.method} contains non-finite output")
    by_method = {result.method: result for result in results}
    classical = [by_method["poca"], by_method["likelihood"]]
    policies = {result.metadata.get("momentum_policy") for result in classical}
    if policies != {"mean"}:
        raise ValueError(
            "PoCA and likelihood must both declare the shared mean-momentum policy"
        )
    assumed = [result.metadata.get("assumed_momentum_mev") for result in classical]
    if None in assumed or not np.isclose(float(assumed[0]), float(assumed[1])):
        raise ValueError("PoCA and likelihood do not use the same assumed momentum")
    _write_json(
        {
            "status": "success",
            "methods": sorted(methods),
            "geometry": first.metadata(),
            "track_fingerprint_sha256": fingerprints.pop(),
            "truth_geometry_checked": truth is not None,
            "truth_geometry_matches": True if truth is not None else None,
            "classical_momentum_policy": "mean",
            "assumed_momentum_mev": float(assumed[0]),
            "spatial_contract": {
                "canonical_bounds_cm": [list(first.lower_cm), list(first.upper_cm)],
                "array_indexing": "volume[x,y,z]",
                "orientation": "+x,+y,+z increase with array indices",
                "munet_normalized_bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            },
        },
        args.output,
    )


def _common_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input")
    parser.add_argument("--format", choices=("auto", "tfrecord", "root"), default="auto")
    parser.add_argument("--config")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="muon-tomo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="inspect an input without reconstructing it")
    _common_input(inspect)
    inspect.add_argument("--output")
    inspect.set_defaults(func=command_inspect)

    truth = sub.add_parser("truth", help="export TFRecord material truth as canonical 1/X0")
    _common_input(truth)
    truth.add_argument("--output", required=True)
    truth.set_defaults(func=command_truth)

    recon = sub.add_parser("reconstruct", help="run one method or all three")
    _common_input(recon)
    recon.add_argument("--method", choices=("all", "munet", "poca", "likelihood"), default="all")
    recon.add_argument("--output-dir", required=True)
    recon.add_argument("--calibration")
    recon.add_argument("--dosage", type=int, choices=(512, 1024))
    recon.add_argument("--weights")
    recon.add_argument("--max-chunks", type=int)
    recon.add_argument("--iterations", type=int)
    recon.add_argument("--likelihood-mode", choices=("full", "angular"), default="full")
    recon.add_argument("--likelihood-update", choices=("mean", "median"))
    recon.add_argument(
        "--classical-momentum-policy",
        choices=("mean", "per-track"),
        help="classical-method momentum information; default comes from config",
    )
    recon.add_argument(
        "--average-momentum-mev",
        type=float,
        help="fixed population-average momentum used by both classical methods",
    )
    recon.set_defaults(func=command_reconstruct)

    calibrate = sub.add_parser("calibrate", help="fit all outputs to funny 1/X0 truth")
    calibrate.add_argument("--dataset", required=True)
    calibrate.add_argument("--results-dir", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--config")
    calibrate.add_argument("--fit-records", type=int, nargs="+", default=[0])
    calibrate.add_argument("--validate-records", type=int, nargs="+", default=[1])
    calibrate.add_argument("--calibration-id", default="funny-v4-robust-mlsd")
    calibrate.set_defaults(func=command_calibrate)

    compare = sub.add_parser("compare", help="compare result files with canonical truth")
    compare.add_argument("--truth", required=True)
    compare.add_argument("results", nargs="+")
    compare.add_argument("--output")
    compare.set_defaults(func=command_compare)

    plot = sub.add_parser("plot", help="plot truth and three calibrated reconstruction slabs")
    plot.add_argument("--truth", required=True)
    plot.add_argument("results", nargs=3)
    plot.add_argument("--calibration")
    plot.add_argument("--center-index", type=int, default=32)
    plot.add_argument("--slab-layers", type=int, default=7)
    plot.add_argument(
        "--full-depth",
        action="store_true",
        help="average all 64 layers along the selected projection axis",
    )
    plot.add_argument(
        "--projection-axis",
        choices=("x", "y", "z"),
        default="z",
        help="axis to average through; the other two axes form the 64x64 plot",
    )
    plot.add_argument("--output", required=True)
    plot.add_argument("--metrics-output")
    plot.set_defaults(func=command_plot)

    plot3d = sub.add_parser(
        "plot3d",
        help="write a rotatable truth/MuNet/PoCA/likelihood voxel-cube viewer",
    )
    plot3d.add_argument("--truth", required=True)
    plot3d.add_argument("results", nargs=3)
    plot3d.add_argument("--calibration")
    plot3d.add_argument(
        "--threshold-cm-inv",
        type=float,
        default=0.1,
        help="display voxels at or above this 1/X0 threshold",
    )
    plot3d.add_argument(
        "--max-cubes",
        type=int,
        default=12000,
        help="maximum displayed cubes per volume",
    )
    plot3d.add_argument(
        "--opacity",
        type=float,
        default=0.6,
        help="maximum voxel opacity; lower-density cubes are more transparent",
    )
    plot3d.add_argument(
        "--include-interior",
        action="store_true",
        help="include interior thresholded voxels; surface cubes are the default",
    )
    plot3d.add_argument("--output", required=True)
    plot3d.add_argument("--metrics-output")
    plot3d.set_defaults(func=command_plot3d)

    validate = sub.add_parser("validate", help="validate an aligned three-method result set")
    validate.add_argument("results", nargs="+")
    validate.add_argument("--truth")
    validate.add_argument("--output")
    validate.set_defaults(func=command_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
