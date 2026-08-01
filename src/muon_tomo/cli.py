"""Command-line workflow for inspecting, reconstructing, calibrating, and comparing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from .calibration import apply_calibration, fit_calibration, load_calibrations, save_calibrations
from .compare import compare_to_truth
from .geometry import GridGeometry
from .io import inspect_tfrecord, load_root_tracks, load_tfrecord_record
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
    calibration = load_calibrations(args.calibration) if args.calibration else {}
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
            )
        result = reconstructor.reconstruct(tracks)
        result.metadata.update(
            {
                "input_source": str(Path(args.input).resolve()),
                "input_format": _format(args.input, args.format),
                "record_index": args.record_index,
                "track_selection": tracks.metadata,
            }
        )
        if method in calibration:
            result = apply_calibration(result, calibration[method])
        target = result.save(output / f"{stem}_{method}.npz")
        created.append(str(target))
        print(target)
    _write_json(
        {"created": created, "geometry": geometry.metadata(), "tracks": len(tracks)},
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
    models = []
    report: dict[str, object] = {
        "dataset": str(Path(args.dataset).resolve()),
        "fit_records": fit_records,
        "validate_records": args.validate_records,
        "methods": {},
    }
    for method in ("munet", "poca", "likelihood"):
        pairs = []
        for record in fit_records:
            result = ReconstructionResult.load(
                Path(args.results_dir) / f"record_{record}_{method}.npz"
            )
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
            original = ReconstructionResult.load(
                Path(args.results_dir) / f"record_{record}_{method}.npz"
            )
            calibrated = apply_calibration(original, model)
            validations.append(
                {"record": record, **compare_to_truth(calibrated, truths[record])}
            )
        report["methods"][method] = {
            "slope": model.slope,
            "intercept": model.intercept,
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


def command_validate(args: argparse.Namespace) -> None:
    results = [ReconstructionResult.load(path) for path in args.results]
    methods = {x.method for x in results}
    expected = {"munet", "poca", "likelihood"}
    if methods != expected:
        raise ValueError(f"expected exactly {sorted(expected)}, got {sorted(methods)}")
    first = results[0].geometry
    for result in results:
        if result.geometry != first or result.inverse_radiation_length_cm_inv.shape != (64, 64, 64):
            raise ValueError("results do not share the mandatory 64^3 geometry")
        if not np.all(np.isfinite(result.inverse_radiation_length_cm_inv)):
            raise ValueError(f"{result.method} contains non-finite output")
    _write_json(
        {"status": "success", "methods": sorted(methods), "geometry": first.metadata()},
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
    recon.set_defaults(func=command_reconstruct)

    calibrate = sub.add_parser("calibrate", help="fit all outputs to funny 1/X0 truth")
    calibrate.add_argument("--dataset", required=True)
    calibrate.add_argument("--results-dir", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--config")
    calibrate.add_argument("--fit-records", type=int, nargs="+", default=[0])
    calibrate.add_argument("--validate-records", type=int, nargs="+", default=[1])
    calibrate.add_argument("--calibration-id", default="funny-v1")
    calibrate.set_defaults(func=command_calibrate)

    compare = sub.add_parser("compare", help="compare result files with canonical truth")
    compare.add_argument("--truth", required=True)
    compare.add_argument("results", nargs="+")
    compare.add_argument("--output")
    compare.set_defaults(func=command_compare)

    validate = sub.add_parser("validate", help="validate an aligned three-method result set")
    validate.add_argument("results", nargs="+")
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
