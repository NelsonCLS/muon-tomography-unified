"""Muon tomography commands; algorithms and workflows are imported on demand."""

from __future__ import annotations

import argparse
import sys

from .config import DEFAULT_CONFIG  # compatibility import for existing callers
from .config import geometry_from_config as _geometry
from .config import load_config as _config

__all__ = ["DEFAULT_CONFIG", "_config", "_geometry", "build_parser", "main"]


def _legacy(name, *args, **kwargs):
    from .workflows import legacy

    return getattr(legacy, name)(*args, **kwargs)


def _format(*args, **kwargs):
    return _legacy("_format", *args, **kwargs)


def _load_input(*args, **kwargs):
    return _legacy("_load_input", *args, **kwargs)


def _write_json(*args, **kwargs):
    return _legacy("_write_json", *args, **kwargs)


def _track_fingerprint(*args, **kwargs):
    return _legacy("_track_fingerprint", *args, **kwargs)


def _save_truth(*args, **kwargs):
    return _legacy("_save_truth", *args, **kwargs)


def _load_truth(*args, **kwargs):
    return _legacy("_load_truth", *args, **kwargs)


def command_inspect(*args, **kwargs):
    return _legacy("command_inspect", *args, **kwargs)


def command_truth(*args, **kwargs):
    return _legacy("command_truth", *args, **kwargs)


def command_reconstruct(*args, **kwargs):
    return _legacy("command_reconstruct", *args, **kwargs)


def command_calibrate(*args, **kwargs):
    return _legacy("command_calibrate", *args, **kwargs)


def command_compare(*args, **kwargs):
    return _legacy("command_compare", *args, **kwargs)


def command_stability(*args, **kwargs):
    return _legacy("command_stability", *args, **kwargs)


def command_plot(*args, **kwargs):
    return _legacy("command_plot", *args, **kwargs)


def command_plot3d(*args, **kwargs):
    return _legacy("command_plot3d", *args, **kwargs)


def command_validate(*args, **kwargs):
    return _legacy("command_validate", *args, **kwargs)


def command_predict(args):
    from .workflows.predict import predict

    result = predict(
        args.input,
        record_index=args.record_index,
        limit=args.limit,
        seed=args.seed,
        method=args.method,
        runs_dir=args.runs_dir,
        config_path=args.config,
        catalog_path=args.catalog,
        open_outputs=args.open,
        evaluate_truth=not args.no_truth,
    )
    print(result)


def command_dataset_list(args):
    import json

    from .data.catalog import DatasetCatalog

    print(json.dumps(DatasetCatalog(catalog_path=args.catalog).list(), indent=2))


def command_dataset_register(args):
    import json

    from .data.catalog import DatasetCatalog

    row = DatasetCatalog(catalog_path=args.catalog).register(
        args.path,
        dataset_id=args.id,
        kind=args.kind,
        parent_id=args.parent_id,
    )
    print(json.dumps(row, indent=2))


def command_run_list(args):
    import json

    from .artifacts.runs import list_runs

    rows = list_runs(args.runs_dir)
    if not args.full:
        keys = (
            "run_id",
            "status",
            "created_at",
            "input_reference",
            "dataset_id",
            "record_index",
            "methods",
            "tracks_used",
            "path",
            "error",
        )
        rows = [{key: row[key] for key in keys if key in row} for row in rows]
    print(json.dumps(rows, indent=2))


def _common_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input")
    parser.add_argument("--format", choices=("auto", "tfrecord", "root"), default="auto")
    parser.add_argument("--config")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    from .reconstruction.registry import METHODS

    method_choices = ("all", *METHODS)
    parser = argparse.ArgumentParser(prog="muon-tomo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser(
        "predict", help="reconstruct a sample, evaluate, and create plots/viewer"
    )
    predict.add_argument("--input", required=True, help="dataset path or registered dataset ID")
    predict.add_argument("--record-index", type=int, default=0)
    predict.add_argument("--limit", type=int, default=8192)
    predict.add_argument("--seed", type=int, default=0)
    predict.add_argument("--method", choices=method_choices, default="all")
    predict.add_argument("--runs-dir", default="runs")
    predict.add_argument("--config")
    predict.add_argument("--catalog", help="optional dataset catalog JSON")
    predict.add_argument("--open", action="store_true", help="open the generated plot and viewer")
    predict.add_argument(
        "--no-truth", action="store_true", help="skip truth loading and evaluation"
    )
    predict.set_defaults(func=command_predict)

    datasets = sub.add_parser("dataset", help="register and locate datasets without moving them")
    dataset_sub = datasets.add_subparsers(dest="dataset_command", required=True)
    dataset_list = dataset_sub.add_parser("list", help="list registered datasets")
    dataset_list.add_argument("--catalog")
    dataset_list.set_defaults(func=command_dataset_list)
    register = dataset_sub.add_parser("register", help="register a file without modifying it")
    register.add_argument("path")
    register.add_argument("--id")
    register.add_argument(
        "--kind", choices=("imported", "derived", "simulated"), default="imported"
    )
    register.add_argument("--parent-id")
    register.add_argument("--catalog")
    register.set_defaults(func=command_dataset_register)

    runs = sub.add_parser("run", help="inspect saved prediction runs")
    run_sub = runs.add_subparsers(dest="run_command", required=True)
    run_list = run_sub.add_parser("list", help="list completed, failed, and running jobs")
    run_list.add_argument("--runs-dir", default="runs")
    run_list.add_argument(
        "--full", action="store_true", help="include complete provenance and artifact checksums"
    )
    run_list.set_defaults(func=command_run_list)

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
    recon.add_argument("--method", choices=method_choices, default="all")
    recon.add_argument("--output-dir", required=True)
    recon.add_argument("--calibration")
    recon.add_argument("--dosage", type=int, choices=(512, 1024))
    recon.add_argument("--weights")
    recon.add_argument("--max-chunks", type=int)
    recon.add_argument("--iterations", type=int)
    recon.add_argument("--likelihood-mode", choices=("full", "angular"))
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

    calibrate = sub.add_parser(
        "calibrate", help="legacy research only: fit outputs to truth (never used by predict)"
    )
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

    stability = sub.add_parser("stability", help="run a paired fixed detector-error stability test")
    stability.add_argument("--dataset", required=True)
    stability.add_argument("--output-dir", required=True)
    stability.add_argument("--config")
    stability.add_argument("--calibration")
    stability.add_argument("--record-index", type=int, default=0)
    stability.add_argument("--limit", type=int, default=8192)
    stability.add_argument("--seed", type=int, default=0)
    stability.add_argument("--error-seed", type=int, default=20260803)
    stability.add_argument("--munet-seed", type=int, default=20260803)
    stability.add_argument("--position-sigma-cm", type=float, default=0.29)
    stability.add_argument("--angular-sigma-deg", type=float, default=0.5)
    stability.add_argument("--iterations", type=int, default=10)
    stability.set_defaults(func=command_stability)

    plot = sub.add_parser("plot", help="plot truth and reconstruction slabs")
    plot.add_argument("--truth", help="optional evaluation truth")
    plot.add_argument("results", nargs="+")
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
    plot3d.add_argument("--truth", help="optional evaluation truth")
    plot3d.add_argument("results", nargs="+")
    plot3d.add_argument("--calibration")
    plot3d.add_argument(
        "--threshold-cm-inv",
        type=float,
        default=0.8,
        help="display voxels strictly above this 1/X0 threshold; lower values are fully transparent",
    )
    plot3d.add_argument(
        "--max-cubes",
        type=int,
        default=None,
        help="optional display cap; default shows all voxels passing the threshold",
    )
    plot3d.add_argument(
        "--opacity",
        type=float,
        default=0.6,
        help="maximum voxel opacity; lower-density cubes are more transparent",
    )
    plot3d.add_argument(
        "--surface-only",
        action="store_true",
        help="draw only thresholded surface voxels (interiors are shown by default)",
    )
    plot3d.add_argument("--output", required=True)
    plot3d.add_argument("--metrics-output")
    plot3d.set_defaults(func=command_plot3d)

    validate = sub.add_parser(
        "validate", help="validate shared geometry and track identity for saved results"
    )
    validate.add_argument("results", nargs="+")
    validate.add_argument("--truth")
    validate.add_argument("--output")
    validate.set_defaults(func=command_validate)
    from .simulation.cli import add_simulation_commands

    add_simulation_commands(sub)
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
