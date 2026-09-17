"""One reproducible prediction run: select, reconstruct, evaluate, then display.

Detector geometry enters reconstruction. Optional truth is read only after all
predictions have been saved. The production fixed prior and algorithms remain
unchanged; this workflow performs no truth-fitted calibration or tuning.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import sys
from pathlib import Path

import numpy as np

from ..artifacts.results import save_truth
from ..artifacts.runs import create_run
from ..config import geometry_from_config, load_config
from ..data.catalog import DatasetCatalog
from ..reconstruction.registry import METHODS, selected_methods
from .reconstruction import reconstruct_selected, select_shared_tracks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity() -> dict:
    """Small package-source hashes, independent of datasets and generated outputs."""
    package = Path(__file__).resolve().parents[1]
    sources = {
        str(path.relative_to(package)): _sha256(path) for path in sorted(package.rglob("*.py"))
    }
    digest = hashlib.sha256()
    for name, checksum in sources.items():
        digest.update(f"{name}:{checksum}\n".encode())
    versions = {}
    for name in ("numpy", "scipy", "tensorflow", "matplotlib", "plotly"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        package_version = importlib.metadata.version("muon-tomography-unified")
    except importlib.metadata.PackageNotFoundError:
        package_version = "uninstalled-source-tree"
    repository = package.parents[1]
    revision = None
    if (repository / ".git").exists():
        import subprocess

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                capture_output=True,
                text=True,
                check=False,
            )
            revision = result.stdout.strip() if result.returncode == 0 else None
        except OSError:
            pass
    return {
        "source_sha256": digest.hexdigest(),
        "source_files": sources,
        "dependencies": versions,
        "source_version": {
            "package_version": package_version,
            "git_commit": revision,
            "identity_basis": "actual package source hashes at run start",
        },
    }


def _load_detector_tracks(path: Path, *, record_index: int, geometry, limit: int, seed: int):
    if path.suffix.lower() == ".tfrecord":
        from ..io.tfrecord import load_tfrecord_detector_tracks

        return load_tfrecord_detector_tracks(
            path,
            record_index,
            geometry=geometry,
            limit=limit,
            seed=seed,
        )
    if path.suffix.lower() == ".root":
        from ..io.root import load_root_detector_tracks

        return load_root_detector_tracks(path, geometry=geometry, limit=limit, seed=seed)
    if path.suffix.lower() == ".npz":
        from ..io.detector import load_detector_npz

        return load_detector_npz(path, limit=limit, seed=seed)
    raise ValueError("prediction supports .tfrecord, .root, and detector .npz inputs")


def _load_evaluation_truth(path: Path, record_index: int, geometry):
    if path.suffix.lower() != ".tfrecord":
        return None
    from ..io.tfrecord import load_tfrecord_truth

    return load_tfrecord_truth(path, record_index, geometry=geometry)


def _prior_provenance(value: float) -> str:
    if value == 1743.4601815051958:
        return "legacy fixed scalar originally derived from Funny0 simulation population"
    return "user configured fixed scalar; external provenance not verified"


def _open_outputs(paths: list[Path]) -> None:
    if sys.platform == "darwin":
        import subprocess

        subprocess.run(["open", *(str(path.resolve()) for path in paths)], check=True)
    else:
        import webbrowser

        for path in paths:
            webbrowser.open(path.resolve().as_uri())


def predict(
    input_or_id: str | Path,
    *,
    record_index: int = 0,
    limit: int = 8192,
    seed: int = 0,
    method: str = "all",
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    open_outputs: bool = False,
    evaluate_truth: bool = True,
) -> Path:
    """Create a unique complete run, or preserve a marked failed/interrupted run."""
    if record_index < 0 or limit <= 0 or seed < 0:
        raise ValueError("record_index and seed must be nonnegative; limit must be positive")
    config = load_config(config_path)
    if config["physics"]["classical_momentum_policy"] != "mean":
        raise ValueError(
            "predict uses a fixed prior only; per-track momentum is a legacy research mode"
        )
    methods = selected_methods(method)
    geometry = geometry_from_config(config)
    catalog = DatasetCatalog(catalog_path=catalog_path)
    source = catalog.resolve(input_or_id)
    if source.suffix.lower() != ".tfrecord" and record_index != 0:
        raise ValueError("record_index applies only to TFRecord inputs; ROOT/NPZ inputs use 0")
    try:
        dataset_entry = catalog.get(str(input_or_id))
    except KeyError:
        dataset_entry = None
    dataset_id = dataset_entry["dataset_id"] if dataset_entry else None
    priors = {}
    for name in methods:
        value = (
            config["munet"]["momentum_estimate_mev"]
            if name == "munet"
            else config["physics"]["average_momentum_mev"]
        )
        priors[name] = {"effective_momentum_mev_c": value, "provenance": _prior_provenance(value)}
    stat = source.stat()
    run = create_run(
        runs_dir,
        resolved_config=config,
        metadata={
            "workflow": "predict-v1",
            "input": str(source),
            "input_reference": str(input_or_id),
            "input_size_bytes": stat.st_size,
            "input_mtime_ns": stat.st_mtime_ns,
            "dataset_id": dataset_id,
            "dataset_version": dataset_entry.get("version") if dataset_entry else None,
            "prior_by_method": priors,
            "munet_inference_seed": config["munet"].get("inference_seed", 0)
            if "munet" in methods
            else None,
            "input_identity": "path/stat plus selected detector geometry fingerprint; no full-file hash",
            "record_index": record_index,
            "requested_tracks": limit,
            "selection_seed": seed,
            "methods": {name: METHODS[name].implementation for name in methods},
            "method_status": {name: METHODS[name].status for name in methods},
            "truth_used_for_inference": False,
            "truth_calibration_applied": False,
            "truth_evaluation_requested": evaluate_truth,
            "momentum_policy": "fixed inverse-square effective momentum",
            "effective_momentum_mev_c": config["physics"]["average_momentum_mev"],
            "prior_provenance": _prior_provenance(config["physics"]["average_momentum_mev"]),
            **_source_identity(),
        },
    )
    with run:
        detectors = _load_detector_tracks(
            source,
            record_index=record_index,
            geometry=geometry,
            limit=limit,
            seed=seed,
        )
        tracks = detectors.to_legacy(assumed_momentum_mev=config["physics"]["average_momentum_mev"])
        selected_count = len(tracks)
        tracks = select_shared_tracks(tracks, methods, config)
        from ..data.selection import selection_manifest

        selection = selection_manifest(
            tracks,
            dataset_id=dataset_id,
            record_index=record_index,
            input_path=source,
            requested_tracks=limit,
            dataset_version=dataset_entry.get("version") if dataset_entry else None,
        )
        fingerprint = selection["detector_fingerprint_sha256"]
        run.write_json("selection.json", selection)
        run.update(
            tracks_selected=selected_count,
            tracks_used=len(tracks),
            tracks_discarded_for_complete_chunks=selected_count - len(tracks),
            selection_policy="same selected tracks for all methods; complete MuNet chunks when included",
            detector_fingerprint_sha256=fingerprint,
            input_fingerprint=fingerprint,
            selected_ids_path="selected_event_ids.npy",
            geometry=geometry.metadata(),
        )
        np.save(run.path / "selected_event_ids.npy", tracks.event_id, allow_pickle=False)
        prediction_dir = run.path / "predictions"
        prediction_dir.mkdir()
        results = []
        for result in reconstruct_selected(tracks, methods, config):
            result.metadata.update(
                run_id=run.manifest["run_id"],
                input_source=str(source),
                record_index=record_index,
                track_count=len(tracks),
                detector_fingerprint_sha256=fingerprint,
                track_selection=tracks.metadata,
                truth_calibration_applied=False,
                truth_used_for_inference=False,
                prior_provenance=priors[result.method]["provenance"],
                dataset_id=dataset_id,
                dataset_version=run.manifest["dataset_version"],
                algorithm_implementation=METHODS[result.method].implementation,
                algorithm_status=METHODS[result.method].status,
            )
            if result.method == "munet":
                weight_path = Path(result.metadata["weights"])
                result.metadata["weights_sha256"] = _sha256(weight_path)
                run.update(munet_weights_sha256=result.metadata["weights_sha256"])
            result.save(prediction_dir / f"{result.method}.npz")
            results.append(result)
            run.update(methods_completed=[item.method for item in results])

        # Truth is first requested here, after every numerical prediction is fixed.
        truth = _load_evaluation_truth(source, record_index, geometry) if evaluate_truth else None
        metrics = []
        if truth is not None:
            from ..compare import compare_to_truth

            save_truth(truth, run.path / "evaluation/truth.npz")
            metrics = [compare_to_truth(result, truth) for result in results]
        run.write_json(
            "evaluation/metrics.json",
            {
                "truth_available": truth is not None,
                "evaluation_stage": "after_prediction",
                "metrics": metrics,
                "mse_unit": "cm^-2",
            },
        )
        from ..plotting import plot_comparison_slab
        from ..plotting3d import plot_comparison_3d

        plots = run.path / "figures"
        plots.mkdir()
        section_path, viewer_path = plots / "cross_section.png", plots / "viewer.html"
        section_report = plot_comparison_slab(truth, results, output=section_path)
        viewer_report = plot_comparison_3d(
            truth,
            results,
            output=viewer_path,
            threshold_cm_inv=0.8,
            max_cubes=None,
            surface_only=False,
        )
        run.write_json("figures/section.json", section_report)
        run.write_json("figures/viewer.json", viewer_report)
        run.complete(
            truth_available=truth is not None,
            display={
                "threshold_cm_inv": 0.8,
                "comparison": ">",
                "max_cubes": None,
                "surface_only": False,
            },
        )
    if open_outputs:
        _open_outputs([section_path, viewer_path])
    return run.path
