"""Configuration and orchestration contracts, without running scientific studies."""

import json
from pathlib import Path

import numpy as np
import pytest

from muon_tomo.cli import build_parser
from muon_tomo.config import load_config
from muon_tomo.reconstruction.registry import build_reconstructor, selected_methods
from muon_tomo.types import DetectorTracks, ReconstructionResult, TruthVolume
from muon_tomo.workflows.reconstruction import select_shared_tracks


def test_partial_config_and_effective_momentum_alias(tmp_path):
    path = tmp_path / "override.yaml"
    path.write_text(
        "physics:\n  effective_momentum_mev: 2000.0\npoca:\n  pooling_sigma_cm: 3.125\n"
    )
    config = load_config(path)
    assert config["physics"]["average_momentum_mev"] == 2000.0
    assert config["poca"]["pooling_sigma_cm"] == 3.125
    assert config["munet"]["dosage"] == 1024
    assert config["likelihood"]["iterations"] == 10
    assert "effective_momentum_mev" not in config["physics"]


@pytest.mark.parametrize(
    "text, message",
    [
        ("unknown: {}", "unknown configuration sections"),
        ("poca: {sigam: 3}", "unknown settings in poca"),
        ("likelihood: null", "must be a mapping"),
        ("physics: {effective_momentum_mev: 2, average_momentum_mev: 3}", "conflict"),
        ("likelihood: {iterations: 0}", "positive integer"),
        ("physics: {average_momentum_mev: .nan}", "finite"),
        ("poca: {pooling_sigma_cm: -1}", "nonnegative"),
        ("[]", "YAML mapping"),
    ],
)
def test_bad_configuration_fails_early(tmp_path, text, message):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_factory_preserves_production_method_settings():
    config = load_config()
    poca = build_reconstructor("poca", config)
    likelihood = build_reconstructor("likelihood", config)
    munet = build_reconstructor("munet", config)
    assert poca.coarse_grid_size == 32
    assert poca.pooling_sigma_cm == 6.25
    assert poca.average_momentum_mev == 1743.4601815051958
    assert likelihood.mode == "full"
    assert likelihood.iterations == 10
    assert likelihood.update_statistic == "median"
    assert munet.dosage == 1024
    assert munet.momentum_estimate_mev == poca.average_momentum_mev
    assert selected_methods("all") == ["munet", "poca", "likelihood"]
    with pytest.raises(ValueError, match="unknown"):
        build_reconstructor("typo", config)


def test_complete_chunk_selection_is_explicit(synthetic_tracks):
    config = load_config()
    selected = synthetic_tracks.take_first(63)
    assert select_shared_tracks(selected, ["poca"], config) is selected
    with pytest.raises(ValueError, match="at least 1024"):
        select_shared_tracks(selected, ["munet", "poca"], config)
    from muon_tomo.types import TrackTable

    # Repeat geometry to exercise the exact 700 -> 512 selection boundary.
    available = TrackTable(
        *(
            np.tile(getattr(synthetic_tracks, key), (8, 1))[:700]
            for key in (
                "entry_position_cm",
                "entry_direction",
                "exit_position_cm",
                "exit_direction",
            )
        ),
        np.tile(synthetic_tracks.momentum_mev, 8)[:700],
    )
    shared = select_shared_tracks(available, ["munet", "poca"], config, overrides={"dosage": 512})
    assert len(shared) == 512
    np.testing.assert_array_equal(shared.event_id, available.event_id[:512])


def test_predict_cli_has_no_truth_calibration_and_display_has_no_cap():
    parser = build_parser()
    args = parser.parse_args(["predict", "--input", "funny"])
    assert args.limit == 8192 and args.record_index == 0 and args.seed == 0
    assert not hasattr(args, "calibration")
    with pytest.raises(SystemExit):
        parser.parse_args(["predict", "--input", "funny", "--calibration", "fit.json"])
    plot = parser.parse_args(["plot3d", "--truth", "truth.npz", "poca.npz", "--output", "out.html"])
    assert plot.max_cubes is None and plot.threshold_cm_inv == 0.8


def test_predict_orders_inference_before_truth_and_records_outputs(
    tmp_path, monkeypatch, synthetic_tracks
):
    from muon_tomo import plotting, plotting3d
    from muon_tomo.geometry import GridGeometry
    from muon_tomo.workflows import predict as workflow

    source = tmp_path / "sample.tfrecord"
    source.write_bytes(b"test fixture; reader is replaced")
    tracks = synthetic_tracks.take_first(32)
    detector = DetectorTracks.from_legacy(tracks)
    geometry = GridGeometry()
    truth = TruthVolume(
        np.zeros(geometry.shape, dtype=np.int32), np.zeros(geometry.shape), geometry
    )
    events = []
    monkeypatch.setattr(workflow, "_source_identity", lambda: {"source_sha256": "fixture"})
    monkeypatch.setattr(workflow, "_load_detector_tracks", lambda *a, **k: detector)

    def reconstruct(allowed_tracks, methods, config):
        events.append("inference")
        assert methods == ["poca"]
        assert np.all(allowed_tracks.momentum_mev == 1743.4601815051958)
        zeros = np.zeros(geometry.shape)
        yield ReconstructionResult(
            "poca", zeros, zeros, zeros, np.ones(geometry.shape, bool), geometry
        )

    def load_truth(*args):
        assert events == ["inference"]
        assert list((tmp_path / "runs").glob("*/predictions/poca.npz"))
        events.append("truth")
        return truth

    def plot(truth_arg, results, *, output, **kwargs):
        assert events == ["inference", "truth"]
        Path(output).write_text("fixture plot")
        return {"methods": [result.method for result in results], **kwargs}

    monkeypatch.setattr(workflow, "reconstruct_selected", reconstruct)
    monkeypatch.setattr(workflow, "_load_evaluation_truth", load_truth)
    monkeypatch.setattr(plotting, "plot_comparison_slab", plot)
    monkeypatch.setattr(plotting3d, "plot_comparison_3d", plot)
    path = workflow.predict(source, method="poca", limit=32, runs_dir=tmp_path / "runs")
    manifest = json.loads((path / "run.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["truth_calibration_applied"] is False
    assert manifest["display"]["comparison"] == ">"
    assert manifest["display"]["max_cubes"] is None
    assert (path / "figures/cross_section.png").is_file()
    assert (path / "figures/viewer.html").is_file()
    assert (path / "selection.json").is_file()
    assert (path / "evaluation/metrics.json").is_file()
    np.testing.assert_array_equal(np.load(path / "selected_event_ids.npy"), tracks.event_id)


def test_predict_failure_keeps_partial_run(tmp_path, monkeypatch, synthetic_tracks):
    from muon_tomo.workflows import predict as workflow

    source = tmp_path / "sample.tfrecord"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(workflow, "_source_identity", lambda: {})
    monkeypatch.setattr(
        workflow,
        "_load_detector_tracks",
        lambda *a, **k: DetectorTracks.from_legacy(synthetic_tracks.take_first(2)),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(workflow, "reconstruct_selected", fail)
    with pytest.raises(RuntimeError, match="solver failed"):
        workflow.predict(source, method="poca", runs_dir=tmp_path / "runs")
    saved = list((tmp_path / "runs").glob("*/run.json"))
    assert len(saved) == 1
    manifest = json.loads(saved[0].read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == "solver failed"


def test_predict_rejects_per_track_policy_before_run(tmp_path):
    from muon_tomo.workflows.predict import predict

    config = tmp_path / "per-track.yaml"
    config.write_text("physics: {classical_momentum_policy: per-track}")
    with pytest.raises(ValueError, match="fixed prior"):
        predict("missing-dataset", config_path=config, runs_dir=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_predict_no_truth_never_reads_labels(tmp_path, monkeypatch, synthetic_tracks):
    from muon_tomo import plotting, plotting3d
    from muon_tomo.geometry import GridGeometry
    from muon_tomo.workflows import predict as workflow

    source = tmp_path / "measurements.tfrecord"
    source.write_bytes(b"no labels in input")
    monkeypatch.setattr(workflow, "_source_identity", lambda: {})
    monkeypatch.setattr(
        workflow,
        "_load_detector_tracks",
        lambda *a, **k: DetectorTracks.from_legacy(synthetic_tracks.take_first(2)),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("truth loader must never be called")

    monkeypatch.setattr(workflow, "_load_evaluation_truth", forbidden)
    geometry = GridGeometry()
    zeros = np.zeros(geometry.shape)
    result = ReconstructionResult(
        "poca", zeros, zeros, zeros, np.ones(geometry.shape, bool), geometry
    )
    monkeypatch.setattr(workflow, "reconstruct_selected", lambda *a: iter([result]))

    def plot(truth, results, *, output, **kwargs):
        assert truth is None
        Path(output).write_text("fixture")
        return {}

    monkeypatch.setattr(plotting, "plot_comparison_slab", plot)
    monkeypatch.setattr(plotting3d, "plot_comparison_3d", plot)
    path = workflow.predict(source, method="poca", runs_dir=tmp_path / "runs", evaluate_truth=False)
    assert not (path / "evaluation/truth.npz").exists()
    manifest = json.loads((path / "run.json").read_text())
    assert manifest["truth_available"] is False
    assert manifest["truth_evaluation_requested"] is False


def test_registered_factory_extends_cli_without_dispatch_edits(monkeypatch):
    from muon_tomo.reconstruction.registry import METHODS, MethodSpec

    sentinel = object()
    monkeypatch.setitem(
        METHODS,
        "future",
        MethodSpec(
            "future",
            "Future method",
            "experimental",
            "test-v1",
            lambda config, geometry, overrides: sentinel,
            fixed_geometry=False,
        ),
    )
    assert build_reconstructor("future", load_config()) is sentinel
    assert "future" not in selected_methods("all")
    assert selected_methods("future") == ["future"]
    args = build_parser().parse_args(["predict", "--input", "measurements", "--method", "future"])
    assert args.method == "future"


def test_validate_new_single_method_and_legacy_schemes(tmp_path):
    from argparse import Namespace

    from muon_tomo.geometry import GridGeometry
    from muon_tomo.workflows.legacy import command_validate

    geometry = GridGeometry()
    zeros = np.zeros(geometry.shape)
    result = ReconstructionResult(
        "poca",
        zeros,
        zeros,
        zeros,
        np.ones(geometry.shape, bool),
        geometry,
        metadata={
            "detector_fingerprint_sha256": "new-identity",
            "momentum_policy": "mean",
            "assumed_momentum_mev": 1743.0,
        },
    )
    new_path = result.save(tmp_path / "new.npz")
    output = tmp_path / "validation.json"
    command_validate(Namespace(results=[new_path], truth=None, output=output))
    report = json.loads(output.read_text())
    assert report["fingerprint_scheme"] == "detector_fingerprint_sha256"
    assert report["methods"] == ["poca"]
    result.metadata = {"track_fingerprint_sha256": "legacy-identity"}
    old_path = result.save(tmp_path / "old.npz")
    command_validate(Namespace(results=[old_path], truth=None, output=output))
    assert json.loads(output.read_text())["fingerprint_scheme"] == "track_fingerprint_sha256"


def test_config_weight_path_is_relative_to_config(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("munet: {weights: checkpoints/weights.h5}")
    config = load_config(config_path)
    assert config["munet"]["weights"] == str(tmp_path / "checkpoints/weights.h5")
    assert build_reconstructor("munet", config).weights_path == tmp_path / "checkpoints/weights.h5"


def test_new_prediction_seeds_munet_without_changing_legacy_factory(monkeypatch, synthetic_tracks):
    import sys
    from types import SimpleNamespace

    from muon_tomo.geometry import GridGeometry
    from muon_tomo.workflows import reconstruction as workflow

    calls = []
    monkeypatch.setitem(
        sys.modules, "tensorflow", SimpleNamespace(random=SimpleNamespace(set_seed=calls.append))
    )
    geometry = GridGeometry()
    zeros = np.zeros(geometry.shape)
    result = ReconstructionResult(
        "munet", zeros, zeros, zeros, np.ones(geometry.shape, bool), geometry
    )
    monkeypatch.setattr(
        workflow,
        "build_reconstructor",
        lambda *args: SimpleNamespace(reconstruct=lambda tracks: result),
    )
    config = load_config()
    config["munet"]["inference_seed"] = 17
    predicted = list(workflow.reconstruct_selected(synthetic_tracks, ["munet"], config))
    assert calls == [17]
    assert predicted[0].metadata["inference_seed"] == 17


def test_inference_seed_rejects_invalid_values(tmp_path):
    path = tmp_path / "bad-seed.yaml"
    path.write_text("munet: {inference_seed: -1}")
    with pytest.raises(ValueError, match="nonnegative integer"):
        load_config(path)


def test_dosage_only_config_selects_matching_frozen_weights(tmp_path):
    path = tmp_path / "dose.yaml"
    path.write_text("munet: {dosage: 512}")
    config = load_config(path)
    assert config["munet"]["weights"] == "model_512.weights.h5"
    assert build_reconstructor("munet", config).weights_path.name == "model_512.weights.h5"
    path.write_text("munet: {dosage: 1024, weights: model_512.weights.h5}")
    with pytest.raises(ValueError, match="must match"):
        load_config(path)
    # The historical dosage CLI override still picks the matching package file.
    assert (
        build_reconstructor("munet", load_config(), overrides={"dosage": 512}).weights_path.name
        == "model_512.weights.h5"
    )


def test_single_record_input_rejects_misleading_record_index(tmp_path):
    from muon_tomo.workflows.predict import predict

    source = tmp_path / "measurements.npz"
    source.write_bytes(b"validation occurs before reading the archive")
    with pytest.raises(ValueError, match="record_index applies only"):
        predict(source, record_index=3, runs_dir=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()
