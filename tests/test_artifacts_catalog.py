"""Run immutability, registry relocation, and cache identity safeguards."""

import json
import shutil

import numpy as np
import pytest

from muon_tomo.artifacts.runs import cache_matches, config_hash, create_run, list_runs
from muon_tomo.artifacts.storage import atomic_json
from muon_tomo.data.catalog import DatasetCatalog
from muon_tomo.reconstruction import PoCAReconstructor
from muon_tomo.types import ReconstructionResult


def test_catalogue_registers_in_place_and_relocates(tmp_path):
    root = tmp_path / "first"
    root.mkdir()
    source = root / "tiny.tfrecord"
    source.write_bytes(b"opaque input, never parsed")
    registry = tmp_path / "catalog.json"
    catalog = DatasetCatalog(root=root, catalog_path=registry)
    entry = catalog.register(source, "tiny")
    assert entry["path"] == "tiny.tfrecord"
    assert entry["sha256"] is None
    assert catalog.resolve("tiny") == source
    assert source.read_bytes() == b"opaque input, never parsed"
    with pytest.raises(FileExistsError):
        catalog.register(source, "tiny")
    with pytest.raises(ValueError, match="parent"):
        catalog.register(source, "derived", parent_id="missing")
    moved = tmp_path / "second"
    shutil.copytree(root, moved)
    assert DatasetCatalog(root=moved, catalog_path=registry).resolve("tiny") == moved / source.name


def test_run_unique_atomic_and_final_state_immutable(tmp_path):
    run = create_run(tmp_path, run_id="sample", resolved_config={"iterations": 10})
    with pytest.raises(FileExistsError):
        create_run(tmp_path, run_id="sample")
    run.write_json("evaluation/metrics.json", {"mse": 0.5})
    with pytest.raises(FileExistsError):
        run.write_json("evaluation/metrics.json", {"mse": 3})
    with pytest.raises(ValueError):
        run.write_json("../escaped.json", {})
    with pytest.raises(ValueError):
        run.update(status="completed")
    run.complete()
    with pytest.raises(RuntimeError):
        run.update(extra=3)
    records = list_runs(tmp_path)
    assert len(records) == 1 and records[0]["status"] == "completed"
    assert "evaluation/metrics.json" in records[0]["artifacts"]


def test_context_records_failures_and_interruptions(tmp_path):
    with pytest.raises(ValueError):
        with create_run(tmp_path, run_id="failure"):
            raise ValueError("bad input")
    assert list_runs(tmp_path)[0]["status"] == "failed"
    with pytest.raises(KeyboardInterrupt):
        with create_run(tmp_path, run_id="interrupt"):
            raise KeyboardInterrupt()
    assert json.loads((tmp_path / "interrupt/run.json").read_text())["status"] == "interrupted"


def test_cache_requires_all_provenance_and_completed_status():
    manifest = {
        "status": "completed",
        "input_fingerprint": "sample-a",
        "config_sha256": config_hash({"iterations": 10}),
        "source_sha256": "implementation-v1",
    }
    identity = {k: manifest[k] for k in ("input_fingerprint", "config_sha256", "source_sha256")}
    assert cache_matches(manifest, **identity)
    for key in identity:
        assert not cache_matches(manifest, **{**identity, key: "changed"})
    assert not cache_matches({**manifest, "status": "running"}, **identity)


def test_result_storage_no_overwrite_and_schema_validation(tmp_path, synthetic_tracks):
    result = PoCAReconstructor().reconstruct(synthetic_tracks.subset(4))
    path = tmp_path / "poca.npz"
    result.save(path)
    with pytest.raises(FileExistsError):
        result.save(path)
    loaded = ReconstructionResult.load(path)
    np.testing.assert_array_equal(
        result.inverse_radiation_length_cm_inv, loaded.inverse_radiation_length_cm_inv
    )
    with np.load(path, allow_pickle=False) as data:
        payload = dict(data)
    metadata = json.loads(str(payload["metadata_json"]))
    metadata["schema_version"] = 999
    payload["metadata_json"] = np.asarray(json.dumps(metadata))
    future = tmp_path / "future.npz"
    np.savez_compressed(future, **payload)
    with pytest.raises(ValueError, match="unsupported"):
        ReconstructionResult.load(future)


def test_atomic_json_rejects_nonfinite_and_does_not_publish(tmp_path):
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        atomic_json(path, {"value": float("nan")})
    assert not path.exists()


def test_installed_catalogue_root_uses_explicit_workspace_or_cwd(tmp_path, monkeypatch):
    from muon_tomo.data import catalog

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MUON_TOMO_PROJECT_ROOT", raising=False)
    monkeypatch.setattr(
        catalog,
        "__file__",
        str(tmp_path / "venv/lib/python3.13/site-packages/muon_tomo/data/catalog.py"),
    )
    assert catalog.project_root() == tmp_path
    explicit = tmp_path / "workspace"
    monkeypatch.setenv("MUON_TOMO_PROJECT_ROOT", str(explicit))
    assert catalog.project_root() == explicit
    registry = tmp_path / "custom-catalogue.json"
    monkeypatch.setenv("MUON_TOMO_CATALOG", str(registry))
    assert catalog.DatasetCatalog().catalog_path == registry


def test_invalid_run_metadata_does_not_reserve_directory(tmp_path):
    with pytest.raises(ValueError):
        create_run(tmp_path, run_id="bad", resolved_config={"sigma": float("nan")})
    assert not (tmp_path / "bad").exists()
    run = create_run(tmp_path, run_id="good")
    with pytest.raises(ValueError):
        run.update(metric=float("nan"))
    assert "metric" not in run.manifest
    run.complete()
    assert run.manifest["artifact_integrity"]["config.resolved.json"]["verification"] == "sha256"
