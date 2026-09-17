"""Strict cache provenance for resumable research; legacy fits are never adopted."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ensure_protocol(path: str | Path, protocol: dict) -> str:
    """Write a new protocol or require the existing one to match exactly."""
    path = Path(path)
    if path.exists():
        stored = json.loads(path.read_text())
        if stored.get("cache_schema_version") != SCHEMA_VERSION:
            raise ValueError("Legacy study has no verifiable cache identity. Preserve it and use --results-dir NEW_DIRECTORY.")
        if object_sha256(stored) != object_sha256(protocol):
            raise ValueError("Study source/configuration/dependencies changed. Use --results-dir NEW_DIRECTORY; existing results are preserved.")
    else:
        if path.parent.exists() and any(path.parent.iterdir()):
            raise ValueError("Results directory contains unregistered artifacts. Use --results-dir NEW_DIRECTORY.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(protocol, stream, indent=2)
    return object_sha256(protocol)


def verify_protocol_sources(protocol_path: str | Path, root: str | Path) -> dict:
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol.get("cache_schema_version") != SCHEMA_VERSION:
        raise ValueError("Legacy study has no verifiable cache identity. Use --results-dir NEW_DIRECTORY.")
    base = Path(root)
    for name, digest in protocol["production_sha256"].items():
        if not (base / name).is_file() or file_sha256(base / name) != digest:
            raise ValueError(f"Study source changed: {name}; create a new results directory.")
    for name, digest in protocol["source_sha256"].items():
        path = base / "experiments/likelihood_physics_study" / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Study runner changed: {name}; create a new results directory.")
    return protocol


def input_identity(path: str | Path, protocol: dict) -> dict:
    """Hash measured arrays plus original selection IDs, never evaluation labels."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        fields = ("entry_position_cm", "entry_direction", "exit_position_cm", "exit_direction")
        count = len(data[fields[0]])
        for name in fields:
            array = np.ascontiguousarray(data[name])
            if array.shape != (count, 3):
                raise ValueError("detector inputs have inconsistent shapes")
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        key = "selected_rows" if "selected_rows" in data else "event_ids"
        if key not in data:
            raise ValueError("Cannot reuse detector inputs without saved original selection IDs")
        ids = np.asarray(data[key], dtype=np.int64)
        if ids.shape != (count,):
            raise ValueError("selected row IDs do not match detector arrays")
        digest.update(ids.tobytes())
        selection_hash = hashlib.sha256(ids.tobytes()).hexdigest()
    return {"schema_version": SCHEMA_VERSION, "protocol_sha256": object_sha256(protocol),
            "detector_selection_sha256": digest.hexdigest(), "selected_rows_sha256": selection_hash,
            "selected_tracks": count}


def verify_fit(directory: str | Path, identity: dict) -> dict:
    directory = Path(directory)
    result = json.loads((directory / "fit_complete.json").read_text())
    if result.get("cache_identity") != identity:
        raise ValueError("Cached fit input/selection/source/configuration does not match. Use --results-dir NEW_DIRECTORY.")
    files = result.get("artifact_sha256")
    if not files:
        raise ValueError("Cached fit lacks artifact hashes; preserve it as legacy.")
    for name, digest in files.items():
        path = directory / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Cached fit artifact missing or changed: {path}")
    return result
