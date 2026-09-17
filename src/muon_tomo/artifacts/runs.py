"""Unique run storage with explicit lifecycle and complete cache identities."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .storage import atomic_json

SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def config_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def cache_matches(
    manifest: dict, *, input_fingerprint: str, config_sha256: str, source_sha256: str
) -> bool:
    """Reuse only completed results with all declared identity components equal."""
    return manifest.get("status") == "completed" and all(
        manifest.get(k) == v
        for k, v in {
            "input_fingerprint": input_fingerprint,
            "config_sha256": config_sha256,
            "source_sha256": source_sha256,
        }.items()
    )


class RunDirectory:
    """Single-writer run. Completed/failed runs are immutable through this API."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "run.json").read_text())
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported run manifest schema")

    def _open(self) -> None:
        if self.manifest["status"] != "running":
            raise RuntimeError("a finished run cannot be changed; create a new run")

    def update(self, **fields) -> None:
        self._open()
        for key in ("schema_version", "run_id", "created_at", "status", "config_sha256"):
            if key in fields:
                raise ValueError(f"reserved run field: {key}")
        updated = {**self.manifest, **fields}
        atomic_json(self.path / "run.json", updated, overwrite=True)
        self.manifest = updated

    def write_json(self, relative: str | Path, value) -> Path:
        self._open()
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts or str(relative) == "run.json":
            raise ValueError("artifact path must stay inside the run and cannot replace run.json")
        target = self.path / relative
        if not target.resolve().is_relative_to(self.path.resolve()):
            raise ValueError("artifact path escapes run through a symbolic link")
        return atomic_json(target, value)

    def _finish(self, status: str, **fields) -> None:
        self._open()
        if fields:
            self.update(**fields)
        updated = {**self.manifest, "status": status, "finished_at": utc_now()}
        updated["artifacts"] = [
            str(p.relative_to(self.path))
            for p in sorted(self.path.rglob("*"))
            if p.is_file() and p.name != "run.json"
        ]
        inventory = {}
        for relative in updated["artifacts"]:
            artifact = self.path / relative
            if not artifact.resolve().is_relative_to(self.path.resolve()):
                inventory[relative] = {
                    "sha256": None,
                    "verification": "external_symlink_not_hashed",
                }
                continue
            try:
                digest = hashlib.sha256()
                with artifact.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                inventory[relative] = {
                    "size_bytes": artifact.stat().st_size,
                    "sha256": digest.hexdigest(),
                    "verification": "sha256",
                }
            except OSError as exc:
                inventory[relative] = {
                    "sha256": None,
                    "verification": "unavailable",
                    "error": str(exc),
                }
        updated["artifact_integrity"] = inventory
        atomic_json(self.path / "run.json", updated, overwrite=True)
        self.manifest = updated

    def complete(self, **fields) -> None:
        self._finish("completed", **fields)

    def fail(self, error) -> None:
        self._finish("failed", error=str(error))

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if self.manifest["status"] == "running":
            if kind is not None:
                self._finish(
                    "interrupted"
                    if issubclass(kind, (KeyboardInterrupt, SystemExit))
                    else "failed",
                    error=str(value),
                )
            else:
                self.complete()
        return False


def create_run(
    runs_dir: str | Path,
    *,
    metadata: dict | None = None,
    resolved_config: dict | None = None,
    run_id: str | None = None,
) -> RunDirectory:
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id must be a single safe directory name")
    directory = Path(runs_dir) / run_id
    config = resolved_config or {}
    manifest = {
        **(metadata or {}),
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "created_at": utc_now(),
        "config_sha256": config_hash(config),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    # Validate all JSON before reserving a run directory.
    json.dumps(manifest, allow_nan=False)
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "config.resolved.json", config)
    atomic_json(directory / "run.json", manifest)
    return RunDirectory(directory)


def list_runs(runs_dir: str | Path = "runs") -> list[dict]:
    """Read small manifests only; never open reconstruction arrays."""
    result = []
    for manifest in sorted(Path(runs_dir).glob("*/run.json"), reverse=True):
        try:
            row = json.loads(manifest.read_text())
            if not isinstance(row, dict) or row.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported schema")
        except (OSError, ValueError) as exc:
            row = {"run_id": manifest.parent.name, "status": "invalid_manifest", "error": str(exc)}
        result.append({**row, "path": str(manifest.parent)})
    return result
