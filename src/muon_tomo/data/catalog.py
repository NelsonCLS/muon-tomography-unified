"""Register existing datasets in place without moving or scanning their contents."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..artifacts.storage import atomic_json

SCHEMA_VERSION = 1


def project_root() -> Path:
    """Locate a workspace, never mistake an installed wheel for a project root."""
    explicit = os.environ.get("MUON_TOMO_PROJECT_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    cwd = Path.cwd().resolve()
    source = Path(__file__).resolve()
    for candidate in (cwd, *cwd.parents, *source.parents):
        marker = candidate / "pyproject.toml"
        if marker.is_file() and 'name = "muon-tomography-unified"' in marker.read_text():
            return candidate
    return cwd


class DatasetCatalog:
    def __init__(self, root: str | Path | None = None, catalog_path: str | Path | None = None):
        self.root = Path(root or os.environ.get("MUON_TOMO_DATA_ROOT") or project_root()).resolve()
        self.catalog_path = (
            Path(catalog_path or os.environ["MUON_TOMO_CATALOG"]).expanduser()
            if catalog_path or os.environ.get("MUON_TOMO_CATALOG")
            else project_root() / "data/catalog/datasets.json"
        )

    def _read(self) -> dict:
        if not self.catalog_path.exists():
            return {"schema_version": SCHEMA_VERSION, "datasets": []}
        data = json.loads(self.catalog_path.read_text())
        if data.get("schema_version") != SCHEMA_VERSION or not isinstance(
            data.get("datasets"), list
        ):
            raise ValueError("unsupported dataset catalogue schema")
        return data

    def list(self) -> list[dict]:
        return self._read()["datasets"]

    def get(self, dataset_id: str) -> dict:
        for entry in self.list():
            if entry["dataset_id"] == dataset_id:
                return entry
        raise KeyError(f"unknown dataset ID: {dataset_id}")

    def resolve(self, input_or_id: str | Path) -> Path:
        candidate = Path(input_or_id).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        try:
            item = self.get(str(input_or_id))
        except KeyError as exc:
            rooted = self.root / candidate
            if rooted.is_file():
                return rooted.resolve()
            raise FileNotFoundError(f"No existing file or dataset ID: {input_or_id}") from exc
        path = Path(item["path"])
        path = path if path.is_absolute() else self.root / path
        if not path.is_file():
            raise FileNotFoundError(f"Dataset {input_or_id} is registered but missing: {path}")
        return path.resolve()

    def register(
        self,
        path: str | Path,
        dataset_id: str | None = None,
        kind: str = "imported",
        parent_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(target)
        dataset_id = dataset_id or target.stem.replace("_", "-")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", dataset_id):
            raise ValueError("dataset_id must contain only letters, digits, _, ., :, -")
        content = self._read()
        if any(item["dataset_id"] == dataset_id for item in content["datasets"]):
            raise FileExistsError(f"dataset ID already registered: {dataset_id}")
        if parent_id is not None and not any(
            item["dataset_id"] == parent_id for item in content["datasets"]
        ):
            raise ValueError(f"register parent dataset first: {parent_id}")
        stat = target.stat()
        try:
            location = str(target.relative_to(self.root))
        except ValueError:
            location = str(target)
        entry = {
            "dataset_id": dataset_id,
            "version": 1,
            "path": location,
            "kind": kind,
            "parent_id": parent_id,
            "size_bytes": stat.st_size,
            "mtime_ns_at_registration": stat.st_mtime_ns,
            "identity_verification": "filesystem_metadata_only",
            "sha256": None,
            "metadata": metadata or {},
        }
        content["datasets"].append(entry)
        atomic_json(self.catalog_path, content, overwrite=True)
        return entry


def resolve_dataset(
    input_or_id: str | Path,
    *,
    catalog_path: str | Path | None = None,
    root: str | Path | None = None,
) -> Path:
    return DatasetCatalog(root=root, catalog_path=catalog_path).resolve(input_or_id)


def index_legacy_artifacts(
    root: str | Path | None = None, output_path: str | Path | None = None
) -> dict:
    """Index directory metadata, preserving historical results exactly in place."""
    base = Path(root or project_root()).resolve()
    entries = []
    classifications = {
        "outputs/likelihood-development": ("stopped_checkpoint", "CHECKPOINT_REPORT.md"),
        "outputs/funny-current-best": ("legacy_assembled_comparison", "metrics.json"),
        "experiments/detector_only_likelihood": ("experimental", "README.md"),
        "experiments/likelihood_physics_study": ("experimental_validated_option", "README.md"),
    }
    for parent in ("outputs", "experiments"):
        location = base / parent
        if not location.exists():
            continue
        for directory in sorted(location.iterdir()):
            if not directory.is_dir():
                continue
            files = [
                p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts
            ]
            relative = str(directory.relative_to(base))
            status, primary = classifications.get(relative, ("legacy_unclassified", None))
            entries.append(
                {
                    "artifact_id": relative.replace("/", ":"),
                    "path": relative,
                    "status": status,
                    "entry_point": primary,
                    "files": len(files),
                    "size_bytes": sum(p.stat().st_size for p in files),
                    "reports": [
                        str(p.relative_to(directory))
                        for p in files
                        if p.name
                        in ("README.md", "REPORT.md", "report.html", "CHECKPOINT_REPORT.md")
                    ],
                    "identity_verification": "filesystem_metadata_only",
                    "modified": False,
                }
            )
    payload = {
        "schema_version": 1,
        "purpose": "discovery_only; no inferred missing scientific provenance",
        "artifacts": entries,
    }
    if output_path:
        atomic_json(output_path, payload, overwrite=True)
    return payload
