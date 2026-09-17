"""Truth artifacts and unchanged legacy track fingerprints."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ..geometry import GridGeometry
from ..types import TruthVolume
from .storage import atomic_npz


def legacy_track_fingerprint(tracks) -> str:
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


def save_truth(truth: TruthVolume, path: str | Path, *, overwrite: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **truth.metadata,
        "schema_version": 1,
        "primary_quantity": "inverse_radiation_length",
        "primary_unit": "cm^-1",
        "geometry": truth.geometry.metadata(),
    }
    atomic_npz(
        target,
        {
            "material_labels": truth.material_labels,
            "inverse_radiation_length_cm_inv": truth.inverse_radiation_length_cm_inv,
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        },
        overwrite=overwrite,
    )
    return target


def load_truth(path: str | Path) -> TruthVolume:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if metadata.get("schema_version", 1) != 1:
            raise ValueError("unsupported truth artifact schema")
        gm = metadata["geometry"]
        geometry = GridGeometry(
            tuple(gm["lower_cm"]),
            tuple(gm["upper_cm"]),
            tuple(gm["shape"]),
            gm["axis_order"],
        )
        return TruthVolume(
            data["material_labels"], data["inverse_radiation_length_cm_inv"], geometry, metadata
        )
