"""Portable measured-track interchange with no momentum or labels.

The NPZ contains physical incoming/outgoing positions in cm and unit directions.
Its writer never includes compatibility caches, simulation energy, or truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..artifacts.storage import atomic_npz
from ..types import DetectorTracks

ARRAY_FIELDS = ("entry_position_cm", "entry_direction", "exit_position_cm", "exit_direction")


def save_detector_npz(tracks: DetectorTracks, path: str | Path, *, overwrite: bool = False) -> Path:
    if not isinstance(tracks, DetectorTracks):
        raise TypeError("save_detector_npz requires DetectorTracks; convert explicitly first")
    metadata = {
        **tracks.metadata,
        "schema": "muon_tomo_detector_tracks",
        "schema_version": 1,
        "position_unit": "cm",
        "direction_unit": "unit_vector",
        "track_order": "physical_incoming_then_outgoing",
        "source": tracks.source,
        "per_track_momentum_present": False,
        "labels_present": False,
    }
    payload = {name: getattr(tracks, name) for name in ARRAY_FIELDS}
    payload.update(
        event_id=tracks.event_id,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
    )
    return atomic_npz(path, payload, overwrite=overwrite)


def load_detector_npz(
    path: str | Path, *, limit: int | None = None, seed: int = 0
) -> DetectorTracks:
    """Read the versioned detector contract without optional simulation fields."""
    with np.load(path, allow_pickle=False) as archive:
        missing = set((*ARRAY_FIELDS, "metadata_json")) - set(archive.files)
        if missing:
            raise ValueError(f"detector NPZ is missing fields: {sorted(missing)}")
        extra = set(archive.files) - set((*ARRAY_FIELDS, "metadata_json", "event_id"))
        if extra:
            raise ValueError(
                f"detector NPZ contains fields outside its detector-only schema: {sorted(extra)}"
            )
        metadata = json.loads(str(archive["metadata_json"]))
        if (
            metadata.get("schema") != "muon_tomo_detector_tracks"
            or metadata.get("schema_version") != 1
        ):
            raise ValueError("unsupported detector NPZ schema")
        if (
            metadata.get("position_unit") != "cm"
            or metadata.get("track_order") != "physical_incoming_then_outgoing"
        ):
            raise ValueError("detector NPZ must declare cm and physical incoming/outgoing order")
        if metadata.get("per_track_momentum_present") or metadata.get("labels_present"):
            raise ValueError("detector NPZ cannot declare momentum or evaluation labels")
        arrays = [archive[name] for name in ARRAY_FIELDS]
        events = archive["event_id"] if "event_id" in archive.files else None
    return DetectorTracks(*arrays, events, str(Path(path).resolve()), metadata).subset(limit, seed)
