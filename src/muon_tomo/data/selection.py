"""Selection identities separate detector observables from hidden momentum."""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from ..types import DetectorTracks


def detector_fingerprint(tracks) -> str:
    detector = tracks if isinstance(tracks, DetectorTracks) else DetectorTracks.from_legacy(tracks)
    digest = hashlib.sha256()
    digest.update(b"muon-tomo-detector-tracks-v1\0")
    for values, dtype in (
        (detector.event_id, "<i8"),
        (detector.entry_position_cm, "<f8"),
        (detector.entry_direction, "<f8"),
        (detector.exit_position_cm, "<f8"),
        (detector.exit_direction, "<f8"),
    ):
        value = np.asarray(values, dtype=dtype)
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def original_rows(
    path: str | Path, record_index: int, selected_rows: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Resolve existing one-generation lineage maps without reading TFRecords.

    Absence of a sidecar is reported as unknown, never as proven raw ancestry.
    """
    path = Path(path)
    rows = np.asarray(selected_rows, dtype=np.int64)
    if np.any(rows < 0):
        raise ValueError("selected rows must be nonnegative")
    packed = path.with_suffix(".original_indices.u32.gz")
    simple = path.with_suffix(".indices.npy")
    mapping = None
    if packed.is_file():
        with gzip.open(packed, "rb") as stream:
            while header := stream.read(16):
                if len(header) != 16:
                    raise ValueError(f"truncated lineage header: {packed}")
                record, count = struct.unpack("<QQ", header)
                if record == record_index:
                    values = stream.read(count * 4)
                    if len(values) != count * 4:
                        raise ValueError(f"truncated lineage rows: {packed}")
                    mapping = np.frombuffer(values, dtype="<u4")
                    break
                stream.seek(count * 4, 1)
        if mapping is None:
            raise IndexError(f"lineage has no record {record_index}")
        sidecar = packed
    elif simple.is_file():
        if record_index != 0:
            raise ValueError("single-record lineage map cannot resolve another record")
        mapping = np.load(simple, allow_pickle=False)
        sidecar = simple
    else:
        return rows.copy(), {"status": "no_lineage_sidecar", "row_scope": "selected_dataset"}
    if mapping.ndim != 1 or mapping.dtype.kind not in "iu" or np.any(mapping < 0):
        raise ValueError("lineage mapping must be a one-dimensional nonnegative integer array")
    if len(rows) and np.max(rows) >= len(mapping):
        raise ValueError("selected row lies outside lineage mapping")
    return mapping[rows].astype(np.int64), {
        "status": "parent_rows_resolved",
        "sidecar": str(sidecar),
        "row_scope": "immediate_parent_dataset",
    }


def selection_manifest(
    tracks,
    *,
    dataset_id: str | None = None,
    dataset_version: int | str | None = None,
    record_index: int | None = None,
    input_path: str | Path | None = None,
    requested_tracks: int | None = None,
) -> dict:
    source = str(input_path or tracks.source)
    record = record_index if record_index is not None else tracks.metadata.get("record_index", 0)
    row_indexed = Path(source).suffix.lower() == ".tfrecord"
    if row_indexed:
        rows, lineage = original_rows(source, record, tracks.event_id)
    else:
        rows = np.asarray(tracks.event_id)
        lineage = {"status": "event_ids_only", "row_scope": "source_event_identifiers"}
    payload = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "input_source": source,
        "record_index": record,
        "selected_event_ids": np.asarray(tracks.event_id).tolist(),
        "selected_dataset_rows": np.asarray(tracks.event_id).tolist() if row_indexed else None,
        "event_id_kind": "record_row_index" if row_indexed else "source_event_id",
        "parent_rows": rows.tolist() if lineage["status"] == "parent_rows_resolved" else None,
        "lineage": lineage,
        "selected_count": len(tracks),
        "requested_tracks": requested_tracks,
        "sampling_seed": tracks.metadata.get("selection_seed"),
        "selection_limit": tracks.metadata.get("selection_limit"),
        "detector_fingerprint_sha256": detector_fingerprint(tracks),
        "dose_basis": "selected_stored_events; acquisition_time_not_inferred",
    }
    identity = {
        k: payload[k]
        for k in (
            "dataset_id",
            "dataset_version",
            "record_index",
            "selected_event_ids",
            "parent_rows",
            "detector_fingerprint_sha256",
        )
    }
    if dataset_id is None:
        identity["uncatalogued_source"] = source
    payload["selection_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    return payload
