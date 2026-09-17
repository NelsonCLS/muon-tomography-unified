"""Validated in-memory and on-disk contracts shared by all methods."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import inverse_to_reference_scattering_density
from .geometry import GridGeometry


def _unit_rows(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n,3)")
    norms = np.linalg.norm(arr, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError(f"{name} contains a zero or non-finite vector")
    return arr / norms[:, None]


@dataclass
class TrackTable:
    entry_position_cm: np.ndarray
    entry_direction: np.ndarray
    exit_position_cm: np.ndarray
    exit_direction: np.ndarray
    momentum_mev: np.ndarray
    event_id: np.ndarray | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entry_position_cm = np.asarray(self.entry_position_cm, dtype=np.float64)
        self.exit_position_cm = np.asarray(self.exit_position_cm, dtype=np.float64)
        self.entry_direction = _unit_rows(self.entry_direction, "entry_direction")
        self.exit_direction = _unit_rows(self.exit_direction, "exit_direction")
        self.momentum_mev = np.asarray(self.momentum_mev, dtype=np.float64).reshape(-1)
        n = len(self.momentum_mev)
        for name in ("entry_position_cm", "entry_direction", "exit_position_cm", "exit_direction"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} length differs from momentum length")
        if np.any(~np.isfinite(self.entry_position_cm)) or np.any(
            ~np.isfinite(self.exit_position_cm)
        ):
            raise ValueError("positions must be finite")
        if np.any(~np.isfinite(self.momentum_mev)) or np.any(self.momentum_mev <= 0):
            raise ValueError("momentum_mev must be finite and positive")
        if self.event_id is None:
            self.event_id = np.arange(n, dtype=np.int64)
        else:
            self.event_id = np.asarray(self.event_id, dtype=np.int64).reshape(-1)
            if len(self.event_id) != n:
                raise ValueError("event_id length differs from momentum length")

    def __len__(self) -> int:
        return len(self.momentum_mev)

    def subset(self, limit: int | None = None, seed: int = 0) -> TrackTable:
        if limit is None or limit >= len(self):
            idx = np.arange(len(self))
        else:
            idx = np.sort(np.random.default_rng(seed).choice(len(self), size=limit, replace=False))
        return TrackTable(
            self.entry_position_cm[idx],
            self.entry_direction[idx],
            self.exit_position_cm[idx],
            self.exit_direction[idx],
            self.momentum_mev[idx],
            self.event_id[idx],
            self.source,
            {**self.metadata, "selection_limit": limit, "selection_seed": seed},
        )

    def take_first(self, count: int) -> TrackTable:
        if count < 0 or count > len(self):
            raise ValueError(f"count must be between 0 and {len(self)}")
        idx = np.arange(count)
        return TrackTable(
            self.entry_position_cm[idx],
            self.entry_direction[idx],
            self.exit_position_cm[idx],
            self.exit_direction[idx],
            self.momentum_mev[idx],
            self.event_id[idx],
            self.source,
            {**self.metadata, "tracks_taken_from_selection": count},
        )

    def projected_scattering_angles(self) -> tuple[np.ndarray, np.ndarray]:
        """Return signed projected xz and yz scattering angles in radians."""
        incoming = self.entry_direction.copy()
        outgoing = self.exit_direction.copy()
        flip = np.sum(incoming * outgoing, axis=1) < 0
        outgoing[flip] *= -1
        ax0 = np.arctan2(incoming[:, 0], np.abs(incoming[:, 2]))
        ay0 = np.arctan2(incoming[:, 1], np.abs(incoming[:, 2]))
        ax1 = np.arctan2(outgoing[:, 0], np.abs(outgoing[:, 2]))
        ay1 = np.arctan2(outgoing[:, 1], np.abs(outgoing[:, 2]))
        return ax1 - ax0, ay1 - ay0


@dataclass
class DetectorTracks:
    """Physical incoming/outgoing states with no momentum or label field.

    Positions use cm and directions are unit vectors oriented along the flight.
    Legacy adapters retain detector-only feature arrays for exact MuNet parity.
    Editing physical arrays invalidates that cache; construct a new instance
    from edited arrays before inference instead of silently using stale values.
    """

    entry_position_cm: np.ndarray
    entry_direction: np.ndarray
    exit_position_cm: np.ndarray
    exit_direction: np.ndarray
    event_id: np.ndarray | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    _legacy_arrays: tuple | None = field(default=None, init=False, repr=False)
    _legacy_physical_signature: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("entry_position_cm", "exit_position_cm"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 3 or not np.isfinite(arr).all():
                raise ValueError(f"{name} must be a finite (N,3) array")
            setattr(self, name, arr)
        self.entry_direction = _unit_rows(self.entry_direction, "entry_direction")
        self.exit_direction = _unit_rows(self.exit_direction, "exit_direction")
        n = len(self.entry_position_cm)
        if any(
            len(getattr(self, key)) != n
            for key in ("exit_position_cm", "entry_direction", "exit_direction")
        ):
            raise ValueError("all detector arrays must have equal row counts")
        if self.event_id is None:
            self.event_id = np.arange(n, dtype=np.int64)
        else:
            values = np.asarray(self.event_id)
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise ValueError("detector event IDs must be a one-dimensional integer array")
            if np.any(values < 0) or np.any(values > np.iinfo(np.int64).max):
                raise ValueError("detector event IDs must fit nonnegative int64")
            self.event_id = values.astype(np.int64)
        if len(self.event_id) != n:
            raise ValueError("event IDs must have one entry per detector track")
        flight = self.exit_position_cm - self.entry_position_cm
        if np.any(np.linalg.norm(flight, axis=1) == 0):
            raise ValueError("incoming and outgoing detector positions must be distinct")
        self.entry_direction[np.sum(self.entry_direction * flight, axis=1) < 0] *= -1.0
        self.exit_direction[np.sum(self.exit_direction * flight, axis=1) < 0] *= -1.0

    def __len__(self) -> int:
        return len(self.entry_position_cm)

    def _physical_signature(self) -> str:
        digest = hashlib.sha256()
        for name in ("entry_position_cm", "entry_direction", "exit_position_cm", "exit_direction"):
            values = np.asarray(getattr(self, name), dtype="<f8")
            digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
            digest.update(values.tobytes(order="C"))
        return digest.hexdigest()

    def _check_legacy_cache(self) -> None:
        if (
            self._legacy_arrays is not None
            and self._physical_signature() != self._legacy_physical_signature
        ):
            raise ValueError(
                "DetectorTracks physical arrays changed after legacy adaptation; construct a new DetectorTracks from the edited arrays before conversion or selection"
            )

    @classmethod
    def from_legacy(cls, tracks: TrackTable) -> DetectorTracks:
        from .reconstruction.common import physical_track_order

        arrays = physical_track_order(tracks)
        result = cls(*arrays, tracks.event_id.copy(), tracks.source, dict(tracks.metadata))
        result.entry_direction = arrays[1]
        result.exit_direction = arrays[3]
        result._legacy_arrays = tuple(
            np.array(getattr(tracks, name), copy=True)
            for name in (
                "entry_position_cm",
                "entry_direction",
                "exit_position_cm",
                "exit_direction",
            )
        )
        result._legacy_physical_signature = result._physical_signature()
        return result

    def select(self, indices: np.ndarray) -> DetectorTracks:
        self._check_legacy_cache()
        indices = np.asarray(indices, dtype=np.int64)
        result = DetectorTracks(
            self.entry_position_cm[indices],
            self.entry_direction[indices],
            self.exit_position_cm[indices],
            self.exit_direction[indices],
            self.event_id[indices],
            self.source,
            dict(self.metadata),
        )
        if self._legacy_arrays is not None:
            result.entry_direction = self.entry_direction[indices].copy()
            result.exit_direction = self.exit_direction[indices].copy()
            result._legacy_arrays = tuple(a[indices].copy() for a in self._legacy_arrays)
            result._legacy_physical_signature = result._physical_signature()
        return result

    def subset(self, limit: int | None = None, seed: int = 0) -> DetectorTracks:
        if limit is not None and limit < 0:
            raise ValueError("limit must be nonnegative")
        idx = (
            np.arange(len(self))
            if limit is None or limit >= len(self)
            else np.sort(np.random.default_rng(seed).choice(len(self), size=limit, replace=False))
        )
        result = self.select(idx)
        result.metadata.update(selection_limit=limit, selection_seed=seed)
        return result

    def to_legacy(self, assumed_momentum_mev: float) -> TrackTable:
        """Adapt an explicit fixed prior; preserve unmodified legacy feature bytes."""
        self._check_legacy_cache()
        if not np.isfinite(assumed_momentum_mev) or assumed_momentum_mev <= 0:
            raise ValueError("assumed momentum must be finite and positive")
        arrays = self._legacy_arrays or (
            self.exit_position_cm,
            self.exit_direction,
            self.entry_position_cm,
            self.entry_direction,
        )
        result = TrackTable(
            *arrays,
            np.full(len(self), assumed_momentum_mev),
            self.event_id.copy(),
            self.source,
            {
                **self.metadata,
                "simulation_momentum_loaded": False,
                "adapter_momentum_policy": "declared_fixed_prior",
                "assumed_momentum_mev": float(assumed_momentum_mev),
            },
        )
        if self._legacy_arrays is not None:
            result.entry_direction = arrays[1].copy()
            result.exit_direction = arrays[3].copy()
        return result


@dataclass
class TruthVolume:
    material_labels: np.ndarray
    inverse_radiation_length_cm_inv: np.ndarray
    geometry: GridGeometry
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.material_labels = np.asarray(self.material_labels, dtype=np.int32)
        self.inverse_radiation_length_cm_inv = np.asarray(
            self.inverse_radiation_length_cm_inv, dtype=np.float32
        )
        if (
            self.material_labels.shape != self.geometry.shape
            or self.inverse_radiation_length_cm_inv.shape != self.geometry.shape
        ):
            raise ValueError(f"truth arrays must have shape {self.geometry.shape}")


@dataclass
class ReconstructionResult:
    method: str
    inverse_radiation_length_cm_inv: np.ndarray
    raw_estimator: np.ndarray
    occupancy_count: np.ndarray
    valid_mask: np.ndarray
    geometry: GridGeometry
    metadata: dict[str, Any] = field(default_factory=dict)
    uncertainty: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in (
            "inverse_radiation_length_cm_inv",
            "raw_estimator",
            "occupancy_count",
            "valid_mask",
        ):
            arr = np.asarray(getattr(self, name))
            if arr.shape != self.geometry.shape:
                raise ValueError(f"{name} must have shape {self.geometry.shape}, got {arr.shape}")
            setattr(self, name, arr)
        self.inverse_radiation_length_cm_inv = self.inverse_radiation_length_cm_inv.astype(
            np.float32
        )
        self.raw_estimator = self.raw_estimator.astype(np.float32)
        self.occupancy_count = self.occupancy_count.astype(np.int64)
        self.valid_mask = self.valid_mask.astype(bool)
        if self.uncertainty is not None:
            self.uncertainty = np.asarray(self.uncertainty, dtype=np.float32)
            if self.uncertainty.shape != self.geometry.shape:
                raise ValueError(f"uncertainty must have shape {self.geometry.shape}")
        if not np.all(np.isfinite(self.inverse_radiation_length_cm_inv)):
            raise ValueError("primary reconstruction contains non-finite values")

    @property
    def radiation_length_cm(self) -> np.ndarray:
        rho = self.inverse_radiation_length_cm_inv
        with np.errstate(over="ignore"):
            return np.divide(1.0, rho, out=np.full_like(rho, np.inf), where=rho > 0)

    @property
    def scattering_density_ref_mrad2_per_cm(self) -> np.ndarray:
        return inverse_to_reference_scattering_density(self.inverse_radiation_length_cm_inv)

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **self.metadata,
            "schema_version": 1,
            "method": self.method,
            "primary_quantity": "inverse_radiation_length",
            "primary_unit": "cm^-1",
            "geometry": self.geometry.metadata(),
        }
        payload = {
            "inverse_radiation_length_cm_inv": self.inverse_radiation_length_cm_inv,
            "radiation_length_cm": self.radiation_length_cm,
            "scattering_density_ref_mrad2_per_cm": self.scattering_density_ref_mrad2_per_cm,
            "raw_estimator": self.raw_estimator,
            "occupancy_count": self.occupancy_count,
            "valid_mask": self.valid_mask,
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        if self.uncertainty is not None:
            payload["uncertainty"] = self.uncertainty
        from .artifacts.storage import atomic_npz

        return atomic_npz(target, payload, overwrite=overwrite)

    @classmethod
    def load(cls, path: str | Path) -> ReconstructionResult:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if metadata.get("schema_version", 1) != 1:
                raise ValueError("unsupported reconstruction artifact schema")
            gm = metadata["geometry"]
            geometry = GridGeometry(
                tuple(gm["lower_cm"]), tuple(gm["upper_cm"]), tuple(gm["shape"]), gm["axis_order"]
            )
            uncertainty = data["uncertainty"] if "uncertainty" in data.files else None
            return cls(
                metadata["method"],
                data["inverse_radiation_length_cm_inv"],
                data["raw_estimator"],
                data["occupancy_count"],
                data["valid_mask"],
                geometry,
                metadata,
                uncertainty,
            )
