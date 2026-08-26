"""Validated in-memory and on-disk contracts shared by all methods."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import GRID_SHAPE, inverse_to_reference_scattering_density
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
            self.material_labels.shape != GRID_SHAPE
            or self.inverse_radiation_length_cm_inv.shape != GRID_SHAPE
        ):
            raise ValueError(f"truth arrays must have shape {GRID_SHAPE}")


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
            if arr.shape != GRID_SHAPE:
                raise ValueError(f"{name} must have shape {GRID_SHAPE}, got {arr.shape}")
            setattr(self, name, arr)
        self.inverse_radiation_length_cm_inv = self.inverse_radiation_length_cm_inv.astype(
            np.float32
        )
        self.raw_estimator = self.raw_estimator.astype(np.float32)
        self.occupancy_count = self.occupancy_count.astype(np.int64)
        self.valid_mask = self.valid_mask.astype(bool)
        if self.uncertainty is not None:
            self.uncertainty = np.asarray(self.uncertainty, dtype=np.float32)
            if self.uncertainty.shape != GRID_SHAPE:
                raise ValueError(f"uncertainty must have shape {GRID_SHAPE}")
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

    def save(self, path: str | Path) -> Path:
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
        np.savez_compressed(target, **payload)
        return target

    @classmethod
    def load(cls, path: str | Path) -> ReconstructionResult:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
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
