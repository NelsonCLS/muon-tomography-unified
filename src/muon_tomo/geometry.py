"""The mandatory common 64^3 reconstruction geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import AXIS_ORDER, GRID_LOWER_CM, GRID_SHAPE, GRID_UPPER_CM


@dataclass(frozen=True)
class GridGeometry:
    """A physical xyz grid whose array indexing is volume[x, y, z]."""

    lower_cm: tuple[float, float, float] = GRID_LOWER_CM
    upper_cm: tuple[float, float, float] = GRID_UPPER_CM
    shape: tuple[int, int, int] = GRID_SHAPE
    axis_order: str = AXIS_ORDER

    def __post_init__(self) -> None:
        if tuple(self.shape) != GRID_SHAPE:
            raise ValueError(f"all methods require grid shape {GRID_SHAPE}; got {self.shape}")
        if self.axis_order != AXIS_ORDER:
            raise ValueError(f"all methods require axis order {AXIS_ORDER!r}")
        lower = np.asarray(self.lower_cm, dtype=float)
        upper = np.asarray(self.upper_cm, dtype=float)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
            raise ValueError("lower_cm and upper_cm must be increasing xyz triples")
        if not np.allclose(lower, GRID_LOWER_CM) or not np.allclose(upper, GRID_UPPER_CM):
            raise ValueError(
                f"frozen Mu-Net requires canonical bounds {GRID_LOWER_CM} to {GRID_UPPER_CM} cm"
            )

    @property
    def lower(self) -> np.ndarray:
        return np.asarray(self.lower_cm, dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.upper_cm, dtype=np.float64)

    @property
    def voxel_size_cm(self) -> np.ndarray:
        return (self.upper - self.lower) / np.asarray(self.shape)

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    def metadata(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "lower_cm": list(self.lower_cm),
            "upper_cm": list(self.upper_cm),
            "voxel_size_cm": self.voxel_size_cm.tolist(),
            "axis_order": self.axis_order,
            "array_indexing": "volume[x,y,z]",
        }

    def point_to_index(self, point_cm: np.ndarray) -> np.ndarray:
        point = np.asarray(point_cm, dtype=np.float64)
        scaled = (point - self.lower) / self.voxel_size_cm
        return np.floor(scaled).astype(np.int64)

    def index_to_center(self, index_xyz: np.ndarray) -> np.ndarray:
        index = np.asarray(index_xyz, dtype=np.float64)
        return self.lower + (index + 0.5) * self.voxel_size_cm

    def contains(self, point_cm: np.ndarray) -> np.ndarray:
        point = np.asarray(point_cm, dtype=np.float64)
        return np.all((point >= self.lower) & (point < self.upper), axis=-1)

    def flat_index(self, index_xyz: np.ndarray) -> np.ndarray:
        idx = np.asarray(index_xyz, dtype=np.int64)
        return np.ravel_multi_index(idx.T, self.shape)

    def trace_segment(
        self, start_cm: np.ndarray, end_cm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ordered xyz voxel indices and exact chord lengths for an AABB-clipped segment."""
        start = np.asarray(start_cm, dtype=np.float64)
        end = np.asarray(end_cm, dtype=np.float64)
        delta = end - start
        norm = float(np.linalg.norm(delta))
        if norm <= 0 or not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64)

        t0, t1 = 0.0, 1.0
        for axis in range(3):
            if abs(delta[axis]) < 1e-15:
                if start[axis] < self.lower[axis] or start[axis] > self.upper[axis]:
                    return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64)
                continue
            a = (self.lower[axis] - start[axis]) / delta[axis]
            b = (self.upper[axis] - start[axis]) / delta[axis]
            t0 = max(t0, min(a, b))
            t1 = min(t1, max(a, b))
        if t1 <= t0:
            return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64)

        crossings = [t0, t1]
        for axis in range(3):
            if abs(delta[axis]) < 1e-15:
                continue
            planes = np.linspace(self.lower[axis], self.upper[axis], self.shape[axis] + 1)
            ts = (planes - start[axis]) / delta[axis]
            crossings.extend(ts[(ts > t0 + 1e-12) & (ts < t1 - 1e-12)].tolist())
        ts = np.unique(np.asarray(crossings, dtype=np.float64))
        mids = (ts[:-1] + ts[1:]) * 0.5
        lengths = (ts[1:] - ts[:-1]) * norm
        indices = self.point_to_index(start[None, :] + mids[:, None] * delta[None, :])
        valid = np.all((indices >= 0) & (indices < np.asarray(self.shape)), axis=1) & (lengths > 0)
        return indices[valid], lengths[valid]
