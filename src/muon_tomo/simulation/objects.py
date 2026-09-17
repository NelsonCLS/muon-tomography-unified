"""Voxelize user-defined solids on the reconstruction package's physical grid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from ..constants import (
    GRID_LOWER_CM,
    GRID_SHAPE,
    GRID_UPPER_CM,
    MATERIAL_NAMES,
    labels_to_inverse_radiation_length,
)


def material_id(value: str | int) -> int:
    if isinstance(value, str):
        try:
            return MATERIAL_NAMES.index(value.lower())
        except ValueError as exc:
            raise ValueError(f"Unknown material {value!r}; choose from {MATERIAL_NAMES}") from exc
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        if 0 <= value < len(MATERIAL_NAMES):
            return int(value)
    raise ValueError("Material must be a known name or integer ID 0..21")


def _vector(value, name: str, *, positive: bool = False) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite numbers")
    if positive and np.any(vector <= 0):
        raise ValueError(f"{name} must be positive")
    return vector


def _positive(value, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _rotation(angles: np.ndarray) -> np.ndarray:
    """Active rotation Rz @ Ry @ Rx in degrees, about the object's centre."""
    x, y, z = np.deg2rad(angles)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def voxel_centres_cm() -> np.ndarray:
    axes = [
        low + (np.arange(n) + 0.5) * (high - low) / n
        for low, high, n in zip(GRID_LOWER_CM, GRID_UPPER_CM, GRID_SHAPE, strict=True)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def read_scene(path: str | Path) -> dict:
    scene = yaml.safe_load(Path(path).read_text())
    if not isinstance(scene, dict):
        raise ValueError("Scene must be a YAML or JSON mapping")
    unknown = scene.keys() - {"version", "name", "background", "shapes"}
    if unknown:
        raise ValueError(f"Unknown scene fields: {sorted(unknown)}")
    if scene.get("version", 1) != 1:
        raise ValueError("Only scene version 1 is supported")
    return scene


def create_voxels(scene: dict) -> np.ndarray:
    """Paint solids in declared order; explicit 'replace' permits cavities/overlap.

    Shapes are sampled at voxel centres and cropped to [-50,50] cm. A cylinder's
    unrotated axis is z. Rotation is applied about center_cm before voxelization.
    """
    background = material_id(scene.get("background", "air"))
    voxels = np.full(GRID_SHAPE, background, dtype=np.int32)
    occupied = np.zeros(GRID_SHAPE, dtype=bool)
    points = voxel_centres_cm()
    shapes = scene.get("shapes", [])
    if not isinstance(shapes, list):
        raise ValueError("shapes must be a list")
    common = {"name", "type", "material", "center_cm", "rotation_deg", "overlap"}
    dimensions = {
        "box": {"size_cm"},
        "hollow_box": {"size_cm", "wall_cm"},
        "sphere": {"radius_cm"},
        "ellipsoid": {"radii_cm"},
        "cylinder": {"radius_cm", "height_cm", "inner_radius_cm"},
    }
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict) or shape.get("type") not in dimensions:
            raise ValueError(f"Shape {index} type must be one of {tuple(dimensions)}")
        kind = shape["type"]
        unknown = shape.keys() - common - dimensions[kind]
        if unknown:
            raise ValueError(f"Unknown fields in shape {index}: {sorted(unknown)}")
        centre = _vector(shape.get("center_cm", [0, 0, 0]), "center_cm")
        rotation = _rotation(_vector(shape.get("rotation_deg", [0, 0, 0]), "rotation_deg"))
        local = (points - centre) @ rotation
        if kind in ("box", "hollow_box"):
            half = _vector(shape["size_cm"], "size_cm", positive=True) / 2
            mask = np.all(np.abs(local) < half, axis=-1)
            if kind == "hollow_box":
                wall = _positive(shape["wall_cm"], "wall_cm")
                if wall >= half.min():
                    raise ValueError("wall_cm must be less than half the smallest box dimension")
                mask &= ~np.all(np.abs(local) < half - wall, axis=-1)
        elif kind in ("sphere", "ellipsoid"):
            radii = (
                np.full(3, _positive(shape["radius_cm"], "radius_cm"))
                if kind == "sphere"
                else _vector(shape["radii_cm"], "radii_cm", positive=True)
            )
            mask = np.sum((local / radii) ** 2, axis=-1) <= 1
        else:
            radius = _positive(shape["radius_cm"], "radius_cm")
            height = _positive(shape["height_cm"], "height_cm")
            inner = float(shape.get("inner_radius_cm", 0))
            if not np.isfinite(inner) or not 0 <= inner < radius:
                raise ValueError("inner_radius_cm must satisfy 0 <= inner < radius_cm")
            radial_squared = local[..., 0] ** 2 + local[..., 1] ** 2
            mask = (radial_squared <= radius**2) & (radial_squared >= inner**2)
            mask &= np.abs(local[..., 2]) < height / 2
        if not np.any(mask):
            raise ValueError(f"Shape {index} contains no voxel centres inside the target cube")
        overlap = shape.get("overlap", "error")
        if overlap not in ("error", "replace"):
            raise ValueError("overlap must be 'error' or 'replace'")
        if overlap == "error" and np.any(mask & occupied):
            raise ValueError(f"Shape {index} overlaps an earlier shape; use overlap: replace")
        voxels[mask] = material_id(shape["material"])
        occupied[mask] = True
    return voxels


def load_voxels(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            values = data["material_labels"]
    else:
        # Read as float first so nonintegral tokens cannot be silently truncated.
        flat = np.loadtxt(path, dtype=np.float64)
        if flat.size != int(np.prod(GRID_SHAPE)):
            raise ValueError("Voxel text must contain exactly 64^3 material IDs")
        if not np.isfinite(flat).all() or not np.equal(flat, np.floor(flat)).all():
            raise ValueError("Voxel labels must be finite integers")
        if flat.min() < 0 or flat.max() >= len(MATERIAL_NAMES):
            raise ValueError("Voxel labels must be in 0..21")
        values = flat.astype(np.int32).reshape(GRID_SHAPE)
    labels_to_inverse_radiation_length(values)
    return values.astype(np.int32)


def save_voxels(voxels: np.ndarray, path: str | Path, *, metadata: dict | None = None) -> Path:
    labels_to_inverse_radiation_length(voxels)
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".npz":
        np.savez_compressed(
            target,
            material_labels=np.asarray(voxels, dtype=np.int32),
            metadata_json=json.dumps(metadata or {}),
        )
    elif target.suffix == ".npy":
        np.save(target, np.asarray(voxels, dtype=np.int32), allow_pickle=False)
    elif target.suffix == ".txt":
        np.savetxt(target, voxels.ravel(order="C"), fmt="%d")
    else:
        raise ValueError("Voxel output must end in .npz, .npy or .txt")
    return target
