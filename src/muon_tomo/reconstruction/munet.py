"""Frozen Mu-Net inference adapter. This module contains no training path."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..constants import MUNET_POSITION_OFFSET, MUNET_POSITION_SCALE_CM
from ..geometry import GridGeometry
from ..types import ReconstructionResult, TrackTable
from .common import path_occupancy


class MuNetReconstructor:
    def __init__(
        self,
        geometry: GridGeometry | None = None,
        *,
        dosage: int = 1024,
        weights_path: str | Path | None = None,
        position_scale_cm: float = MUNET_POSITION_SCALE_CM,
        position_offset: float = MUNET_POSITION_OFFSET,
        max_chunks: int | None = None,
    ) -> None:
        self.geometry = geometry or GridGeometry()
        if dosage not in {512, 1024}:
            raise ValueError("frozen Mu-Net supports dosage 512 or 1024")
        self.dosage = dosage
        default = Path(__file__).parents[1] / "_vendor" / "munet" / "weights" / f"model_{dosage}.weights.h5"
        self.weights_path = Path(weights_path) if weights_path else default
        self.position_scale_cm = float(position_scale_cm)
        self.position_offset = float(position_offset)
        self.max_chunks = max_chunks
        expected_lower = np.full(3, -self.position_offset * self.position_scale_cm)
        expected_upper = np.full(3, (1.0 - self.position_offset) * self.position_scale_cm)
        if not np.allclose(self.geometry.lower, expected_lower) or not np.allclose(
            self.geometry.upper, expected_upper
        ):
            raise ValueError(
                "Mu-Net position normalization and GridGeometry bounds describe different physical domains"
            )

    def _features(self, tracks: TrackTable) -> np.ndarray:
        incoming = tracks.entry_direction.astype(np.float32)
        outgoing = tracks.exit_direction.astype(np.float32)
        return np.concatenate(
            [
                tracks.entry_position_cm.astype(np.float32) / self.position_scale_cm + self.position_offset,
                incoming,
                tracks.exit_position_cm.astype(np.float32) / self.position_scale_cm + self.position_offset,
                outgoing,
                np.zeros((len(tracks), 1), dtype=np.float32),
                np.linalg.norm(outgoing - incoming, axis=1, keepdims=True).astype(np.float32),
            ],
            axis=1,
        )

    def reconstruct(self, tracks: TrackTable) -> ReconstructionResult:
        try:
            import tensorflow as tf
            from muon_tomo._vendor.munet.models.agg_3d import Agg3D
        except ImportError as exc:
            raise RuntimeError("Mu-Net inference requires the 'munet' extra (TensorFlow)") from exc
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"frozen Mu-Net weights not found: {self.weights_path}")
        chunks = len(tracks) // self.dosage
        if self.max_chunks is not None:
            chunks = min(chunks, self.max_chunks)
        if chunks < 1:
            raise ValueError(f"Mu-Net {self.dosage} needs at least {self.dosage} tracks")
        features = self._features(tracks)[: chunks * self.dosage]
        features = features.reshape(chunks, self.dosage, 14)
        model = Agg3D(
            point_size=1,
            downward_convs=[1, 2, 3, 4, 5],
            downward_filters=[8, 16, 32, 64, 128],
            upward_convs=[4, 3, 2, 1],
            upward_filters=[64, 32, 16, 8],
            resolution=64,
            threshold=1e-8,
        )
        _ = model(tf.zeros((1, self.dosage, 14), dtype=tf.float32), training=False)
        model.load_weights(str(self.weights_path))
        total = np.zeros(self.geometry.shape, dtype=np.float64)
        for chunk in features:
            prediction = model(tf.convert_to_tensor(chunk[None, ...]), training=False).numpy()[0, ..., 0]
            if prediction.shape != self.geometry.shape:
                raise ValueError(f"Mu-Net returned {prediction.shape}, expected {self.geometry.shape}")
            total += prediction
        raw = (total / chunks).astype(np.float32)
        used = tracks.take_first(chunks * self.dosage)
        occupancy = path_occupancy(used, self.geometry)
        return ReconstructionResult(
            method="munet",
            inverse_radiation_length_cm_inv=np.clip(raw, 0.0, None),
            raw_estimator=raw,
            occupancy_count=occupancy,
            valid_mask=np.ones(self.geometry.shape, dtype=bool),
            geometry=self.geometry,
            metadata={
                "algorithm": "frozen Mu-Net inference",
                "weights": str(self.weights_path.resolve()),
                "weights_modified": False,
                "training_performed": False,
                "dosage": self.dosage,
                "chunks": chunks,
                "tracks_used": chunks * self.dosage,
                "raw_estimator_unit": "cm^-1 (learned target)",
                "feature_count": 14,
                "position_scale_cm": self.position_scale_cm,
                "position_offset": self.position_offset,
                "momentum_feature_policy": "zero, matching the verified frozen-weight inference notebook",
                "occupancy_definition": "number of straight detector chords crossing each voxel",
            },
        )
