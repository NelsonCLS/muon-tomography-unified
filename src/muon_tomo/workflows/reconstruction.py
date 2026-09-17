"""Paired track selection and execution, independent of CLI and truth data."""

from __future__ import annotations

from ..reconstruction.registry import build_reconstructor
from ..types import TrackTable


def select_shared_tracks(
    tracks: TrackTable,
    methods: list[str],
    config: dict,
    *,
    overrides: dict | None = None,
) -> TrackTable:
    """Preserve the established common-selection policy for MuNet comparisons.

    When MuNet participates, every method receives the same leading complete
    MuNet chunks. Classical-only runs retain every selected track. Any random
    selection has already happened in the input adapter, with recorded IDs.
    """
    if len(tracks) == 0:
        raise ValueError("prediction requires at least one track")
    if "munet" not in methods:
        return tracks
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    dosage = int(overrides.get("dosage", config["munet"]["dosage"]))
    chunks = len(tracks) // dosage
    if overrides.get("max_chunks") is not None:
        chunks = min(chunks, int(overrides["max_chunks"]))
    if chunks < 1:
        raise ValueError(f"Mu-Net {dosage} needs at least {dosage} tracks")
    return tracks.take_first(chunks * dosage)


def reconstruct_selected(tracks: TrackTable, methods: list[str], config: dict):
    """Yield unchanged raw algorithm outputs. Truth is deliberately not an input."""
    for method in methods:
        inference_seed = None
        if method == "munet":
            # The frozen vendor model samples positions for near-parallel tracks
            # even at training=False. Seed its existing behavior, not its math.
            import tensorflow as tf

            inference_seed = int(config["munet"].get("inference_seed", 0))
            tf.random.set_seed(inference_seed)
        reconstructor = build_reconstructor(method, config)
        result = reconstructor.reconstruct(tracks)
        if inference_seed is not None:
            result.metadata["inference_seed"] = inference_seed
            result.metadata["inference_seed_scope"] = (
                "TensorFlow global seed set before frozen model construction"
            )
        yield result
