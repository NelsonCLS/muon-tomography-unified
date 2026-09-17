"""Build aligned, uniquely identified panels for arbitrary reconstruction methods."""

from dataclasses import dataclass

import numpy as np

from ..compare import compare_to_truth
from ..types import ReconstructionResult, TruthVolume

LABELS = {"munet": "MuNet", "poca": "PoCA", "likelihood": "Likelihood"}


@dataclass(frozen=True)
class VolumePanel:
    key: str
    label: str
    volume: np.ndarray
    metrics: dict | None
    is_truth: bool = False


def prepare_panels(truth: TruthVolume | None, results: list[ReconstructionResult]):
    if truth is None and not results:
        raise ValueError("provide at least one reconstruction or truth volume")
    geometry = truth.geometry if truth is not None else results[0].geometry
    panels = []
    if truth is not None:
        panels.append(
            VolumePanel("truth", "Truth", truth.inverse_radiation_length_cm_inv, None, True)
        )
    ordered = list(results)
    if len(ordered) == 3 and {r.method for r in ordered} == set(LABELS):
        ordered.sort(key=lambda r: list(LABELS).index(r.method))
    used = {"truth"} if truth is not None else set()
    for result in ordered:
        if result.geometry != geometry:
            raise ValueError(f"{result.method} geometry differs from the other panels")
        key = str(result.metadata.get("method_id", result.method))
        base = key
        serial = 2
        while key in used:
            key = f"{base}_{serial}"
            serial += 1
        used.add(key)
        label = str(result.metadata.get("display_name", LABELS.get(result.method, result.method)))
        if key != base:
            label += f" ({serial - 1})"
        metrics = compare_to_truth(result, truth) if truth is not None else None
        panels.append(VolumePanel(key, label, result.inverse_radiation_length_cm_inv, metrics))
    return geometry, panels


def momentum_caption(results):
    descriptions = []
    for result in results:
        meta = result.metadata
        label = LABELS.get(result.method, result.method)
        p = meta.get("assumed_momentum_mev", meta.get("momentum_estimate_mev"))
        policy = str(meta.get("momentum_policy", ""))
        if "external" in policy:
            description = "external spectrum"
        elif p is not None:
            description = f"fixed p estimate {float(p):.1f} MeV/c"
        else:
            description = policy or "momentum policy not recorded"
        descriptions.append(f"{label}: {description}")
    return "; ".join(descriptions)
