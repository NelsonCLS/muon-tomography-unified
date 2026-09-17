"""Explicit method registration and shared construction of frozen algorithms.

Register a MethodSpec with a factory to extend prediction and CLI selection.
Factories interpret settings; numerical estimators remain in their own modules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import geometry_from_config
from ..geometry import GridGeometry


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    label: str
    status: str
    implementation: str
    factory: Callable[[dict, GridGeometry, dict], Any]
    fixed_geometry: bool = True


def _momentum(config, overrides):
    physics = config["physics"]
    return (
        overrides.get("classical_momentum_policy", physics["classical_momentum_policy"]),
        float(overrides.get("average_momentum_mev", physics["average_momentum_mev"])),
    )


def _munet(config, geometry, overrides):
    from .munet import MuNetReconstructor

    values = config["munet"]
    dosage = overrides.get("dosage", values["dosage"])
    weights = overrides.get("weights")
    configured_weights = values.get("weights")
    if weights is None and configured_weights:
        # Bare built-in filenames preserve the historical package lookup.
        # Other values are explicit user paths, resolved at config load time.
        builtin_names = {"model_512.weights.h5", "model_1024.weights.h5"}
        if configured_weights not in builtin_names:
            weights = Path(configured_weights)
    return MuNetReconstructor(
        geometry,
        dosage=dosage,
        weights_path=weights,
        position_scale_cm=float(values["position_scale_cm"]),
        position_offset=float(values["position_offset"]),
        momentum_estimate_mev=float(values["momentum_estimate_mev"]),
        max_chunks=overrides.get("max_chunks"),
    )


def _poca(config, geometry, overrides):
    from .poca import PoCAReconstructor

    physics, values = config["physics"], config["poca"]
    policy, momentum = _momentum(config, overrides)
    return PoCAReconstructor(
        geometry,
        highland_constant_mev=float(physics["highland_constant_mev"]),
        minimum_coverage=int(values["minimum_coverage"]),
        momentum_policy=policy,
        average_momentum_mev=momentum,
        coarse_grid_size=int(values["grid_size"]),
        pooling_sigma_cm=float(values["pooling_sigma_cm"]),
    )


def _likelihood(config, geometry, overrides):
    from .likelihood import LikelihoodReconstructor

    physics, values = config["physics"], config["likelihood"]
    policy, momentum = _momentum(config, overrides)
    return LikelihoodReconstructor(
        geometry,
        iterations=overrides.get("iterations", int(values["iterations"])),
        reference_momentum_mev=float(physics["reference_momentum_mev"]),
        highland_constant_mev=float(physics["highland_constant_mev"]),
        initial_inverse_radiation_length_cm_inv=float(
            values["initial_inverse_radiation_length_cm_inv"]
        ),
        minimum_coverage=int(values["minimum_coverage"]),
        epsilon=float(values["epsilon"]),
        mode=overrides.get("likelihood_mode", values.get("mode", "full")),
        momentum_policy=policy,
        average_momentum_mev=momentum,
        update_statistic=overrides.get("likelihood_update", values["update_statistic"]),
        angular_resolution_rad=float(values["angular_resolution_rad"]),
        position_resolution_cm=float(values["position_resolution_cm"]),
        maximum_inverse_radiation_length_cm_inv=float(
            values["maximum_inverse_radiation_length_cm_inv"]
        ),
    )


METHODS = {
    "munet": MethodSpec("munet", "MuNet", "production", "frozen-munet-v1", _munet),
    "poca": MethodSpec("poca", "PoCA", "production", "exposure-pooled-poca-v1", _poca),
    "likelihood": MethodSpec(
        "likelihood", "Likelihood", "production", "covariance-mlem-v1", _likelihood
    ),
}


def selected_methods(requested: str) -> list[str]:
    if requested == "all":
        return [name for name, spec in METHODS.items() if spec.status == "production"]
    if requested not in METHODS:
        raise ValueError(f"unknown method {requested!r}; choose all or {', '.join(METHODS)}")
    return [requested]


def build_reconstructor(
    method: str,
    config: dict,
    *,
    geometry: GridGeometry | None = None,
    overrides: dict | None = None,
):
    """Dispatch an explicitly registered factory without an implicit fallback."""
    if method not in METHODS:
        raise ValueError(f"unknown reconstruction method: {method}")
    geometry = geometry or geometry_from_config(config)
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    return METHODS[method].factory(config, geometry, overrides)
