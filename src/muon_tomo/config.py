"""Resolve and validate reconstruction settings without invoking algorithms.

The packaged default YAML is authoritative. User files are partial overrides;
unknown keys fail early. ``average_momentum_mev`` is the historical spelling of
an inverse-square effective momentum, not necessarily an arithmetic mean.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from .geometry import GridGeometry

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "default.yaml"
_ALLOWED_EXTRA = {
    "physics": {"effective_momentum_mev"},
    "likelihood": {"mode"},
    "munet": {"inference_seed"},
}


def geometry_from_config(config: dict) -> GridGeometry:
    values = config["geometry"]
    return GridGeometry(
        tuple(values["lower_cm"]),
        tuple(values["upper_cm"]),
        tuple(values["shape"]),
        values["axis_order"],
    )


def _positive(value, name: str, *, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def validate_config(config: dict) -> None:
    """Validate physical ranges and supported settings before loading a dataset."""
    geometry_from_config(config)
    physics = config["physics"]
    for key in ("highland_constant_mev", "reference_momentum_mev", "average_momentum_mev"):
        _positive(physics[key], f"physics.{key}")
    if physics["classical_momentum_policy"] not in {"mean", "per-track"}:
        raise ValueError("physics.classical_momentum_policy must be mean or per-track")
    munet = config["munet"]
    inference_seed = munet.get("inference_seed", 0)
    if (
        isinstance(inference_seed, bool)
        or not isinstance(inference_seed, int)
        or inference_seed < 0
    ):
        raise ValueError("munet.inference_seed must be a nonnegative integer")
    if munet["dosage"] not in (512, 1024):
        raise ValueError("munet.dosage must be 512 or 1024")
    weights = munet.get("weights")
    if (
        weights in {"model_512.weights.h5", "model_1024.weights.h5"}
        and weights != f"model_{munet['dosage']}.weights.h5"
    ):
        raise ValueError("munet.weights built-in filename must match munet.dosage")
    for key in ("position_scale_cm", "momentum_estimate_mev"):
        _positive(munet[key], f"munet.{key}")
    if not np.isfinite(munet["position_offset"]):
        raise ValueError("munet.position_offset must be finite")
    poca = config["poca"]
    if poca["grid_size"] not in (32, 64):
        raise ValueError("poca.grid_size must be 32 or 64")
    _positive(poca["pooling_sigma_cm"], "poca.pooling_sigma_cm", zero=True)
    likelihood = config["likelihood"]
    if likelihood.get("mode", "full") not in {"full", "angular"}:
        raise ValueError("likelihood.mode must be full or angular")
    if likelihood["update_statistic"] not in {"mean", "median"}:
        raise ValueError("likelihood.update_statistic must be mean or median")
    for section, key in (
        ("poca", "minimum_coverage"),
        ("likelihood", "minimum_coverage"),
        ("likelihood", "iterations"),
    ):
        value = config[section][key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{section}.{key} must be a positive integer")
    for key in (
        "initial_inverse_radiation_length_cm_inv",
        "epsilon",
        "maximum_inverse_radiation_length_cm_inv",
    ):
        _positive(likelihood[key], f"likelihood.{key}")
    for key in ("angular_resolution_rad", "position_resolution_cm"):
        _positive(likelihood[key], f"likelihood.{key}", zero=True)


def load_config(path: str | Path | None = None) -> dict:
    """Merge a partial YAML override onto the packaged defaults and validate it."""
    defaults = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config = deepcopy(defaults)
    if path is not None:
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"configuration not found: {target}")
        override = yaml.safe_load(target.read_text())
        if not isinstance(override, dict):
            raise ValueError("configuration must be a YAML mapping")
        unknown = set(override) - set(defaults)
        if unknown:
            raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
        for section, values in override.items():
            if not isinstance(values, dict):
                raise ValueError(f"configuration section {section!r} must be a mapping")
            unknown = set(values) - set(defaults[section]) - _ALLOWED_EXTRA.get(section, set())
            if unknown:
                raise ValueError(f"unknown settings in {section}: {sorted(unknown)}")
            values = dict(values)
            if section == "physics" and "effective_momentum_mev" in values:
                effective = values.pop("effective_momentum_mev")
                if "average_momentum_mev" in values and values["average_momentum_mev"] != effective:
                    raise ValueError("effective_momentum_mev and average_momentum_mev conflict")
                values["average_momentum_mev"] = effective
            if section == "munet" and "dosage" in values and "weights" not in values:
                values["weights"] = f"model_{values['dosage']}.weights.h5"
            if section == "munet" and values.get("weights"):
                weights = str(values["weights"])
                if weights not in {"model_512.weights.h5", "model_1024.weights.h5"}:
                    location = Path(weights).expanduser()
                    values["weights"] = str(
                        (target.parent / location).resolve()
                        if not location.is_absolute()
                        else location
                    )
            config[section].update(values)
    validate_config(config)
    return config
