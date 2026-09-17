"""Aligned cross-section plots for all reconstruction methods."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .compare import compare_to_truth
from .constants import INVERSE_RADIATION_LENGTH_CM_INV, MATERIAL_NAMES
from .types import ReconstructionResult, TruthVolume

# Fixed physical scale for all 2-D reconstruction comparison plots.
COMPARISON_VMAX_CM_INV = 3.2


def _z_slab(volume: np.ndarray, center_index: int, slab_layers: int) -> tuple[np.ndarray, int, int]:
    if slab_layers < 1 or slab_layers % 2 != 1:
        raise ValueError("slab_layers must be a positive odd number")
    half = slab_layers // 2
    first = center_index - half
    last = center_index + half + 1
    if first < 0 or last > volume.shape[2]:
        raise ValueError("requested z slab falls outside the volume")
    return np.mean(volume[:, :, first:last], axis=2), first, last - 1


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _axis_projection(
    volume: np.ndarray,
    center_index: int,
    slab_layers: int,
    full_depth: bool,
    projection_axis: str = "z",
) -> tuple[np.ndarray, int, int]:
    if projection_axis not in AXIS_INDEX:
        raise ValueError("projection_axis must be x, y, or z")
    axis = AXIS_INDEX[projection_axis]
    if full_depth:
        return np.mean(volume, axis=axis), 0, volume.shape[axis] - 1
    if projection_axis == "z":
        return _z_slab(volume, center_index, slab_layers)
    if slab_layers < 1 or slab_layers % 2 != 1:
        raise ValueError("slab_layers must be a positive odd number")
    half = slab_layers // 2
    first = center_index - half
    last = center_index + half + 1
    if first < 0 or last > volume.shape[axis]:
        raise ValueError(f"requested {projection_axis} slab falls outside the volume")
    selection = [slice(None)] * volume.ndim
    selection[axis] = slice(first, last)
    return np.mean(volume[tuple(selection)], axis=axis), first, last - 1


def _z_projection(
    volume: np.ndarray,
    center_index: int,
    slab_layers: int,
    full_depth: bool,
) -> tuple[np.ndarray, int, int]:
    """Backward-compatible z projection helper."""
    return _axis_projection(volume, center_index, slab_layers, full_depth, "z")


def plot_comparison_slab(
    truth: TruthVolume,
    results: list[ReconstructionResult],
    *,
    output: str | Path,
    center_index: int = 32,
    slab_layers: int = 7,
    full_depth: bool = False,
    projection_axis: str = "z",
) -> dict[str, object]:
    """Write a 2x2 truth/MuNet/PoCA/likelihood projection comparison."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise RuntimeError("plotting requires the 'plot' optional dependency") from exc

    by_method = {result.method: result for result in results}
    expected = {"munet", "poca", "likelihood"}
    if set(by_method) != expected or len(results) != 3:
        raise ValueError(f"expected exactly one result for each of {sorted(expected)}")
    for result in results:
        if result.geometry != truth.geometry:
            raise ValueError(f"{result.method} geometry differs from truth")

    truth_slab, first, last = _axis_projection(
        truth.inverse_radiation_length_cm_inv,
        center_index,
        slab_layers,
        full_depth,
        projection_axis,
    )
    panels = [("Truth", truth_slab, 0.0)]
    metrics: dict[str, object] = {}
    for method, label in (
        ("munet", "MuNet"),
        ("poca", "PoCA"),
        ("likelihood", "Likelihood"),
    ):
        result = by_method[method]
        slab, _, _ = _axis_projection(
            result.inverse_radiation_length_cm_inv,
            center_index,
            slab_layers,
            full_depth,
            projection_axis,
        )
        row = compare_to_truth(result, truth)
        metrics[method] = row
        panels.append((label, slab, float(row["mse_full_grid_cm_inv_squared"])))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), constrained_layout=True)
    vmax = COMPARISON_VMAX_CM_INV
    plane_axes = [name for name in ("x", "y", "z") if name != projection_axis]
    horizontal, vertical = plane_axes
    images = []
    for axis, (label, slab, mse) in zip(axes.flat, panels, strict=True):
        image = axis.imshow(
            slab.T,
            origin="lower",
            interpolation="nearest",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            extent=(-50.0, 50.0, -50.0, 50.0),
        )
        images.append(image)
        axis.set_title(f"{label}\nfull 64³ MSE = {mse:.4g} (cm⁻¹)²")
        axis.set_xlabel(f"{horizontal} [cm] (64 voxels)")
        axis.set_ylabel(f"{vertical} [cm] (64 voxels)")
        axis.set_aspect("equal")

    labels = sorted(int(x) for x in np.unique(truth.material_labels) if int(x) != 0)
    material_text = "\n".join(
        f"{MATERIAL_NAMES[label].title()}: {INVERSE_RADIATION_LENGTH_CM_INV[label]:.3f} cm⁻¹"
        for label in labels
    )
    axes.flat[0].text(
        0.02,
        0.98,
        material_text,
        transform=axes.flat[0].transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.62, "edgecolor": "none", "pad": 4},
    )
    mean_p = by_method["poca"].metadata.get("assumed_momentum_mev")
    munet_momentum = by_method["munet"].metadata.get("momentum_estimate_mev")
    if by_method["likelihood"].metadata.get("momentum_policy") == "external direction-conditioned population prior":
        policy_text = (
            f"PoCA: fixed p = {float(mean_p):.1f} MeV; "
            "likelihood: external direction-conditioned spectrum; "
            f"MuNet: fixed p estimate = {float(munet_momentum):.1f} MeV"
        )
    else:
        policy_text = (
        f"PoCA + likelihood: one shared p = {float(mean_p):.1f} MeV; "
        f"MuNet: fixed p estimate = {float(munet_momentum):.1f} MeV"
        if mean_p is not None and munet_momentum is not None
        else "Momentum policy not recorded"
        )
    depth_text = (
        f"full-depth mean over {projection_axis}=0-{last} ({last + 1} layers)"
        if full_depth
        else (f"mean {projection_axis} slab {first}-{last} ({slab_layers} layers)")
    )
    source_name = Path(truth.metadata.get("source", "dataset")).name
    record_index = truth.metadata.get("record_index", "unknown")
    fig.suptitle(
        f"{source_name} · record {record_index} — {depth_text}\n{policy_text}", fontsize=13
    )
    fig.colorbar(
        images[0],
        ax=axes,
        label="inverse radiation length 1/X₀ [cm⁻¹]",
        shrink=0.9,
        extend="max",
        ticks=np.linspace(0.0, vmax, 5),
    )
    material_colors = {
        0: "#111111",  # air
        17: "#BFC5CC",  # silver
        19: "#58636D",  # lead
        21: "#D9A441",  # uranium
    }
    norm = plt.Normalize(vmin=0.0, vmax=vmax)
    legend_handles = [
        Patch(
            facecolor=material_colors.get(
                label, plt.get_cmap("magma")(norm(INVERSE_RADIATION_LENGTH_CM_INV[label]))
            ),
            edgecolor="white",
            linewidth=0.8,
            label=(
                f"{MATERIAL_NAMES[label].title()}  "
                f"{INVERSE_RADIATION_LENGTH_CM_INV[label]:.3f} cm⁻¹"
            ),
        )
        for label in [0, *labels]
    ]
    axes.flat[0].legend(
        handles=legend_handles,
        loc="lower left",
        title="Truth material / 1/X₀",
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        framealpha=0.92,
        facecolor="#222222",
        edgecolor="white",
        labelcolor="white",
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return {
        "output": str(target.resolve()),
        "projection_axis": projection_axis,
        "center_axis_index": None if full_depth else center_index,
        "slab_first_axis_index": first,
        "slab_last_axis_index": last,
        "slab_layers": last - first + 1,
        "projection": (
            f"full-depth {projection_axis} mean" if full_depth else f"{projection_axis}-slab mean"
        ),
        "mse_scope": "full aligned 64^3 grid",
        "display_color_scale_cm_inv": [0.0, vmax],
        "display_values_above_scale": "saturated; MSE uses unclipped values",
        "classical_momentum_policy": by_method["poca"].metadata.get("momentum_policy"),
        "assumed_momentum_mev": mean_p,
        "methods": metrics,
    }
