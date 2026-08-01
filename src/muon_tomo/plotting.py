"""Aligned cross-section plots for all reconstruction methods."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .compare import compare_to_truth
from .constants import INVERSE_RADIATION_LENGTH_CM_INV, MATERIAL_NAMES
from .types import ReconstructionResult, TruthVolume


def _z_slab(
    volume: np.ndarray, center_index: int, slab_layers: int
) -> tuple[np.ndarray, int, int]:
    if slab_layers < 1 or slab_layers % 2 != 1:
        raise ValueError("slab_layers must be a positive odd number")
    half = slab_layers // 2
    first = center_index - half
    last = center_index + half + 1
    if first < 0 or last > volume.shape[2]:
        raise ValueError("requested z slab falls outside the volume")
    return np.mean(volume[:, :, first:last], axis=2), first, last - 1


def plot_comparison_slab(
    truth: TruthVolume,
    results: list[ReconstructionResult],
    *,
    output: str | Path,
    center_index: int = 32,
    slab_layers: int = 7,
) -> dict[str, object]:
    """Write a 2x2 truth/MuNet/PoCA/likelihood z-slab comparison."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires the 'plot' optional dependency") from exc

    by_method = {result.method: result for result in results}
    expected = {"munet", "poca", "likelihood"}
    if set(by_method) != expected or len(results) != 3:
        raise ValueError(f"expected exactly one result for each of {sorted(expected)}")
    for result in results:
        if result.geometry != truth.geometry:
            raise ValueError(f"{result.method} geometry differs from truth")

    truth_slab, first, last = _z_slab(
        truth.inverse_radiation_length_cm_inv, center_index, slab_layers
    )
    panels = [("Truth", truth_slab, 0.0)]
    metrics: dict[str, object] = {}
    for method, label in (
        ("munet", "MuNet"),
        ("poca", "PoCA"),
        ("likelihood", "Likelihood"),
    ):
        result = by_method[method]
        slab, _, _ = _z_slab(
            result.inverse_radiation_length_cm_inv, center_index, slab_layers
        )
        row = compare_to_truth(result, truth)
        metrics[method] = row
        panels.append((label, slab, float(row["mse_full_grid_cm_inv_squared"])))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), constrained_layout=True)
    vmax = float(np.max(truth.inverse_radiation_length_cm_inv))
    images = []
    for axis, (label, slab, mse) in zip(axes.flat, panels):
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
        axis.set_title(f"{label}\nfull 64³ MSE = {mse:.4g} cm⁻²")
        axis.set_xlabel("x [cm] (64 voxels)")
        axis.set_ylabel("y [cm] (64 voxels)")
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
    policy_text = (
        f"PoCA + likelihood: one shared p = {float(mean_p):.1f} MeV; "
        "MuNet: existing frozen model with zero momentum feature"
        if mean_p is not None
        else "Momentum policy not recorded"
    )
    fig.suptitle(
        f"Funny record comparison — mean z slab {first}–{last} ({slab_layers} layers)\n"
        f"{policy_text}",
        fontsize=13,
    )
    fig.colorbar(
        images[0],
        ax=axes,
        label="inverse radiation length 1/X₀ [cm⁻¹]",
        shrink=0.9,
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return {
        "output": str(target.resolve()),
        "center_z_index": center_index,
        "slab_first_z_index": first,
        "slab_last_z_index": last,
        "slab_layers": slab_layers,
        "mse_scope": "full aligned 64^3 grid",
        "classical_momentum_policy": by_method["poca"].metadata.get(
            "momentum_policy"
        ),
        "assumed_momentum_mev": mean_p,
        "methods": metrics,
    }
