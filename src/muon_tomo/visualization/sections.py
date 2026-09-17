"""Cross sections for arbitrary methods, with optional evaluation truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..constants import INVERSE_RADIATION_LENGTH_CM_INV, MATERIAL_NAMES
from ..types import ReconstructionResult, TruthVolume
from .panels import momentum_caption, prepare_panels

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
    truth: TruthVolume | None,
    results: list[ReconstructionResult],
    *,
    output: str | Path,
    center_index: int | None = None,
    slab_layers: int | None = None,
    full_depth: bool = False,
    projection_axis: str = "z",
) -> dict[str, object]:
    """Render raw-volume slabs; metrics always use full, unthresholded arrays."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires the 'plot' optional dependency") from exc
    geometry, panels = prepare_panels(truth, results)
    if projection_axis not in AXIS_INDEX:
        raise ValueError("projection_axis must be x, y, or z")
    depth_size = geometry.shape[AXIS_INDEX[projection_axis]]
    if center_index is None:
        center_index = depth_size // 2
    if slab_layers is None:
        slab_layers = min(7, 2 * min(center_index, depth_size - center_index - 1) + 1)
    columns = min(2, len(panels))
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(6.25 * columns, 5.25 * rows), constrained_layout=True, squeeze=False
    )
    plane_axes = [i for i in range(3) if i != AXIS_INDEX.get(projection_axis, -1)]
    metrics = {}
    first = last = 0
    for axis, panel in zip(axes.flat, panels, strict=False):
        slab, first, last = _axis_projection(
            panel.volume, center_index, slab_layers, full_depth, projection_axis
        )
        xaxis, yaxis = plane_axes
        image = axis.imshow(
            slab.T,
            origin="lower",
            interpolation="nearest",
            cmap="magma",
            vmin=0.0,
            vmax=COMPARISON_VMAX_CM_INV,
            extent=(
                geometry.lower_cm[xaxis],
                geometry.upper_cm[xaxis],
                geometry.lower_cm[yaxis],
                geometry.upper_cm[yaxis],
            ),
        )
        title = panel.label
        if panel.is_truth:
            title += "\nreference 1/X₀"
        elif panel.metrics is not None:
            metrics[panel.key] = panel.metrics
            mse = panel.metrics["mse_full_grid_cm_inv_squared"]
            title += f"\nfull-grid MSE = {mse:.4g} cm⁻²"
        axis.set_title(title)
        axis.set_xlabel(f"{'xyz'[xaxis]} [cm]")
        axis.set_ylabel(f"{'xyz'[yaxis]} [cm]")
        axis.set_aspect("equal")
    for axis in list(axes.flat)[len(panels) :]:
        axis.set_visible(False)
    if truth is not None:
        labels = sorted(int(x) for x in np.unique(truth.material_labels) if int(x) != 0)
        material_text = "\n".join(
            f"{MATERIAL_NAMES[label].title()}: {INVERSE_RADIATION_LENGTH_CM_INV[label]:.3f} cm⁻¹"
            for label in labels
            if 0 <= label < len(MATERIAL_NAMES)
        )
        if material_text:
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
    source_meta = truth.metadata if truth is not None else results[0].metadata
    source = Path(
        str(source_meta.get("source", source_meta.get("input_source", "detector data")))
    ).name
    record = source_meta.get("record_index", "unknown")
    depth = f"mean {projection_axis} layers {first}-{last}"
    fig.suptitle(f"{source} | record {record} - {depth}\n{momentum_caption(results)}", fontsize=11)
    fig.colorbar(
        image,
        ax=list(axes.flat)[: len(panels)],
        label="inverse radiation length 1/X₀ [cm⁻¹]",
        shrink=0.9,
        extend="max",
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
        "projection": f"{'full-depth' if full_depth else 'slab'} {projection_axis} mean",
        "mse_scope": "full aligned grid" if truth is not None else "not evaluated: no truth",
        "display_color_scale_cm_inv": [0.0, COMPARISON_VMAX_CM_INV],
        "display_values_above_scale": "saturated; MSE uses unclipped values",
        "methods": metrics,
    }
