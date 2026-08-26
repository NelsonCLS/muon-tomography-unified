"""Interactive 3D voxel-cube views for aligned tomography volumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

from .compare import compare_to_truth
from .types import ReconstructionResult, TruthVolume

CUBE_OFFSETS = np.asarray(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float64,
)

CUBE_TRIANGLES = np.asarray(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ],
    dtype=np.int64,
)


def _surface_mask(mask: np.ndarray, separate_regions: bool = False) -> np.ndarray:
    """Return six-neighbor boundary voxels, optionally per discrete value region."""
    mask = np.asarray(mask)
    if separate_regions:
        surface = np.zeros(mask.shape, dtype=bool)
        for value in np.unique(mask):
            if value == 0:
                continue
            region = mask == value
            surface |= region & ~binary_erosion(region)
        return surface
    region = mask.astype(bool)
    return region & ~binary_erosion(region)


def _select_voxels(
    volume: np.ndarray,
    threshold_cm_inv: float,
    max_cubes: int,
    *,
    surface_only: bool,
    separate_regions: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Select a deterministic responsive subset of physical voxels."""
    values = np.asarray(volume, dtype=np.float64)
    eligible = np.isfinite(values) & (values >= threshold_cm_inv)
    if surface_only:
        if separate_regions:
            region_ids = np.where(eligible, values, 0.0)
            eligible &= _surface_mask(region_ids, separate_regions=True)
        else:
            eligible &= _surface_mask(eligible)
    indices = np.argwhere(eligible)
    selected_values = values[tuple(indices.T)] if len(indices) else np.empty(0)
    candidate_count = len(indices)
    if candidate_count > max_cubes:
        chosen = np.sort(
            np.random.default_rng(seed).choice(candidate_count, size=max_cubes, replace=False)
        )
        indices = indices[chosen]
        selected_values = selected_values[chosen]
    return indices, selected_values, candidate_count


def _density_opacity_groups(
    values: np.ndarray,
    maximum_value: float,
    maximum_opacity: float,
    bins: int = 6,
) -> list[tuple[np.ndarray, float]]:
    """Group cubes into monotonic density-opacity bands for Plotly meshes."""
    values = np.asarray(values, dtype=np.float64)
    if maximum_value <= 0.0 or bins < 1:
        raise ValueError("maximum_value and bins must be positive")
    normalized = np.clip(values / maximum_value, 0.0, 1.0)
    bin_ids = np.minimum((normalized * bins).astype(np.int64), bins - 1)
    groups = []
    for bin_id in np.unique(bin_ids):
        selected = bin_ids == bin_id
        group_opacity = max(float(maximum_opacity * np.mean(normalized[selected])), 1.0e-4)
        groups.append((selected, group_opacity))
    return groups


def _voxel_cube_mesh(
    indices: np.ndarray,
    values: np.ndarray,
    lower_cm: tuple[float, float, float],
    voxel_size_cm: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    """Build one indexed triangular mesh containing independent voxel cubes."""
    indices = np.asarray(indices, dtype=np.float64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(indices) != len(values):
        raise ValueError("indices and values must have equal length")
    lower = np.asarray(lower_cm, dtype=np.float64)
    size = np.asarray(voxel_size_cm, dtype=np.float64)
    centers = lower + (indices + 0.5) * size
    vertices = centers[:, None, :] + 0.5 * CUBE_OFFSETS[None, :, :] * size
    offsets = 8 * np.arange(len(indices), dtype=np.int64)[:, None, None]
    triangles = CUBE_TRIANGLES[None, :, :] + offsets
    flat_vertices = vertices.reshape(-1, 3)
    flat_triangles = triangles.reshape(-1, 3)
    return {
        "x": flat_vertices[:, 0],
        "y": flat_vertices[:, 1],
        "z": flat_vertices[:, 2],
        "i": flat_triangles[:, 0],
        "j": flat_triangles[:, 1],
        "k": flat_triangles[:, 2],
        "intensity": np.repeat(values, 8),
    }


def _region_edges(
    lower_cm: tuple[float, float, float],
    upper_cm: tuple[float, float, float],
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    lower = np.asarray(lower_cm, dtype=float)
    upper = np.asarray(upper_cm, dtype=float)
    corners = np.asarray(
        [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], upper[1], lower[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], upper[2]],
            [lower[0], upper[1], upper[2]],
        ]
    )
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    xyz: list[list[float | None]] = [[], [], []]
    for first, second in edges:
        for axis in range(3):
            xyz[axis].extend([corners[first, axis], corners[second, axis], None])
    return xyz[0], xyz[1], xyz[2]


def plot_comparison_3d(
    truth: TruthVolume,
    results: list[ReconstructionResult],
    *,
    output: str | Path,
    threshold_cm_inv: float = 0.7,
    max_cubes: int = 12000,
    opacity: float = 0.6,
    surface_only: bool = False,
) -> dict[str, object]:
    """Write a rotatable HTML viewer with selectable aligned voxel-cube volumes."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("3D plotting requires the plot optional dependency") from exc

    if not np.isfinite(threshold_cm_inv) or threshold_cm_inv < 0.0:
        raise ValueError("threshold_cm_inv must be finite and nonnegative")
    if max_cubes < 1:
        raise ValueError("max_cubes must be positive")
    if not np.isfinite(opacity) or not 0.0 < opacity <= 1.0:
        raise ValueError("opacity must be in (0, 1]")
    by_method = {result.method: result for result in results}
    expected = {"munet", "poca", "likelihood"}
    if set(by_method) != expected or len(results) != 3:
        raise ValueError(f"expected exactly one result for each of {sorted(expected)}")
    for result in results:
        if result.geometry != truth.geometry:
            raise ValueError(f"{result.method} geometry differs from truth")

    geometry = truth.geometry
    panels = [
        ("Truth", truth.inverse_radiation_length_cm_inv, None, True),
        (
            "MuNet",
            by_method["munet"].inverse_radiation_length_cm_inv,
            compare_to_truth(by_method["munet"], truth),
            False,
        ),
        (
            "PoCA",
            by_method["poca"].inverse_radiation_length_cm_inv,
            compare_to_truth(by_method["poca"], truth),
            False,
        ),
        (
            "Likelihood",
            by_method["likelihood"].inverse_radiation_length_cm_inv,
            compare_to_truth(by_method["likelihood"], truth),
            False,
        ),
    ]
    vmax = float(np.max(truth.inverse_radiation_length_cm_inv))
    edge_x, edge_y, edge_z = _region_edges(geometry.lower_cm, geometry.upper_cm)
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line={"color": "rgba(120,120,120,0.75)", "width": 4},
                hoverinfo="skip",
                name="64^3 reconstruction boundary",
                showlegend=False,
            )
        ]
    )
    report_panels: dict[str, object] = {}
    button_labels: list[str] = []
    panel_trace_indices: list[list[int]] = []
    for panel_index, (label, volume, metrics, discrete) in enumerate(panels):
        # Keep the truth volume complete so solid material is not made sparse
        # by the responsiveness cap used for reconstructed volumes.
        panel_max_cubes = max_cubes
        if discrete:
            panel_max_cubes = max(
                max_cubes,
                int(np.count_nonzero(np.isfinite(volume) & (volume >= threshold_cm_inv))),
            )
        indices, values, candidates = _select_voxels(
            volume,
            threshold_cm_inv,
            panel_max_cubes,
            surface_only=surface_only,
            separate_regions=discrete,
            seed=panel_index,
        )
        mse = 0.0 if metrics is None else float(metrics["mse_full_grid_cm_inv_squared"])
        button_label = f"{label} ({len(indices):,} cubes)"
        button_labels.append(button_label)
        trace_indices = []
        opacity_values = []
        for group_index, (selected, group_opacity) in enumerate(
            _density_opacity_groups(values, vmax, opacity)
        ):
            mesh = _voxel_cube_mesh(
                indices[selected],
                values[selected],
                geometry.lower_cm,
                geometry.voxel_size_cm,
            )
            figure.add_trace(
                go.Mesh3d(
                    **mesh,
                    name=label,
                    visible=panel_index == 0,
                    opacity=group_opacity,
                    colorscale="Magma",
                    cmin=0.0,
                    cmax=vmax,
                    flatshading=True,
                    showscale=group_index == 0,
                    colorbar={
                        "title": "1/X0 [cm^-1]",
                        "len": 0.72,
                        "thickness": 18,
                    },
                    lighting={
                        "ambient": 0.72,
                        "diffuse": 0.55,
                        "specular": 0.08,
                        "roughness": 0.9,
                        "fresnel": 0.05,
                    },
                    lightposition={"x": 150, "y": 120, "z": 180},
                    hovertemplate=(
                        f"{label}<br>x=%{{x:.2f}} cm<br>y=%{{y:.2f}} cm"
                        "<br>z=%{z:.2f} cm<br>1/X0=%{intensity:.3f} cm^-1"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            trace_indices.append(len(figure.data) - 1)
            opacity_values.append(group_opacity)
        panel_trace_indices.append(trace_indices)
        report_panels[label.lower()] = {
            "eligible_cubes": candidates,
            "displayed_cubes": len(indices),
            "mse_full_grid_cm_inv_squared": mse,
            "minimum_display_opacity": (float(min(opacity_values)) if opacity_values else 0.0),
            "maximum_display_opacity": (float(max(opacity_values)) if opacity_values else 0.0),
        }

    buttons = []
    for selected, label in enumerate(button_labels):
        visible = [False] * len(figure.data)
        visible[0] = True
        for trace_index in panel_trace_indices[selected]:
            visible[trace_index] = True
        panel_name = panels[selected][0]
        mse = report_panels[panel_name.lower()]["mse_full_grid_cm_inv_squared"]
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": {
                            "text": (
                                f"{panel_name} - rotatable 64^3 voxel view"
                                f"<br><sup>full-grid MSE = {mse:.4g} cm^-2; "
                                f"threshold = {threshold_cm_inv:.3g} cm^-1; "
                                f"max opacity = {opacity:.0%}</sup>"
                            )
                        }
                    },
                ],
            }
        )

    axis_style = {
        "range": [float(geometry.lower_cm[0]), float(geometry.upper_cm[0])],
        "backgroundcolor": "rgba(0,0,0,0)",
        "gridcolor": "rgba(128,128,128,0.25)",
        "zerolinecolor": "rgba(128,128,128,0.45)",
        "showbackground": False,
    }
    figure.update_layout(
        title={
            "text": (
                "Truth - rotatable 64^3 voxel view"
                f"<br><sup>full-grid MSE = 0 cm^-2; "
                f"threshold = {threshold_cm_inv:.3g} cm^-1; "
                f"max opacity = {opacity:.0%}</sup>"
            ),
            "x": 0.5,
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.01,
                "xanchor": "left",
                "y": 1.08,
                "yanchor": "top",
            }
        ],
        scene={
            "xaxis": {**axis_style, "title": "x [cm]"},
            "yaxis": {**axis_style, "title": "y [cm]"},
            "zaxis": {**axis_style, "title": "z [cm]"},
            "aspectmode": "cube",
            "camera": {
                "eye": {"x": 1.55, "y": 1.55, "z": 1.25},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
            },
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 95},
        height=820,
        template="plotly_white",
    )
    target = Path(output)
    if target.suffix.lower() != ".html":
        raise ValueError("3D plot output must use the .html extension")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        target,
        include_plotlyjs=True,
        full_html=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )
    return {
        "output": str(target.resolve()),
        "geometry": geometry.metadata(),
        "threshold_cm_inv": threshold_cm_inv,
        "max_cubes_per_volume": max_cubes,
        "maximum_opacity": opacity,
        "opacity_mapping": "six monotonic density bands",
        "surface_only": surface_only,
        "color_scale_cm_inv": [0.0, vmax],
        "panels": report_panels,
    }
