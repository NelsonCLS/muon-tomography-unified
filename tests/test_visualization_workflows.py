import numpy as np
import pytest

from muon_tomo.geometry import GridGeometry
from muon_tomo.plotting import plot_comparison_slab
from muon_tomo.plotting3d import _select_voxels, plot_comparison_3d
from muon_tomo.types import ReconstructionResult
from muon_tomo.visualization.panels import prepare_panels


def result(method="poca"):
    g = GridGeometry()
    v = np.zeros(g.shape, np.float32)
    v[31:34, 31:34, 31:34] = 1.2
    return ReconstructionResult(method, v, v.copy(), (v > 0).astype(int), v > 0, g)


def test_strict_threshold_retains_all_interior_voxels_without_cap():
    v = np.ones((25, 25, 25), np.float32)
    v[0, 0, 0] = np.float32(0.8)
    indices, values, total = _select_voxels(v, 0.8, None, surface_only=False)
    assert total == 25**3 - 1
    assert len(indices) == total > 12000
    assert np.any(np.all(indices == (12, 12, 12), axis=1))
    assert np.all(values > np.float32(0.8))


def test_arbitrary_duplicate_method_panels_have_unique_identity():
    _, panels = prepare_panels(None, [result(), result(), result("new_method")])
    assert [p.key for p in panels] == ["poca", "poca_2", "new_method"]
    assert all(p.metrics is None for p in panels)


def test_single_method_without_truth_renders_both_formats(tmp_path):
    r = result()
    before = r.inverse_radiation_length_cm_inv.copy()
    section = plot_comparison_slab(None, [r], output=tmp_path / "section.png")
    viewer = plot_comparison_3d(None, [r], output=tmp_path / "viewer.html")
    assert (tmp_path / "section.png").stat().st_size > 1000
    assert (tmp_path / "viewer.html").stat().st_size > 1000
    assert not section["methods"]
    assert viewer["panels"]["poca"]["mse_full_grid_cm_inv_squared"] is None
    assert viewer["panels"]["poca"]["displayed_cubes"] == 27
    assert viewer["max_cubes_per_volume"] is None
    np.testing.assert_array_equal(before, r.inverse_radiation_length_cm_inv)


def test_geometry_mismatch_is_not_silently_plotted():
    r = result()
    other = result()
    other.geometry = GridGeometry(lower_cm=(-10, -10, -10), upper_cm=(10, 10, 10))
    with pytest.raises(ValueError, match="geometry"):
        prepare_panels(None, [r, other])


def test_empty_coverage_has_full_grid_metrics_but_no_covered_metrics():
    from muon_tomo.evaluation.metrics import compare_to_truth
    from muon_tomo.geometry import GridGeometry
    from muon_tomo.types import ReconstructionResult, TruthVolume

    geometry = GridGeometry(shape=(2, 2, 2))
    zeros = np.zeros(geometry.shape, dtype=np.float32)
    result = ReconstructionResult(
        "empty", zeros, zeros.copy(), zeros.astype(int), zeros.astype(bool), geometry
    )
    truth = TruthVolume(zeros.astype(int), np.ones(geometry.shape), geometry)
    metrics = compare_to_truth(result, truth)
    assert metrics["mse_full_grid_cm_inv_squared"] == 1.0
    assert metrics["valid_voxels"] == 0
    assert metrics["rmse_cm_inv"] is None


def test_small_asymmetric_grid_plots_with_automatic_slab(tmp_path):
    geometry = GridGeometry(shape=(2, 4, 6), lower_cm=(-1, -2, -3), upper_cm=(1, 2, 3))
    values = np.ones(geometry.shape, np.float32)
    result = ReconstructionResult(
        "new_grid", values, values.copy(), values.astype(int), values.astype(bool), geometry
    )
    section = plot_comparison_slab(None, [result], output=tmp_path / "small.png")
    viewer = plot_comparison_3d(None, [result], output=tmp_path / "small.html")
    assert section["center_axis_index"] == 3
    assert section["slab_layers"] == 5
    assert viewer["panels"]["new_grid"]["displayed_cubes"] == 48
