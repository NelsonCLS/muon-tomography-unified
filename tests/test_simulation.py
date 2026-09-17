from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from muon_tomo.cli import build_parser
from muon_tomo.constants import MATERIAL_NAMES
from muon_tomo.simulation.objects import create_voxels, load_voxels, save_voxels
from muon_tomo.simulation.pipeline import SourceConfig, generate_dataset, run_simulation
from muon_tomo.simulation.records import CSV_COLUMNS, read_simulation_csv, write_records


def box(**kwargs):
    return {"type": "box", "material": "lead", "size_cm": [25, 12.5, 6.25], **kwargs}


def test_physical_grid_and_rotation():
    # Off-centre rectangular solid: known cell boundaries, cardinal rotation swaps extents.
    a = create_voxels({"shapes": [box(center_cm=[12.5, -12.5, 0])]})
    assert (a == 19).sum() == 16 * 8 * 4
    indices = np.argwhere(a)
    np.testing.assert_array_equal(indices.min(axis=0), [32, 20, 30])
    np.testing.assert_array_equal(indices.max(axis=0), [47, 27, 33])
    b = create_voxels({"shapes": [box(rotation_deg=[0, 0, 90])]})
    np.testing.assert_array_equal(np.ptp(np.argwhere(b), axis=0) + 1, [8, 16, 4])
    # Mirror symmetry around the true origin catches half-voxel centre shifts.
    c = create_voxels({"shapes": [box()]})
    for axis in range(3):
        np.testing.assert_array_equal(c, np.flip(c, axis=axis))


def test_cavity_and_overlap_validation():
    solid = box(size_cm=[50, 50, 50])
    cavity = box(material="air", size_cm=[25, 25, 25])
    with pytest.raises(ValueError, match="overlaps"):
        create_voxels({"shapes": [solid, cavity]})
    cavity["overlap"] = "replace"
    carved = create_voxels({"shapes": [solid, cavity]})
    assert np.all(carved[24:40, 24:40, 24:40] == 0)
    assert (carved == 19).sum() == 32**3 - 16**3
    hollow = create_voxels({"shapes": [{**solid, "type": "hollow_box", "wall_cm": 12.5}]})
    np.testing.assert_array_equal(hollow, carved)


@pytest.mark.parametrize(
    "shape",
    [
        {"type": "sphere", "radius_cm": 10},
        {"type": "ellipsoid", "radii_cm": [8, 10, 12]},
        {"type": "cylinder", "radius_cm": 10, "inner_radius_cm": 5, "height_cm": 20},
    ],
)
def test_curved_solids_are_symmetric(shape):
    voxels = create_voxels({"shapes": [dict(shape, material="water")]})
    assert set(np.unique(voxels)) == {0, 3}
    for axis in range(3):
        np.testing.assert_array_equal(voxels, np.flip(voxels, axis=axis))


@pytest.mark.parametrize(
    "shape",
    [
        box(material="bad"),
        box(material=22),
        box(size_cm=[0, 1, 1]),
        box(center_cm=[float("nan"), 0, 0]),
        box(center_cm=[1000, 0, 0]),
        box(misspelled_size=10),
        {
            "type": "cylinder",
            "material": "lead",
            "radius_cm": 1,
            "height_cm": 2,
            "inner_radius_cm": 2,
        },
    ],
)
def test_invalid_scene_is_rejected(shape):
    with pytest.raises(ValueError):
        create_voxels({"shapes": [shape]})


@pytest.mark.parametrize("extension", [".npy", ".npz", ".txt"])
def test_voxel_file_order_and_no_overwrite(tmp_path, extension):
    labels = np.zeros((64, 64, 64), dtype=np.int32)
    labels[5, 9, 12] = 21
    path = save_voxels(labels, tmp_path / ("grid" + extension))
    np.testing.assert_array_equal(load_voxels(path), labels)
    if extension == ".txt":
        assert int(path.read_text().splitlines()[5 * 64 * 64 + 9 * 64 + 12]) == 21
    with pytest.raises(FileExistsError):
        save_voxels(labels, path)


def rows(n=3):
    x = np.zeros((n, 13), dtype=np.float64)
    x[:, 0] = np.arange(n) + 0.123456789
    x[:, 2] = -1000
    x[:, 5] = -3000
    x[:, 8] = 1000
    x[:, 11] = -1
    x[:, 12] = 3000
    return x


@pytest.mark.parametrize("header", [False, True])
def test_csv_preserves_first_track_and_precision(tmp_path, header):
    data = rows()
    path = tmp_path / "tracks.csv"
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=",".join(CSV_COLUMNS) if header else "",
        comments="",
        fmt="%.17g",
    )
    np.testing.assert_array_equal(read_simulation_csv(path), data)


def test_bad_csv_does_not_create_dataset(tmp_path):
    pytest.importorskip("tensorflow")
    output = tmp_path / "invalid.tfrecord"
    data = rows()
    data[0, 9:12] = 0
    with pytest.raises(ValueError, match="unit"):
        write_records(output, [(data, np.zeros((64, 64, 64), dtype=np.int32))])
    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_tfrecord_roundtrip_with_existing_loader(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from muon_tomo.io.tfrecord import _decode, load_tfrecord_record

    labels = create_voxels({"shapes": [box()]})
    path = tmp_path / "custom.tfrecord"
    write_records(path, [(rows(3), labels), (rows(2), np.zeros_like(labels))])
    data = list(tf.data.TFRecordDataset(str(path), compression_type="GZIP"))
    assert len(data) == 2
    x, y = _decode(data[0])
    assert x.dtype == np.float64 and y.dtype == np.int32
    np.testing.assert_array_equal(x, rows(3))
    np.testing.assert_array_equal(y, labels)
    tracks, truth = load_tfrecord_record(path, 0)
    np.testing.assert_array_equal(tracks.entry_position_cm, rows(3)[:, :3] / 10)
    np.testing.assert_array_equal(tracks.momentum_mev, [3000] * 3)
    np.testing.assert_array_equal(truth.material_labels, labels)
    assert set(np.unique(truth.inverse_radiation_length_cm_inv)) == {
        0,
        np.float32(1 / 0.5612334801762114),
    }
    with pytest.raises(FileExistsError):
        write_records(path, [(rows(), labels)])


def test_source_macro_and_cli():
    macro = SourceConfig().macro()
    assert "/gps/ene/alpha -2.7" in macro
    assert "biasAlpha" not in macro
    assert "/run/beamOn" not in macro
    with pytest.raises(ValueError):
        SourceConfig(energy_min_gev=-1).macro()
    args = build_parser().parse_args(
        [
            "generate-dataset",
            "--scene",
            "a.yaml",
            "b.yaml",
            "--output",
            "test.tfrecord",
            "--seed",
            "7",
        ]
    )
    assert args.scene == ["a.yaml", "b.yaml"]
    assert args.seed == 7
    assert len(MATERIAL_NAMES) == 22


EXECUTABLE = Path(os.environ.get("MUON_TOMO_SIMULATOR", "/nonexistent/muon_tomo_sim"))


@pytest.mark.skipif(not EXECUTABLE.is_file(), reason="set MUON_TOMO_SIMULATOR for real Geant4 test")
def test_geant4_seeded_planes_and_packaging(tmp_path):
    pytest.importorskip("tensorflow")
    # Empty target is the transparent control: every short vertical ray reaches the detector.
    source = SourceConfig(angular_mode="vertical", energy_mode="mono", source_half_width_cm=5)
    path = save_voxels(np.zeros((64, 64, 64), np.int32), tmp_path / "empty.txt")
    first = run_simulation(EXECUTABLE, path, tmp_path / "one", events=32, seed=47, source=source)
    second = run_simulation(EXECUTABLE, path, tmp_path / "two", events=32, seed=47, source=source)
    x = read_simulation_csv(first["csv"])
    np.testing.assert_array_equal(x, read_simulation_csv(second["csv"]))
    assert len(x) == 32
    np.testing.assert_allclose(x[:, 8], 1000, rtol=0, atol=1e-6)
    np.testing.assert_allclose(x[:, 2], -1000, rtol=0, atol=1e-6)
    np.testing.assert_allclose(x[:, 12], 3000, rtol=0, atol=1e-6)
    assert np.all(np.linalg.norm(x[:, 3:6], axis=1) < 3110)
    # Exercise full scene -> Geant4 -> TFRecord integration with an asymmetric solid.
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps({"name": "lead", "shapes": [box(center_cm=[10, 0, 0])]}))
    result = generate_dataset(
        [scene],
        tmp_path / "sample.tfrecord",
        executable=EXECUTABLE,
        events=32,
        seed=71,
        source=source,
    )
    from muon_tomo.io.tfrecord import load_tfrecord_record

    tracks, truth = load_tfrecord_record(result["tfrecord"])
    assert len(tracks) == result["records"][0]["accepted_tracks"]
    assert set(np.unique(truth.material_labels)) == {0, 19}
