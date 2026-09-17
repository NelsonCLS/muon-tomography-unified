"""Detector-only boundaries and legacy byte compatibility."""

import gzip
import struct

import numpy as np
import pytest

from muon_tomo.data.selection import detector_fingerprint, original_rows, selection_manifest
from muon_tomo.reconstruction.munet import MuNetReconstructor
from muon_tomo.types import DetectorTracks


def test_detector_contract_drops_momentum_and_preserves_legacy_features(synthetic_tracks):
    detector = DetectorTracks.from_legacy(synthetic_tracks)
    assert not hasattr(detector, "momentum_mev")
    assert np.all(detector.entry_position_cm[:, 2] >= detector.exit_position_cm[:, 2])
    assert np.all(
        np.einsum(
            "ij,ij->i",
            detector.entry_direction,
            detector.exit_position_cm - detector.entry_position_cm,
        )
        > 0
    )
    restored = detector.to_legacy(1743.4601815051958)
    for name in ("entry_position_cm", "entry_direction", "exit_position_cm", "exit_direction"):
        np.testing.assert_array_equal(getattr(synthetic_tracks, name), getattr(restored, name))
    model = MuNetReconstructor()
    np.testing.assert_array_equal(model._features(synthetic_tracks), model._features(restored))
    assert np.unique(restored.momentum_mev) == 1743.4601815051958


def test_detector_identity_is_independent_of_per_track_momentum(synthetic_tracks):
    first = detector_fingerprint(synthetic_tracks)
    synthetic_tracks.momentum_mev *= np.arange(len(synthetic_tracks)) + 1
    assert detector_fingerprint(synthetic_tracks) == first
    synthetic_tracks.entry_position_cm[0, 0] += 0.1
    assert detector_fingerprint(synthetic_tracks) != first


def test_detector_contract_validates_dimensions():
    with pytest.raises(ValueError, match="finite"):
        DetectorTracks(np.zeros((2, 2)), np.ones((2, 3)), np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="zero or non-finite"):
        DetectorTracks(np.zeros((2, 3)), np.zeros((2, 3)), np.ones((2, 3)), np.ones((2, 3)))


def test_existing_filtered_original_rows_are_resolved(tmp_path, synthetic_tracks):
    path = tmp_path / "filtered.tfrecord"
    values = np.array([3, 11, 14, 20], dtype="<u4")
    with gzip.open(path.with_suffix(".original_indices.u32.gz"), "wb") as stream:
        stream.write(struct.pack("<QQ", 0, len(values)))
        stream.write(values.tobytes())
    mapped, provenance = original_rows(path, 0, np.array([0, 3]))
    np.testing.assert_array_equal(mapped, [3, 20])
    assert provenance["status"] == "parent_rows_resolved"
    detector = DetectorTracks.from_legacy(synthetic_tracks.subset(2))
    detector.event_id = np.array([0, 3])
    manifest = selection_manifest(detector, input_path=path, dataset_id="filtered")
    assert manifest["parent_rows"] == [3, 20]
    assert manifest["selected_dataset_rows"] == [0, 3]
    with pytest.raises(ValueError):
        original_rows(path, 0, np.array([4]))


def test_unknown_lineage_is_not_claimed_as_original(tmp_path):
    rows, metadata = original_rows(tmp_path / "unknown.tfrecord", 0, np.array([2, 7]))
    np.testing.assert_array_equal(rows, [2, 7])
    assert metadata["status"] == "no_lineage_sidecar"


def _write_record(tf, path, labels):
    x = np.array(
        [
            [1, 2, -1000, 11, 14, -3000, 4, 7, 1000, 0, 0, -1],
            [2, 3, -1000, 12, 17, -2000, 3, 6, 1000, 0, 0, -1],
        ],
        dtype=np.float64,
    )
    features = {
        "x": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()])
        ),
        "y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels])),
    }
    with tf.io.TFRecordWriter(str(path), options="GZIP") as writer:
        writer.write(
            tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
        )


def test_detector_tfrecord_loader_never_decodes_labels(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from muon_tomo.io import load_tfrecord_detector_tracks, load_tfrecord_truth

    path = tmp_path / "sealed.tfrecord"
    _write_record(tf, path, b"deliberately not a TensorProto")
    tracks = load_tfrecord_detector_tracks(path, limit=1, seed=0)
    assert len(tracks) == 1
    assert tracks.metadata["truth_decoded"] is False
    assert tracks.metadata["simulation_momentum_loaded"] is False
    with pytest.raises(tf.errors.InvalidArgumentError):
        load_tfrecord_truth(path)


def test_tfrecord_detector_roundtrip_features_match_legacy(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from muon_tomo.io import load_tfrecord_detector_tracks, load_tfrecord_record

    path = tmp_path / "valid.tfrecord"
    _write_record(tf, path, tf.io.serialize_tensor(np.zeros((64, 64, 64), dtype=np.int32)).numpy())
    legacy, _ = load_tfrecord_record(path, limit=1, seed=3)
    detector = load_tfrecord_detector_tracks(path, limit=1, seed=3)
    restored = detector.to_legacy(1743.4601815051958)
    np.testing.assert_array_equal(
        MuNetReconstructor()._features(legacy), MuNetReconstructor()._features(restored)
    )
    for name in ("entry_position_cm", "entry_direction", "exit_position_cm", "exit_direction"):
        np.testing.assert_array_equal(getattr(legacy, name), getattr(restored, name))


def test_measured_detector_npz_contains_no_momentum_or_labels(tmp_path, synthetic_tracks):
    from muon_tomo.io import load_detector_npz, save_detector_npz

    tracks = DetectorTracks.from_legacy(synthetic_tracks)
    path = tmp_path / "measured.npz"
    save_detector_npz(tracks, path)
    with np.load(path, allow_pickle=False) as content:
        assert set(content.files) == {
            "entry_position_cm",
            "entry_direction",
            "exit_position_cm",
            "exit_direction",
            "event_id",
            "metadata_json",
        }
    loaded = load_detector_npz(path)
    np.testing.assert_allclose(loaded.entry_direction, tracks.entry_direction, rtol=0, atol=2e-16)
    np.testing.assert_array_equal(loaded.entry_position_cm, tracks.entry_position_cm)
    np.testing.assert_array_equal(loaded.event_id, tracks.event_id)
    assert not hasattr(loaded, "momentum_mev")
    with pytest.raises(FileExistsError):
        save_detector_npz(tracks, path)


def test_root_detector_loader_needs_only_position_branches(tmp_path):
    uproot = pytest.importorskip("uproot")
    from muon_tomo.io import load_root_detector_tracks

    path = tmp_path / "positions.root"
    with uproot.recreate(path) as root:
        root["hits"] = {
            "eventNumber": np.array([8, 8, 8, 8]),
            "hitPositionX_truth": np.array([0.0, 0.0, 1.0, 2.0]),
            "hitPositionY_truth": np.array([0.0, 0.0, 0.0, 0.0]),
            "hitPositionZ_truth": np.array([700.0, 600.0, -600.0, -700.0]),
        }
    tracks = load_root_detector_tracks(path)
    assert len(tracks) == 1
    assert tracks.event_id.tolist() == [8]
    assert tracks.metadata["simulation_momentum_loaded"] is False


def test_selection_versions_and_event_ids_have_explicit_scope(synthetic_tracks):
    first = selection_manifest(synthetic_tracks, dataset_id="physical-events", dataset_version=1)
    second = selection_manifest(synthetic_tracks, dataset_id="physical-events", dataset_version=2)
    assert first["selection_id"] != second["selection_id"]
    assert first["selected_dataset_rows"] is None
    assert first["selected_event_ids"] == synthetic_tracks.event_id.tolist()
    assert first["event_id_kind"] == "source_event_id"


def test_detector_npz_rejects_hidden_extra_fields(tmp_path, synthetic_tracks):
    from muon_tomo.io import load_detector_npz, save_detector_npz

    path = tmp_path / "clean.npz"
    save_detector_npz(DetectorTracks.from_legacy(synthetic_tracks), path)
    with np.load(path, allow_pickle=False) as d:
        payload = dict(d)
    payload["momentum_mev"] = synthetic_tracks.momentum_mev
    polluted = tmp_path / "polluted.npz"
    np.savez_compressed(polluted, **payload)
    with pytest.raises(ValueError, match="outside its detector-only schema"):
        load_detector_npz(polluted)


def test_detector_direction_sign_is_oriented_along_measured_flight():
    tracks = DetectorTracks(
        np.array([[0.0, 0.0, 50.0]]),
        np.array([[0.0, 0.0, 7.0]]),
        np.array([[0.0, 0.0, -50.0]]),
        np.array([[0.0, 0.0, -3.0]]),
    )
    np.testing.assert_array_equal(tracks.entry_direction, [[0.0, 0.0, -1.0]])
    np.testing.assert_array_equal(tracks.exit_direction, [[0.0, 0.0, -1.0]])


@pytest.mark.parametrize("mapping", [np.array([1.4, 2.0]), np.array([-1, 2])])
def test_corrupt_simple_lineage_is_rejected(tmp_path, mapping):
    path = tmp_path / "filtered.tfrecord"
    np.save(path.with_suffix(".indices.npy"), mapping, allow_pickle=False)
    with pytest.raises(ValueError, match="nonnegative integer"):
        original_rows(path, 0, np.array([0]))


@pytest.mark.parametrize("action", ["to_legacy", "select"])
def test_edited_physical_arrays_cannot_silently_use_old_feature_cache(synthetic_tracks, action):
    tracks = DetectorTracks.from_legacy(synthetic_tracks)
    tracks.entry_position_cm[0, 0] += 1.0
    with pytest.raises(ValueError, match="physical arrays changed"):
        if action == "to_legacy":
            tracks.to_legacy(1743.0)
        else:
            tracks.select(np.array([0]))
    rebuilt = DetectorTracks(
        tracks.entry_position_cm,
        tracks.entry_direction,
        tracks.exit_position_cm,
        tracks.exit_direction,
        tracks.event_id,
    )
    result = rebuilt.to_legacy(1743.0)
    # New detector-only objects use the documented outgoing-first legacy order.
    assert result.exit_position_cm[0, 0] == tracks.entry_position_cm[0, 0]


def test_detector_event_ids_are_not_silently_truncated(synthetic_tracks):
    with pytest.raises(ValueError, match="integer array"):
        DetectorTracks(
            synthetic_tracks.entry_position_cm,
            synthetic_tracks.entry_direction,
            synthetic_tracks.exit_position_cm,
            synthetic_tracks.exit_direction,
            np.arange(len(synthetic_tracks), dtype=float) + 0.5,
        )
