"""Reproduce the 8192-row sample using the existing target-crossing filter."""
from pathlib import Path
import json
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
filter_path = ROOT / 'dataset_target_crossing/filter_muon_target_datasets.py'
legacy_filter = runpy.run_path(str(filter_path))
crossing_mask = legacy_filter['crossing_mask']

import tensorflow as tf
from muon_tomo.geometry import GridGeometry
from muon_tomo.io.tfrecord import _decode, load_tfrecord_record
from muon_tomo.reconstruction.common import prepare_poca_paths
from muon_tomo.simulation.objects import save_voxels
from muon_tomo.simulation.pipeline import sha256
from muon_tomo.simulation.records import write_records

base = ROOT / 'dataset/generated_custom'
raw = base / 'lead_block_25cm_raw.tfrecord'
out = base / 'lead_block_25cm_filtered_8192.tfrecord'
manifest_path = out.with_suffix(out.suffix + '.json')
ids_path = base / 'lead_block_25cm_filtered_8192.indices.npy'
truth_path = base / 'lead_block_25cm_truth.npz'
for path in (out, manifest_path, ids_path, truth_path):
    if path.exists():
        raise FileExistsError(path)
raw_records = list(tf.data.TFRecordDataset(str(raw), compression_type='GZIP'))
assert len(raw_records) == 1
x, y = _decode(raw_records[0])
expected = np.zeros((64, 64, 64), dtype=np.int32)
expected[24:40, 24:40, 24:40] = 19
np.testing.assert_array_equal(y, expected)
mask, invalid = crossing_mask(x)
eligible = np.flatnonzero(mask)
if len(eligible) < 8192:
    raise RuntimeError(f'Only {len(eligible)} filtered tracks: more simulation required')
ids = np.sort(np.random.default_rng(0).choice(eligible, size=8192, replace=False))
selected = x[ids]
assert len(np.unique(ids)) == 8192
assert crossing_mask(selected)[0].all()
summary = write_records(out, [(selected, y)])
reread = list(tf.data.TFRecordDataset(str(out), compression_type='GZIP'))
assert len(reread) == 1
xx, yy = _decode(reread[0])
np.testing.assert_array_equal(xx, selected)
np.testing.assert_array_equal(yy, y)
assert xx.dtype == np.float64 and yy.dtype == np.int32
tracks, truth = load_tfrecord_record(out)
assert len(tracks) == 8192
paths, lengths, _, kinked = prepare_poca_paths(tracks, GridGeometry())
assert all(len(path) > 0 for path in paths)
assert all(float(np.sum(length)) > 0 for length in lengths)
np.save(ids_path, ids, allow_pickle=False)
save_voxels(y, truth_path, metadata={'center_cm': [0, 0, 0], 'size_cm': [25, 25, 25], 'material': 'lead'})
raw_manifest = json.loads(raw.with_suffix(raw.suffix + '.json').read_text())
report = {
    'schema': 'muon_tomo_filtered_custom_v1',
    'tfrecord': str(out), 'sha256': sha256(out), 'compression': 'GZIP',
    'scene': {'center_cm': [0, 0, 0], 'size_cm': [25, 25, 25], 'material': 'lead', 'material_id': 19, 'lead_voxels': 4096, 'grid_shape': [64, 64, 64], 'grid_bounds_cm': [-50, 50]},
    'source_tfrecord': str(raw), 'source_sha256': sha256(raw),
    'source_manifest': str(raw.with_suffix(raw.suffix + '.json')),
    'simulation': raw_manifest['records'][0],
    'filter': {'algorithm': legacy_filter['ALGORITHM'], 'source': str(filter_path), 'source_sha256': sha256(filter_path), 'description': 'Positive-length inferred path through the [-50,50] cm reconstruction cube, using an in-cube PoCA kink or straight fallback. Includes side crossings and air-only paths. No selection on lead intersection, truth, energy, or scattering-angle magnitude.', 'detected_tracks': len(x), 'invalid_geometry': invalid, 'passing_tracks': len(eligible), 'rejected_tracks': int((~mask).sum())},
    'selection': {'count': 8192, 'seed': 0, 'method': 'NumPy default_rng choice without replacement; sorted original row indices', 'original_row_indices': str(ids_path), 'indices_sha256': sha256(ids_path)},
    'records': summary,
    'validation': {'exact_track_count': True, 'all_rows_unchanged_from_source': True, 'unique_source_rows': True, 'truth_unchanged': True, 'current_loader_passed': True, 'all_selected_have_positive_production_path': True, 'total_target_path_cm': float(sum(np.sum(length) for length in lengths)), 'poca_kink_paths': kinked},
    'physical_model_note': raw_manifest['physical_model_note'],
}
manifest_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'output': str(out), 'detected': len(x), 'filter_passed': len(eligible), 'stored': 8192, 'lead_voxels': 4096, 'manifest': str(manifest_path)}, indent=2))
