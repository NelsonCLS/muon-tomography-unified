"""Run frozen ablations, save fits, then evaluate labels in a separate stage."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/muon-mpl')
import numpy as np
from scipy.stats import rankdata
from muon_tomo.research.provenance import (SCHEMA_VERSION, ensure_protocol, file_sha256, input_identity, verify_fit, verify_protocol_sources)
from muon_tomo.research.likelihood_physics_study.core import DetectorTracks, MAP_METHODS, fit_all, nll, prepare_both

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'results'
DEVELOPMENT = ['funny_0', 'prediction_0', 'prediction_5']
VALIDATION = [11, 31, 71, 131, 211, 311]
CASES = DEVELOPMENT + [f'validation_{i}' for i in VALIDATION]


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(cache_schema_version=SCHEMA_VERSION,
        dependencies={n: importlib.metadata.version(n) for n in ("numpy", "scipy")}, development=DEVELOPMENT, new_validation_records=VALIDATION,
        data='original unfiltered dataset', tracks_per_object=8192, sampling_seed=0,
        reserved_track_split_seed=20260915, training_tracks=6144, reserved_tracks=2048,
        methods=['raw10', 'pool10', 'pool40', 'local_pool10', 'local_pool40', 'mean_local_pool40']+list(MAP_METHODS),
        comparison='same external Reyna expected inverse-p-squared, C=15 MeV, beta=1',
        em_steps=[10, 40], em_rho_bounds=[1e-9, 20], em_initial=4e-5,
        map_grid=32, map_rho_bounds=[1e-9, 20], map_iterations=160,
        map_strength=.3, huber_delta_cm_inv=.1, map_start='pooled raw10',
        pool_grid=32, pool_sigma_cm=3.125,
        validation_claim='new indices relative to prior local study manifests; not an external dataset',
        scoring='all maps saved before labels are loaded; no method or parameter selected by these new labels',
        source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
        production_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in (ROOT/'src').rglob('*.py')})
    ensure_protocol(OUT/'protocol.json', protocol)
    return protocol


def extract():
    import tensorflow as tf
    source = ROOT/'dataset/voxels_prediction_val.tfrecord'
    for i, raw in enumerate(tf.data.TFRecordDataset(str(source), compression_type='GZIP')):
        if i > max(VALIDATION):
            break
        if i not in VALIDATION:
            continue
        p = OUT/f'validation_{i}'
        p.mkdir(exist_ok=True)
        if (p/'detector_inputs.npz').exists() or (p/'evaluation_label.tensor').exists():
            raise FileExistsError(f'Extraction already exists: {p}; use a new results directory')
        example = tf.train.Example.FromString(bytes(raw.numpy()))
        x = tf.io.parse_tensor(example.features.feature['x'].bytes_list.value[0], out_type=tf.double).numpy()
        ids = np.sort(np.random.default_rng(0).choice(len(x), 8192, replace=False))
        x = x[ids]
        t = DetectorTracks(x[:, :3]/10, x[:, 3:6], x[:, 6:9]/10, x[:, 9:12])
        np.savez_compressed(p/'detector_inputs.npz', **{k: getattr(t, k) for k in t.__dataclass_fields__}, selected_rows=ids)
        # Serialized y is sealed for later evaluation, never decoded for fitting.
        (p/'evaluation_label.tensor').write_bytes(example.features.feature['y'].bytes_list.value[0])
        print('EXTRACTED', i, flush=True)


def run_case(name):
    if name not in CASES:
        raise ValueError(f'Unknown study case: {name}')
    p = OUT/name
    p.mkdir(exist_ok=True)
    protocol = verify_protocol_sources(OUT/'protocol.json', ROOT)
    src = ROOT/'experiments/detector_only_likelihood/results'/name if name in DEVELOPMENT else p
    identity = input_identity(src/'detector_inputs.npz', protocol)
    if (p/'fit_complete.json').exists():
        verify_fit(p, identity)
        print('VERIFIED CACHE', name, flush=True)
        return
    if (p/'fit_started.json').exists() or any((p/f).exists() for f in ('full_maps.npz', 'training_maps.npz', 'reserved_scores.npz', 'coverage.npz')):
        raise FileExistsError(f'Incomplete or unverified fit exists: {p}; use --results-dir NEW_DIRECTORY')
    (p/'fit_started.json').write_text(json.dumps(identity, indent=2))
    with np.load(src/'detector_inputs.npz') as d:
        tracks = DetectorTracks(*(d[k] for k in DetectorTracks.__dataclass_fields__))
    started = time.monotonic()
    paper, local, coeff = prepare_both(tracks)
    order = np.random.default_rng(20260915).permutation(len(tracks))
    train_events, test_events = order[:6144], order[6144:]
    train_rows = np.flatnonzero(np.isin(local.selected, train_events))
    test_rows = np.flatnonzero(np.isin(local.selected, test_events))
    print('PREPARED', name, len(local.selected), 'usable tracks', flush=True)
    train, metadata_train = fit_all(paper.subset(train_rows), local.subset(train_rows), coeff[train_rows])
    np.savez_compressed(p/'training_maps.npz', **train)
    test = local.subset(test_rows)
    scores = {f'{k}__{s}': nll(v.ravel(), test, coeff[test_rows], student=(s == 'student'))
              for k, v in train.items() for s in ('gaussian', 'student')}
    np.savez_compressed(p/'reserved_scores.npz', **scores, selected_test_events=local.selected[test_rows])
    print('RESERVED SCORED', name, round(time.monotonic()-started, 1), 'sec', flush=True)
    volumes, metadata_full = fit_all(paper, local, coeff)
    np.savez_compressed(p/'full_maps.npz', **volumes)
    np.savez_compressed(p/'coverage.npz', exposure=np.asarray(local.matrices[0].sum(axis=0)).reshape((64,)*3))
    coverage = np.asarray(local.matrices[0].getnnz(axis=0))
    result = dict(case=name, selected_tracks=len(tracks), usable_tracks=len(local.selected),
        train_usable=len(train_rows), test_usable=len(test_rows),
        nonempty_voxel_fraction=float(np.mean(coverage > 0)),
        median_tracks_per_covered_voxel=float(np.median(coverage[coverage > 0])),
        seconds=time.monotonic()-started, training=metadata_train, full=metadata_full,
        fitting_truth_loaded=False, cache_identity=identity,
        artifact_sha256={name: file_sha256(p/name) for name in ('training_maps.npz', 'reserved_scores.npz', 'full_maps.npz', 'coverage.npz')})
    (p/'fit_complete.json').write_text(json.dumps(result, indent=2))
    print('FIT COMPLETE', name, round(result['seconds'], 1), 'sec', flush=True)


def evaluate():
    protocol = verify_protocol_sources(OUT/'protocol.json', ROOT)
    for name in CASES:
        source = ROOT/'experiments/detector_only_likelihood/results'/name if name in DEVELOPMENT else OUT/name
        verify_fit(OUT/name, input_identity(source/'detector_inputs.npz', protocol))
    import tensorflow as tf
    from muon_tomo.constants import labels_to_inverse_radiation_length
    results = []
    for name in CASES:
        p = OUT/name
        if name in DEVELOPMENT:
            truth = np.load(ROOT/'experiments/detector_only_likelihood/results'/name/'evaluation_truth.npz')['inverse_radiation_length_cm_inv']
        else:
            labels = tf.io.parse_tensor((p/'evaluation_label.tensor').read_bytes(), out_type=tf.int32).numpy()
            truth = labels_to_inverse_radiation_length(labels)
        np.savez_compressed(p/'truth_evaluation_only.npz', rho=truth)
        maps, scores = np.load(p/'full_maps.npz'), np.load(p/'reserved_scores.npz')
        mask = truth > 0
        n1, n0 = int(mask.sum()), int((~mask).sum())
        models = {}
        for method in maps.files:
            v = maps[method]
            squared = (v-truth)**2
            ranks = rankdata(v.ravel())
            models[method] = dict(mse=float(squared.mean()), material_mse=float(squared[mask].mean()),
                background_mse=float(squared[~mask].mean()), mean_density=float(v.mean()),
                material_mean_density=float(v[mask].mean()), max_density=float(v.max()),
                auc=float((ranks[mask.ravel()].sum()-n1*(n1+1)/2)/(n1*n0)),
                test_gaussian_nll=float(scores[f'{method}__gaussian'].mean()),
                test_student_nll=float(scores[f'{method}__student'].mean()))
        row = dict(case=name, group='development' if name in DEVELOPMENT else 'validation',
            zero_mse=float(np.mean(truth**2)), material_fraction=float(mask.mean()),
            truth_material_mean=float(truth[mask].mean()), models=models)
        results.append(row)
        print('EVALUATED', name, {k: round(v['mse'], 5) for k, v in models.items()}, flush=True)
    (OUT/'metrics.json').write_text(json.dumps(results, indent=2))
    summary = {}
    for group in ('development', 'validation', 'all'):
        rows = [r for r in results if group == 'all' or r['group'] == group]
        means = {m: {metric: float(np.mean([r['models'][m][metric] for r in rows]))
                     for metric in rows[0]['models'][m]} for m in rows[0]['models']}
        for m in means:
            means[m]['mse_wins_over_raw10'] = sum(r['models'][m]['mse'] < r['models']['raw10']['mse'] for r in rows)
            means[m]['mse_wins_over_pool10'] = sum(r['models'][m]['mse'] < r['models']['pool10']['mse'] for r in rows)
            delta = np.array([r['models'][m]['mse']-r['models']['pool10']['mse'] for r in rows])
            boot = np.random.default_rng(615).choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
            means[m]['paired_mse_delta_vs_pool10_bootstrap95'] = np.quantile(boot, [.025, .975]).tolist()
        summary[group] = dict(cases=len(rows), zero_mse=float(np.mean([r['zero_mse'] for r in rows])), methods=means)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    original = json.loads((OUT/'protocol.json').read_text())['production_sha256']
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest() == digest for f, digest in original.items())
    (OUT/'integrity.json').write_text(json.dumps(dict(production_unchanged=True, evaluated_cases=len(results)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['extract', 'fit', 'evaluate'])
    parser.add_argument('cases', nargs='*')
    parser.add_argument('--results-dir', type=Path, default=OUT, help='Fresh study root; legacy or mismatched cached fits are never reused')
    args = parser.parse_args()
    OUT = args.results_dir.resolve()
    initialize()
    if args.stage == 'extract':
        extract()
    elif args.stage == 'fit':
        for name in args.cases or CASES:
            run_case(name)
    else:
        evaluate()
