"""Extra optimizer-budget check, specified before new validation labels open.

Continue all five MAP variants on Funny 0 and validation 11 from their
160-step fits for up to 1000 further iterations. Includes reserved-track fits.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from muon_tomo.research.provenance import input_identity, verify_fit, verify_protocol_sources
from muon_tomo.research.likelihood_physics_study.core import DetectorTracks, MAP_METHODS, coarsen, map_fit, nll, prepare_both
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'results'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.results_dir.resolve()
    protocol = verify_protocol_sources(OUT/'protocol.json', ROOT)
    for case in ('funny_0', 'validation_11'):
        source = ROOT/'experiments/detector_only_likelihood/results'/case if case == 'funny_0' else OUT/case
        verify_fit(OUT/case, input_identity(source/'detector_inputs.npz', protocol))
    if (OUT/'convergence_protocol.json').exists():
        raise FileExistsError('Convergence outputs already exist; preserve them and use a new study root')
    (OUT/'convergence_protocol.json').write_text(json.dumps(dict(
        cases=['funny_0', 'validation_11'], additional_iterations=1000,
        reason='all 160-step MAP fits reached iteration budget; specified before validation evaluation',
        methods=list(MAP_METHODS)), indent=2))
    for name in ('funny_0', 'validation_11'):
        p = OUT/name
        src = ROOT/'experiments/detector_only_likelihood/results'/name if name == 'funny_0' else p
        with np.load(src/'detector_inputs.npz') as d:
            tracks = DetectorTracks(*(d[k] for k in DetectorTracks.__dataclass_fields__))
        _, local, coeff = prepare_both(tracks)
        data = coarsen(local)
        order = np.random.default_rng(20260915).permutation(len(tracks))
        train_rows = np.flatnonzero(np.isin(local.selected, order[:6144]))
        test_rows = np.flatnonzero(np.isin(local.selected, order[6144:]))
        volumes, scores, metadata = {}, {}, {}
        for split in ('training', 'full'):
            old = np.load(p/f'{split}_maps.npz')
            d = data.subset(train_rows) if split == 'training' else data
            c = coeff[train_rows] if split == 'training' else coeff
            for method, args in MAP_METHODS.items():
                started = time.monotonic()
                v, meta = map_fit(d, c, old[method][::2, ::2, ::2], **args, maxiter=1000)
                metadata[f'{split}__{method}'] = meta
                if split == 'full':
                    volumes[method] = v.repeat(2, 0).repeat(2, 1).repeat(2, 2)
                else:
                    for score in ('gaussian', 'student'):
                        scores[f'{method}__{score}'] = nll(v.ravel(), data.subset(test_rows), coeff[test_rows], student=score == 'student')
                print(name, split, method, meta['iterations'], meta['success'], round(time.monotonic()-started, 2), flush=True)
        np.savez_compressed(p/'convergence_maps.npz', **volumes)
        np.savez_compressed(p/'convergence_reserved_scores.npz', **scores)
        (p/'convergence.json').write_text(json.dumps(metadata, indent=2))
