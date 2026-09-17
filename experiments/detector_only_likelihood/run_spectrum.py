"""Fixed external-spectrum experiment; all fits saved before truth evaluation."""
from pathlib import Path
import json,hashlib
import numpy as np
from muon_tomo.research.detector_only_likelihood.method import DetectorTracks,SETTINGS,prepare
from muon_tomo.research.detector_only_likelihood.spectrum_method import PRIOR,prepare_spectrum,fit,baseline_rho,marginal_nll,spectrum
from muon_tomo.reconstruction.common import physical_grid_intersections
from muon_tomo.geometry import GridGeometry
P=Path(__file__).resolve().parent/'results'
cases=json.loads((P/'scores.json').read_text())
(P/'external_spectrum_protocol.json').write_text(json.dumps(dict(prior=PRIOR,settings=SETTINGS,
    cases=[c['case'] for c in cases],source_sha256=hashlib.sha256((P.parent/'spectrum_method.py').read_bytes()).hexdigest(),
    policy='fixed before physical MSE evaluation; no changes based on sample scores'),indent=2))
rows=[]
for row in cases:
 p=P/row['case'];d=np.load(p/'detector_inputs.npz')
 t=DetectorTracks(*(d[n] for n in DetectorTracks.__dataclass_fields__))
 tr=t.subset(d['train_indices']);test=t.subset(d['test_indices'])
 data,prior=prepare_spectrum(tr);te,tprior=prepare_spectrum(test)
 rho,meta=fit(data,prior)
 br,bm=baseline_rho(tr)
 te64=prepare(test,64);assert np.array_equal(te.selected,te64.selected)
 bs=marginal_nll(br.ravel(),te64,tprior);cs=marginal_nll(rho.ravel(),te,tprior)
 r=dict(case=row['case'],baseline_mean_test_nll=float(bs.mean()),candidate_mean_test_nll=float(cs.mean()),test_tracks=len(cs))
 np.savez_compressed(p/'spectrum_test.npz',baseline_nll=bs,candidate_nll=cs)
 # Refit with all 8192, no validation-driven changes.
 full,pm=prepare_spectrum(t);rv,rm=fit(full,pm);bv,bmeta=baseline_rho(t)
 rv=rv.repeat(2,0).repeat(2,1).repeat(2,2)
 np.savez_compressed(p/'inverse_radiation_length.npz',candidate_cm_inv=rv,baseline_cm_inv=bv)
 (p/'inverse_metadata.json').write_text(json.dumps(dict(candidate=rm,baseline=bmeta,train_candidate=meta),indent=2))
 rows.append(r);(P/'spectrum_track_scores.json').write_text(json.dumps(rows,indent=2));print(json.dumps(r),flush=True)
(P/'spectrum_fits_complete.json').write_text(json.dumps(dict(truth_used=False,case_count=len(rows))))
