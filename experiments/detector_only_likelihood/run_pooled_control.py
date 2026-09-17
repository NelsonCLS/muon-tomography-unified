from pathlib import Path
import json,hashlib
import numpy as np
from muon_tomo.research.detector_only_likelihood.method import DetectorTracks,SETTINGS,prepare
from muon_tomo.research.detector_only_likelihood.spectrum_method import prepare_spectrum,baseline_rho,marginal_nll
from muon_tomo.research.detector_only_likelihood.exposure_pooled_likelihood import pool_with_exposure
P=Path(__file__).resolve().parent/'results';rows=[]
(P/'pooled_control_protocol.json').write_text(json.dumps(dict(
 settings=SETTINGS,description='fixed pooling control; same width and prior as earlier experiments, no parameter sweep',
 source_sha256=hashlib.sha256((P.parent/'exposure_pooled_likelihood.py').read_bytes()).hexdigest()),indent=2))
for r in json.loads((P/'spectrum_track_scores.json').read_text()):
 p=P/r['case'];d=np.load(p/'detector_inputs.npz');t=DetectorTracks(*(d[n] for n in DetectorTracks.__dataclass_fields__))
 tr=t.subset(d['train_indices']);test=t.subset(d['test_indices'])
 br,_=baseline_rho(tr);pr=pool_with_exposure(br,tr)
 te=prepare(test,64);_,tprior=prepare_spectrum(test)
 score=marginal_nll(pr.ravel(),te,tprior)
 raw=np.load(p/'inverse_radiation_length.npz')['baseline_cm_inv']
 full=pool_with_exposure(raw,t)
 np.savez_compressed(p/'pooled_likelihood.npz',inverse_radiation_length_cm_inv=full,test_nll=score)
 rows.append(dict(case=r['case'],pooled_mean_test_nll=float(score.mean())))
 print(rows[-1],flush=True)
# Scores are calculated only after every output above has been saved.
from scipy.stats import rankdata
for row in rows:
 p=P/row['case'];v=np.load(p/'pooled_likelihood.npz')['inverse_radiation_length_cm_inv'];y=np.load(p/'evaluation_truth.npz')['inverse_radiation_length_cm_inv']
 mask=(y>0).ravel();n1=mask.sum();n0=len(mask)-n1
 row.update(mse=float(np.mean((v-y)**2)),material_mse=float(np.mean((v[y>0]-y[y>0])**2)),background_mse=float(np.mean(v[y==0]**2)),auc=float((rankdata(v.ravel())[mask].sum()-n1*(n1+1)/2)/(n1*n0)))
 print(row,flush=True)
(P/'pooled_control_scores.json').write_text(json.dumps(rows,indent=2))
