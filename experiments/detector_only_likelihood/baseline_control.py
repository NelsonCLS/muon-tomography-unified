"""Matched 80-iteration baseline control; candidate settings are never changed."""
from pathlib import Path
import json
import numpy as np
from muon_tomo.research.detector_only_likelihood.method import DetectorTracks,prepare,likelihood
from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction.common import prepare_poca_paths
from muon_tomo.reconstruction.likelihood import LikelihoodReconstructor
P=Path(__file__).resolve().parent/'results';rows=[]
for row in json.loads((P/'scores.json').read_text()):
 p=P/row['case'];data=np.load(p/'detector_inputs.npz')
 t=DetectorTracks(*(data[n] for n in DetectorTracks.__dataclass_fields__))
 tr=t.subset(data['train_indices']);te=prepare(t.subset(data['test_indices']),64)
 g=GridGeometry();paths,lens,occ,_=prepare_poca_paths(tr,g)
 solver=LikelihoodReconstructor(iterations=80,reference_momentum_mev=1,highland_constant_mev=1,average_momentum_mev=1,initial_inverse_radiation_length_cm_inv=1e-9,epsilon=1e-9,maximum_inverse_radiation_length_cm_inv=.0005,update_statistic='median')
 v,history,clipped=solver._full_covariance_mlem(tr,np.ones(len(tr)),paths,lens)
 v=np.where(occ.ravel()>0,v,0)
 score=likelihood(v,te)
 np.savez_compressed(p/'baseline_80.npz',power_rad2_cm=v.reshape((64,)*3),test_student_nll=score)
 r=dict(case=row['case'],baseline_80_mean_test_nll=float(score.mean()),candidate_mean_test_nll=row['mean_test_student_nll']['pooled_student'],clipped=clipped)
 rows.append(r);(P/'baseline_80_control.json').write_text(json.dumps(rows,indent=2));print(json.dumps(r),flush=True)
