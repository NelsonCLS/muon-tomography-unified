"""Post-fit physical 1/X0 evaluation; never calls a reconstruction or fitter."""
import os
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR','/tmp/muon-mpl')
from pathlib import Path
import json,hashlib,html
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from muon_tomo.plotting import COMPARISON_VMAX_CM_INV
P=Path(__file__).resolve().parent/'results'
assert json.loads((P/'spectrum_fits_complete.json').read_text())['case_count']==9
rows=[]
for r in json.loads((P/'spectrum_track_scores.json').read_text()):
 p=P/r['case'];d=np.load(p/'inverse_radiation_length.npz');t=np.load(p/'evaluation_truth.npz')['inverse_radiation_length_cm_inv']
 row=dict(r,unit='(cm^-1)^2',zero_mse=float(np.mean(t.astype(float)**2)))
 mask=(t>0).ravel();n1=mask.sum();n0=len(mask)-n1
 for method in ('baseline','candidate'):
  v=d[f'{method}_cm_inv']
  row[f'{method}_mse']=float(np.mean((v-t)**2))
  row[f'{method}_material_mse']=float(np.mean((v[t>0]-t[t>0])**2))
  row[f'{method}_background_mse']=float(np.mean(v[t==0]**2))
  row[f'{method}_maximum_cm_inv']=float(v.max())
  ranks=rankdata(v.ravel());row[f'{method}_auc']=float((ranks[mask].sum()-n1*(n1+1)/2)/(n1*n0))
 scores=np.load(p/'spectrum_test.npz');delta=scores['candidate_nll']-scores['baseline_nll']
 boots=np.random.default_rng(2026).choice(delta,size=(1000,len(delta))).mean(axis=1)
 row['candidate_minus_baseline_nll_95pct']=[float(delta.mean()),*np.quantile(boots,[.025,.975]).tolist()]
 fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
 vols=[t,d['baseline_cm_inv'],d['candidate_cm_inv']]
 vmax=COMPARISON_VMAX_CM_INV
 for ax,v,title in zip(axes,vols,['Truth','Existing median + external prior','Pooled spectrum likelihood']):
  mse=float(np.mean((v-t)**2))
  im=ax.imshow(v[:,:,29:36].mean(axis=2).T,origin='lower',extent=(-50,50,-50,50),cmap='magma',vmin=0,vmax=vmax)
  ax.set_title(f'{title}\nFull 64³ MSE = {mse:.4g} (cm⁻¹)²');ax.set_xlabel('x [cm]');ax.set_ylabel('y [cm]')
 fig.colorbar(im,ax=axes,label='1/X0 [cm⁻¹]; colors saturate at 3.2 cm⁻¹',shrink=.7)
 fig.suptitle(f"{r['case']} · 8,192 tracks · mean z slab 29–35\nSame external Reyna spectrum for every object; raw predictions, no fitted calibration")
 fig.savefig(p/'inverse_cross_section.png',dpi=150);plt.close(fig)
 rows.append(row);print(json.dumps(row),flush=True)
(P/'inverse_scores.json').write_text(json.dumps(rows,indent=2))
prod=Path('src/muon_tomo/reconstruction/likelihood.py')
assert hashlib.sha256(prod.read_bytes()).hexdigest()==json.loads((P/'frozen_protocol.json').read_text())['production_sha256']
summary=dict(cases=9,mse_improvements=sum(r['candidate_mse']<r['baseline_mse'] for r in rows),
    heldout_nll_improvements=sum(r['candidate_mean_test_nll']<r['baseline_mean_test_nll'] for r in rows),
    auc_improvements=sum(r['candidate_auc']>r['baseline_auc'] for r in rows),
    baseline_mean_mse=float(np.mean([r['baseline_mse'] for r in rows])),
    candidate_mean_mse=float(np.mean([r['candidate_mse'] for r in rows])),production_unchanged=True)
(P/'inverse_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
