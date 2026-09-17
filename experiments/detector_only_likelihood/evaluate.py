"""Separate post-fit evaluation. Labels are read only after all refits are saved."""
import os
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR','/tmp/muon-mpl')
from pathlib import Path
import json
import numpy as np
from scipy.stats import rankdata,spearmanr
from muon_tomo.research.detector_only_likelihood.method import DetectorTracks,prepare,reconstruct,baseline
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
CASES=json.loads((OUT/'scores.json').read_text())
assert json.loads((OUT/'complete.json').read_text())['truth_loaded'] is False
for row in CASES:
 p=OUT/row['case'];d=np.load(p/'detector_inputs.npz')
 t=DetectorTracks(*(d[n] for n in DetectorTracks.__dataclass_fields__))
 b,bm=baseline(t);s,sm=reconstruct(prepare(t));s=s.repeat(2,0).repeat(2,1).repeat(2,2)
 np.savez_compressed(p/'full_8192.npz',baseline_power_rad2_cm=b,candidate_power_rad2_cm=s)
 (p/'full_metadata.json').write_text(json.dumps(dict(baseline=bm,candidate=sm),indent=2))
 print('refit saved',row['case'],flush=True)

# No fitting calls after this boundary. No truth is passed to method.py.
import tensorflow as tf
from muon_tomo.constants import labels_to_inverse_radiation_length
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
allrows=[]
for group,filename,wanted in [('funny','voxels_prediction_funny.tfrecord',[0]),('prediction','voxels_prediction.tfrecord',list(range(8)))]:
 for i,raw in enumerate(tf.data.TFRecordDataset(str(Path('dataset')/filename),compression_type='GZIP')):
  if i>max(wanted):break
  if i not in wanted:continue
  p=OUT/f'{group}_{i}'
  example=tf.io.parse_single_example(raw,{'y':tf.io.FixedLenFeature([],tf.string)})
  labels=tf.io.parse_tensor(example['y'],out_type=tf.int32).numpy()
  truth=labels_to_inverse_radiation_length(labels)
  d=np.load(p/'full_8192.npz');predictions={k:d[k] for k in d.files}
  mask=(truth>0).ravel();n1=mask.sum();n0=len(mask)-n1
  row=dict(case=p.name,material_voxels=int(n1),models={})
  for name,v in predictions.items():
   ranks=rankdata(v.ravel());auc=(ranks[mask].sum()-n1*(n1+1)/2)/(n1*n0)
   row['models'][name]=dict(material_vs_air_roc_auc=float(auc),spearman_density=float(spearmanr(v.ravel(),truth.ravel()).statistic),maximum_power_rad2_cm=float(v.max()))
  np.savez_compressed(p/'evaluation_truth.npz',inverse_radiation_length_cm_inv=truth)
  fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
  im=axes[0].imshow(truth[:,:,29:36].mean(axis=2).T,origin='lower',extent=(-50,50,-50,50),cmap='magma',vmin=0)
  axes[0].set_title('Truth (evaluation only)');fig.colorbar(im,ax=axes[0],label='1/X0 [cm^-1]',shrink=.75)
  arrays=[predictions[k][:,:,29:36].mean(axis=2).T*1e6 for k in predictions]
  hi=max(float(np.max(a)) for a in arrays);lo=max(hi*1e-5,1e-5)
  for ax,a,name in zip(axes[1:],arrays,['Existing full median','Pooled Student-t candidate']):
   im=ax.imshow(np.maximum(a,lo),origin='lower',extent=(-50,50,-50,50),cmap='magma',norm=LogNorm(lo,hi))
   ax.set_title(name);fig.colorbar(im,ax=ax,label='Effective scattering power [mrad²/cm], log',shrink=.75)
  for ax in axes:ax.set_xlabel('x [cm]');ax.set_ylabel('y [cm]')
  fig.suptitle(f'{group} record {i} · 8,192 tracks · z slab 29–35\nFrozen detector-only algorithm; physical 1/X0 conversion unavailable')
  fig.savefig(p/'cross_section.png',dpi=150);plt.close(fig)
  allrows.append(row);print(json.dumps(row),flush=True)
(OUT/'localization_scores.json').write_text(json.dumps(allrows,indent=2))
