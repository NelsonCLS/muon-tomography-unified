"""Read-only reconstruction audit; diagnostic outputs only, no source changes."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import tensorflow as tf
from muon_tomo.io.tfrecord import _decode, load_tfrecord_record
from muon_tomo.cli import _track_fingerprint
from muon_tomo.geometry import GridGeometry
from muon_tomo.types import TrackTable, ReconstructionResult
from muon_tomo.reconstruction import LikelihoodReconstructor
from muon_tomo.reconstruction.common import physical_grid_intersections, prepare_poca_paths

OUT=Path(__file__).parent
DATA=Path('dataset/voxels_prediction.tfrecord')
def emit(name, value):
    print(name, json.dumps(value), flush=True)
    (OUT/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
def quant(a):
    a=np.asarray(a); a=a[np.isfinite(a)]
    return dict(zip(['min','p1','p10','p50','p90','p99','max'],np.percentile(a,[0,1,10,50,90,99,100]).tolist()))
raw=next(iter(tf.data.TFRecordDataset(str(DATA),compression_type='GZIP').skip(7).take(1)))
x,y=_decode(raw)
t,truth=load_tfrecord_record(DATA,7)
g=GridGeometry()
idx=np.sort(np.random.default_rng(0).choice(len(t),8192,replace=False))
p_initial=np.sqrt(x[:,12]*(x[:,12]+2*105.6583755))
a,di,b,do=physical_grid_intersections(t,g)
faces=np.isfinite(a).all(1)&np.isfinite(b).all(1)&np.all((a[:,:2]>=-50)&(a[:,:2]<50),1)&np.all((b[:,:2]>=-50)&(b[:,:2]<50),1)
dtx,dty=t.projected_scattering_angles()
theta2=.5*(dtx**2+dty**2)
stats={'tracks':len(t),'first_position_z':quant(x[:,2]),'second_position_z':quant(x[:,8]),'exit_momentum_MeV':quant(t.momentum_mev),'initial_kinetic_energy_MeV':quant(x[:,12]),'initial_momentum_MeV':quant(p_initial),'momentum_ratio_exit_initial':quant(t.momentum_mev/p_initial),'angles_rad':quant(np.sqrt(theta2)),'both_faces':int(faces.sum()),'zero_truth_mse':float(np.mean(truth.inverse_radiation_length_cm_inv.astype(float)**2)),'p_eff_exit':float(1/np.sqrt(np.mean(1/t.momentum_mev**2))),'p_eff_initial':float(1/np.sqrt(np.mean(1/p_initial**2)))}
for name,sel in [('all',np.arange(len(t))),('8192',idx)]:
    extreme=sel[np.argsort(theta2[sel])[-10:][::-1]]
    stats[name+'_largest_angle_tracks']=[{'id':int(i),'angle_rms_rad':float(np.sqrt(theta2[i])),'p_exit':float(t.momentum_mev[i]),'p_initial':float(p_initial[i]),'both_faces':bool(faces[i])} for i in extreme]
emit('input_stats',stats)

def subset(sel, momentum):
    return TrackTable(t.entry_position_cm[sel],t.entry_direction[sel],t.exit_position_cm[sel],t.exit_direction[sel],momentum[sel],t.event_id[sel],t.source)
def metrics(v):
    v=np.asarray(v,dtype=float); target=truth.inverse_radiation_length_cm_inv.astype(float)
    err=(v-target)**2
    result={'mse':float(err.mean()),'max':float(v.max()),'positive':int((v>0).sum()),'at_20':int((v>=19.999).sum()),'top_100_voxel_fraction_SSE':float(np.sort(err.ravel())[-100:].sum()/err.sum()),'background_mse':float(err[y==0].mean())}
    for label in np.unique(y):
        result[f'label_{label}']={'mean':float(v[y==label].mean()),'median':float(np.median(v[y==label])),'truth':float(np.median(target[y==label]))}
    return result
rows=[]
cases=[
 ('full_8192_eps1e9',idx,'full','mean',t.momentum_mev,1e-9,10),
 ('full_8192_eps1e12',idx,'full','mean',t.momentum_mev,1e-12,10),
 ('full_all_eps1e9',np.arange(len(t)),'full','mean',t.momentum_mev,1e-9,10),
 ('full_all_exit_momentum',np.arange(len(t)),'full','per-track',t.momentum_mev,1e-9,10),
 ('full_all_initial_momentum',np.arange(len(t)),'full','per-track',p_initial,1e-9,10),
 ('angular_all_exit_momentum',np.arange(len(t)),'angular','per-track',t.momentum_mev,1e-12,10),
 ('angular_all_initial_momentum',np.arange(len(t)),'angular','per-track',p_initial,1e-12,10),
]
for name,sel,mode,policy,momentum,eps,it in cases:
    ts=subset(sel,momentum)
    start=time.monotonic()
    r=LikelihoodReconstructor(g,iterations=it,mode=mode,momentum_policy=policy,average_momentum_mev=1743.4601815051958,epsilon=eps).reconstruct(ts)
    row={'name':name,'tracks':len(ts),'mode':mode,'momentum_policy':policy,'epsilon':eps,'seconds':time.monotonic()-start,**metrics(r.inverse_radiation_length_cm_inv),'fingerprint':_track_fingerprint(ts),'history':r.metadata['convergence_history']}
    rows.append(row);r.save(OUT/f'{name}.npz');emit('experiments',rows)

