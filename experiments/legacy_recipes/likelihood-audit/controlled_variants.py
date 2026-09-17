"""Isolated audit variants: never writes or changes production reconstruction modules."""
from pathlib import Path
import inspect, json, textwrap, time, types
import numpy as np
from muon_tomo.io.tfrecord import load_tfrecord_record
from muon_tomo.geometry import GridGeometry
from muon_tomo.types import TrackTable
from muon_tomo.reconstruction import LikelihoodReconstructor
from muon_tomo.reconstruction.common import physical_grid_intersections
import muon_tomo.reconstruction.likelihood as likelihood_module

out=Path(__file__).parent
t,truth=load_tfrecord_record('dataset/voxels_prediction.tfrecord',7)
g=GridGeometry(); a,di,b,do=physical_grid_intersections(t,g)
faces=np.all((a[:,:2]>=-50)&(a[:,:2]<50),1)&np.all((b[:,:2]>=-50)&(b[:,:2]<50),1)
gaps=np.load(out/'downstream_gaps_cm.npy')
target=truth.inverse_radiation_length_cm_inv.astype(float)
source=textwrap.dedent(inspect.getsource(LikelihoodReconstructor._full_covariance_mlem))
assert source.count('for lens in lengths:')==1
source=source.replace('for lens in lengths:', 'for path_index, lens in enumerate(lengths):')
source=source.replace('w = np.empty((len(lens), 2, 2), dtype=np.float64)', 'remaining = remaining + self.audit_gaps[path_index]\n        w = np.empty((len(lens), 2, 2), dtype=np.float64)')
namespace=dict(vars(likelihood_module)); exec(source,namespace)
gap_variant=namespace['_full_covariance_mlem']
(out/'isolated_gap_variant.txt').write_text(source)
rows=[]
cases=[
 ('full_all_gap_fixed_10',False,'mean','median',10,True),
 ('full_all_gap_exit_10',False,'per-track','median',10,True),
 ('full_all_original_fixed_median_50',False,'mean','median',50,False),
 ('full_all_original_exit_mean_50',False,'per-track','mean',50,False),
 ('full_faces_exit_mean_50',True,'per-track','mean',50,False),
 ('full_faces_exit_median_50',True,'per-track','median',50,False),
]
for name,only_faces,policy,statistic,it,gap_fix in cases:
    sel=np.flatnonzero(faces) if only_faces else np.arange(len(t))
    ts=TrackTable(t.entry_position_cm[sel],t.entry_direction[sel],t.exit_position_cm[sel],t.exit_direction[sel],t.momentum_mev[sel],t.event_id[sel],t.source)
    r=LikelihoodReconstructor(g,iterations=it,mode='full',momentum_policy=policy,average_momentum_mev=1743.4601815051958,epsilon=1e-9,update_statistic=statistic)
    if gap_fix:
        r.audit_gaps=gaps[sel]
        r._full_covariance_mlem=types.MethodType(gap_variant,r)
    start=time.monotonic(); result=r.reconstruct(ts)
    v=result.inverse_radiation_length_cm_inv.astype(float)
    row={'name':name,'tracks':len(ts),'momentum':policy,'update':statistic,'iterations':it,'gap_variant':gap_fix,'mse':float(np.mean((v-target)**2)),'maximum':float(v.max()),'at_clip':int((v>=19.999).sum()),'material_mean':float(v[target>0].mean()),'material_median':float(np.median(v[target>0])),'history':result.metadata['convergence_history'],'seconds':time.monotonic()-start}
    result.save(out/f'{name}.npz'); rows.append(row)
    (out/'controlled_variants.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(row),flush=True)
