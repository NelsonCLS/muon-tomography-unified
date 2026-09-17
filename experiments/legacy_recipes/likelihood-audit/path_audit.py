from pathlib import Path
import json
import numpy as np
from muon_tomo.io.tfrecord import load_tfrecord_record
from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction.common import physical_grid_intersections,prepare_poca_paths,prepare_paths,closest_approach
from muon_tomo.types import ReconstructionResult

out=Path(__file__).parent
t,truth=load_tfrecord_record('dataset/voxels_prediction.tfrecord',7)
g=GridGeometry();a,di,b,do=physical_grid_intersections(t,g)
paths,lens,occ,k=prepare_poca_paths(t,g)
faces=np.all((a[:,:2]>=-50)&(a[:,:2]<50),1)&np.all((b[:,:2]>=-50)&(b[:,:2]<50),1)
gaps=np.zeros(len(t));duplicates=0
for i,(start,end,din,dout,ids) in enumerate(zip(a,b,di,do,paths)):
    if not len(ids):continue
    duplicates+=int(len(np.unique(ids))!=len(ids))
    p=closest_approach(start,din,end,dout)
    last_start=p if p is not None and bool(g.contains(p)) else start
    delta=end-last_start
    hi=1.
    for ax in range(3):
        if abs(delta[ax])>1e-15:
            hi=min(hi,max((-50-last_start[ax])/delta[ax],(50-last_start[ax])/delta[ax]))
    gaps[i]=np.linalg.norm(delta)*max(0,1-hi)
dxang=np.arctan(do[:,0]/abs(do[:,2]))-np.arctan(di[:,0]/abs(di[:,2]))
dyang=np.arctan(do[:,1]/abs(do[:,2]))-np.arctan(di[:,1]/abs(di[:,2]))
travel=(b[:,2]-a[:,2])/di[:,2]
disp=b-(a+travel[:,None]*di)
result={'paths_nonempty':sum(bool(len(p)) for p in paths),'both_faces':int(faces.sum()),'paths_with_duplicate_voxels':duplicates,'paths_with_missing_downstream_gap':int((gaps>1e-6).sum()),'max_missing_gap_cm':float(gaps.max()),'positive_angle_displacement_product_fraction':float(np.mean(dxang*disp[:,0]+dyang*disp[:,1]>0))}
ap,al,ao=prepare_paths(t,g)
r=ReconstructionResult.load('outputs/random-prediction/record-7/angular-all/record_7_likelihood_angular.npz')
v=r.inverse_radiation_length_cm_inv.ravel(); target=truth.inverse_radiation_length_cm_inv.ravel();error=(v.astype(float)-target)**2
rank=np.argsort(error)[-10:][::-1]
events={int(j):[] for j in rank}
q=.5*(dxang**2+dyang**2)*(1743.4601815051958/15)**2
for i,(ids,ll) in enumerate(zip(ap,al)):
    for j in set(ids).intersection(events):
        events[int(j)].append({'track':i,'total_path_cm':float(ll.sum()),'voxel_path_cm':float(ll[np.where(ids==j)[0][0]]),'q_cm_inv_times_cm':float(q[i]),'p_exit_MeV':float(t.momentum_mev[i]),'both_faces':bool(faces[i])})
result['angular_top10_sse_fraction']=float(error[rank].sum()/error.sum())
result['angular_worst_voxels']=[{'xyz':list(map(int,np.unravel_index(int(j),g.shape))),'prediction':float(v[j]),'truth':float(target[j]),'events':events[int(j)]} for j in rank]
np.save(out/'downstream_gaps_cm.npy',gaps)
(out/'path_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
