"""Raw PoCA exposure-normalization experiment; truth is used for scoring only."""
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from muon_tomo.io.tfrecord import load_tfrecord_record
from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction.common import prepare_poca_paths, physical_grid_intersections, closest_approach
from muon_tomo.cli import _track_fingerprint

out=Path(__file__).parent
g=GridGeometry()
rows=[]
for record in (0,1):
    t,truth=load_tfrecord_record('dataset/voxels_prediction_funny.tfrecord',record,limit=8192,seed=0)
    paths,lengths,occ,k=prepare_poca_paths(t,g)
    a,di,b,do=physical_grid_intersections(t,g)
    angles=t.projected_scattering_angles()
    q=.5*(angles[0]**2+angles[1]**2)*(1743.4601815051958/15)**2
    signal=np.zeros((32,32,32));exposure=np.zeros_like(signal);counts=np.zeros_like(signal)
    for i,(ids,lens) in enumerate(zip(paths,lengths)):
        if not len(ids): continue
        coarse=np.asarray(np.unravel_index(ids,g.shape)).T//2
        np.add.at(exposure,tuple(coarse.T),lens)
        p=closest_approach(a[i],di[i],b[i],do[i])
        if p is None or not bool(g.contains(p)): continue
        index=tuple(g.point_to_index(p)//2)
        signal[index]+=q[i];counts[index]+=1
    target=truth.inverse_radiation_length_cm_inv.astype(float)
    for sigma in (0.,1.):
        numerator=gaussian_filter(signal,sigma,mode='constant') if sigma else signal
        denominator=gaussian_filter(exposure,sigma,mode='constant') if sigma else exposure
        volume=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
        v=volume.repeat(2,0).repeat(2,1).repeat(2,2)
        row={'record':record,'sigma_coarse_voxels':sigma,'tracks':len(t),'fingerprint':_track_fingerprint(t),'mse':float(np.mean((v-target)**2)),'zero_mse':float(np.mean(target**2)),'material_mse':float(np.mean((v[target>0]-target[target>0])**2)),'material_mean':float(v[target>0].mean()),'background_mean':float(v[target==0].mean()),'maximum':float(v.max())}
        rows.append(row);print(json.dumps(row),flush=True)
        np.savez_compressed(out/f'record_{record}_sigma_{sigma}.npz',prediction=v,signal=signal,exposure_cm=exposure,poca_count=counts)
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
