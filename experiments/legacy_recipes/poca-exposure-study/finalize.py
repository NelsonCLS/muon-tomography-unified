from pathlib import Path
import json
import numpy as np
from muon_tomo.io.tfrecord import load_tfrecord_record
from muon_tomo.reconstruction import PoCAReconstructor
from muon_tomo.types import ReconstructionResult
from muon_tomo.cli import _track_fingerprint
from muon_tomo.plotting import plot_comparison_slab
from muon_tomo.plotting3d import plot_comparison_3d
out=Path('outputs/poca-exposure-study');rows=[]
for i in (0,1):
 t,truth=load_tfrecord_record('dataset/voxels_prediction_funny.tfrecord',i,limit=8192,seed=0)
 t=t.take_first(8192)
 r=PoCAReconstructor(average_momentum_mev=1743.4601815051958).reconstruct(t)
 r.metadata.update(record_index=i,track_count=len(t),track_fingerprint_sha256=_track_fingerprint(t),track_selection=t.metadata,input_source=str(Path('dataset/voxels_prediction_funny.tfrecord').resolve()))
 r.save(out/f'record_{i}_poca.npz')
 y=truth.inverse_radiation_length_cm_inv
 rows.append(dict(record=i,mse=float(np.mean((r.inverse_radiation_length_cm_inv.astype(float)-y)**2)),zero_mse=float(np.mean(y.astype(float)**2)),metadata=r.metadata))
 if i==0:
  others=[ReconstructionResult.load('outputs/funny-munet-fixed-momentum/record_0_munet.npz'),ReconstructionResult.load('outputs/funny-after-side-exit-8192/record_0_likelihood.npz')]
  assert all(x.metadata['track_fingerprint_sha256']==_track_fingerprint(t) for x in others)
  plot_comparison_slab(truth,[others[0],r,others[1]],output=out/'record_0_cross_section.png')
  info=plot_comparison_3d(truth,[others[0],r,others[1]],output=out/'record_0_viewer.html',threshold_cm_inv=float(np.nextafter(np.float32(.8),np.float32(np.inf))),max_cubes=262144,surface_only=False)
  (out/'viewer_metadata.json').write_text(json.dumps(info,indent=2))
 print(rows[-1]['record'],rows[-1]['mse'],flush=True)
(out/'final_results.json').write_text(json.dumps(rows,indent=2))
