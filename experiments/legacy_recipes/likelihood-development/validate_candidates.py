"""Fixed 64-grid, 40-step local-frame median+pool validation on unseen objects.

No parameter selection is performed from these labels. Fresh validation
indices [17, 53, 101, 257, 503, 997] were specified before their data were read.
Includes original and target-crossing-only populations, original Funny 0,
and filtered Funny 0. All default settings are identical for every object.
"""
import sys, json, time, importlib.util, shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,'/private/tmp')
from likelihood_em_control import *
from muon_tomo.io import load_tfrecord_record
from muon_tomo.cli import _track_fingerprint
from muon_tomo.reconstruction import LikelihoodReconstructor
from spectrum_method import baseline_rho
import tensorflow as tf

CASES=[('funny_original','dataset/voxels_prediction_funny.tfrecord',0),('funny_filtered','dataset_target_crossing/voxels_prediction_funny.tfrecord',0)]
CASES += [(f'val_{i}_{tag}',f'{folder}/voxels_prediction_val.tfrecord',i) for i in (17,53,101,257,503,997) for tag,folder in [('original','dataset'),('filtered','dataset_target_crossing')]]

def main():
    out=OUT/'validation';out.mkdir(exist_ok=True)
    shutil.copy2(__file__,OUT/'validate_candidates.py')
    (out/'protocol.json').write_text(json.dumps(dict(cases=CASES,parameters=dict(grid=64,iterations=40,post_pool_grid=32,post_pool_sigma_cm=3.125,observation_frame='incoming transverse',momentum='fixed external Reyna',evaluation='no truth-derived scaling or case-specific settings')),indent=2))
    for name,path,record in CASES:
        case=out/name;case.mkdir(exist_ok=True)
        if (case/'metrics.json').exists(): continue
        t=time.monotonic()
        # Decode only x for inference. y is decoded after all predictions save.
        raw=next(iter(tf.data.TFRecordDataset(str(ROOT/path),compression_type='GZIP').skip(record).take(1)))
        example=tf.train.Example.FromString(bytes(raw.numpy()))
        x=tf.io.parse_tensor(example.features.feature['x'].bytes_list.value[0],out_type=tf.double).numpy()
        chosen=np.sort(np.random.default_rng(0).choice(len(x),8192,replace=False));x=x[chosen]
        tracks=DetectorTracks(x[:,:3]/10,x[:,3:6],x[:,6:9]/10,x[:,9:12])
        np.savez_compressed(case/'detector_inputs.npz',entry_position_cm=tracks.entry_position_cm,entry_direction=tracks.entry_direction,exit_position_cm=tracks.exit_position_cm,exit_direction=tracks.exit_direction,selected_rows=chosen)
        data=prepare(tracks,64)
        local=local_observations(tracks,data)
        _,inc,_,_=physical_grid_intersections(tracks,GridGeometry())
        co,lw=spectrum(abs(inc[:,2]));coeff=np.sum(np.exp(lw)*co,axis=1)
        fits=em(local,coeff[data.selected],steps=40)
        candidate=post(fits[40][0],local)
        legacy=LikelihoodReconstructor(iterations=10,epsilon=1e-9,average_momentum_mev=1743.4601815051958,momentum_policy='mean').reconstruct(tracks).inverse_radiation_length_cm_inv
        baseline,meta=baseline_rho(tracks)
        pool=post(baseline,data)
        np.savez_compressed(case/'predictions.npz',candidate=candidate,legacy_fixed=legacy,external_raw=baseline,external_pooled=pool)
        labels=tf.io.parse_tensor(example.features.feature['y'].bytes_list.value[0],out_type=tf.int32).numpy()
        from muon_tomo.constants import labels_to_inverse_radiation_length
        truth=labels_to_inverse_radiation_length(labels)
        np.savez_compressed(case/'truth_evaluation_only.npz',truth=truth,labels=labels)
        results={}
        for label,v in dict(candidate=candidate,legacy_fixed=legacy,external_raw=baseline,external_pooled=pool,zero=np.zeros_like(truth)).items():
            err=v.astype(float)-truth
            results[label]=dict(mse=float(np.mean(err**2)),object_mse=float(np.mean(err[truth>0]**2)),background_mse=float(np.mean(err[truth==0]**2)),maximum=float(v.max()),mean=float(v.mean()))
        row=dict(case=name,input=path,record=record,tracks=8192,target_tracks=len(data.selected),seconds=time.monotonic()-t,results=results)
        (case/'metrics.json').write_text(json.dumps(row,indent=2))
        print(json.dumps(row),flush=True)
    rows=[json.loads(p.read_text()) for p in out.glob('*/metrics.json')]
    (out/'metrics.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
