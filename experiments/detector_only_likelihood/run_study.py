"""Fit every predeclared case before opening labels for optional evaluation."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/muon-mpl')
import json,hashlib
from pathlib import Path
import numpy as np
from muon_tomo.research.detector_only_likelihood.method import DetectorTracks,prepare,reconstruct,baseline,likelihood,SETTINGS

HERE=Path(__file__).resolve().parent
OUT=HERE/'results';OUT.mkdir(exist_ok=True)
CASES=[('funny',0)]+[('prediction',i) for i in range(8)]

def detector_records(path,wanted):
    import tensorflow as tf
    spec={'x':tf.io.FixedLenFeature([],tf.string)} # Deliberately no y feature.
    for i,raw in enumerate(tf.data.TFRecordDataset(str(path),compression_type='GZIP')):
        if i>max(wanted): break
        if i not in wanted:continue
        example=tf.io.parse_single_example(raw,spec)
        x=tf.io.parse_tensor(example['x'],out_type=tf.double).numpy()
        indices=np.sort(np.random.default_rng(0).choice(len(x),8192,replace=False))
        x=x[indices]
        # The file encodes directions as vectors with arbitrary scale. Only
        # unit directions survive this adapter. Extra columns are discarded.
        tracks=DetectorTracks(x[:,:3]/10,x[:,3:6],x[:,6:9]/10,x[:,9:12])
        yield i,tracks,indices

def main():
    production=Path('src/muon_tomo/reconstruction/likelihood.py')
    manifest=dict(settings=SETTINGS,cases=CASES,tracks=8192,split_seed=20260908,
                  test_fraction=.25,selection='fixed list before scoring; no case selection from outcomes',
                  source_sha256=hashlib.sha256((HERE/'method.py').read_bytes()).hexdigest(),
                  production_sha256=hashlib.sha256(production.read_bytes()).hexdigest())
    (OUT/'frozen_protocol.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    for name,filename,wanted in [('funny','voxels_prediction_funny.tfrecord',[0]),('prediction','voxels_prediction.tfrecord',list(range(8)))]:
        for record,tracks,event_ids in detector_records(Path('dataset')/filename,wanted):
            tag=f'{name}_{record}';p=OUT/tag;p.mkdir(exist_ok=True)
            print('START',tag,flush=True)
            rng=np.random.default_rng(20260908);indices=rng.permutation(len(tracks));cut=6144
            train=tracks.subset(indices[:cut]);test=tracks.subset(indices[cut:])
            tr=prepare(train);te=prepare(test);te64=prepare(test,64)
            assert np.array_equal(te.selected,te64.selected)
            raw,meta=baseline(train);np.savez_compressed(p/'baseline.npz',power_rad2_cm=raw)
            scores={'baseline':likelihood(raw.ravel(),te64)}
            metadata={'baseline':meta}
            for label,robust in [('pooled_gaussian',False),('pooled_student',True)]:
                v,m=reconstruct(tr,student=robust)
                scores[label]=likelihood(v.ravel(),te)
                np.savez_compressed(p/f'{label}.npz',power_rad2_cm=v.repeat(2,0).repeat(2,1).repeat(2,2),coarse_power_rad2_cm=v)
                metadata[label]=m
            # Constant spatial field is an additional non-reconstruction comparator.
            from scipy.optimize import minimize_scalar
            fit=minimize_scalar(lambda z:float(likelihood(np.full(32**3,np.exp(z)),tr).mean()),bounds=(-30,0),method='bounded')
            scores['uniform']=likelihood(np.full(32**3,np.exp(fit.x)),te)
            row=dict(case=tag,training_tracks=6144,test_tracks=2048,
                     training_with_path=len(tr.observations),test_with_path=len(te.observations),
                     mean_test_student_nll={k:float(v.mean()) for k,v in scores.items()},
                     settings=SETTINGS)
            delta=scores['pooled_student']-scores['baseline']
            boots=np.random.default_rng(44).choice(delta,size=(1000,len(delta)),replace=True).mean(axis=1)
            row['student_minus_baseline_mean_nll_95pct_bootstrap']=[float(delta.mean()),*np.quantile(boots,[.025,.975]).tolist()]
            row['student_beats_baseline']=bool(delta.mean()<0)
            np.savez_compressed(p/'test_scores.npz',**scores,test_event_ids=event_ids[indices[cut:]][te.selected])
            # Save detector-only inputs, without momentum or material arrays.
            np.savez_compressed(p/'detector_inputs.npz',**{n:getattr(tracks,n) for n in tracks.__dataclass_fields__},event_ids=event_ids,train_indices=indices[:cut],test_indices=indices[cut:])
            (p/'metadata.json').write_text(json.dumps(metadata,indent=2))
            rows.append(row);(OUT/'scores.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(row),flush=True)
    assert hashlib.sha256(production.read_bytes()).hexdigest()==manifest['production_sha256']
    (OUT/'complete.json').write_text(json.dumps(dict(cases=len(rows),production_unchanged=True,truth_loaded=False)))

if __name__=='__main__':main()
