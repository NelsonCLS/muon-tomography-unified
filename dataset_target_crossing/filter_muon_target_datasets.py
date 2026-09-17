"""Non-destructive TFRecord filtering by canonical target-path intersection."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import time
import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(2)
FIELDS=['record','before','after','invalid_geometry','output_sha256','truth_sha256']
ALGORITHM='canonical_poca_kink_or_straight_fallback_positive_cube_path_v1'

def segment_hits(a,b):
    delta=b-a;lo=np.zeros(len(a));hi=np.ones(len(a))
    valid=np.isfinite(a).all(1)&np.isfinite(b).all(1)
    for j in range(3):
        parallel=np.abs(delta[:,j])<1e-15
        valid&=~parallel|((a[:,j]>=-50)&(a[:,j]<50))
        den=np.where(parallel,1.,delta[:,j])
        u=(-50-a[:,j])/den;v=(50-a[:,j])/den
        lo=np.maximum(lo,np.where(parallel,-np.inf,np.minimum(u,v)))
        hi=np.minimum(hi,np.where(parallel,np.inf,np.maximum(u,v)))
    return valid&(hi>lo)&(np.linalg.norm(delta,axis=1)>0)

def crossing_mask(x):
    if x.ndim!=2 or x.shape[1]<12:raise ValueError(f'Unsupported x shape {x.shape}')
    a0=x[:,:3]/10.;b0=x[:,6:9]/10.
    d0=x[:,3:6].copy();e0=x[:,9:12].copy()
    nd=np.linalg.norm(d0,axis=1);ne=np.linalg.norm(e0,axis=1)
    valid=np.isfinite(x[:,:12]).all(1)&(nd>0)&(ne>0)
    d0/=np.where(nd>0,nd,1.)[:,None];e0/=np.where(ne>0,ne,1.)[:,None]
    upper=a0[:,2]>=b0[:,2]
    a=np.where(upper[:,None],a0,b0);b=np.where(upper[:,None],b0,a0)
    d=np.where(upper[:,None],d0,e0);e=np.where(upper[:,None],e0,d0)
    flight=b-a;d[np.sum(d*flight,axis=1)<0]*=-1;e[np.sum(e*flight,axis=1)<0]*=-1
    valid&=(np.abs(d[:,2])>1e-12)&(np.abs(e[:,2])>1e-12)
    dz=np.where(np.abs(d[:,2])>1e-12,d[:,2],1.);ez=np.where(np.abs(e[:,2])>1e-12,e[:,2],1.)
    a=a+((50-a[:,2])/dz)[:,None]*d;b=b+((-50-b[:,2])/ez)[:,None]*e
    w=a-b;aa=np.sum(d*d,axis=1);bb=np.sum(d*e,axis=1);cc=np.sum(e*e,axis=1)
    dd=np.sum(d*w,axis=1);ee=np.sum(e*w,axis=1);det=aa*cc-bb*bb
    defined=np.abs(det)>=1e-12;den=np.where(defined,det,1.)
    p=.5*(a+((bb*ee-cc*dd)/den)[:,None]*d+b+((aa*ee-bb*dd)/den)[:,None]*e)
    inside=defined&np.all((p>=-50)&(p<50),axis=1)
    return valid&(inside|segment_hits(a,b)),int((~valid).sum())

def dataset(path):
    with path.open('rb') as f:compressed=f.read(2)==b'\x1f\x8b'
    options=tf.data.Options();options.threading.private_threadpool_size=2
    return tf.data.TFRecordDataset(str(path),compression_type='GZIP' if compressed else '',
        buffer_size=8*1024*1024).with_options(options).prefetch(4)

def decode_x(example):
    x=tf.io.parse_tensor(example.features.feature['x'].bytes_list.value[0],out_type=tf.float64).numpy()
    if x.ndim!=2 or x.shape[1]<12:raise ValueError(f'Unsupported x shape {x.shape}')
    return x

def process(index,raw):
    ex=tf.train.Example.FromString(raw)
    if 'x' not in ex.features.feature or 'y' not in ex.features.feature:raise ValueError('Expected x/y TFRecord schema')
    x=decode_x(ex);mask,invalid=crossing_mask(x)
    selected=np.ascontiguousarray(x[mask]);ids=np.flatnonzero(mask).astype('<u4')
    ex.features.feature['x'].bytes_list.value[0]=tf.io.serialize_tensor(selected).numpy()
    out=ex.SerializeToString()
    row=dict(record=index,before=len(x),after=len(ids),invalid_geometry=invalid,
        output_sha256=hashlib.sha256(out).hexdigest(),truth_sha256=hashlib.sha256(ex.features.feature['y'].SerializeToString()).hexdigest())
    return out,ids,row

def self_check(root,sources):
    sys.path.insert(0,str(root/'src'))
    from muon_tomo.geometry import GridGeometry
    from muon_tomo.types import TrackTable
    from muon_tomo.reconstruction.common import prepare_poca_paths
    checks=[]
    for source in sources:
        raw=next(iter(dataset(source))).numpy();ex=tf.train.Example.FromString(raw);x=decode_x(ex)
        mask,_=crossing_mask(x)
        tracks=TrackTable(x[:,:3]/10.,x[:,3:6],x[:,6:9]/10.,x[:,9:12],np.ones(len(x)))
        paths,_,_,_=prepare_poca_paths(tracks,GridGeometry())
        np.testing.assert_array_equal(mask,np.array([len(p)>0 for p in paths]))
        serialized,ids,row=process(0,raw);out=tf.train.Example.FromString(serialized)
        np.testing.assert_array_equal(decode_x(out),x[mask])
        assert crossing_mask(decode_x(out))[0].all()
        for key,value in ex.features.feature.items():
            if key!='x':assert value.SerializeToString()==out.features.feature[key].SerializeToString()
        if source.name=='voxels_prediction_funny.tfrecord':assert len(x)==44240 and mask.sum()==14943
        check=dict(dataset=source.name,input_tracks=len(x),retained=int(mask.sum()),matches_production_geometry=True,unchanged_truth=True)
        checks.append(check);print('CHECK',json.dumps(check),flush=True)
    a=np.array([[0,0,50],[75,0,50],[75,0,50],[50,0,50],[0,0,50]],float)
    b=np.array([[0,0,-50],[0,0,-50],[75,0,-50],[50,0,-50],[100,0,50]],float)
    np.testing.assert_array_equal(segment_hits(a,b),[True,True,False,False,False])
    return checks

def snapshot(p):
    s=p.stat();return dict(size_bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)

def verify(path,counts,indices):
    n=0;total=0;last=time.monotonic()
    with counts.open(newline='') as f,gzip.open(indices,'rb') as provenance:
        expected=iter(csv.DictReader(f))
        for n,raw in enumerate(dataset(path),start=1):
            row=next(expected,None);assert row is not None
            raw=raw.numpy();assert hashlib.sha256(raw).hexdigest()==row['output_sha256']
            ex=tf.train.Example.FromString(raw);x=decode_x(ex)
            assert len(x)==int(row['after']) and crossing_mask(x)[0].all()
            assert hashlib.sha256(ex.features.feature['y'].SerializeToString()).hexdigest()==row['truth_sha256']
            record,k=struct.unpack('<QQ',provenance.read(16));assert record==n-1 and k==len(x)
            ids=np.frombuffer(provenance.read(4*k),dtype='<u4')
            assert len(ids)==k and (np.diff(ids.astype(np.int64))>0).all()
            assert not len(ids) or ids[-1]<int(row['before'])
            total+=len(x)
            if time.monotonic()-last>25:
                print(f'VERIFY {path.name}: {n:,} samples, {total:,} tracks checked',flush=True);last=time.monotonic()
        assert next(expected,None) is None and provenance.read(1)==b''
    return dict(samples_verified=n,tracks_verified=total,serialized_records_match=True,
        truth_payloads_unchanged=True,all_tracks_cross_cube=True,provenance_indices_verified=True)

def convert(source,dest,workers):
    target=dest/source.name;partial=target.with_suffix(target.suffix+'.partial')
    counts=dest/(source.stem+'.counts.csv');indices=dest/(source.stem+'.original_indices.u32.gz')
    manifest=dest/(source.stem+'.summary.json');before=snapshot(source)
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved.get('complete') and saved['source_snapshot']==before and target.exists():
            print('REUSE',target.name,flush=True);return saved
        raise FileExistsError(f'Existing manifest cannot be reused: {manifest}')
    for p in (target,partial,counts,indices):
        if p.exists():raise FileExistsError(f'Refusing to overwrite {p}')
    start=time.monotonic();last=start;rows=[]
    options=tf.io.TFRecordOptions(compression_type='GZIP',compression_level=1)
    with counts.open('x',newline='') as f,gzip.open(indices,'xb',compresslevel=1) as prov,tf.io.TFRecordWriter(str(partial),options=options) as writer,ThreadPoolExecutor(max_workers=workers) as pool:
        csv_writer=csv.DictWriter(f,fieldnames=FIELDS);csv_writer.writeheader();pending=deque()
        def consume():
            nonlocal last
            raw,ids,row=pending.popleft().result();writer.write(raw);csv_writer.writerow(row)
            prov.write(struct.pack('<QQ',row['record'],len(ids)));prov.write(ids.tobytes())
            rows.append((row['before'],row['after'],row['invalid_geometry']))
            if time.monotonic()-last>25:
                f.flush();print(f'FILTER {source.name}: {len(rows):,} samples, mean retained {np.mean([r[1] for r in rows]):,.1f}, elapsed {time.monotonic()-start:.0f}s',flush=True);last=time.monotonic()
        for i,raw in enumerate(dataset(source)):
            pending.append(pool.submit(process,i,raw.numpy()))
            if len(pending)>=2*workers:consume()
        while pending:consume()
    assert snapshot(source)==before,'Source changed during conversion'
    print(f'FILTERED {source.name}: {len(rows):,} samples; verifying output',flush=True)
    verified=verify(partial,counts,indices);assert snapshot(source)==before
    partial.rename(target);arr=np.array(rows,dtype=np.int64)
    r=dict(source=str(source),output=str(target),algorithm=ALGORITHM,source_snapshot=before,samples=len(rows),
        total_input_tracks=int(arr[:,0].sum()),total_retained_tracks=int(arr[:,1].sum()),
        mean_retained_tracks_per_sample=float(arr[:,1].mean()),min_retained_tracks_per_sample=int(arr[:,1].min()),
        max_retained_tracks_per_sample=int(arr[:,1].max()),retained_fraction=float(arr[:,1].sum()/arr[:,0].sum()),
        empty_samples=int((arr[:,1]==0).sum()),invalid_geometry_tracks=int(arr[:,2].sum()),
        output_bytes=target.stat().st_size,elapsed_seconds=time.monotonic()-start,verification=verified,original_source_unchanged=True,complete=True)
    with manifest.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print('DONE',json.dumps(r),flush=True);return r

def report(dest,results,checks,sources):
    n=sum(r['samples'] for r in results);total=sum(r['total_retained_tracks'] for r in results)
    summary=dict(algorithm=ALGORITHM,bounds_cm=[-50,50],datasets=results,geometry_checks=checks,
        selection_information='Positions and normalized directions only. No truth, momentum magnitude, RPC coincidence, or top-bottom-only cut.',
        definition='Nonzero cube path: entry-PoCA-exit when PoCA is inside, otherwise straight segment between fitted z-face endpoints. Includes side crossings. This is an inferred path, not stored transport truth.',
        source_files=[str(p) for p in sources],all_datasets_complete=len(results)==len(sources),total_samples=n,
        total_retained_tracks=total,pooled_mean_retained_tracks_per_sample=total/n,
        pooled_min_retained_tracks_per_sample=min(r['min_retained_tracks_per_sample'] for r in results),
        pooled_max_retained_tracks_per_sample=max(r['max_retained_tracks_per_sample'] for r in results))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (dest/'summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['dataset','samples','mean_retained','minimum','maximum','retained_percent'])
        for r in results:writer.writerow([Path(r['source']).name,r['samples'],r['mean_retained_tracks_per_sample'],r['min_retained_tracks_per_sample'],r['max_retained_tracks_per_sample'],100*r['retained_fraction']])
    lines=['# Target-crossing dataset copies','',summary['definition'],'',summary['selection_information'],'',
        'Originals are read-only inputs. All samples remain in their original order, including empty samples. Every retained x row preserves every original value and column. Serialized y and extra feature payloads are unchanged. Outputs use GZIP compression.','',
        '| Dataset | Samples | Mean retained | Min | Max | Retained (%) |','|---|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {Path(r['source']).name} | {r['samples']:,} | {r['mean_retained_tracks_per_sample']:,.2f} | {r['min_retained_tracks_per_sample']:,} | {r['max_retained_tracks_per_sample']:,} | {100*r['retained_fraction']:.2f} |")
    lines+=['','All output files were fully reread and checked against serialized record hashes, truth hashes, retained geometry, and row provenance. Source size, modification time and inode were unchanged.','',
        'Each original_indices.u32.gz decompresses to repeated little-endian uint64 record index, uint64 retained count, then uint32 original row indices[count]. Indices are zero-based and ascending. Per-sample counts CSVs record input/output counts and hashes.','',
        'Dose semantics change: 8192 rows drawn from these files are target-crossing tracks when a sample contains enough. No cut on truth object material is applied; paths through air inside the cube still qualify.', '',f'Completed {len(results)} / {len(sources)} datasets.']
    (dest/'README.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--check-only',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    root=a.root.resolve();sources=sorted((root/'dataset').glob('*.tfrecord'),key=lambda p:p.stat().st_size)
    if not sources:raise RuntimeError('No datasets found')
    checks=self_check(root,sources)
    if a.check_only:return
    dest=root/'dataset_target_crossing';dest.mkdir(exist_ok=True)
    script=dest/'filter_muon_target_datasets.py'
    if not script.exists():shutil.copyfile(__file__,script)
    results=[]
    for source in sources:
        results.append(convert(source,dest,a.workers));report(dest,results,checks,sources)
    print('ALL_DATASETS_COMPLETE',str(dest/'summary.json'),flush=True)

if __name__=='__main__':main()
