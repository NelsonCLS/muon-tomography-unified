"""Audit measured track geometry; no material or momentum inputs to statistics."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from muon_tomo.io.tfrecord import load_tfrecord_record
from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction.common import physical_track_order,physical_grid_intersections,prepare_poca_paths,closest_approach
from muon_tomo.cli import _track_fingerprint

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'dataset/voxels_prediction_funny.tfrecord'
G=GridGeometry(); FACE=['x−','x+','y−','y+','bottom','top']

def clip(a,b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():return None
    d=b-a; lo=0.;hi=1.
    for j in range(3):
        if abs(d[j])<1e-15:
            if a[j]<-50 or a[j]>50:return None
        else:
            t1=(-50-a[j])/d[j];t2=(50-a[j])/d[j]
            lo=max(lo,min(t1,t2));hi=min(hi,max(t1,t2))
    if hi<=lo:return None
    return a+lo*d,a+hi*d,float((hi-lo)*np.linalg.norm(d))

def face(p):
    residual=np.array([abs(p[0]+50),abs(p[0]-50),abs(p[1]+50),abs(p[1]-50),abs(p[2]+50),abs(p[2]-50)])
    assert residual.min()<1e-6
    return FACE[int(residual.argmin())]

def distribution(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if not len(x):return None
    return dict(count=len(x),mean=float(x.mean()),std=float(x.std()),**{f'p{p}':float(np.percentile(x,p)) for p in (0,1,5,25,50,75,95,99,100)})

def analyze(t,label):
    n=len(t);a,di,b,do=physical_grid_intersections(t,G)
    top,inc,bot,out=physical_track_order(t)
    paths,lens,_,kinks=prepare_poca_paths(t,G)
    pathlen=np.array([sum(v) for v in lens]);hit=pathlen>0
    incoming_hit=np.zeros(n,bool);outgoing_hit=np.zeros(n,bool)
    poca_inside=np.zeros(n,bool);poca_defined=np.zeros(n,bool)
    entry=np.full(n,'miss',dtype='<U10');exit=np.full(n,'miss',dtype='<U10')
    counts={32:np.zeros(32**3,int),64:np.zeros(64**3,int)}
    exposure={32:np.zeros(32**3,int),64:np.zeros(64**3,int)}
    poca_xyz=np.full((n,3),np.nan)
    independent_lengths=[]
    for i in range(n):
        in_end=a[i]+((-50-a[i,2])/di[i,2])*di[i]
        out_start=b[i]+((50-b[i,2])/do[i,2])*do[i]
        incoming_hit[i]=clip(a[i],in_end) is not None
        outgoing_hit[i]=clip(out_start,b[i]) is not None
        p=closest_approach(a[i],di[i],b[i],do[i])
        poca_defined[i]=p is not None
        if p is not None:
            poca_xyz[i]=p;poca_inside[i]=bool(G.contains(p))
        segments=[(a[i],p),(p,b[i])] if poca_inside[i] else [(a[i],b[i])]
        cuts=[c for u,v in segments if (c:=clip(u,v)) is not None]
        independent_lengths.append(sum(c[2] for c in cuts))
        if cuts:
            entry[i]=face(cuts[0][0]);exit[i]=face(cuts[-1][1])
        for size in (32,64):
            factor=64//size
            if len(paths[i]):
                xyz=np.array(np.unravel_index(paths[i],G.shape)).T//factor
                ids=np.unique(np.ravel_multi_index(xyz.T,(size,)*3))
                exposure[size][ids]+=1
            if hit[i] and poca_inside[i]:
                xyz=G.point_to_index(p)//factor
                counts[size][np.ravel_multi_index(xyz,(size,)*3)]+=1
    np.testing.assert_allclose(independent_lengths,pathlen,rtol=1e-9,atol=1e-8)
    assert np.array_equal(entry!='miss',hit)
    through=(entry=='top')&(exit=='bottom')
    top_in_square=np.all((a[:,:2]>=-50)&(a[:,:2]<=50),axis=1)
    bottom_in_square=np.all((b[:,:2]>=-50)&(b[:,:2]<=50),axis=1)
    assert np.array_equal(through,top_in_square&bottom_in_square)
    angle=np.degrees(np.arccos(np.clip(np.sum(inc*out,axis=1),-1,1)))
    zenith=np.degrees(np.arccos(np.clip(-inc[:,2],-1,1)))
    dtx,dty=t.projected_scattering_angles();q=.5*(dtx*dtx+dty*dty)
    valid=hit&poca_inside;q_valid=q[valid]
    contributions={}
    for pct in (0.1,1,5,10):
        k=max(1,int(np.ceil(len(q_valid)*pct/100)))
        contributions[str(pct)]=dict(tracks=k,fraction_of_deposited_angular_signal=float(np.sort(q_valid)[-k:].sum()/q_valid.sum()))
    face_pairs=Counter(f'{en} → {ex}' for en,ex in zip(entry[hit],exit[hit]))
    classes={'top_to_bottom':int(through.sum()),'top_to_side':int(((entry=='top')&hit&(exit!='bottom')).sum()),
        'side_to_bottom':int(((entry!='top')&hit&(exit=='bottom')).sum()),
        'side_to_side':int(((entry!='top')&hit&(exit!='bottom')).sum()),'miss':int((~hit).sum())}
    assert sum(classes.values())==n
    report=dict(tracks=n,fingerprint=_track_fingerprint(t),classes=classes,intersects_current_reconstruction_path=int(hit.sum()),
        incoming_fitted_line_intersects=int(incoming_hit.sum()),outgoing_fitted_line_intersects=int(outgoing_hit.sum()),
        both_fitted_lines_intersect=int((incoming_hit&outgoing_hit).sum()),either_fitted_line_intersects=int((incoming_hit|outgoing_hit).sum()),
        reconstruction_hit_neither_fitted_line_hit=int((hit&~incoming_hit&~outgoing_hit).sum()),
        fitted_line_hit_reconstruction_miss=int(((incoming_hit|outgoing_hit)&~hit).sum()),
        entry_faces=dict(Counter(entry[hit])),exit_faces=dict(Counter(exit[hit])),face_pairs=dict(face_pairs),
        poca_defined=int(poca_defined.sum()),poca_inside=int(poca_inside.sum()),poca_outside=int((poca_defined&~poca_inside).sum()),
        poca_parallel_undefined=int((~poca_defined).sum()),signal_depositing_tracks=int(valid.sum()),
        exposure_only_tracks=int((hit&~poca_inside).sum()),path_length_cm=distribution(pathlen[hit]),
        short_paths_below_10cm=int(((pathlen>0)&(pathlen<10)).sum()),short_paths_below_25cm=int(((pathlen>0)&(pathlen<25)).sum()),
        incoming_zenith_deg=distribution(zenith),deflection_deg=distribution(angle),
        deposited_signal_tail_contributions=contributions,
        endpoint_z_cm=dict(first=distribution(t.entry_position_cm[:,2]),second=distribution(t.exit_position_cm[:,2])),
        voxel_statistics={})
    for size in (32,64):
        e=exposure[size];c=counts[size]
        report['voxel_statistics'][str(size)]=dict(total_voxels=size**3,voxels_with_path=int((e>0).sum()),
            voxels_with_poca=int((c>0).sum()),poca_count_per_occupied_voxel=distribution(c[c>0]),
            track_count_per_traversed_voxel=distribution(e[e>0]),
            fraction_voxels_with_path=float((e>0).mean()),fraction_voxels_with_poca=float((c>0).mean()))
    np.savez_compressed(OUT/f'{label}_tracks.npz',event_id=t.event_id,path_length_cm=pathlen,entry_face=entry,exit_face=exit,
        incoming_line_hit=incoming_hit,outgoing_line_hit=outgoing_hit,poca_inside=poca_inside,poca_xyz_cm=poca_xyz,
        incoming_zenith_deg=zenith,deflection_deg=angle,angular_signal=q,
        path_count_32=exposure[32].reshape((32,)*3),path_count_64=exposure[64].reshape((64,)*3),
        poca_count_32=counts[32].reshape((32,)*3),poca_count_64=counts[64].reshape((64,)*3))
    print(label,json.dumps({k:report[k] for k in ('tracks','classes','intersects_current_reconstruction_path','poca_inside','exposure_only_tracks')}),flush=True)
    return report,pathlen,zenith,angle

full,_=load_tfrecord_record(SOURCE,0)
selected,_=load_tfrecord_record(SOURCE,0,limit=8192,seed=0);selected=selected.take_first(8192)
expected=np.load(ROOT/'outputs/poca-grid-sigma-study/selected_event_ids.npy')
np.testing.assert_array_equal(selected.event_id,expected)
all_r,all_l,all_z,all_a=analyze(full,'full')
sub_r,sub_l,sub_z,sub_a=analyze(selected,'selected8192')
protocol=dict(dataset=str(SOURCE),record=0,bounds_cm=[-50,50],top_z_cm=50,bottom_z_cm=-50,
    selected_count=8192,seed=0,selected_matches_previous_poca_sweep=True,
    momentum_or_material_information_used=False,
    interpretation='Endpoint fits do not provide the true internal trajectory. Main face counts use the current clipped entry-PoCA-exit path, or straight fallback. Independent fitted-line intersection counts are also reported.',
    validation='Independent segment-box clipping lengths agree with production voxel tracing; face categories partition all records; top-bottom matches fitted z-plane square acceptance.')
(OUT/'statistics.json').write_text(json.dumps(dict(protocol=protocol,full=all_r,selected8192=sub_r),indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
names=['top_to_bottom','top_to_side','side_to_bottom','side_to_side','miss'];x=np.arange(5)
for j,(r,label,color) in enumerate([(all_r,'Full Funny 0','#406783'),(sub_r,'Selected 8,192','#b77537')]):
    axes[0,0].bar(x+(j-.5)*.36,[100*r['classes'][k]/r['tracks'] for k in names],width=.36,label=label,color=color)
axes[0,0].set_xticks(x,['Top → bottom','Top → side','Side → bottom','Side → side','Miss'],rotation=15)
axes[0,0].set_ylabel('Tracks (%)');axes[0,0].set_xlabel('Reconstruction-path face category');axes[0,0].legend();axes[0,0].set_title('Entry and exit composition')
for ax,fa,sa,bins,title,xlabel in [(axes[0,1],all_l[all_l>0],sub_l[sub_l>0],np.linspace(0,180,37),'In-volume path lengths','Path length (cm)'),
    (axes[1,0],all_z,sub_z,np.linspace(0,90,37),'Incoming directions','Zenith angle (degrees)'),
    (axes[1,1],all_a,sub_a,np.linspace(0,max(all_a.max(),sub_a.max())+.01,60),'Measured direction changes','3D deflection angle (degrees)')]:
    for v,label,color in [(fa,'Full Funny 0','#406783'),(sa,'Selected 8,192','#b77537')]:
        ax.hist(v,bins=bins,weights=np.ones(len(v))*100/len(v),histtype='step',linewidth=1.7,label=label,color=color)
    ax.set_title(title);ax.set_xlabel(xlabel);ax.set_ylabel('Tracks per bin (%)');ax.legend()
axes[1,1].set_yscale('log');axes[1,1].set_ylabel('Tracks per bin (%, log scale)')
fig.suptitle(f'Funny 0: {len(full):,} stored tracks vs the same 8,192-track subset (seed 0)\nDetector geometry only; cube x,y,z ∈ [−50,50] cm',fontsize=14)
fig.savefig(OUT/'composition.png',dpi=160);plt.close(fig)
