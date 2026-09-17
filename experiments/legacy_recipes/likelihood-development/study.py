"""Development comparison. Estimators receive detector arrays only.

Protocol fixed before evaluation: 32^3 nonnegative Gaussian basis, sigma
6.25 cm; 120 L-BFGS iterations; external Reyna expected inverse momentum^2;
Gaussian and covariance-parameterized Student(df=4), angular/full observations.
The existing externally scaled, post-pooled median method is a control.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import sys
from pathlib import Path
import json, hashlib, time, shutil
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.special import gammaln

ROOT=Path('/Users/longsangchan/MuonTomo_local/muon-tomography-unified-main')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments/detector_only_likelihood')]
from method import DetectorTracks,prepare,Observations
from spectrum_method import spectrum,baseline_rho
from exposure_pooled_likelihood import pool_with_exposure
from muon_tomo.reconstruction.common import physical_grid_intersections
from muon_tomo.geometry import GridGeometry

OUT=ROOT/'outputs/likelihood-development'

def setup():
    OUT.mkdir(exist_ok=True)
    protected=[ROOT/'src/muon_tomo/reconstruction/poca.py',ROOT/'src/muon_tomo/reconstruction/munet.py',ROOT/'src/muon_tomo/reconstruction/common.py',Path('/Users/longsangchan/MuonTomo_local/muonTruck/recon/likelihood.py'),Path('/Users/longsangchan/MuonTomo_local/muonTruck/recon/utils_likelihood.py')]
    manifest=OUT/'protected_before.json'
    if not manifest.exists():
        manifest.write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},indent=2))
        shutil.copy2(ROOT/'src/muon_tomo/reconstruction/likelihood.py',OUT/'legacy_likelihood.py')
        shutil.copy2(ROOT/'src/muon_tomo/configs/default.yaml',OUT/'legacy_default.yaml')
    shutil.copy2(__file__,OUT/'study.py')

def local_observations(tracks,data):
    starts,inc,ends,out=physical_grid_intersections(tracks,GridGeometry())
    # A right-handed orthonormal pair perpendicular to the incoming direction.
    axis=np.eye(3)[np.argmin(np.abs(inc),axis=1)]
    e1=np.cross(inc,axis);e1/=np.linalg.norm(e1,axis=1)[:,None]
    e2=np.cross(inc,e1)
    longitudinal=np.sum(inc*out,axis=1)
    angles=np.column_stack([np.arctan2(np.sum(out*e,axis=1),longitudinal) for e in (e1,e2)])
    displacements=np.column_stack([np.sum((ends-starts)*e,axis=1)/100 for e in (e1,e2)])
    return Observations(data.matrices,np.stack((angles,displacements),axis=2)[data.selected],data.selected,data.grid_size)

def objective(rho,data,coefficient,mode,student):
    a=data.matrices[0]@rho*coefficient+1e-12
    xx=np.sum(data.observations[:,:,0]**2,axis=1)
    if mode=='angular':
        q=xx/a; dim=2; logdet=np.log(a)
    else:
        b=data.matrices[1]@rho*coefficient
        c=data.matrices[2]@rho*coefficient+1e-10
        determinant=np.maximum(a*c-b*b,1e-40)
        ia=c/determinant;ib=-b/determinant;ic=a/determinant
        xy=np.sum(data.observations[:,:,0]*data.observations[:,:,1],axis=1)
        yy=np.sum(data.observations[:,:,1]**2,axis=1)
        q=ia*xx+2*ib*xy+ic*yy;dim=4;logdet=np.log(determinant)
    if student:
        nu=4.
        h=(nu+dim)/(2*(nu-2+q))
        values=logdet+(nu+dim)/2*np.log1p(q/(nu-2)) + gammaln(nu/2)-gammaln((nu+dim)/2)+dim/2*np.log((nu-2)*np.pi)
    else:
        h=.5
        values=logdet+.5*q+dim/2*np.log(2*np.pi)
    if mode=='angular':
        grad=data.matrices[0].T@(coefficient*(1/a-h*xx/a**2))
    else:
        da=ia-h*(ia*ia*xx+2*ia*ib*xy+ib*ib*yy)
        db=2*ib-2*h*(ia*ib*xx+(ia*ic+ib*ib)*xy+ib*ic*yy)
        dc=ic-h*(ib*ib*xx+2*ib*ic*xy+ic*ic*yy)
        grad=sum(m.T@(coefficient*d) for m,d in zip(data.matrices,(da,db,dc)))
    return float(np.mean(values)),grad/len(values)

def fit(data,coeff,mode,student,sigma=6.25):
    n=data.grid_size
    length=data.matrices[0]@np.ones(n**3)
    observed=np.mean(data.observations[:,:,0]**2,axis=1)
    initial=max(float(np.median(observed/(coeff*length)))/np.log(2),1e-6)
    def pool(x): return gaussian_filter(x.reshape((n,)*3),sigma/(100/n),mode='reflect').ravel()
    history=[]
    def fun(z):
        amplitude=initial*np.exp(z)
        value,grad=objective(pool(amplitude),data,coeff,mode,student)
        return value,pool(grad)*amplitude
    res=minimize(fun,np.zeros(n**3),jac=True,method='L-BFGS-B',bounds=[(-20,15)]*(n**3),options={'maxiter':120,'ftol':1e-8,'gtol':1e-7,'maxls':30},callback=lambda z:history.append(fun(z)[0]))
    rho=pool(initial*np.exp(res.x)).reshape((n,)*3)
    return rho,dict(initial=initial,iterations=int(res.nit),success=bool(res.success),message=str(res.message),objective=history,mode=mode,student=student,sigma_cm=sigma)

def run_case(name):
    case=OUT/name;case.mkdir(exist_ok=True)
    src=ROOT/'experiments/detector_only_likelihood/results'/name
    with np.load(src/'detector_inputs.npz') as d:
        tracks=DetectorTracks(*(d[k] for k in ('entry_position_cm','entry_direction','exit_position_cm','exit_direction')))
    t=time.monotonic();data=local_observations(tracks,prepare(tracks))
    _,inc,_,_=physical_grid_intersections(tracks,GridGeometry())
    co,logw=spectrum(abs(inc[data.selected,2]));coeff=np.sum(np.exp(logw)*co,axis=1)
    # A fixed row split, independent of object labels, for model assessment.
    rng=np.random.default_rng(20260914);order=rng.permutation(len(data.selected));split=int(.75*len(order))
    train,test=order[:split],order[split:]
    print(name,'prepare',len(data.selected),time.monotonic()-t,flush=True)
    volumes={};metas={}
    for mode in ('angular','full'):
        for student in (False,True):
            label=f'{mode}_{"student" if student else "gaussian"}'
            cached=case/f'{label}.npz'
            if cached.exists():continue
            t=time.monotonic()
            rho,meta=fit(data,coeff,mode,student)
            # Current pass compares fixed candidates; a final policy is frozen
            # before evaluating the previously unseen validation file.
            np.savez_compressed(cached,rho=rho.repeat(2,0).repeat(2,1).repeat(2,2))
            (case/f'{label}.json').write_text(json.dumps(meta,indent=2))
            print(name,label,'fit secs',time.monotonic()-t,flush=True)
    with np.load(src/'evaluation_truth.npz') as d:
        print('truth keys',d.files,flush=True)
        target=d[d.files[0]]
    for file in case.glob('*.npz'):
        with np.load(file) as d:v=d['rho']
        row=dict(mse=float(np.mean((v-target)**2)),maximum=float(v.max()),mean=float(v.mean()),object_mse=float(np.mean((v[target>0]-target[target>0])**2)),background_mse=float(np.mean(v[target==0]**2)))
        print(name,file.stem,row,flush=True)
        metas[file.stem]=row
    (case/'evaluation.json').write_text(json.dumps(metas,indent=2))

if __name__=='__main__':
    os.chdir(ROOT);setup()
    for name in sys.argv[1:] or ['funny_0','prediction_0','prediction_5']:run_case(name)
