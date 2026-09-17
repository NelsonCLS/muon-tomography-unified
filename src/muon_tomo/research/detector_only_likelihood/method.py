"""Experimental detector-only covariance likelihood. Never returns 1/X0.

Four measured arrays are the entire public input. The effective scattering
power includes the unknown momentum spectrum. No truth or momentum arguments.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction.common import (
    physical_grid_intersections, prepare_poca_paths, downstream_path_gaps,
)

SETTINGS = dict(grid_size=32, pooling_sigma_cm=3.125, student_df=4.,
                iterations=80, ftol=1e-8, angular_noise_rad=1e-6,
                position_noise_cm=1e-3, displacement_scale_cm=100.)

@dataclass(frozen=True)
class DetectorTracks:
    entry_position_cm: np.ndarray
    entry_direction: np.ndarray
    exit_position_cm: np.ndarray
    exit_direction: np.ndarray

    def __post_init__(self):
        for name in ('entry_position_cm','exit_position_cm','entry_direction','exit_direction'):
            a=np.array(getattr(self,name),dtype=float,copy=True)
            if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all():
                raise ValueError('finite (n,3) detector arrays required')
            if 'direction' in name:
                norms=np.linalg.norm(a,axis=1)
                if np.any(norms==0): raise ValueError('zero direction')
                a/=norms[:,None]
            object.__setattr__(self,name,a)
        if len({len(getattr(self,n)) for n in self.__dataclass_fields__})!=1:
            raise ValueError('inconsistent track counts')

    def __len__(self): return len(self.entry_position_cm)
    def subset(self,indices):
        return DetectorTracks(*(getattr(self,n)[indices] for n in self.__dataclass_fields__))
    def take_first(self, count):
        return self.subset(np.arange(min(int(count), len(self))))
    def projected_scattering_angles(self):
        incoming = self.entry_direction.copy()
        outgoing = self.exit_direction.copy()
        flip = np.sum(incoming * outgoing, axis=1) < 0
        outgoing[flip] *= -1
        ax0 = np.arctan2(incoming[:, 0], np.abs(incoming[:, 2]))
        ay0 = np.arctan2(incoming[:, 1], np.abs(incoming[:, 2]))
        ax1 = np.arctan2(outgoing[:, 0], np.abs(outgoing[:, 2]))
        ay1 = np.arctan2(outgoing[:, 1], np.abs(outgoing[:, 2]))
        return ax1 - ax0, ay1 - ay0

@dataclass
class Observations:
    matrices: tuple
    observations: np.ndarray
    selected: np.ndarray
    grid_size: int

    def subset(self, indices):
        return Observations(tuple(m[indices] for m in self.matrices),
                            self.observations[indices],self.selected[indices],self.grid_size)

def prepare(tracks,grid_size=32):
    g=GridGeometry();paths,lengths,_,_=prepare_poca_paths(tracks,g)
    starts,inc,ends,out=physical_grid_intersections(tracks,g)
    ti=np.arctan(inc[:,:2]/np.abs(inc[:,2,None]));to=np.arctan(out[:,:2]/np.abs(out[:,2,None]))
    angles=to-ti
    travel=(ends[:,2]-starts[:,2])/inc[:,2]
    projected=starts+travel[:,None]*inc
    disp=(ends[:,:2]-projected[:,:2])*np.cos(ti)*np.sqrt(1+np.sum(np.tan(ti)**2,axis=1))[:,None]*np.cos(to)/np.cos(angles)
    obs=np.stack((angles,disp/SETTINGS['displacement_scale_cm']),axis=2)
    gaps=downstream_path_gaps(tracks,g)
    rows=[];cols=[];weights=[[],[],[]];selected=[]
    for i,(ids,lens) in enumerate(zip(paths,lengths)):
        if not len(ids) or not np.isfinite(obs[i]).all():continue
        coarse=np.array(np.unravel_index(ids,g.shape))//(64//grid_size)
        flat=np.ravel_multi_index(coarse,(grid_size,)*3)
        remaining=np.cumsum(lens[::-1])[::-1]-lens+gaps[i]
        rows.extend([len(selected)]*len(ids));cols.extend(flat)
        weights[0].extend(lens)
        weights[1].extend((lens*lens/2+lens*remaining)/100.)
        weights[2].extend((lens**3/3+lens*lens*remaining+lens*remaining**2)/10000.)
        selected.append(i)
    matrices=tuple(csr_matrix((w,(rows,cols)),shape=(len(selected),grid_size**3)) for w in weights)
    return Observations(matrices,obs[selected],np.array(selected),grid_size)

def likelihood(power,data,student=True,gradient=False):
    a,b,c=(m@power for m in data.matrices)
    a=a+SETTINGS['angular_noise_rad']**2
    c=c+(SETTINGS['position_noise_cm']/100.)**2
    det=np.maximum(a*c-b*b,1e-40)
    ia=c/det;ib=-b/det;ic=a/det
    x=data.observations[:,:,0];y=data.observations[:,:,1]
    xx=np.sum(x*x,axis=1);xy=np.sum(x*y,axis=1);yy=np.sum(y*y,axis=1)
    q=ia*xx+2*ib*xy+ic*yy
    if student:
        nu=SETTINGS['student_df']
        # Covariance parameterization of a 4D Student-t; scale=(nu-2)/nu * Cov.
        values=np.log(det)+(nu+4)/2*np.log1p(q/(nu-2))
        h=(nu+4)/(2*(nu-2+q))
    else:
        values=np.log(det)+q/2
        h=np.full(len(q),.5)
    if not gradient:return values
    ux=ia[:,None]*x+ib[:,None]*y
    uy=ib[:,None]*x+ic[:,None]*y
    da=ia-h*np.sum(ux*ux,axis=1)
    db=2*ib-2*h*np.sum(ux*uy,axis=1)
    dc=ic-h*np.sum(uy*uy,axis=1)
    grad=sum(m.T@v for m,v in zip(data.matrices,(da,db,dc)))/len(q)
    return float(values.mean()),grad

def reconstruct(data,student=True):
    n=data.grid_size
    exposure=data.matrices[0]@np.ones(n**3)
    angular=np.mean(data.observations[:,:,0]**2,axis=1)
    initial=max(float(np.median(angular/exposure)),1e-12)
    sigma=SETTINGS['pooling_sigma_cm']/(100/n)
    def pool(x):return gaussian_filter(x.reshape((n,)*3),sigma,mode='constant').ravel()
    history=[]
    def objective(z):
        amplitudes=initial*np.exp(z)
        power=pool(amplitudes)+1e-14
        value,grad=likelihood(power,data,student,True)
        # Constant-padding symmetric Gaussian is its own transpose.
        return value,pool(grad)*amplitudes
    result=minimize(objective,np.zeros(n**3),jac=True,method='L-BFGS-B',
                    bounds=[(-20,20)]*(n**3),
                    options={'maxiter':SETTINGS['iterations'],'ftol':SETTINGS['ftol'],'gtol':1e-7,'maxls':30},
                    callback=lambda z:history.append(objective(z)[0]))
    volume=(pool(initial*np.exp(result.x))+1e-14).reshape((n,)*3)
    return volume,dict(settings=SETTINGS,student=student,initial_power_rad2_cm=initial,
                       optimizer_success=bool(result.success),message=str(result.message),
                       iterations=int(result.nit),objective=history,unit='rad^2/cm',
                       interpretation='effective covariance scattering power, no momentum conversion')

def baseline(tracks):
    """Existing full-mode median update, expressed directly in scattering power.

    Unit scale 1 is an algebraic adapter, NOT a physical momentum assumption.
    Production iteration count, initializer expressed without hidden prior,
    covariance jitter and upper bound are explicitly documented in metadata.
    """
    from muon_tomo.reconstruction.likelihood import LikelihoodReconstructor
    g=GridGeometry();paths,lens,occ,_=prepare_poca_paths(tracks,g)
    # Public default reference 3000 and Highland 15 only define the old numerical
    # initializer/unit conversion. No 1743.46 sample-derived prior is used.
    initial=4e-5*(15/3000)**2
    solver=LikelihoodReconstructor(iterations=10,reference_momentum_mev=1.,
        highland_constant_mev=1.,average_momentum_mev=1.,
        initial_inverse_radiation_length_cm_inv=initial,epsilon=1e-9,
        maximum_inverse_radiation_length_cm_inv=20*(15/3000)**2,update_statistic='median')
    power,history,clipped=solver._full_covariance_mlem(tracks,np.ones(len(tracks)),paths,lens)
    return np.where(occ>0,power.reshape(g.shape),0),dict(
        iterations=10,update_statistic='median',initial_power_rad2_cm=initial,
        epsilon=1e-9,upper_power_rad2_cm=20*(15/3000)**2,clipped=clipped,
        objective_history=history,unit='rad^2/cm',physical_momentum_used=False)
