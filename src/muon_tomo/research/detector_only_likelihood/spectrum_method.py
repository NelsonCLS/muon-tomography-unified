"""Experimental 1/X0 reconstruction with a fixed external Reyna spectrum.

No momentum measurements: integrate over unobserved momentum. The same
published conditional distribution is used for every object. The direction
conditions the distribution, not a fitted per-track momentum estimate.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from numpy.polynomial.legendre import leggauss
from muon_tomo.research.detector_only_likelihood.method import DetectorTracks,prepare,SETTINGS
from muon_tomo.reconstruction.common import physical_grid_intersections,prepare_poca_paths
from muon_tomo.geometry import GridGeometry

PRIOR=dict(name='Reyna 2006, equations 2 and 3, combined fit',
           source='https://arxiv.org/pdf/hep-ph/0604145',
           coefficients=[.00253,.2455,1.288,-.2555,.0209],
           minimum_p_gev=1.,maximum_p_cos_theta_gev=2000.,quadrature_nodes=64,
           highland_constant_mev=15.,beta_assumption=1.,
           component_student_df=4.,
           applicability='sea-level open-sky incident spectrum; no attenuation or energy-loss conditioning')

def spectrum(cos_theta):
    """Weights for f(p | measured zenith); no sample distribution fitting."""
    cos_theta=np.maximum(np.asarray(cos_theta),1e-8)
    nodes,w=leggauss(PRIOR['quadrature_nodes'])
    lo=np.log(PRIOR['minimum_p_gev']);hi=np.log(PRIOR['maximum_p_cos_theta_gev']/cos_theta)
    logp=lo+(nodes[None,:]+1)*(hi[:,None]-lo)/2
    p=np.exp(logp);zeta=p*cos_theta[:,None];y=np.log10(zeta)
    c1,c2,c3,c4,c5=PRIOR['coefficients']
    logflux=np.log(c1)+3*np.log(cos_theta[:,None])-(c2+c3*y+c4*y*y+c5*y**3)*np.log(zeta)
    logw=logflux+logp+np.log(w[None,:]*(hi[:,None]-lo)/2)
    logw-=logsumexp(logw,axis=1)[:,None]
    coefficient=(PRIOR['highland_constant_mev']/(1000*p))**2
    return coefficient,logw

def prepare_spectrum(tracks):
    data=prepare(tracks)
    _,incoming,_,_=physical_grid_intersections(tracks,GridGeometry())
    return data,spectrum(np.abs(incoming[data.selected,2]))

def marginal_nll(rho,data,prior,gradient=False):
    coeff,logw=prior
    base=[m@rho for m in data.matrices]
    a=base[0][:,None]*coeff+SETTINGS['angular_noise_rad']**2
    b=base[1][:,None]*coeff
    c=base[2][:,None]*coeff+(SETTINGS['position_noise_cm']/100)**2
    det=np.maximum(a*c-b*b,1e-40);ia=c/det;ib=-b/det;ic=a/det
    x=data.observations[:,:,0];y=data.observations[:,:,1]
    xx=np.sum(x*x,axis=1)[:,None];xy=np.sum(x*y,axis=1)[:,None];yy=np.sum(y*y,axis=1)[:,None]
    q=ia*xx+2*ib*xy+ic*yy;nu=PRIOR['component_student_df']
    components=np.log(det)+(nu+4)/2*np.log1p(q/(nu-2))
    logs=logw-components;normalizer=logsumexp(logs,axis=1)
    values=-normalizer
    if not gradient:return values
    posterior=np.exp(logs-normalizer[:,None])
    h=(nu+4)/(2*(nu-2+q))
    uxx=ia*ia*xx+2*ia*ib*xy+ib*ib*yy
    uxy=ia*ib*xx+(ia*ic+ib*ib)*xy+ib*ic*yy
    uyy=ib*ib*xx+2*ib*ic*xy+ic*ic*yy
    da=np.sum(posterior*coeff*(ia-h*uxx),axis=1)
    db=np.sum(posterior*coeff*(2*ib-2*h*uxy),axis=1)
    dc=np.sum(posterior*coeff*(ic-h*uyy),axis=1)
    grad=sum(m.T@v for m,v in zip(data.matrices,(da,db,dc)))/len(values)
    return float(values.mean()),grad

def fit(data,prior):
    n=data.grid_size;coeff,logw=prior
    expected=np.sum(np.exp(logw)*coeff,axis=1)
    exposure=data.matrices[0]@np.ones(n**3)
    observed=np.mean(data.observations[:,:,0]**2,axis=1)
    initial=max(float(np.median(observed/(expected*exposure))),1e-8)
    sigma=SETTINGS['pooling_sigma_cm']/(100/n)
    def pool(x):return gaussian_filter(x.reshape((n,)*3),sigma,mode='constant').ravel()
    history=[]
    def objective(z):
        amplitudes=initial*np.exp(z);rho=pool(amplitudes)+1e-12
        value,grad=marginal_nll(rho,data,prior,True)
        return value,pool(grad)*amplitudes
    result=minimize(objective,np.zeros(n**3),jac=True,method='L-BFGS-B',bounds=[(-20,20)]*(n**3),
        options={'maxiter':SETTINGS['iterations'],'ftol':SETTINGS['ftol'],'gtol':1e-7,'maxls':30},
        callback=lambda z:history.append(objective(z)[0]))
    rho=(pool(initial*np.exp(result.x))+1e-12).reshape((n,)*3)
    return rho,dict(settings=SETTINGS,external_prior=PRIOR,unit='cm^-1',initial_rho=initial,
        iterations=int(result.nit),optimizer_success=bool(result.success),message=str(result.message),
        objective=history,truth_used=False,measured_momentum_used=False)

def reconstruct_inverse_radiation_length(entry_position_cm,entry_direction,exit_position_cm,exit_direction):
    """Only four detector arrays; returns raw 64³ 1/X0 [cm^-1] and provenance."""
    tracks=DetectorTracks(entry_position_cm,entry_direction,exit_position_cm,exit_direction)
    data,prior=prepare_spectrum(tracks);rho,metadata=fit(data,prior)
    return rho.repeat(2,0).repeat(2,1).repeat(2,2),metadata

def baseline_rho(tracks,iterations=10):
    """Existing median update with the SAME external expected inverse-p².

    Unit coefficients adapt the old routine to rho directly; no observed p.
    """
    from muon_tomo.reconstruction.likelihood import LikelihoodReconstructor
    g=GridGeometry();paths,lens,occ,_=prepare_poca_paths(tracks,g)
    _,inc,_,_=physical_grid_intersections(tracks,g)
    coeff,logw=spectrum(np.abs(inc[:,2]));expected=np.sum(np.exp(logw)*coeff,axis=1)
    solver=LikelihoodReconstructor(iterations=iterations,reference_momentum_mev=1.,highland_constant_mev=1.,
        average_momentum_mev=1.,initial_inverse_radiation_length_cm_inv=4e-5,
        epsilon=1e-9,maximum_inverse_radiation_length_cm_inv=20.,update_statistic='median')
    rho,history,clipped=solver._full_covariance_mlem(tracks,1/np.sqrt(expected),paths,lens)
    return np.where(occ>0,rho.reshape(g.shape),0),dict(iterations=iterations,external_prior=PRIOR,
        algorithm='existing full median update; external expected inverse-p² in covariance',
        internal_coefficient_adapter='1/sqrt(expected variance coefficient), not measured momentum',
        upper_clip_applications=clipped,unit='cm^-1')
