import numpy as np
from scipy.sparse import csr_matrix
from muon_tomo.research.detector_only_likelihood.method import DetectorTracks,Observations,likelihood,reconstruct,prepare

def test_covariance_gradient():
    rng=np.random.default_rng(42)
    a=rng.uniform(.2,4,(11,8));b=a*rng.uniform(.1,.3,(11,8));c=a*.3
    data=Observations(tuple(csr_matrix(x) for x in (a,b,c)),rng.normal(0,.01,(11,2,2)),np.arange(11),2)
    power=rng.uniform(1e-5,1e-4,8)
    for robust in (False,True):
        _,grad=likelihood(power,data,robust,True)
        numerical=[]
        for j in range(8):
            step=np.zeros(8);step[j]=1e-9
            numerical.append((likelihood(power+step,data,robust).mean()-likelihood(power-step,data,robust).mean())/2e-9)
        np.testing.assert_allclose(grad,numerical,rtol=1e-5,atol=1e-4)

def test_input_contains_only_measured_geometry_and_ignores_vector_scale():
    entry=np.array([[1,1,100.],[4,2,100.]])
    exit=np.array([[3,2,-100.],[2,1,-100.]])
    incoming=np.array([[0,0,-1.],[.01,0,-1.]])
    outgoing=np.array([[.03,.01,-1.],[-.01,.01,-1.]])
    first=DetectorTracks(entry,incoming,exit,outgoing)
    second=DetectorTracks(entry,incoming*np.array([[100.],[.001]]),exit,outgoing*np.array([[.03],[1e6]]))
    assert not hasattr(first,'momentum_mev')
    assert not hasattr(first,'material_labels')
    a,b=prepare(first),prepare(second)
    np.testing.assert_allclose(a.observations,b.observations,atol=1e-12)
    for x,y in zip(a.matrices,b.matrices):np.testing.assert_allclose(x.toarray(),y.toarray(),atol=1e-8)

def test_uniform_covariance_recovery_without_momentum():
    # Independent forward simulation: one common physical coefficient across rays
    # of varied lengths; no material label or momentum is passed to the optimizer.
    rng=np.random.default_rng(91);n=20000
    length=rng.uniform(10,100,n)
    power=2e-5
    matrices=tuple(csr_matrix(v[:,None]) for v in (length,length*.2,length*.08))
    covariance=np.array([[1.,.2],[.2,.08]])
    normal=rng.normal(size=(n,2,2))@np.linalg.cholesky(covariance).T
    scale=np.sqrt(power*length*2/rng.chisquare(4,n))
    obs=normal*scale[:,None,None]
    data=Observations(matrices,obs,np.arange(n),1)
    v,meta=reconstruct(data)
    np.testing.assert_allclose(v.item(),power,rtol=.04)
    assert meta['optimizer_success']

def test_external_spectrum_marginal_gradient():
    from muon_tomo.research.detector_only_likelihood.spectrum_method import spectrum,marginal_nll
    rng=np.random.default_rng(53);a=rng.uniform(.2,4,(11,8))
    data=Observations(tuple(csr_matrix(x) for x in (a,a*.2,a*.3)),rng.normal(0,.005,(11,2,2)),np.arange(11),2)
    prior=spectrum(np.linspace(.4,1,11));rho=rng.uniform(.1,2,8)
    np.testing.assert_allclose(np.exp(prior[1]).sum(axis=1),1)
    _,grad=marginal_nll(rho,data,prior,True)
    numerical=[]
    for j in range(8):
        step=np.zeros(8);step[j]=1e-5
        numerical.append((marginal_nll(rho+step,data,prior).mean()-marginal_nll(rho-step,data,prior).mean())/2e-5)
    np.testing.assert_allclose(grad,numerical,rtol=1e-5,atol=1e-7)

def test_exposure_pool_preserves_constant_and_identical_blocks():
    from muon_tomo.research.detector_only_likelihood.exposure_pooled_likelihood import pool_with_exposure
    entry=np.array([[1,1,100.],[4,2,100.]])
    exit=np.array([[3,2,-100.],[2,1,-100.]])
    t=DetectorTracks(entry,np.array([[0,0,-1.],[.01,0,-1.]]),exit,np.array([[.03,.01,-1.],[-.01,.01,-1.]]))
    v=pool_with_exposure(np.full((64,)*3,1.7),t)
    np.testing.assert_allclose(v[v>0],1.7)
    for axis in range(3):
        np.testing.assert_array_equal(np.take(v,range(0,64,2),axis=axis),np.take(v,range(1,64,2),axis=axis))
