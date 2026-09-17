import numpy as np
import pytest
from scipy.sparse import csr_matrix
from muon_tomo.research.likelihood_physics_study.core import (DetectorTracks, Observations, coarsen, em, local_frame, map_fit,
                  nll, penalty, prepare_both, ANGULAR_VARIANCE, POSITION_VARIANCE)


def toy_data(seed=7, n=2, count=6000):
    rng = np.random.default_rng(seed)
    rows = np.arange(count)
    ids = rng.integers(0, n**3, count)
    length = rng.uniform(2, 10, count)
    remaining = rng.uniform(0, 40, count)
    a = length
    b = (length**2/2+length*remaining)/100
    c = (length**3/3+length**2*remaining+length*remaining**2)/10000
    mats = tuple(csr_matrix((v, (rows, ids)), shape=(count, n**3)) for v in (a, b, c))
    truth = rng.uniform(.2, 2, n**3)
    coeff = np.full(count, 4e-5)
    cov = np.zeros((count, 2, 2))
    cov[:, 0, 0] = coeff*a*truth[ids]+ANGULAR_VARIANCE
    cov[:, 0, 1] = cov[:, 1, 0] = coeff*b*truth[ids]
    cov[:, 1, 1] = coeff*c*truth[ids]+POSITION_VARIANCE
    noise = rng.normal(size=(count, 2, 2))
    obs = np.einsum('nij,nkj->nki', np.linalg.cholesky(cov), noise)
    return Observations(mats, obs, rows, n), coeff, truth


@pytest.mark.parametrize('angular,student', [(False, False), (False, True), (True, False), (True, True)])
def test_likelihood_gradient(angular, student):
    data, coeff, rho = toy_data(count=100)
    _, grad = nll(rho, data, coeff, angular=angular, student=student, gradient=True)
    for i in range(len(rho)):
        shift = np.zeros_like(rho)
        shift[i] = 1e-5
        numeric = (nll(rho+shift, data, coeff, angular=angular, student=student).mean()
                   - nll(rho-shift, data, coeff, angular=angular, student=student).mean())/2e-5
        np.testing.assert_allclose(grad[i], numeric, atol=1e-8, rtol=1e-5)


@pytest.mark.parametrize('kind', ['quadratic', 'huber'])
def test_regularization_gradient(kind):
    rng = np.random.default_rng(9)
    rho = rng.random(27)
    direction = rng.normal(size=27)
    _, grad = penalty(rho, 3, kind)
    numeric = (penalty(rho+1e-6*direction, 3, kind)[0]-penalty(rho-1e-6*direction, 3, kind)[0])/2e-6
    np.testing.assert_allclose(grad@direction, numeric, rtol=1e-7)
    assert penalty(np.ones(27), 3, kind)[0] == 0


def test_local_frame_rotation_invariance():
    rng = np.random.default_rng(8)
    inc, out = rng.normal(size=(2, 30, 3))
    inc /= np.linalg.norm(inc, axis=1)[:, None]
    out /= np.linalg.norm(out, axis=1)[:, None]
    delta = rng.normal(size=(30, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    a = local_frame(inc, out, delta)
    b = local_frame(inc@rotation, out@rotation, delta@rotation)
    # Isotropic joint likelihood depends on angle norm, displacement norm,
    # and their dot product, all invariant under the transverse basis choice.
    np.testing.assert_allclose(np.einsum('nki,nkj->nij', a, a), np.einsum('nki,nkj->nij', b, b), atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(a[:, :, 0], axis=1), np.arccos(np.sum(inc*out, axis=1)))


def test_synthetic_density_recovery_and_scale_degeneracy():
    data, coeff, truth = toy_data(count=16000)
    fitted, meta = map_fit(data, coeff, np.ones((2,)*3), strength=0, maxiter=200)
    assert meta['success']
    np.testing.assert_allclose(fitted.ravel(), truth, rtol=.07)
    np.testing.assert_allclose(nll(truth, data, coeff), nll(truth*4, data, coeff/4))


def test_coarse_covariance_integrals():
    data, coeff, _ = toy_data(n=4, count=100)
    coarse = coarsen(data)
    rho = np.random.default_rng(3).uniform(.2, 1, (2,)*3)
    fine = rho.repeat(2, 0).repeat(2, 1).repeat(2, 2).ravel()
    for a, b in zip(data.matrices, coarse.matrices):
        np.testing.assert_allclose(a@fine, b@rho.ravel())


def test_vectorized_em_matches_existing_solver():
    from muon_tomo.reconstruction.likelihood import LikelihoodReconstructor
    from muon_tomo.reconstruction.common import prepare_poca_paths
    from muon_tomo.geometry import GridGeometry
    rng = np.random.default_rng(1)
    n = 45
    entry = np.column_stack((rng.uniform(-10, 10, (n, 2)), np.full(n, 100.)))
    inc = np.tile([0., 0., -1.], (n, 1))
    out = np.column_stack((rng.normal(0, .02, (n, 2)), -np.ones(n)))
    end = entry+200*out
    end[:, 2] = -100
    tracks = DetectorTracks(entry, inc, end, out)
    paper, _, coeff = prepare_both(tracks)
    grid = GridGeometry()
    paths, lengths, coverage, _ = prepare_poca_paths(tracks, grid)
    solver = LikelihoodReconstructor(iterations=10, reference_momentum_mev=1, highland_constant_mev=1,
        epsilon=1e-9, update_statistic='median')
    old, _, _ = solver._full_covariance_mlem(tracks, 1/np.sqrt(coeff), paths, lengths)
    old = np.where(coverage > 0, old.reshape(grid.shape), 0)
    new = em(paper, coeff, steps=10)[10]
    np.testing.assert_allclose(old, new, atol=1e-10, rtol=1e-9)
