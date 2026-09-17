"""Physics ablations; detector arrays only, no truth-based calibration.

Reuses the existing ray tracing and fixed external spectrum. All displacement
coordinates are divided by 100 cm, including detector covariance.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from muon_tomo.research.detector_only_likelihood.method import DetectorTracks, Observations, prepare
from muon_tomo.research.detector_only_likelihood.spectrum_method import spectrum
from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction.common import physical_grid_intersections

FLOOR = 1e-9
ANGULAR_VARIANCE = 1e-12 + 1e-9
POSITION_VARIANCE = (1e-3 / 100)**2 + 1e-9 / 10000


def local_frame(incoming, outgoing, delta):
    axes = np.eye(3)[np.argmin(abs(incoming), axis=1)]
    e1 = np.cross(incoming, axes)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(incoming, e1)
    t = np.column_stack([np.sum(outgoing * e, axis=1) for e in (e1, e2)])
    sine = np.linalg.norm(t, axis=1)
    angle = np.arctan2(sine, np.sum(incoming * outgoing, axis=1))
    scale = np.divide(angle, sine, out=np.ones_like(angle), where=sine > 1e-15)
    displacement = np.column_stack([np.sum(delta * e, axis=1) / 100 for e in (e1, e2)])
    return np.stack((t * scale[:, None], displacement), axis=2)


def prepare_both(tracks):
    paper = prepare(tracks, 64)
    start, inc, end, out = physical_grid_intersections(tracks, GridGeometry())
    obs = local_frame(inc, out, end - start)[paper.selected]
    local = Observations(paper.matrices, obs, paper.selected, 64)
    co, lw = spectrum(abs(inc[paper.selected, 2]))
    return paper, local, np.sum(np.exp(lw) * co, axis=1)


def coarsen(data):
    n = data.grid_size
    ijk = np.array(np.unravel_index(np.arange(n**3), (n,)*3)) // 2
    ids = np.ravel_multi_index(ijk, (n//2,)*3)
    aggregation = csr_matrix((np.ones(n**3), (np.arange(n**3), ids)), shape=(n**3, (n//2)**3))
    return Observations(tuple(m @ aggregation for m in data.matrices), data.observations, data.selected, n//2)


def nll(rho, data, coeff, *, angular=False, student=False, gradient=False):
    a = data.matrices[0] @ rho * coeff + ANGULAR_VARIANCE
    x, y = data.observations[:, :, 0], data.observations[:, :, 1]
    xx = np.sum(x*x, axis=1)
    if angular:
        q, ld, dim = xx/a, np.log(a), 2
    else:
        b = data.matrices[1] @ rho * coeff
        c = data.matrices[2] @ rho * coeff + POSITION_VARIANCE
        det = a*c-b*b
        if np.any(det <= 0):
            raise ValueError('nonpositive covariance determinant')
        ia, ib, ic = c/det, -b/det, a/det
        xy, yy = np.sum(x*y, axis=1), np.sum(y*y, axis=1)
        q, ld, dim = ia*xx+2*ib*xy+ic*yy, np.log(det), 4
    h = (4+dim)/(2*(2+q)) if student else .5
    values = ld + (4+dim)/2*np.log1p(q/2) if student else ld + .5*q
    if not gradient:
        return values
    if angular:
        grad = data.matrices[0].T @ (coeff*(1/a-h*xx/a**2))
    else:
        da = ia-h*(ia*ia*xx+2*ia*ib*xy+ib*ib*yy)
        db = 2*ib-2*h*(ia*ib*xx+(ia*ic+ib*ib)*xy+ib*ic*yy)
        dc = ic-h*(ib*ib*xx+2*ib*ic*xy+ic*ic*yy)
        grad = sum(m.T @ (coeff*d) for m, d in zip(data.matrices, (da, db, dc)))
    return float(values.mean()), grad/len(values)


def em(data, coeff, statistic='median', steps=40):
    mats = data.matrices
    rows, ids = mats[0].nonzero()
    weights = np.stack([np.asarray(m[rows, ids]).ravel() for m in mats], axis=1)
    counts = np.bincount(ids, minlength=data.grid_size**3)
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    use = np.flatnonzero(counts >= 3)
    x, y = data.observations[:, :, 0], data.observations[:, :, 1]
    xx, xy, yy = np.sum(x*x, axis=1), np.sum(x*y, axis=1), np.sum(y*y, axis=1)
    rho = np.full(data.grid_size**3, 4e-5)
    snapshots = {}
    for step in range(1, steps+1):
        a, b, c = [m @ rho * coeff for m in mats]
        a += ANGULAR_VARIANCE
        c += POSITION_VARIANCE
        det = a*c-b*b
        ia, ib, ic = c/det, -b/det, a/det
        da = ia-.5*(ia*ia*xx+2*ia*ib*xy+ib*ib*yy)
        db = 2*ib-(ia*ib*xx+(ia*ic+ib*ib)*xy+ib*ic*yy)
        dc = ic-.5*(ib*ib*xx+2*ib*ic*xy+ic*ic*yy)
        grad = (weights[:, 0]*da[rows]+weights[:, 1]*db[rows]+weights[:, 2]*dc[rows])*coeff[rows]
        scores = rho[ids]-.5*rho[ids]**2*grad
        new = np.divide(np.bincount(ids, weights=scores, minlength=len(rho)), counts,
                        out=np.zeros_like(rho), where=counts > 0)
        if statistic == 'median':
            ordered = scores[np.lexsort((scores, ids))]
            new[use] = .5*(ordered[offsets[use]+(counts[use]-1)//2]+ordered[offsets[use]+counts[use]//2])
        rho = np.clip(new, FLOOR, 20)
        if step in (10, 40):
            snapshots[step] = np.where(counts > 0, rho, 0).reshape((data.grid_size,)*3)
    return snapshots


def pool(rho, data, sigma_cm=3.125):
    n = data.grid_size
    exposure = np.asarray(data.matrices[0].sum(axis=0)).reshape((n,)*3)
    num = exposure*rho
    if n == 64:
        num = num.reshape(32, 2, 32, 2, 32, 2).sum((1, 3, 5))
        exposure = exposure.reshape(32, 2, 32, 2, 32, 2).sum((1, 3, 5))
    num = gaussian_filter(num, sigma_cm/3.125, mode='constant')
    den = gaussian_filter(exposure, sigma_cm/3.125, mode='constant')
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0).repeat(2, 0).repeat(2, 1).repeat(2, 2)


def penalty(rho, n, kind):
    """Mean nearest-neighbor penalty on rho/(1 cm^-1), grid spacing 3.125 cm."""
    volume = rho.reshape((n,)*3)
    grad = np.zeros_like(volume)
    value = 0.
    edges = 3*(n-1)*n*n
    for axis in range(3):
        left, right = [slice(None)]*3, [slice(None)]*3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        diff = volume[right]-volume[left]
        if kind == 'quadratic':
            value += .5*np.sum(diff**2)
            deriv = diff
        else:
            root = np.sqrt(diff**2+.1**2)
            value += np.sum(root-.1)
            deriv = diff/root
        grad[right] += deriv
        grad[left] -= deriv
    return value/edges, grad.ravel()/edges


def map_fit(data, coeff, start, *, angular=False, student=False, kind='huber', strength=.3, maxiter=160):
    n = data.grid_size
    history = []
    def fun(rho):
        value, grad = nll(rho, data, coeff, angular=angular, student=student, gradient=True)
        if strength:
            v, g = penalty(rho, n, kind)
            value, grad = value+strength*v, grad+strength*g
        return value, grad
    result = minimize(fun, np.clip(start.ravel(), FLOOR, 20), jac=True, method='L-BFGS-B',
                      bounds=[(FLOOR, 20)]*(n**3),
                      options={'maxiter': maxiter, 'ftol': 1e-9, 'gtol': 1e-7, 'maxls': 40},
                      callback=lambda r: history.append(fun(r)[0]))
    return result.x.reshape((n,)*3), dict(iterations=int(result.nit), success=bool(result.success),
        message=str(result.message), objective=history, final_objective=float(result.fun),
        gradient_max=float(np.max(abs(result.jac))), strength=strength, kind=kind,
        angular=angular, student=student)


MAP_METHODS = {
    'gaussian_ml': dict(strength=0.),
    'gaussian_quadratic': dict(kind='quadratic'),
    'gaussian_huber': dict(),
    'student_huber': dict(student=True),
    'angular_huber': dict(angular=True),
}


def fit_all(paper, local, coeff):
    raw = em(paper, coeff)
    corrected = em(local, coeff)
    mean = em(local, coeff, statistic='mean')
    volumes = dict(raw10=raw[10], pool10=pool(raw[10], paper), pool40=pool(raw[40], paper),
                   local_pool10=pool(corrected[10], local), local_pool40=pool(corrected[40], local),
                   mean_local_pool40=pool(mean[40], local))
    coarse = coarsen(local)
    start = volumes['pool10'][::2, ::2, ::2]
    metas = {}
    for name, args in MAP_METHODS.items():
        rho, metas[name] = map_fit(coarse, coeff, start, **args)
        volumes[name] = rho.repeat(2, 0).repeat(2, 1).repeat(2, 2)
    return volumes, metas


def reconstruct_local_pooled(entry_position_cm, entry_direction,
                             exit_position_cm, exit_direction, *, iterations=40):
    """Opt-in experimental reconstruction returning 64³ 1/X0 and provenance.

    Only detector geometry is accepted. Absolute scale is conditional on the
    fixed external Reyna population spectrum; no calibration is fitted.
    """
    if iterations not in (10, 40):
        raise ValueError('validated iteration choices are 10 or 40')
    tracks = DetectorTracks(entry_position_cm, entry_direction, exit_position_cm, exit_direction)
    _, local, coefficient = prepare_both(tracks)
    if not len(local.selected):
        raise ValueError('no usable tracks through the reconstruction volume')
    rho = em(local, coefficient, steps=iterations)[iterations]
    output = pool(rho, local)
    return output, dict(algorithm='experimental local transverse median update plus exposure pooling',
        iterations=iterations, unit='cm^-1', output_shape=[64, 64, 64],
        inference_grid_size=64, pooling_grid_size=32, pooling_sigma_cm=3.125,
        tracks_input=len(tracks), tracks_with_path=len(local.selected),
        momentum_policy='external Reyna expected inverse momentum squared, beta=1, C=15 MeV',
        prior_source='https://arxiv.org/abs/hep-ph/0604145',
        truth_used=False, per_event_momentum_used=False,
        effective_resolution='spatially pooled; 64³ output does not establish 1.5625 cm resolution',
        caveat='absolute density conditional on spectrum and approximate internal paths')
