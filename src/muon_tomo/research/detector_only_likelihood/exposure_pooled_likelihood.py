"""Experimental fixed spatial regularization of existing likelihood estimates.

This is a post-reconstruction exposure-weighted pool, not a new exact MLE.
Input: only measured positions/unit directions. Output: 1/X0 [cm^-1].
Scale comes from the same fixed external Reyna prior as spectrum_method.py.
No fitted calibration, clipping of pooled values, or truth inputs.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from muon_tomo.research.detector_only_likelihood.method import DetectorTracks,prepare,SETTINGS
from muon_tomo.research.detector_only_likelihood.spectrum_method import baseline_rho

def pool_with_exposure(rho,tracks):
    data=prepare(tracks,64)
    exposure=np.asarray(data.matrices[0].sum(axis=0)).reshape((64,)*3)
    # Coarsen sufficient weighted sums and exposure before dividing. Integrals
    # of path exposure and exposure*rho are preserved in each coarse cell.
    def coarse(v):return v.reshape(32,2,32,2,32,2).sum(axis=(1,3,5))
    sigma=SETTINGS['pooling_sigma_cm']/3.125
    numerator=gaussian_filter(coarse(exposure*rho),sigma,mode='constant')
    denominator=gaussian_filter(coarse(exposure),sigma,mode='constant')
    result=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
    return result.repeat(2,0).repeat(2,1).repeat(2,2)

def reconstruct_inverse_radiation_length(entry_position_cm,entry_direction,exit_position_cm,exit_direction):
    tracks=DetectorTracks(entry_position_cm,entry_direction,exit_position_cm,exit_direction)
    rho,meta=baseline_rho(tracks)
    result=pool_with_exposure(rho,tracks)
    meta.update(algorithm='existing full median likelihood + fixed exposure-weighted spatial pooling',
                pooling_sigma_cm=SETTINGS['pooling_sigma_cm'],grid_size=32,
                truth_used=False,measured_momentum_used=False,
                uncertainty='not estimated; post-pooling is heuristic regularization')
    return result,meta
