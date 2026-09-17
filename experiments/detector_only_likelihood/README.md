# Detector-only likelihood experiment — not installed in production

The useful variant is `exposure_pooled_likelihood.py`. It returns **inverse radiation length, 1/X0, in cm^-1** from exactly four arrays: incoming/outgoing positions in cm and incoming/outgoing directions. It is saved outside `src/`; the production likelihood source SHA-256 is verified unchanged in `results/integrity.json`.

Open `report.html` for all nine results and figures. `results/final_comparison.json` contains raw metrics. `results/funny_0/pooled_viewer.html` displays all voxels strictly above 0.8 cm^-1, including interiors, without a count cap.

## Result

For Funny 0, full 64³ MSE falls from **0.718055 to 0.217222 (cm^-1)^2**, comparing the existing full median update and its pooled version under the **same external prior**. Across Funny 0 and main-dataset records 0–7, mean MSE falls from **1.373266 to 0.312692**, a 77.23% reduction. All nine individual MSEs improve, as do reserved-track conditional likelihood scores. Record 6 remains above 1 (1.221332); record 3 only barely beats its empty-volume baseline. See JSON for unrounded values; this is a small simulation panel, not proof of universal accuracy.

Earlier production results with the hidden-simulation-derived 1743.46 MeV/c prior are not the baseline here. That number is not used by either experimental estimator.

## Inputs and external assumption

The public function is:

```python
from exposure_pooled_likelihood import reconstruct_inverse_radiation_length
rho_cm_inv, metadata = reconstruct_inverse_radiation_length(
    entry_position_cm, entry_direction, exit_position_cm, exit_direction
)
```

Run from this directory or add this directory to the Python module search path. Dependencies come from the existing repository environment. Coordinates and measurement-array conventions are those of the existing geometry utilities. Unit directions are computed from the stored vectors; their norms are discarded. No material labels, truth volumes, simulated momenta, extra simulation columns, or object-specific metadata are passed to reconstruction.

The user authorized one fixed external momentum spectrum to identify the absolute density scale. We use D. Reyna (2006), equations 2 and 3 with the published combined-fit coefficients:

- I(p,theta) = cos(theta)^3 I_vertical(p*cos(theta)).
- I_vertical(zeta) = 0.00253 * zeta^-[0.2455 + 1.288 log10(zeta) - 0.2555 log10(zeta)^2 + 0.0209 log10(zeta)^3].
- Integrate 1 < p < 2000/cos(theta) GeV/c, the paper's stated applicability interval, using 64 Gauss-Legendre points in log(p).
- Normalize the conditional distribution at each **measured incoming direction**. Use its expected (15 MeV / p)^2 in the old covariance likelihood. This is an angle-conditioned population expectation, not a measured or freely fitted momentum for an individual muon.
- The same spectrum, integration limits, coefficient 15 MeV, and beta=1 approximation apply to every object.

Source: [Reyna, A Simple Parameterization of the Cosmic-Ray Muon Momentum Spectra at the Surface as a Function of Zenith Angle](https://arxiv.org/pdf/hep-ph/0604145). The likelihood covariance structure follows the existing implementation based on [Schultz et al. (2007)](https://www.hep.phy.cam.ac.uk/~hommels/CosmicConcrete_dir/MuonPapersForBart/Paper_IEEE_16_8_Aug2007_StatImageReco.pdf).

This prior is not verified to match the TFRecord generator or the real detector's accepted spectrum. Muons below 1 GeV/c, attenuation, energy loss, overburden and momentum-dependent detector acceptance are not modeled. Absolute 1/X0 remains conditional on these assumptions. Without an independent spectrum/reference, scattering observations alone do not identify absolute radiation length.

## Useful method

1. Run the existing full-covariance median update for 10 iterations on the canonical 64³ grid, with the external expected scattering coefficient. Retain its side-exit correction, detector-error settings, and 20 cm^-1 upper cap.
2. Trace measured entry–PoCA–exit paths, with the existing straight fallback, and sum each voxel's total path exposure E in cm.
3. Form the weighted density sum E*rho. Sum E*rho and E separately into 32³ coarse cells.
4. Gaussian-pool each coarse sum with **sigma=3.125 cm**, one coarse cell, and divide pooled(E*rho) by pooled(E). Output zero only where pooled exposure is zero.
5. Repeat each coarse voxel into eight identical fine voxels to return 64³. No additional clipping, thresholding, affine fit, normalization to truth, or material assignment is applied.

This is **post-reconstruction spatial regularization**, not a derivation of a new exact MLE. In particular, E*rho is a weighted sum of already fitted likelihood estimates; unlike PoCA's q deposits, it is not an independent measured sufficient statistic. Its benefit is suppressing poorly supported local extremes while sharing information spatially. The tradeoff is bias, smoothing and loss of small structures. The inherited cap acts before pooling; the saved result is the raw output of the specified pipeline, not an unconstrained estimator.

## Validation protocol and failed alternatives

The same nine cases and seed-0 8192-track selections were used throughout. Fits for track-score comparison used 6144 training tracks and 2048 reserved tracks; only nonempty valid paths enter the likelihood. Final maps use all 8192. Full-grid MSE and material/background errors use unsmoothed 64³ labels. No labels are passed into the reconstruction stage. Labels are scored separately after all maps are saved.

Scoring uses a common spectrum-marginalized Student-t covariance likelihood. Reserved paths are reconstructed using the reserved measured PoCA itself, so the likelihood is conditional on estimated paths, not an unconditional prediction of new detector observations. Scores omit common additive constants. The source, settings and cases were recorded before each stage. Smoothing width and spectrum parameters were not adjusted per object or swept against truth.

The initial detector-only `method.py` pooled Student-t likelihood recovers effective scattering power without a physical momentum scale. Its baseline/80-iteration controls and scores remain available. After authorization of an external spectrum, `spectrum_method.py` implemented an explicit marginalization over that spectrum and returned physical 1/X0. It improved localization and reserved-track likelihood, but **worsened absolute MSE in all nine cases** (mean 12.7187 versus 1.3733 for the external-prior baseline). It is saved as an unsuccessful absolute-density experiment, not the recommended variant.

The simpler exposure-pooling control was evaluated afterward and is the useful variant. **Approach selection was informed by this panel's results**, although no per-object parameters, truth-based scales or optimized masks were used. This is therefore not a completely blind benchmark. A separate object set and detector acceptance model are still needed for a stronger generalization claim. The initial marginal-likelihood fits stopped at the fixed 80-iteration budget and are not claimed to be fully converged.

## Reproduction and tests

From the repository root, use the existing `.venv/bin/python` and set `OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/muon-mpl`:

1. `run_study.py` — detector-only input extraction and initial fixed comparison. This parses only the TFRecord x field and strips it to four measured arrays.
2. `evaluate.py` — refit and separately export evaluation labels/localization figures.
3. `baseline_control.py` — initial 80-iteration baseline control (not the external-prior MSE baseline).
4. `run_spectrum.py` — fixed external-spectrum baseline and marginalized candidate.
5. `report_spectrum.py` — evaluate those saved physical outputs.
6. `run_pooled_control.py` — useful fixed exposure-pooling control.
7. `final_report.py` — final figures, viewer, report, and source-integrity check.

All scripts are under `experiments/detector_only_likelihood/`. They write only experimental outputs. Five focused tests pass with `.venv/bin/pytest -q experiments/detector_only_likelihood/test_method.py`: analytic covariance gradients, marginal-spectrum gradients, direction-scale invariance/no momentum field, synthetic scattering-power recovery, and exposure-pooling constant preservation/block identity. Synthetic forward truth is generated only for the numerical recovery test, never supplied to its optimizer.
