# Scientific contracts

## Geometry, units, and direction conventions

Canonical prediction/truth arrays have shape 64³ and order `volume[x,y,z]` over
[-50,50] cm on each axis. Cell edges are 1.5625 cm apart. Positions inside Python
are centimetres; angles radians; momentum parameter names use MeV/c convention.
Density is inverse radiation length `1/X0` in cm⁻¹; its squared error has cm⁻² units.

Legacy TFRecords use `x` with at least 12 columns (zero-based indices):

- `0:3`: lower/outgoing position in mm.
- `3:6`: simulated outgoing momentum vector in MeV/c; its norm is outgoing momentum.
- `6:9`: upper/incoming position in mm.
- `9:12`: incoming unit direction, **not** a momentum vector.
- Optional column `12`: initial/source kinetic energy T in MeV in the validated
  13-column generator schema. Convert to source momentum using
  `p = sqrt(T * (T + 2*m_mu))`, with `m_mu = 105.6583755 MeV/c²` and c=1 in this
  energy-momentum expression. This is not measured upper-plane momentum.

Older 12-column records do not contain initial energy; never invent it from a
unit-direction norm. Verify generator provenance for other imported schemas.
`y` is a 64³ material-label array.
Legacy lower/upper endpoint ordering is not physical incoming/outgoing ordering.
Use the shared physical-intersection helpers; do not infer flight order from a
field named `entry` alone. Detector-only loaders normalize directions and exclude
simulation momentum and material labels from solver inputs.

## Momentum policies

Scattering variance depends approximately on inverse-square momentum. Thus
`p_eff = <1/p²>^(-1/2)` is not the arithmetic mean momentum. Historical production
uses 1743.4601815051958 MeV/c, estimated from Funny 0's simulated outgoing population.
Freezing that scalar removes per-event momentum use but does not make its source
independent of simulation. MuNet receives the same assumed scalar as a feature.

The research Reyna prior is fixed externally and conditioned on incoming zenith.
It is not a measured per-track momentum. Its cutoff/beta/open-sky assumptions live
in `research/detector_only_likelihood/spectrum_method.py:PRIOR`. Never silently
substitute it for production defaults during a structural refactor.

The current simulator uses a kinetic-energy power law `T^-2.7` over 1–200 GeV,
independent of its configured angular distribution. Kinetic energy is not momentum.
Legacy datasets were generated differently. Source spectrum, detector acceptance,
and reconstruction momentum prior must be recorded as distinct objects.

## No truth leakage

Normal inference must not inspect truth, labels, material-region statistics,
per-event simulation momentum, or fitted output calibration. Truth is permitted
for metrics, comparison figures, and requested region statistics **after** a
prediction and its settings are fixed. Selecting a method/width using Funny 0
truth is development-set selection; do not call the resulting benchmark blind.

Legacy `calibrate` commands/files remain for reproducibility of historical work.
They are not part of `predict` and are not standard reconstruction. Preserve old
outputs without presenting them as current raw results.

## PoCA and likelihood interpretation

Production PoCA accumulates scattering signal at localized PoCA points and path
exposure from all eligible traversing paths. It pools numerator and denominator
separately before division on 32³, then repeats cells to 64³. This is not equivalent
to smoothing a field of per-voxel ratios. σ=6.25 cm is a regularization width,
not measured detector spatial resolution.

Production full likelihood uses both projected angle and displacement. Angular
mode uses only projected angles. The current candidate's external spectrum,
coordinate basis, iteration changes and exposure pooling are experimental choices,
not defaults or a guarantee of exact marginal maximum likelihood.

## Comparisons and figures

Match dataset identity, record index, selected original row IDs/order, track cuts,
geometry, configuration, weights, source version, and momentum policy. Equal stored
counts in original and target-filtered datasets do not imply equal detector dose.

Full-grid MSE includes air and uncovered cells, before display thresholds. Other
legacy metrics use valid-mask support; label their scope. A zero predictor is an
essential reference for sparse objects. Region mean/standard deviation is spatial
variation, not a confidence interval. Plot densities in cm⁻¹ and MSE in cm⁻².

The standard viewer uses a strict > 0.8 cm⁻¹ cut, retains interior voxels, and applies
no 12,000-voxel cap. An empty weak-material view need not indicate missing data.
