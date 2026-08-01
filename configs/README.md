# Configuration assets

- `default.yaml` defines the mandatory shared 64³ geometry and validated method
  defaults. Classical reconstruction assigns every event the same fixed
  1743.4601815051958 MeV inverse-square effective momentum prior and never uses
  per-muon momentum magnitudes. Full-covariance likelihood uses physical
  upper-to-lower ordering, grid-face observations, PoCA-kinked paths, ten
  iterations, and the robust median update.
- `funny-v4-robust-mlsd.calibration.json` is generated from funny record 0 and
  includes held-out record-1 metrics. Likelihood fits covered-region means;
  Mu-Net and PoCA fit eroded material medians. All maps target cm^-1 and are
  bounded to the material truth range represented by funny.
- `funny-v3-mean-momentum.calibration.json` and
  `funny-v2-spatial.calibration.json` are historical angular-likelihood
  calibrations retained for provenance. They are not compatible with the
  corrected full-covariance estimator.

Calibration files are estimator-, geometry-, update-, iteration-, and
momentum-policy-specific. Never apply an angular-likelihood calibration to a
full-covariance result, or a per-track calibration to a fixed-prior result.
Refit whenever any of those settings changes.
