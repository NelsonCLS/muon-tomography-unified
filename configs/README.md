# Configuration assets

- `default.yaml` defines the shared mandatory 64³ geometry and method defaults.
- `funny-v3-mean-momentum.calibration.json` is the current calibration for the
  fixed 3000 MeV population-average policy. PoCA and likelihood use that same
  scalar for every event and never use per-muon momentum magnitudes. It fits
  funny record 0 and records held-out record-1 metrics.
- `funny-v2-spatial.calibration.json` is the retained calibration generated
  from the 8,192-track-per-record angular-likelihood workflow documented in the
  main README. It fits funny record 0 on the corrected canonical ±50 cm target
  cube and contains validation metrics for record 1.
  The likelihood entry uses the native cm^-1 identity map; likelihood affine
  fits are intentionally disabled because sparse material medians can invert or
  strongly amplify an already physical estimator.

Calibration files are estimator- and momentum-policy-specific. Do not apply an
angular-likelihood calibration to a full-covariance result, or a per-track
calibration to a fixed-average result; refit with the same mode and settings
used for reconstruction.
