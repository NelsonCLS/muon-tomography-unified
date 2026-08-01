# Configuration assets

- `default.yaml` defines the shared mandatory 64³ geometry and method defaults.
- `funny-v2-spatial.calibration.json` is the retained calibration generated
  from the 8,192-track-per-record angular-likelihood workflow documented in the
  main README. It fits funny record 0 on the corrected canonical ±50 cm target
  cube and contains validation metrics for record 1.
  The likelihood entry uses the native cm^-1 identity map; likelihood affine
  fits are intentionally disabled because sparse material medians can invert or
  strongly amplify an already physical estimator.

Calibration files are estimator-specific. Do not apply this angular-likelihood
calibration to a full-covariance likelihood result; refit with the same mode and
settings used for reconstruction.
