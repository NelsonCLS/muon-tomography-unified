# Configuration assets

- `default.yaml` defines the shared mandatory 64³ geometry and method defaults.
- `funny-v1.calibration.json` is the retained calibration generated from the
  8,192-track-per-record angular-likelihood workflow documented in the main
  README. It fits funny record 0 and contains validation metrics for record 1.

Calibration files are estimator-specific. Do not apply this angular-likelihood
calibration to a full-covariance likelihood result; refit with the same mode and
settings used for reconstruction.
