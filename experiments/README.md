# Experiments

Production methods live under `src/muon_tomo/reconstruction/`. This directory holds
explicit research protocols/runners and historical sources, not runtime defaults.

- `detector_only_likelihood/`: fixed external-prior and pooling comparisons.
- `likelihood_physics_study/`: covariance coordinates, EM/MAP and regularization ablations.
- `momentum_study/`: fixed/reference momentum policies and labelled simulation oracles.
- `legacy_recipes/`: byte-for-byte copies of source once stored next to outputs.
- `helper_migration.json`: provenance of numerical helpers moved into the package.

Reusable helpers are importable as `muon_tomo.research.*`; do not add `sys.path`
mutations or import another experiment as a bare module. Saved `results/` are local,
ignored artifacts. Read [the research guide](../docs/research.md) before continuing
an old runner: its cache/protocol is not a fresh held-out benchmark.
