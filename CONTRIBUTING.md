# Contributing

Read [AGENTS.md](AGENTS.md), [architecture](docs/architecture.md), and
[scientific contracts](docs/scientific_contracts.md) before changing behavior.
Use [testing](docs/testing.md) for setup and the relevant validation tier.

Keep structural changes separate from estimator/physics changes. A refactor must
preserve selected inputs and numerical results, not just a headline MSE. New
features belong behind explicit configuration and method/status metadata.

The current defaults deliberately remain frozen: no truth-fitted calibration or
per-object tuning in normal prediction. An external momentum prior and an assumed
historical scalar are different scientific policies. Truth is evaluated only after
predictions are fixed. Preserve model weights and source data.

Use installed `muon_tomo` imports. New shared solvers belong in the package, explicit
experimental variants in `muon_tomo.research`, and study runners in `experiments/`.
Do not place new executable source inside output/data folders or add `sys.path`
mutations. Old snapshots in `experiments/legacy_recipes/` are reference-only.

Use registered dataset identities and fresh run directories. Do not commit large
TFRecords/ROOT files, arrays, figures, generated presentations, environment folders,
or credentials. Review small catalogues for machine-local paths before sharing.

Update the appropriate short documentation page and provenance manifest whenever
behavior or storage changes; keep the root project summary a navigation index.
Archive useful old reports without promoting them to current validation claims.

Read `LICENSE` and `NOTICE` before redistribution. The maintainer has confirmed
permission to redistribute the muonTruck adaptations; this does not assign the
Apache licence to that upstream code. Preserve component attribution and consult
the maintainer about permission terms for further redistribution.
