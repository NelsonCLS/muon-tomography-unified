# Working in this repository

Read `project_summary.md` first, then only the linked document for the task.
This file applies to the whole repository. User instructions override it.

## Scientific contract

- Output is inverse radiation length `1/X0` in `cm^-1`; MSE is `cm^-2`.
- Canonical arrays are `volume[x,y,z]`, 64³, bounds −50…50 cm on all axes.
- Stored TFRecord endpoint order is legacy lower-then-upper, not physical time
  order. Use shared geometry adapters; never repair it by swapping arrays ad hoc.
- Reconstruction uses measured positions/directions and an explicit momentum
  policy. Do not pass truth/material labels or event-level simulated momentum
  to a detector-only method. Truth is allowed for evaluation after prediction.
- No truth-fitted calibration, truth masking, material-specific adjustment, or
  object-specific hyperparameter tuning in normal prediction.
- Historical fixed 1743.4601815 MeV/c is inverse-square effective momentum from
  Funny 0 simulation, not arithmetic mean or an external measured spectrum.
- Production algorithms, experimental candidates, and historical reports are
  distinct. Do not promote a candidate or change defaults as part of a refactor.
- Preserve source datasets, model weights, and old outputs. New runs get new IDs.

## Navigation and changes

- `src/muon_tomo/reconstruction/`: production methods and geometry helpers.
- `src/muon_tomo/workflows/`: orchestration; `data/` and `artifacts/`: storage APIs.
- `src/muon_tomo/research/`: opt-in experimental numerical helpers.
- `experiments/`: runners/protocols; `experiments/legacy_recipes/`: frozen references.
- `data/catalog/`: dataset and legacy-artifact inventory; large inputs stay local.
- `docs/architecture.md`, `docs/scientific_contracts.md`: boundaries and invariants.
- `docs/workflows.md`, `docs/testing.md`: commands and relevant validation tiers.

Use installed package imports, not `sys.path` manipulation or sibling experiment
imports. Keep CLI handlers thin. Add a new method through the registry. Keep
inference independent of truth, plotting, and filesystem output. Use pathlib and
configured paths; do not add a developer's absolute home directory to active code.

## Verification and reporting

- Run focused tests and the portable suite; state skipped integration checks.
- For refactors, preserve selected row IDs/configuration and compare full arrays,
  not only MSE. Do not touch production numerical code without parity evidence.
- Standard prediction means raw arrays, full-grid metrics where truth exists,
  four-panel cross sections, and a 3D viewer. Viewer cuts do not affect metrics.
- Default viewer intent: every voxel strictly greater than 0.8, interior included.
- Prefer saved arrays/manifests for plots; do not rescan or rehash 50 GB datasets
  merely to find a file. Catalogue stat information is not a content checksum.
- Update the relevant short document or manifest when behavior changes. Keep
  historical narrative in `docs/archive/`, not this entry point.
