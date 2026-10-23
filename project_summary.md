# Project entry point

Updated 2026-10-23. Read this page for navigation; actual source/configuration and
saved run metadata remain authoritative. No absolute checkout path is required.
The earlier detailed September 9 reference is preserved in
[the archive](docs/archive/project_summary_2026-09-09.md), including old numerical
results and mistake history. It is not a current operating manual.

## What is current

This is a Python research framework for common-grid muon tomography: frozen
MuNet, exposure-normalized PoCA, production full/angular likelihood, plus a
headless Geant4 simulator and isolated likelihood research. It is not a trained
8192-track MuNet model or a validated deployed detector product.

Production behavior is intentionally preserved during the structural refactor:

- Geometry: 64³, xyz, −50…50 cm, output `1/X0` in cm⁻¹.
- MuNet: frozen 1024-dosage weights; complete chunks averaged; assumed momentum
  feature 1743.4601815051958 MeV/c. Managed `predict` records inference seed 0;
  legacy inference keeps its earlier random behavior. No truth calibration.
- PoCA: 32³ internal grid; signal divided by path exposure after separate Gaussian
  pooling with σ=6.25 cm; repeated to 64³.
- Likelihood: full covariance by default, median updates, 10 iterations, same
  fixed effective momentum. The improved external-prior candidates remain research.
- The 1743 value is simulation-derived inverse-square effective momentum. It is
  not an arithmetic mean or an independent external spectrum. The fixed Reyna
  prior lives in an explicit experimental module.
- Truth may define evaluation regions and metrics after predictions are fixed.
  It must not calibrate, mask, or rescale normal predictions.

## Find the right file quickly

| Task | Start here |
|---|---|
| Install / standard prediction | [README](README.md), [workflows](docs/workflows.md) |
| Architecture / adding a method | [architecture](docs/architecture.md), `src/muon_tomo/reconstruction/registry.py` |
| Detector arrays / track selection | `src/muon_tomo/data/`, [scientific contracts](docs/scientific_contracts.md) |
| Dataset locations / parentage | `data/catalog/datasets.json`, [storage](docs/storage.md) |
| New run / artifact metadata | `src/muon_tomo/artifacts/`, [storage](docs/storage.md) |
| Current default parameters | `src/muon_tomo/configs/default.yaml`, `src/muon_tomo/config.py` |
| PoCA formula / grid / pooling | `src/muon_tomo/reconstruction/poca.py` |
| Likelihood full/angular / side exits | `src/muon_tomo/reconstruction/likelihood.py`, `common.py` |
| MuNet features / frozen weights | `src/muon_tomo/reconstruction/munet.py`, `_vendor/munet/` |
| Momentum sensitivity / source-energy schema | `experiments/momentum_study/README.md`, `src/muon_tomo/research/momentum_study.py` |
| External spectrum / candidate likelihood | `src/muon_tomo/research/`, [research index](docs/research.md) |
| Cross sections / 3D meshes | `src/muon_tomo/visualization/` (old plotting imports remain compatible) |
| Custom source/object simulation | [simulation guide](docs/custom_simulation.md), `src/muon_tomo/simulation/` |
| Tests / numerical parity | [testing](docs/testing.md), [refactor evidence](docs/refactor_report.md), `tests/`, `benchmarks/` |
| Old results / why earlier numbers differ | `data/catalog/legacy_artifacts.json`, [archive](docs/archive/project_summary_2026-09-09.md) |
| Agent rules | [AGENTS.md](AGENTS.md) |

## Avoid recurring mistakes

Dataset ID plus record index identifies a sample: `record_0` alone does not.
A seed is insufficient without the source/filtering stage and selected row IDs.
8192 stored rows need not be 8192 target-crossing tracks, and neither means one
minute of accepted detector exposure. Keep source spectrum, acceptance, and
generated/accepted/selected counts separate.

Low full-volume MSE can reflect mostly-air support. Report zero baseline, object
bias, false positives, and uncertainty when assessing scientific progress.
The historical 0.1103/0.2987/0.3064 screenshot used different processing and must
not be a refactor's numerical target. Do not recover it with calibration.

A 64³ output does not imply 1.56 cm effective resolution: PoCA pooling width is
6.25 cm. A strict > 0.8 viewer may legitimately show little or no weak material.
The former 12,000 display cap did not indicate model sparsity.
