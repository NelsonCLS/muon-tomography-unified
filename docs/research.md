# Research and historical results

## Production versus experiments

Production numerical implementations remain under `src/muon_tomo/reconstruction/`.
The likelihood candidate was stopped and preserved; reorganizing files does not
promote it to the default. Method and prior must be read from artifact metadata,
not a directory name such as `current-best`.

Reusable experimental helpers now live in the installed package:

- `muon_tomo.research.detector_only_likelihood.method`: detector-array type,
  covariance observations, Gaussian/Student scattering-power experiments.
- `...spectrum_method`: fixed external Reyna prior and 1/X0 experiments.
- `...exposure_pooled_likelihood`: post-fit exposure-weighted pooling.
- `...all_methods_external`: experimental MuNet/PoCA external-prior adapters.
- `muon_tomo.research.likelihood_physics_study.core`: coordinate/EM/MAP ablations.

These modules keep their computational definitions unchanged. The migration
manifest `experiments/helper_migration.json` records original/moved hashes and
function/class AST equivalence. They are intentionally excluded from automatic
formatting/lint during this parity-preserving move, like vendored historical code.
Further style/typing cleanup should be a separate verified change.

The old helper paths in `experiments/` are compatibility imports. Runners import
the package directly; they no longer modify `sys.path` or import a sibling study's
bare module name. Run from the repository root with an installed package:

```sh
python experiments/likelihood_physics_study/run.py --help
python -m pytest -q experiments/detector_only_likelihood/test_method.py experiments/likelihood_physics_study/test_core.py
```

The physics-study runner has `--results-dir` and verifies source/config/input/row
identities and output hashes before accepting a cached fit. Configuration identity also
records the NumPy/SciPy versions. Old fits without these
identities are refused as current caches; preserve them and choose a new directory.
Other historical study runners retain their fixed protocols/output locations.
They are not aliases for the new managed prediction command. Inspect their README
and input requirements before running; their result caches are excluded from Git.

## Historical sources preserved

`experiments/legacy_recipes/index.json` identifies byte-for-byte source snapshots
formerly stored under `outputs/` and generated dataset directories. Original files
and results stay untouched. The copies are **reference only** until their old
relative/absolute path assumptions have been ported deliberately.

`data/catalog/legacy_artifacts.json` inventories existing artifacts. The original
September 9 narrative, scientific mistake chain and result caveats remain in
`docs/archive/project_summary_2026-09-09.md`. Old calibrated/surface-only displays
must not be mixed with raw outputs or newer all-voxel viewers.

## Adding an experiment

Create `experiments/<descriptive_name>/README.md` plus a parameterized runner.
State the question, fixed protocol, data split, allowed inference inputs, expected
outputs, and stop criteria. Reuse installed helpers and normal storage APIs.
Write new results to a unique run, not into the source directory. Freeze candidate
settings before holdout evaluation. Keep source, metrics, raw arrays, and figures
linked by the same run identity; move only reusable solvers into `research/`.

## Momentum sensitivity study

The opt-in [momentum study](../experiments/momentum_study/README.md) compares
frozen MuNet and PoCA across independent simulation priors and explicitly labelled
truth-based oracles. Numerical helpers live in `muon_tomo.research.momentum_study`.
The 2026-10-20 completed results are intended for `outputs/momentum-study/20261020-v3`;
check its `manifest.json` completion field before using them. No production policy
or trained weight is changed. Initial/source momentum is derived from stored
kinetic energy, never the norm of the incoming unit-direction vector.
