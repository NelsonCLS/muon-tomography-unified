# Testing and numerical validation

## Fast checks

```sh
python -m pip install -e '.[test,plot,dev]'
ruff check src tests
ruff format --check src tests
MPLBACKEND=Agg python -m pytest -q
```

The configured suite includes portable production tests and the two synthetic
experimental test modules plus cache-provenance regressions. It requires no private
dataset. Tests needing TensorFlow, a real input path, or Geant4 explicitly skip when
unavailable; a green portable run is not a full real-data/model validation claim.

Existing tests exercise geometry/path integration, axis conventions, PoCA exposure
and pooling, fixed-momentum invariance, result schemas, frozen weights, numerical
likelihood gradients, synthetic density recovery, plotting selection, and simulation
object/record conversion. New storage/config/workflow tests protect the refactored
boundaries. Historical numerical research modules and `_vendor` are excluded from
mass lint/reformatting during the parity-preserving migration.

## Integration tiers

```sh
python -m pip install -e '.[all]'
FUNNY_TFRECORD=/path/to/funny.tfrecord python -m pytest -q tests/test_tfrecord_integration.py tests/test_munet_integration.py
GEANT4_ROOT_SAMPLE=/path/to/hits.root python -m pytest -q tests/test_root_integration.py
MUON_TOMO_SIMULATOR=/path/to/muon_tomo_sim python -m pytest -q tests/test_simulation.py
```

Use a ROOT file with the expected hits tree, not a TFRecord or unrelated ROOT
format. Geant4 tests require its native data/environment. These commands do not
select a new default source spectrum or train MuNet.

CI retains portable tests on supported Python versions, lint/format, and package
build validation. A separate TensorFlow job exercises synthetic TFRecord round trips;
a wheel is installed and invoked outside the checkout to catch source-tree imports
and missing package resources. The portable suite also runs from the extracted
source distribution, catching omitted shared fixtures and research runners. The
dataset-free detector example is exercised on each portable platform. CI configuration is not evidence that remote jobs
have run successfully; report actual local/remote results separately.

## Refactor versus scientific improvement

Before moving code, preserve a source snapshot and exact selected track inputs.
After each stage compare complete density arrays, occupancy, masks, geometry,
selected IDs/order, and metadata with the previous implementation. State numerical
tolerances and platform. Equal MSE alone is insufficient evidence of parity.

Truth and hidden-momentum perturbation tests should demonstrate that detector-only
inference does not depend on them. Such tests do not prove a simulation-derived
fixed prior is external; provenance is a separate requirement.

A scientific benchmark must freeze candidate settings before holdout evaluation,
include zero prediction/background/material metrics, and retain the full protocol.
See [benchmarks](../benchmarks/README.md). Do not rerun all large datasets for a small
stylistic change or regenerate historical output to match a preferred score.

## Packaging

```sh
python -m build
python -m twine check dist/*
```

Build and install the wheel in a fresh environment; invoke the CLI from outside the
repository, then check packaged default config, weights and simulator source.
The local reference environment snapshot is not a complete multi-platform lock.

## Repeatable public-command verification

```sh
python benchmarks/verify_workflows.py --output-dir /tmp/muon-workflows-new
```

The output directory must be new. This exercises the public CLI on a small synthetic
dataset and retains every command/log plus `workflow_report.json`. Add `--dataset
/path/to/funny.tfrecord` for a real sample and `--simulator /path/to/muon_tomo_sim`
for native generation. It is software workflow verification, not a claim of
scientific reconstruction accuracy.

The 2026-09-18 refactor's retained evidence is under
`outputs/refactor-validation/` (parity arrays/report, managed runs, and CLI logs).
The packaging build/outside-checkout wheel smoke is summarized in
[the packaging record](../benchmarks/package_smoke_2026-09-18.json). Inspect the
actual evidence and final refactor report for the tested snapshot and scope.
