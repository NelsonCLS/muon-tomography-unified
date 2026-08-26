# Contributing

Thank you for helping improve Muon Tomography Unified. Changes should preserve
the common physical contract and remain reproducible from metadata.

## Development setup

Create an isolated environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all,dev]'
```

Before submitting a change, run:

```bash
ruff check src tests
ruff format --check src tests
pytest -q
python -m build
python -m twine check dist/*
```

The normal test suite does not require private datasets. Real-data integration
tests are enabled only when their read-only input paths are supplied:

```bash
FUNNY_TFRECORD=/path/to/funny.tfrecord pytest -q tests/test_tfrecord_integration.py
GEANT4_ROOT_SAMPLE=/path/to/sample.root pytest -q tests/test_root_integration.py
```

## Scientific changes

Changes to geometry, units, track selection, momentum policy, calibration, or a
reconstruction algorithm must include focused tests and updated provenance.
Do not silently reuse calibration parameters after changing an estimator or
its configuration. Keep the MuNet weights frozen unless a separate training
project is explicitly introduced.

Code under `src/muon_tomo/_vendor/` is preserved from the upstream MuNet
implementation and is excluded from automatic formatting. Keep any necessary
vendor modification minimal and document it in `NOTICE`.

## Data and generated files

Do not commit TFRecords, ROOT files, reconstructions, figures, local paths,
credentials, or generated presentation workspaces. The repository's ignore
rules cover the normal local locations, but contributors remain responsible
for reviewing staged files before every commit.

## Licensing

Read `LICENSE` and `NOTICE` before redistributing this project. In particular,
the licence for the adapted muonTruck material has not been confirmed, so a
public redistribution requires permission or a compatible upstream licence.
