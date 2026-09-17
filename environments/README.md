# Environment reproduction

`pyproject.toml` is the supported dependency range and optional-feature definition.
Create an environment in the checkout location; do not copy `.venv` between
machines/directories or commit it.

- `constraints-reference.txt`: direct dependency versions observed on the user's
  macOS arm64/Python 3.13 environment on 2026-09-18.
- `reference-macos-arm64-py313.json`: full installed-package metadata snapshot.

These are **reference snapshots, not a dependency solver lock**, wheel archive,
or a tested Linux/CUDA environment. They make the environment behind results
inspectable without claiming unavailable cross-platform reproducibility. To try
a matching environment on a compatible platform:

```sh
python -m pip install -c environments/constraints-reference.txt -e '.[all]'
```

Otherwise install supported ranges from `pyproject.toml` and run the relevant tests.
Record the actual dependency versions with scientific runs. Geant4 physics data,
CMake, compiler, and executable hashes need separate simulator provenance.
Do not upgrade an existing scientific environment during ordinary plotting work.
