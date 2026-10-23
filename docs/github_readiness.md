# GitHub preparation audit — 23 October 2026

Scope: prepare the working tree for public checkout and source/wheel installation.
Base commit: `8495a6ba2a029a783e9e362d4b7c847c5e249787`. This audit includes local
preparation changes and the previously uncommitted momentum-study source/tests.
It does not publish a release or push to GitHub.

## Changes

- README now states what is bundled, which datasets/results remain local, how to
  install, and how to produce a plot/viewer without private data or TensorFlow.
- Added a deterministic detector-only example. It is a software fixture, with no
  truth volume, momentum field, material claim or accuracy score.
- Corrected the validated TFRecord schema: outgoing momentum is in columns 3:6;
  incoming columns 9:12 are unit directions; optional column 12 is initial kinetic
  energy. This documentation change does not change the loader's numerical output.
- Fixed source distributions omitting shared test fixtures; included benchmark
  runners, the existing momentum study, detector example and metadata catalogues.
  Added package project URLs and corrected the build-backend minimum requirement.
- CI now exercises the dataset-free demo and tests the extracted source archive
  against an installed wheel, as well as its existing platform/adapter checks.
- Archived August's validation record, whose calibrated scores must not be mistaken
  for current raw results. Updated research entry points and package imports.
- Recorded the maintainer's confirmation of muonTruck redistribution permission
  in LICENSE/NOTICE, preserving attribution and component-specific terms.
- Replaced five tracked reading PDFs with publisher/arXiv links. Local PDF copies
  remain on disk and are ignored; previously published Git history is unchanged.

## Checks actually run

On this macOS ARM64 computer with Python 3.13:

- Full suite: **152 passed, zero skipped**, including real Funny TFRecord, frozen
  MuNet, Geant4 ROOT input and native Geant4 simulation checks. Ten warnings are
  upstream TensorFlow/gast AST deprecations.
- Fresh temporary environment, built wheel, extracted source distribution, no
  TensorFlow or private datasets: **144 passed, eight expected skips** (optional
  TensorFlow adapters, real input paths and native Geant4).
- **18 public command workflows passed**: synthetic conversion, inspection,
  reconstruction by all three methods, evaluation, plots/viewers, stability,
  catalog registration/listing, managed prediction and run listing.
- Dataset-free detector demo generated its cross-section PNG and HTML viewer in
  the fresh environment using only the classical/plot installation.
- Wheel and source distribution build; Twine metadata validation; archive resource
  checks; Ruff lint/format (74 files); Git whitespace checks passed.
- Inspected active documentation links and candidate tracked file inventory.
  A limited common credential-pattern scan found no matches. This is not an
  exhaustive security audit or a scan of all Git history.

The clean-source test initially found 24 setup errors because `tests/conftest.py`
was missing from the archive. The manifest was fixed and the clean-source test
rerun successfully. This is the reason for the additional CI packaging check.

Local detailed evidence is preserved under `outputs/github-readiness/20261023/`:
`audit.json`, suite/install/build logs, per-command workflow logs and the clean
installation's demo artifacts. Those generated files are intentionally not shipped.
Remote GitHub Actions have not been run for these unpushed changes. Platform and
Python-version coverage in CI is configured, not claimed as locally executed.

## Preserved scientific behavior and limits

No production reconstruction algorithm, default momentum prior, model weight,
source dataset or historical prediction was changed. The only production Python
edit corrects the TFRecord missing-dependency message to name the `tfrecord` extra.
The pre-existing momentum research changes remain opt-in.

Software readiness does not establish accurate material reconstruction on real
detector data. The fixed momentum prior remains simulation-derived; the sparse
simulation benchmarks and historical calibrated reports retain their limitations.
See [scientific contracts](scientific_contracts.md), [testing](testing.md), and
[LICENSE](../LICENSE) / [NOTICE](../NOTICE) for those distinct concerns.
