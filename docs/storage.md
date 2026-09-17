# Dataset and result storage

Source code, small catalogues, and scientific records belong in version control.
Large datasets, intermediate arrays, generated figures, and local environments do not.
The refactor registers existing files **in place**; no raw data is moved or deleted.

```text
data/catalog/datasets.json          small dataset identity/location catalogue
data/catalog/legacy_artifacts.json  discovery index of old outputs/studies
data/raw/                          optional location for future original inputs
data/derived/                      optional location for future transformed inputs
data/cache/                        disposable, reproducible intermediates
runs/<unique-run-id>/               newly managed predictions and results
outputs/                           existing historical artifacts, left in place
experiments/<study>/                source, protocol and documentation
experiments/<study>/results/        existing local research outputs, left in place
```

The three data payload directories are conventions for new work, not a forced
migration or an automatic conversion. Existing `dataset/` and
`dataset_target_crossing/` stay valid. Use catalogue IDs rather than inventing
machine-specific paths inside source code.

## Dataset catalogue

```sh
muon-tomo dataset list
muon-tomo dataset register /path/to/new.tfrecord --id my-simulation --kind simulated
muon-tomo dataset register /path/to/filtered.tfrecord --id my-filtered --kind derived --parent-id my-simulation
muon-tomo predict --input legacy-funny --record-index 0
```

Entries include ID, path, kind, optional parent, file size/mtime, checksum status,
and metadata. Registration does not scan TFRecord contents or hash the entire
file. `filesystem_metadata_only` must not be described as verified content identity.
Original-to-derived row maps remain beside filtered files; the selection manifest
uses them where available and states when such mapping cannot be established.

`--catalog PATH` selects a custom catalogue. Dataset catalogue API root / the
`MUON_TOMO_DATA_ROOT` environment variable controls resolution of relative entries.
`MUON_TOMO_PROJECT_ROOT` locates the checkout and `MUON_TOMO_CATALOG` selects a
catalogue for installed/outside-checkout use.
Absolute external paths may be registered but are machine-local; review them before
sharing a catalogue. Duplicate IDs and missing parent IDs fail rather than replace
existing entries. Dataset kind labels are provenance descriptions, not scientific
validation status.

## Detector-only interchange

The versioned detector NPZ schema is `muon_tomo_detector_tracks`, version 1.
It contains four physical incoming/outgoing position/direction arrays, event IDs,
and JSON metadata. Positions are in cm; directions are unit vectors. There are
no momentum or material-label arrays; unexpected fields are rejected.

`muon_tomo.io.save_detector_npz` and `load_detector_npz` provide the Python APIs.
The writer refuses replacement unless explicitly requested. `predict --input
/path/to/detector.npz --no-truth` uses the same managed workflow as simulated inputs.
This format describes measured tracks; a truth volume belongs in a separate
explicit evaluation artifact.

## New run layout

```text
runs/<run-id>/
  run.json                    version, lifecycle, input/method/source identity
  config.resolved.json         complete resolved reconstruction settings
  selection.json              selected event/order/filter provenance
  selected_event_ids.npy      selected row IDs (paired with selection.json)
  predictions/
    munet.npz
    poca.npz
    likelihood.npz
  evaluation/
    truth.npz                 only when truth is requested and available
    metrics.json              truth availability and post-prediction metrics
  figures/
    cross_section.png
    viewer.html
    section.json
    viewer.json
```

A single-method run contains only that method's NPZ. All methods share the same
selected tracks; including MuNet can trim to complete dosage chunks, recorded in
the manifest. `--no-truth` skips truth loading entirely. Predictions are saved
before truth is requested for evaluation.

Run directories are unique and refuse collisions. Status is `running`,
`completed`, `failed`, or `interrupted`; incomplete artifacts remain available
for diagnosis. The API prevents writes to a finished run. This is workflow
immutability, not filesystem permissions or tamper-proof storage.

Input path/stat information, detector selection fingerprint, effective prior,
source hashes, dependency versions, and MuNet weight hash are recorded. Full large
input hashes are not silently recomputed. Run lists read small manifests only.
Use `--runs-dir /external/disk/runs` to store new artifacts outside this checkout.

## Historical results and caches

Historical directories are indexed, not retroactively declared valid. Preserve
original names, files, and scientific caveats. Small report/source indices are
versionable; large arrays/HTML remain local. Empty directories and guessed source
hashes are not evidence of complete experiments.

The physics study now checks its protocol, code/configuration, selected measured
inputs, and output hashes before reusing a completed fit. Legacy fits with no cache
identity remain readable historical artifacts but are refused as current cached
fits. Use a fresh `--results-dir`; do not manufacture provenance for old files.
