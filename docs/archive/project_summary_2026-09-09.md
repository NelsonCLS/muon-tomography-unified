# Project reference: unified muon tomography

Last inspected: **2026-09-09**, on the user's Apple Silicon Mac, in
`/Users/longsangchan/Muon/muon-tomography-unified-main`.

This is an operational reference for future agents working in this repository. It records the current implementation, experiment provenance, user decisions, reproducible commands, and known traps. It is not a claim that every historical output is scientifically valid. The repository has evolved substantially beyond its original README and August validation report.

**Read sections 1–3 first. Then use the index to read only the section relevant to the next task. Do not rescan all datasets, outputs, or this entire document on every turn.** Verify the specific source/configuration and artifact metadata before changing behavior or making an exact numerical comparison. This document is a dated snapshot, not an executable specification.

## Index and economical navigation

- [1. Essential state and standing user instructions](#1-essential-state-and-standing-user-instructions)
- [2. Three different meanings of current prediction](#2-three-different-meanings-of-current-prediction)
- [3. Task-to-file quick lookup](#3-task-to-file-quick-lookup)
- [4. Repository layout, dependencies, and local runtime](#4-repository-layout-dependencies-and-local-runtime)
- [5. Dataset inventory and TFRecord schema](#5-dataset-inventory-and-tfrecord-schema)
- [6. Spatial contract, materials, and units](#6-spatial-contract-materials-and-units)
- [7. Core data structures and saved-file schemas](#7-core-data-structures-and-saved-file-schemas)
- [8. Track selection and provenance](#8-track-selection-and-provenance)
- [9. Shared path geometry and the side-exit fix](#9-shared-path-geometry-and-the-side-exit-fix)
- [10. MuNet implementation and momentum features](#10-munet-implementation-and-momentum-features)
- [11. Production PoCA: exposure-normalized pooling](#11-production-poca-exposure-normalized-pooling)
- [12. Production likelihood: full and angular](#12-production-likelihood-full-and-angular)
- [13. Detector-only likelihood experiments](#13-detector-only-likelihood-experiments)
- [14. External spectrum across all three methods](#14-external-spectrum-across-all-three-methods)
- [15. Configuration and command-line workflow](#15-configuration-and-command-line-workflow)
- [16. Plotting, 3D viewers, and interpretation](#16-plotting-3d-viewers-and-interpretation)
- [17. Evaluation and material-region statistics](#17-evaluation-and-material-region-statistics)
- [18. Saved results and experiment history](#18-saved-results-and-experiment-history)
- [19. Calibration, leakage, and unresolved scientific limitations](#19-calibration-leakage-and-unresolved-scientific-limitations)
- [20. Tests, packaging, and validation status](#20-tests-packaging-and-validation-status)
- [21. External references, licensing, and preservation](#21-external-references-licensing-and-preservation)
- [22. Recipes and future update checklist](#22-recipes-and-future-update-checklist)

Quick navigation from the repository root:

```sh
rg -n '^## ' project_summary.md
rg -n 'side-exit|1743|current-best|fingerprint|pool_with_exposure' project_summary.md
rg -n '^def |^class ' src/muon_tomo/reconstruction/common.py
```

Prefer narrow reads of the file named in section 3. Most routine prediction/plotting requests can use existing saved arrays and do not require opening the original 50 GiB TFRecord. For current source behavior, inspect `src/`; for the experimental likelihood, inspect its explicitly named module, not whichever file happens to contain a similarly named function.

## 1. Essential state and standing user instructions

### What this project does

The project puts three muon-tomography reconstruction approaches on a shared spatial/output contract:

1. Frozen, pretrained **MuNet** inference using a vendored TensorFlow/Keras model.
2. **PoCA** reconstruction, now using track-path exposure normalization and spatial pooling.
3. **Likelihood** reconstruction, with a production full angular-and-displacement model and an older angular-only option.

It reads simulated TFRecord data and Geant4 ROOT hits; exports canonical truth and reconstruction arrays; computes metrics; produces four-panel cross sections and rotatable voxel-cube HTML viewers; and contains calibration, detector-error stability, and later research workflows. There is no continuously running backend required to view the saved HTML. This is primarily a Python scientific workflow, not a deployed web service.

### User decisions that must survive future turns

- **“Apply prediction” includes both a truth/MuNet/PoCA/likelihood cross-sectional plot and a 3D viewer.** Merely producing NPZ files does not fulfill this recurring request.
- Standard recent comparisons use **8,192 tracks, selection seed 0**, unless the user specifies otherwise. Preserve the same selected events across methods when comparing them.
- The predicted physical quantity is **`1/X0` in `cm^-1`**. Do not substitute an arbitrary scattering-power score while labeling it `1/X0`.
- Keep output raw in the sense of **no truth-fitted rescaling, affine calibration, material-based fitting, truth masking, or per-object optimization**. Raw here still includes each method's explicit model constraints and regularization; e.g. current likelihood caps and MuNet nonnegative clipping exist in source.
- The detector-only research method may use **only entering/exiting positions and directions**. No per-track simulation momentum magnitude, material identity, truth volume, or hidden tensor column may enter reconstruction.
- The user explicitly authorized **one fixed external cosmic-muon spectrum**, independent of the simulation and shared across objects. Direction-conditioned population information is allowed; measured/simulated individual momentum is not.
- The user later asked to apply that external-prior idea to all methods. This was done in **experimental adapters**, not by replacing the production CLI defaults.
- **Do not zero MuNet's momentum feature arbitrarily.** A typical assumed momentum was requested earlier; the current production scalar and its provenance are documented below. The later external-prior adapter is a separate inference experiment, not retraining.
- The viewer should display **every voxel with prediction strictly greater than `0.8 cm^-1`, including interior voxels**, with **no 12,000-voxel display limit**. Source CLI defaults still need explicit overrides.
- Do not change production likelihood merely because an experimental method has better scores. The user requested that improved likelihood code be saved separately, without integrating it yet. Later requests used those results; there has not been an explicit instruction to replace production likelihood.
- Preserve datasets, model weights, and historical research artifacts unless deletion is explicitly requested. The unusable cosmic lead-box dataset and its outputs were already requested deleted; do not recreate them from stale paths.
- Report full-volume MSE before display cuts. Truth can be used for **evaluation after prediction is fixed**, and to define requested material-region statistics, without feeding it back into reconstruction.

### Critical provenance caveat

The frozen value **`1743.4601815051958 MeV/c` was derived from the simulated momentum population in Funny record 0**, as stated in the current default YAML. It is an inverse-square effective momentum, not an independent external measurement. No event-level momentum is used under the production `mean` policy, but that does **not** make this scalar independent of simulation truth. Do not repeat older claims that all frozen-1743 results are strictly detector-only or use no truth information whatsoever.

Similarly, PoCA's current pooling width was selected after examining Funny truth MSE. The latest “best per method” bundle was selected using Funny performance. No final affine calibration was applied to those arrays, but the development/selection process was not blind. Distinguish **inference inputs**, **prior provenance**, and **model/hyperparameter selection** in every scientific claim.

## 2. Three different meanings of current prediction

### A. Production source and CLI defaults

- MuNet: frozen 1,024-track model, scalar momentum feature `1743.4601815051958 MeV/c`, chunk averaging.
- PoCA: 32³ coarse cells, exposure-normalized weighted scattering signal, Gaussian pooling sigma **6.25 cm**, repeat each coarse prediction into its eight 64³ children; same scalar momentum under `mean` policy.
- Likelihood: **full** angular-and-displacement mode, **median** update, **10 iterations**, same scalar momentum, side-exit geometry fix, **no exposure-pooling postprocess**.
- Calibration occurs only if explicitly passed. It should not be passed for current user requests.

Active default configuration: `src/muon_tomo/configs/default.yaml`. The root `configs/default.yaml` is a duplicate checked by tests, not the file implicitly loaded by the CLI.

### B. Experimental detector-only / external-spectrum methods

Located under `experiments/detector_only_likelihood/`:

- Recommended experimental likelihood entry point:
  **`exposure_pooled_likelihood.reconstruct_inverse_radiation_length`**.
- Fixed external Reyna spectrum, direction-dependent effective scattering coefficient.
- Existing full median likelihood followed by **exposure-weighted 32³ pooling, sigma 3.125 cm**, expanded to 64³.
- External-spectrum MuNet and PoCA adapters live in `all_methods_external.py`.
- These accept a four-array detector representation and do not need material labels or per-track momentum magnitude.
- They are not registered as replacement production CLI methods.

### C. Most recent saved Funny 0 “current best” bundle

Directory: **`outputs/funny-current-best/`**.

This is a **mixed-prior, best-per-method saved comparison**, not a fresh invocation of one unified configuration:

- MuNet: fixed 1743, full-grid MSE **0.08930092771316271**.
- PoCA: fixed 1743, current exposure-normalized pooling, MSE **0.11800038120305795**.
- Likelihood: external-spectrum experimental exposure pooling, MSE **0.21722170958022824**.

All MSE values have units `cm^-2`; all predictions have units `cm^-1`. Track count is 8,192, seed 0.

Files include `record_0_truth.npz`, `record_0_munet.npz`, `record_0_poca.npz`, `record_0_likelihood.npz`, `cross_section.png`, `viewer.html`, `viewer_metadata.json`, `metrics.json`, and `material_region_statistics.json`.

Sources:

- MuNet and PoCA arrays: `outputs/detector-only-comparison/funny-fixed-1743/record_0_*.npz`.
- Likelihood array: `experiments/detector_only_likelihood/results/all_methods_external/funny_0/predictions.npz`, key `likelihood`.

**Export limitation:** the current-best likelihood wrapper has placeholder zero occupancy, an all-true validity mask, and no standard selected-track fingerprint. Its density array is useful; its coverage fields are not real coverage measurements. Do not claim standard `validate` passes or interpret this occupancy physically. There is no persistent dedicated current-best generation script; the package was assembled from saved arrays.

The `metrics.json` flag `truth_used_for_scoring_only: true` describes the immediate scoring step, not the complete provenance of the frozen prior and best-method selection.

## 3. Task-to-file quick lookup

- **Run a standard production prediction:** `src/muon_tomo/cli.py`, active YAML, section 15 recipe.
- **Use current experimental likelihood:** `experiments/detector_only_likelihood/exposure_pooled_likelihood.py`; inputs/geometry in `method.py`; external prior/baseline in `spectrum_method.py`.
- **Use external spectrum for all three methods:** `all_methods_external.py`; selection/driver in `run_all_methods_external.py`.
- **Understand selected 8,192 tracks:** `TrackTable.subset` / `take_first` in `types.py`, `command_reconstruct` and `_track_fingerprint` in `cli.py`.
- **Verify TFRecord columns / decode one record:** `src/muon_tomo/io/tfrecord.py`; detector-only decoder `load_x` is in the experimental `run_all_methods_external.py`.
- **Verify ROOT hit fitting:** `src/muon_tomo/io/root.py`.
- **Check axes, bounds, ray lengths:** `geometry.py`, `reconstruction/common.py`, `tests/test_spatial_contract.py`.
- **Check side-exit handling:** `physical_grid_intersections`, `prepare_poca_paths`, `downstream_path_gaps` in `common.py`; full covariance in `likelihood.py`.
- **Understand current PoCA:** `reconstruction/poca.py`; history in `outputs/poca-exposure-study/README.md` and `experiment.py`.
- **Understand MuNet input and weights:** `reconstruction/munet.py`, `_vendor/munet/models/agg_3d.py`, `_vendor/munet/layers/agg_3d.py`, `_vendor/munet/layers/poca.py`.
- **Compute full-grid MSE:** `compare.py`; read section 17 before reusing its other metrics.
- **Compute silver/lead/uranium statistics:** saved current-best truth labels plus each density array; section 17. Do not use the default eroded calibration statistics for this request.
- **Change plotting threshold, density cubes, cap:** `plotting3d.py`, `command_plot3d` / argument defaults in `cli.py`.
- **Replot cross section without reconstruction:** `plotting.py`, `cli plot` using NPZ files.
- **Investigate old wrong scores:** `outputs/likelihood-audit/REPORT.md`, `outputs/mismatch-check/`, sections 18–19.
- **Inspect latest Funny result:** `outputs/funny-current-best/metrics.json` and `material_region_statistics.json`.
- **Inspect matched external vs fixed-prior likelihood:** experimental `results/all_methods_external/matched_likelihood.json`.
- **Inspect pooled likelihood across nine examples:** experimental `results/final_comparison.json`, `final_summary.json`, and `report.html`.
- **Check historical affine calibration:** `calibration.py`, `configs/*.calibration.json`; do not invoke it by default.
- **Check detector-error simulation:** `stability.py`, `command_stability` in `cli.py`.
- **Check dependencies / build contents:** `pyproject.toml`, `MANIFEST.in`, `tests/test_packaging.py`, `.github/workflows/`.

## 4. Repository layout, dependencies, and local runtime

### Layout

```text
muon-tomography-unified-main/
  project_summary.md                  this reference
  README.md                          original usage; contains stale calibration workflows
  VALIDATION.md                      historical August validation, not current scientific status
  CONTRIBUTING.md / NOTICE / LICENSE / LICENSES/ / CITATION.cff
  pyproject.toml / MANIFEST.in        packaging and optional dependencies
  .github/workflows/                  CI, lint/tests, package build checks
  .venv/                             working local Python environment
  configs/                           root default copy and historical calibration JSONs
  src/muon_tomo/
    cli.py / constants.py / geometry.py / types.py
    compare.py / calibration.py / postprocess.py / stability.py
    plotting.py / plotting3d.py
    configs/default.yaml             CLI's implicit default configuration
    io/{tfrecord,root}.py
    reconstruction/{common,munet,poca,likelihood}.py
    _vendor/munet/{models,layers,weights}/
  src/muon_tomography_unified.egg-info/ generated editable-install metadata
  tests/                             portable and opt-in real-input tests
  third_party/                       provenance and original-import reference material
  dataset/                           large local inputs, not ordinary source
  outputs/                           historical and current predictions / audits / plots
  experiments/detector_only_likelihood/ isolated research implementations and saved results
```

At inspection this is a downloaded archive with **no `.git` directory**. Do not claim a clean working tree, current branch, commit, or ability to revert using Git. No `AGENTS.md` was found during this inspection. Recheck if the user later initializes Git or adds guidance.

`.gitignore` excludes `.venv`, caches, `outputs/`, calibration output directories, ROOT/TFRecord files, logs, and environment files. **`experiments/` is not globally ignored** and contains potentially large generated arrays/HTML. If Git is introduced, review tracking deliberately rather than adding everything.

### Working environment, verified 2026-09-09

- Python **3.13.3**, architecture **arm64**.
- NumPy 2.5.3; SciPy 1.18.1; TensorFlow 2.21.0.
- Matplotlib 3.11.1; Plotly 6.9.0.
- uproot 5.7.6; awkward 2.13.0; PyYAML 6.0.3; h5py 3.14.0.
- pytest 9.1.1.
- `ruff`, `build`, and `twine` were **not installed** in this environment at inspection.

Use `.venv/bin/python` and `.venv/bin/muon-tomo`. Do not rely on whatever `python` resolves to outside this environment.

```sh
.venv/bin/python -m muon_tomo.cli --help
.venv/bin/muon-tomo --help
```

There is no package `__main__.py`: **`.venv/bin/python -m muon_tomo` is not the entry point**.

For noninteractive plots/tests, use:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/muon-mpl .venv/bin/python -m muon_tomo.cli --help
```

The macOS GUI Matplotlib backend previously aborted with status 134. `Agg` avoids that. `MPLCONFIGDIR` avoids attempts to populate a read-only default font cache. Optional `OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` reduce thread variability/oversubscription in small scientific checks. TensorFlow CPU-frequency, XLA, or retracing messages do not by themselves prove inference failed.

### Installation / packaging contract

`pyproject.toml` declares `muon-tomography-unified` version 0.1.0, Python `>=3.11,<3.14`, setuptools/wheel build backend, and a `muon-tomo` console script.

- Core: NumPy, SciPy, h5py, PyYAML, uproot, awkward.
- `[munet]`: TensorFlow.
- `[plot]`: Matplotlib and Plotly.
- `[test]`: pytest and coverage tooling.
- `[dev]`: build, ruff, twine.
- `[all]`: inference, plots, and tests; **not all developer/build tools**.

The local environment is already working. Do not reinstall or upgrade dependencies during ordinary prediction or documentation work without a concrete need.

## 5. Dataset inventory and TFRecord schema

### Local inputs

- **`dataset/voxels_prediction_funny.tfrecord`**: 5,372,706 bytes, approximately 5.1 MiB, GZIP. Two records. Funny 0 has **44,240 tracks**; Funny 1 has **44,323 tracks**. Their material object is the same Funny shape, with different simulated tracks; they are not two independent object designs.
- **`dataset/voxels_prediction.tfrecord`**: 53,987,661,706 bytes, approximately 50 GiB, GZIP. First record has **44,660 tracks**, `x` shape `(44660,13)`. Main records 0–7 have been used in local studies. **The full record count was not established in this inspection.** Do not describe sampling among those eight as uniform sampling from the entire file.
- **`dataset/voxels_prediction_test.tfrecord`**: approximately 4 GiB; not fully rescanned for this summary.
- **`dataset/voxels_prediction_val.tfrecord`**: approximately 4 GiB; not fully rescanned for this summary.
- **`dataset/custom_object.tfrecord`**: 2,503,594 bytes, approximately 2.4 MiB. Historical first-record inspection: `(44294,12)` tracks, air/water/lead labels.
- **`cosmic_leadBox_200000.tfrecord` and `outputs/cosmic-leadbox/`**: deleted by user request. Old browser URLs and inspection JSONs are stale references, not evidence the files should exist.

Useful existing inspection artifacts:

- `outputs/local-check/dataset-inspection.json`: historical first-record inspections, including the now-deleted cosmic file; not a current filesystem inventory.
- `outputs/funny-after-side-exit/record_0_inspection.json`: Funny schema, record information, label counts. Contains simulation momentum statistics; those are inspection/evaluation data, not detector-only reconstruction inputs.

The number in a filename can refer to generated muons, not detected/retained tracks. The deleted cosmic file contained only 11,057 detector tracks and all-air truth; it did not contain 200,000 usable reconstructed tracks. Likewise, main record 7's **44,272 tracks** must not be confused with Funny 0's 44,240.

### Serialized schema

Reader: `src/muon_tomo/io/tfrecord.py`.

- Stream is `tf.data.TFRecordDataset(..., compression_type="GZIP")`.
- Each `tf.train.Example` has byte-string features `x` and `y`.
- `x` is a serialized **float64 tensor**, shape `(N, >=12)`.
- `y` is a serialized **int32 tensor**, shape `(64,64,64)`, material indices.
- `x[:,0:3]`: first stored position, in **mm**, normally the lower/outgoing endpoint.
- `x[:,3:6]`: first stored direction/momentum vector. Its **norm encodes simulation momentum magnitude** in the existing loader; direction is normalized for geometry.
- `x[:,6:9]`: second stored position, in **mm**, normally the upper/incoming endpoint.
- `x[:,9:12]`: second stored direction/momentum vector.
- Some files have a thirteenth input column; current standard reconstruction ignores columns after index 11. Historical generator context identifies the extra scalar with energy information. Do not use it as an allowed detector feature.
- Positions become centimeters by division by 10.

The standard loader produces both `TrackTable` and `TruthVolume`. It derives `TrackTable.momentum_mev` from a vector norm even when downstream `mean` methods ignore that field. Thus the adapter is not itself a minimal detector-only boundary.

The strict experimental decoder requests only feature `x`, extracts the first twelve columns, normalizes vectors, and discards their magnitudes. It does not decode `y` until the separate evaluation phase. `DetectorTracks` then exposes only four position/direction arrays.

### Avoid expensive or invalid input navigation

- `inspect_tfrecord()` scans **all records**, despite common-input CLI flags such as `--record-index` being accepted by the parser. Do not run it casually on the 50 GiB input.
- `.skip(k)` on this GZIP stream must traverse/decompress preceding records; it is not indexed random access.
- Seeking raw TFRecord headers in the compressed file is invalid. An earlier attempt incorrectly inferred a tiny record count this way.
- Use cached `detector_inputs.npz`, saved truth, and known case indices for repeat experiments. For a genuinely new random full-dataset selection, establish a valid record index/count or sampling strategy rather than pretending the known subset is the whole population.

### ROOT adapter

`io/root.py` reads the `hits` tree through uproot. Required branches include `eventNumber`, `hitPositionX_truth/Y_truth/Z_truth`, and `hitMomentumX_truth/Y_truth/Z_truth`.

It groups by event, splits above/below-target hits, needs at least two hits on each side, fits lines with SVD, orients them consistently, extrapolates to detector planes at `z=±100 cm`, and returns the lower endpoint first to match the MuNet storage convention. Input hit positions are converted mm→cm; momentum GeV→MeV. This existing adapter uses simulation truth branches and momentum; **it is not a ready-made real-detector-only input adapter**.

## 6. Spatial contract, materials, and units

Authoritative files: `constants.py`, `geometry.py`, `tests/test_geometry.py`, `tests/test_spatial_contract.py`, `tests/test_units.py`.

- Array convention: **`volume[x, y, z]`**, not z-y-x.
- Shape: **64×64×64 = 262,144 voxels**.
- Physical target bounds: **[-50,50) cm on each axis**.
- Fine voxel width: **1.5625 cm**.
- Coarse 32³ voxel width: **3.125 cm**.
- Centers: `lower + (index + 0.5) * pitch`.
- Coordinate-to-index mapping: floor of `(point-lower)/pitch`.
- Membership is lower-inclusive, upper-exclusive. A point exactly on +50 is not an interior voxel point.
- Detector reference planes are at `z=±100 cm`; target observation planes for likelihood geometry are at `z=±50 cm`.
- MuNet normalized positions: `position_cm / 100 + 0.5`. The target maps to `[0,1]`; detector points may lie outside that interval.

`GridGeometry` is a frozen dataclass with positional argument order **lower bounds, upper bounds, shape, axis order**. It enforces the canonical 64³ xyz contract. **Do not instantiate it with a 32³ shape to implement coarse PoCA.** Current coarse methods aggregate canonical fine paths or use separate sparse experimental matrices.

`trace_segment` clips a segment against the target box, intersects voxel planes, removes duplicate parameters, assigns segments via midpoints, and returns ordered xyz indices and physical lengths in centimeters. Common reconstruction helpers frequently convert those indices to **flattened** voxel IDs; do not assume every path is already an `(N,3)` array.

### Material mapping

The material index list in `constants.py` is:

```text
0 air           1 benzene       2 methanol       3 water
4 magnesium     5 concrete      6 gypsum         7 calcium
8 sulfur        9 silicon      10 aluminium     11 caesium
12 manganese   13 iron         14 iodine        15 nickel
16 molybdenum  17 silver       18 polonium      19 lead
20 gold        21 uranium
```

`labels_to_inverse_radiation_length()` validates integer labels/range/shape, returns float32 canonical density, and maps air to zero.

Funny truth regions:

- Air: 226,384 voxels.
- Silver, label 17: **30,384 voxels**, approximately **1.1705686 cm^-1**.
- Lead, label 19: **2,592 voxels**, approximately **1.7817897 cm^-1**.
- Uranium, label 21: **2,784 voxels**, approximately **3.1583333 cm^-1**.
- Total nonair: **35,760 voxels**.

### Physical quantity conversions

Let `rho = 1/X0` in `cm^-1`. The code uses Highland constant **C=15 MeV** and reference momentum **p_ref=3000 MeV/c**.

```text
reference scattering density [mrad²/cm] = 10^6 (C/p_ref)² rho = 25 rho
reference scattering density [rad²/cm]  =      (C/p_ref)² rho = 0.000025 rho
```

These are explicit convention/unit conversions, not permission to fit arbitrary scale factors to truth. Momentum factors must use consistent MeV/GeV units. Angle covariance is in radians, not degrees or milliradians unless explicitly converted.

**Density and its standard deviation:** `cm^-1`.
**Density MSE:** `cm^-2`, equivalently `(cm^-1)^2`.
An MSE caption displaying `cm^-2` is correct and should not be changed to `cm^-1`.

## 7. Core data structures and saved-file schemas

### `TrackTable` (`types.py`)

Fields include `entry_position_cm`, `entry_direction`, `exit_position_cm`, `exit_direction`, `momentum_mev`, optional `event_id`, `source`, and `metadata`.

The names are misleading if interpreted as flight order: **the TFRecord-compatible table stores lower/outgoing first and upper/incoming second**. Use physical-order helpers for transport calculations. Directions are normalized on construction; finite shapes and positive momenta are validated.

- `subset(limit, seed)`: seeded sampling without replacement, then sorts selected indices to preserve original order. `None` or a limit at least the total returns all tracks; zero yields empty tracks.
- `take_first(count)`: deterministic truncation of the current ordering.
- Both operations construct/normalize a new table. This can cause tiny direction-bit changes even when event selection is unchanged.
- Projected scattering angles use `atan2(direction_x_or_y, abs(direction_z))` with alignment of opposed directions.

### `TruthVolume`

Holds material labels, density, geometry, metadata. It **does not have `.save()`**. Standard helpers are `cli._save_truth()` and `cli._load_truth()`.

Canonical truth NPZ keys:

```text
material_labels
inverse_radiation_length_cm_inv
metadata_json
```

Metadata contains schema version 1, geometry, primary quantity and unit, and available source/record information.

### `ReconstructionResult`

Holds method, canonical density, raw estimator, occupancy, validity mask, geometry, metadata, optional uncertainty. Density/raw are stored as float32, occupancy as int64, mask as bool; all primary arrays must have canonical shape and finite density.

Use `.save(path)` and `ReconstructionResult.load(path)`. Saved arrays include density plus derived reciprocal radiation length, reference scattering density, raw estimator, occupancy, valid mask, JSON metadata, and optional uncertainty. Zero density leads to infinite reciprocal radiation length by design; do not confuse that derived infinity with invalid primary output.

Exact standard result NPZ keys:

```text
inverse_radiation_length_cm_inv
radiation_length_cm
scattering_density_ref_mrad2_per_cm
raw_estimator
occupancy_count
valid_mask
metadata_json
uncertainty                         optional
```

Read NPZs with `allow_pickle=False`. Metadata is JSON stored as a scalar string, typically decoded with `json.loads(str(data['metadata_json']))`.

**Experimental NPZs often have different schemas.** Examples: `predictions.npz` has keys `munet`, `poca`, `likelihood`; `inverse_radiation_length.npz` contains candidate/baseline maps; early studies contain scattering-power maps. They are not automatically loadable through `ReconstructionResult.load`. Inspect `data.files` and metadata before converting; do not relabel an early scattering-power array as `1/X0`.

## 8. Track selection and provenance

The standard comparison uses `np.random.default_rng(0).choice(N, 8192, replace=False)`, sorted ascending. This is a random subset, **not the first 8,192 events**, highest-angle selection, or truth-based selection.

When `--method all` includes MuNet, `command_reconstruct` trims the selected table to complete MuNet chunks **before running any method**. With dosage 1024:

- `--limit 8192`: 8 complete chunks, all methods receive 8,192.
- `--limit 8000`: only 7 complete chunks, all methods receive **7,168**.
- `--max-chunks` may reduce this further.
- A PoCA-only or likelihood-only run has no MuNet chunk requirement; omitting `--limit` uses all tracks loaded from that record.

Selecting all input rows is also different from the number of rays that intersect the target or possess an in-box PoCA. Inspect path/valid-event metadata before interpreting acceptance.

### Existing fingerprint mechanism

`cli._track_fingerprint` hashes event IDs and the four geometric arrays **plus `momentum_mev`**, using explicit little-endian int64/float64 C-order bytes. It is not a detector-only fingerprint.

Known historical Funny 0 seed-0 8192 fingerprints:

- CLI all-method path, including another `take_first` normalization:
  `df466b5ec169d2dbf83ff03afafb0f46b7e4b2f432e81857b0f21f28444658d5`
- A PoCA-only subset path:
  `b7beabcdbde207181c2aefccc85f268fc2b7c99395a6536ced560ef607d245cb`

A hash difference can reflect normalization-bit differences, not necessarily different sampled IDs. **Do not dismiss mismatches automatically.** Compare source, record, event IDs, and numerical geometry explicitly. A future strict detector-only fingerprint should include IDs and the four allowed arrays, not hidden momentum; that is not the current general helper.

A cached main-record-0 CLI fingerprint is:
`c9ad07d9b6e1b4d22ef3e0c50b56cfb31af6dc55d80e2e32595a43fc8efdc196`.

### Random-sample naming traps

`record_0` alone does not identify a dataset. `outputs/random-prediction/record-0-exposure/` uses **main** record 0, not Funny 0. Earlier “another random sample” choices drew from a small set of known record indices, not a verified full-file population. Record 7's all-track experiments refer to the main dataset.

Future manifests should record input filename, record index, selected IDs/count/seed, geometry, momentum policy and provenance, algorithm parameters, weights/source hashes, calibration status, and output schema. Never combine files solely because their filename begins `record_0`.

## 9. Shared path geometry and the side-exit fix

Authoritative implementation: `src/muon_tomo/reconstruction/common.py`.

- `reconstruction_momenta`: `mean` fills a supplied scalar; it does not estimate a mean from the current object's tracks. `per-track` uses stored magnitudes and is forbidden in the strict detector-only workflow.
- `closest_approach`: midpoint of the shortest connector between two infinite lines. Near-parallel lines return no PoCA (`abs(denominator)<1e-12`).
- `physical_track_order`: reorders detector endpoints into actual upper-to-lower flight order and orients directions along flight.
- `physical_grid_intersections`: extrapolates the measured lines to the **fixed top and bottom target planes**, even when an intersection's x/y lies outside the target face.
- `prepare_poca_paths`: traces entry→PoCA→exit if PoCA lies in the box; otherwise uses a chord. Trace clipping finds the actual in-box segments. Returns flat voxel IDs, physical lengths, occupancy, and a kinked-path count.
- `downstream_path_gaps`: accounts for distance between the actual box exit and the fixed downstream observation plane.
- `prepare_paths` is the older straight-path helper; do not interchange it with PoCA-broken transport without checking which observations the caller uses.

### Why the side-exit correction matters

Full likelihood uses both angular deflection and transverse displacement. Scattering in an upstream voxel continues to affect displacement while the muon travels downstream, including the portion after leaving through a side face before reaching the fixed observation plane.

Previously the downstream lever arm could stop at the last in-box segment, effectively using the side exit as the observation plane while displacement was measured relative to the fixed bottom plane. The fix restores the downstream gap in the covariance's `T` terms. The path's material integral still stops at the actual box boundary.

This was a real geometry bug and has been fixed in current shared/full-likelihood code. It does **not** imply every remaining poor score is another geometry bug, or that all model assumptions have been validated. Keep observation definitions, path clipping, and covariance lever arms consistent in future variants.

Top-entry/bottom-exit-only cuts were explored earlier. They are a selection change, not a substitute for correct side-exit covariance on the general accepted sample.

## 10. MuNet implementation and momentum features

Wrapper: `src/muon_tomo/reconstruction/munet.py`.
Architecture: `src/muon_tomo/_vendor/munet/models/agg_3d.py` and its layers.

### Current inference path

- Dosage is either 512 or 1024; default is 1024 with corresponding packaged weights.
- Feature rows have **14 columns**:
  1. first position (3), normalized `cm/100+0.5`;
  2. first unit direction (3);
  3. second position (3), same normalization;
  4. second unit direction (3);
  5. standardized assumed momentum (**column index 12, the thirteenth feature**);
  6. norm of the direction difference (**index 13, the fourteenth feature**).
- Momentum feature standardization constants: mean **5585.2666 MeV**, standard deviation **13839.263 MeV**.
- Production fills the feature with a fixed `momentum_estimate_mev`, currently 1743.4601815051958; it does not consume `TrackTable.momentum_mev` for this feature.
- A zero **standardized** feature represents the training normalization mean, not zero physical momentum. Do not conflate these operations.
- Full dosage chunks are passed with `training=False`, then their predicted volumes are averaged arithmetically.
- Primary output is clipped nonnegative; `raw_estimator` preserves the unclipped model result. Current output is not affine-calibrated unless requested through the old calibration path.
- Occupancy comes from track geometry; the MuNet validity mask covers the full grid, not just occupied voxels.

Architecture construction uses `Agg3D`, resolution 64, point size 1, aggregation threshold `1e-8`, downsampling stages `[1,2,3,4,5]` with filters `[8,16,32,64,128]`, and upsampling `[4,3,2,1]` with `[64,32,16,8]`. Vendored PoCA/aggregation layers map geometric features into a voxel tensor before the convolutional encoder-decoder. ConvNeXt/residual/support layers are part of the frozen model implementation, not separate reconstruction methods.

There is no complete local training pipeline or completed retraining experiment in the current workflow. **Do not assert that the original training used the exact current fixed-momentum policy; that provenance was not independently established here.** The experimental external-prior feature is explicitly an inference-time distribution change with unchanged weights.

### Weight locations and hashes

- `src/muon_tomo/_vendor/munet/weights/model_512.weights.h5`
  SHA256 `6fc8764ddb899c969329f5c44f649da8872fc9dd669a8a5108485e7954ce67ec`
- `src/muon_tomo/_vendor/munet/weights/model_1024.weights.h5`
  SHA256 `d8383322563156697d93dacf1ffaa5caf0588a6c69096462908d9a6aba6e28d5`

Treat weights as immutable input artifacts. Preserve checksums in future provenance. Small floating-point differences in reruns can occur; saved fixed-1743 Funny MSE values include 0.0893678407 and 0.0893009277. This is not justification for forcing results back to a historical 0.1103 using calibration.

## 11. Production PoCA: exposure-normalized pooling

Implementation: `src/muon_tomo/reconstruction/poca.py`.
Development record: `outputs/poca-exposure-study/`.

### Current estimator

For each accepted ray, compute the two projected angular changes and a scalar scattering contribution:

```text
q_i = 0.5 (delta_theta_x_i² + delta_theta_y_i²) (p_i / C)²
```

Under current production `mean` policy every `p_i` is the same supplied scalar. The external experimental adapter substitutes the direction-conditioned effective population momentum.

Accumulate:

```text
S_j = sum of q_i for rays whose PoCA is assigned to coarse cell j
E_j = sum of physical path lengths through cell j from ALL traversing rays
rho_j = GaussianPool(S)_j / GaussianPool(E)_j
```

Use zero where pooled exposure is zero. Pool numerator and denominator **separately before dividing**. The default coarse grid is 32³; Gaussian sigma is 6.25 cm, i.e. two coarse cells. Gaussian boundary mode is constant/zero. Each coarse prediction is repeated into the corresponding 2×2×2 fine voxel block.

All valid path exposure contributes to `E`, including rays with no in-box PoCA. A ray can provide evidence of weak scattering even when its PoCA is elsewhere or undefined. This is the main reason the density scale is much better than the earlier conditional event average.

### Interpretation

Old PoCA averaged large scattering events **conditional on being assigned a PoCA in the cell**. It omitted the many rays that crossed the cell without depositing a PoCA there. This preferentially selected stronger deflections, especially with sparse events and fine voxels.

The new denominator measures illumination/exposure. For a fixed exposure, more PoCA events or larger deflections increase the estimate. For fixed accumulated signal, more traversing path length reduces it. It therefore combines event-count evidence with angular scattering strength, although it is **not a counts-only method**.

The dimensional heuristic follows angular variance proportional to `(C/p)² L/X0`; `q` is dimensionless and `E` is in cm, so the ratio is in `cm^-1`. PoCA still assigns distributed multiple scattering to an approximate point, so this is not an unbiased exact local inverse-radiation-length measurement.

### Implementation details and pitfalls

- Supported coarse sizes are 32 and 64 while public geometry stays 64³.
- Fine path segments are grouped by integer index division; **all fine lengths within a coarse cell must be summed**.
- A previous 32³ implementation lost within-cell length contributions, severely inflating density; current production aggregation fixes that.
- `occupancy` reports PoCA event counts, repeated into children, **not path-exposure centimeters**.
- Default `minimum_coverage=1` uses the pooled-exposure validity behavior. Values above one also restrict by unsmoothed PoCA counts; inspect code before assuming this is a minimum-path-count cut.
- Uncertainty is not estimated.
- Constructor defaults and YAML defaults can differ, especially scalar momentum. Use the CLI configuration or explicit arguments for reproducible comparisons.

### Historical Funny 0 MSE progression

These are different algorithms/settings, not interchangeable regression targets:

- Old 64³ conditional PoCA: approximately **20.1327**.
- Initially broken 32³ conditional implementation: approximately **221.24**.
- Corrected length aggregation, still conditional 32³ estimator: approximately **110.29393**.
- Exposure normalization with no spatial pool: approximately **35.295**.
- Exposure pooling sigma 3.125 cm: approximately **0.347759**.
- Current sigma 6.25 cm: **0.11800038**.
- A tested sigma 9.375 cm: approximately **0.111478**.

The selected 6.25 cm width balanced smoothing/performance after examining Funny scores. It is not a hyperparameter selected independently of that benchmark. Funny 1 also improved (about 0.471779 under the study), but shares the same object shape.

## 12. Production likelihood: full and angular

Implementation: `src/muon_tomo/reconstruction/likelihood.py`.
Detailed audit: `outputs/likelihood-audit/REPORT.md` and supporting scripts.

### Full mode — current production default

Each track contributes angular deflection and transverse displacement in two projections. The covariance accumulates scattering along the approximated in-box path, with material density per voxel and momentum scaling.

For a segment length `L` and remaining downstream distance `T`, the geometric moments are:

```text
W_angle,angle = L
W_angle,disp  = L²/2 + L T
W_disp,disp   = L³/3 + L² T + L T²
```

The model sums those matrices over traversed voxels, multiplied by reference scattering density and relative momentum squared, and adds angular/position resolution covariance. The side-exit gap contributes to `T`.

The update computes per-track/per-voxel contributions involving quadratic residual terms minus covariance traces. Conceptually its sufficient-score form is `2 lambda + relative_p² lambda² (quadratic - trace)`, followed by the relevant reduction and factor of one half.

Current defaults:

- 10 iterations.
- Initial `rho = 0.00004 cm^-1`.
- Reference momentum 3000 MeV/c and Highland constant 15 MeV.
- Fixed scalar effective momentum 1743.4601815051958 under `mean` policy.
- Update statistic **median** where enough track contributions are present; mean is another option.
- Angular resolution `1e-6 rad`; position resolution `1e-3 cm`.
- CLI epsilon `1e-9` (class default differs).
- Maximum density `20 cm^-1`.
- No spatial pooling in production likelihood.

Median aggregation is a robust heuristic; it should not be described as the exact Gaussian maximum-likelihood EM update. The iteration count and cap are part of the estimator. A better score after early stopping or pooling does not establish statistical optimality.

### Angular mode — older alternative

Uses projected angular changes and straight-path lengths, without displacement covariance. Its update resembles:

```text
rho_j <- rho_j * [sum_i L_ij q_i / predicted_variance_i] / [sum_i L_ij]
```

The existing implementation is a legacy multiplicative rule. The audit found it is **not algebraically identical to the intended Gaussian angular-variance likelihood update**. A one-voxel diagnostic produced approximately 0.0091818 versus analytic 0.0055. Do not call it interchangeable with the original `recon` angular algorithm merely because both use angles.

Its code uses a separate hardcoded cap of 1000 and does not honor the full-mode median/cap behavior in the same way, even when metadata includes those settings. Always inspect the selected branch rather than inferring behavior from shared metadata.

### Relation to the paper

The full model implements angular/displacement covariance ideas from the cited statistical reconstruction paper, but includes practical deviations: approximate PoCA paths, robust median reduction, frozen/population momentum approximations, numerical caps, and limited iterations. The geometry fix addresses one concrete defect; it does not remove model mismatch from energy loss, unknown momentum, detector acceptance, or non-Gaussian scattering.

The original source comparison and synthetic algebra/path checks are saved in `outputs/likelihood-audit/`. These contain diagnostic uses of simulated momentum; those runs are not detector-only deployable results.

## 13. Detector-only likelihood experiments

Directory: **`experiments/detector_only_likelihood/`**. This intentionally sits outside production `src/`.

### Recommended saved method

Import from **`exposure_pooled_likelihood.py`**, not from the similarly named public function in `spectrum_method.py`:

```python
reconstruct_inverse_radiation_length(
    entry_position_cm, entry_direction, exit_position_cm, exit_direction
) -> (density_64_xyz_cm_inv, metadata)
```

The endpoint order follows the repository-compatible four-array convention. Geometry helpers resolve physical flight order. `DetectorTracks` normalizes direction vectors and has **no momentum, labels, or truth fields**.

The recommended method:

1. Uses `spectrum_method.baseline_rho()` to reconstruct the existing full median likelihood with a fixed external direction-conditioned scattering coefficient.
2. Uses corrected PoCA paths to compute fine-grid path exposure `E`.
3. Sums `E*rho` and `E` into 32³ cells.
4. Gaussian-pools each sum with sigma **3.125 cm** (one coarse cell).
5. Divides pooled weighted density by pooled exposure.
6. Repeats each coarse cell into eight fine voxels.

```text
rho_out = expand_2x2x2( G[coarse(E*rho)] / G[coarse(E)] )
```

This is **post-reconstruction exposure-weighted regularization**, not a newly derived exact MLE. It uses the same ratio-of-pooled-sums principle as the improved PoCA, but the numerator comes from an initial likelihood estimate rather than deposited angular signal. There is no additional truth-fitted scaling or pooled-output clipping; the initial baseline itself retains its numerical density cap.

### File roles and chronology

- `method.py`: four-array `DetectorTracks`; sparse observation/covariance construction; early pooled Gaussian/Student objective and gradients; coarse positive field optimization; early baseline. Early outputs represent scattering power `kappa` in `rad²/cm`, **not automatically rho**. Later helper additions permit MuNet-style truncation and projected-angle access.
- `spectrum_method.py`: fixed Reyna prior, quadrature, direction-conditioned coefficients, full median external-prior baseline, and the **failed spectrum-marginalized optimization candidate**. Its `reconstruct_inverse_radiation_length()` is not the recommended pooled method.
- `exposure_pooled_likelihood.py`: recommended experimental wrapper and reusable `pool_with_exposure(rho, tracks)`.
- `run_study.py`: initial predeclared cases Funny 0 and main 0–7; 8,192 sampled tracks split into 6,144 fit / 2,048 held-out; Gaussian/Student/uniform controls.
- `evaluate.py`: full-sample refits, then separately decoded truth for early localization metrics and plots.
- `baseline_control.py`: early 80-iteration baseline control in scattering-power space; not the later matched fixed-prior rho benchmark.
- `run_spectrum.py`: external-prior baseline and spectrum-marginal candidate, held-out scoring and full-sample fits.
- `report_spectrum.py`: later truth-based evaluation and plots of that candidate.
- `run_pooled_control.py`: simple exposure-pooling control after the more elaborate candidate failed density MSE; saves predictions before label-based evaluation.
- `final_report.py`: current pooled-method report, pooled cross sections, Funny viewer, and integrity report.
- `all_methods_external.py`: experimental PoCA and frozen-MuNet external-prior adapters.
- `run_all_methods_external.py`: Funny 0 and main 0 all-method comparison; detector-only `load_x` helper is defined here.
- `matched_likelihood_compare.py`: applies the same pooling to frozen-1743 likelihood for a fairer prior comparison.
- `test_method.py`: five focused experimental tests.
- `README.md`, `ALL_METHODS.md`, `report.html`: research explanations/results. `ALL_METHODS.md` has an old feature-number wording error: momentum is feature 13/index 12, not feature 14.

### Fixed external momentum prior

Source: Reyna 2006 atmospheric muon parameterization, equations 2/3, coefficients `[0.00253, 0.2455, 1.288, -0.2555, 0.0209]` in the implemented spectrum.

- Momentum integration: **1 GeV/c to `2000/cos(theta)` GeV/c**.
- 64 log-momentum quadrature points.
- Fixed Highland constant 15 MeV and beta approximately 1.
- Spectrum depends on the physical incoming zenith direction.
- Effective momentum is **`p_eff(theta) = E[1/p² | theta]^(-1/2)`**, not arithmetic mean momentum and not an inferred individual measurement.
- Example effective values from this fixed model: approximately 2190 MeV/c vertically, 2893 MeV/c at cos(theta)=0.5, 9540 MeV/c at cos(theta)=0.1.

The baseline substitutes the expected scattering coefficient into the full model. Internal numerical effective-momentum representations in `baseline_rho` can use rescaled units to reuse algebra; do not interpret every intermediate variable called momentum as a measured MeV value.

The prior assumes an open-sky external population with its fixed low-energy cutoff. It does not model object-dependent survival, energy loss, overburden, or detailed detector acceptance. Those are plausible general sources of absolute-scale error. Do not silently tune the spectrum to Funny truth to fix them.

### Results across the nine predefined cases

Full-grid MSE, external-prior unpooled baseline → recommended exposure pool, in `cm^-2`:

```text
case          unpooled         pooled          all-zero predictor
Funny 0       0.7180547        0.2172217       0.2961453
main 0        0.4627808        0.0750294       0.1122567
main 1        1.1844931        0.1963314       0.3114326
main 2        1.5549511        0.2070679       0.2624820
main 3        0.5853262        0.0598847       0.0601093
main 4        0.8205439        0.1103698       0.1343039
main 5        2.4167548        0.6160672       0.9456348
main 6        3.6775214        1.2213322       1.7508656
main 7        0.9389642        0.1109248       0.1141526
mean          1.3732656        0.3126921
```

Mean reduction is about **77.23%**, with lower MSE on all nine. Held-out track scores also improved in the saved comparison. These are development-panel results, not a new blind external validation. In some cases the improvement over simply predicting zero is small; do not infer accurate material density from a good global MSE alone.

The more elaborate spectrum-marginalized Student candidate performed **worse** in absolute density MSE: mean approximately **12.7187**, Funny approximately **4.961**. Better localization/held-out likelihood did not guarantee correct `1/X0`. Preserve these failed results and their units; do not substitute them for the pooled method or post-calibrate them into apparent success.

### Research output map

Under `experiments/detector_only_likelihood/results/`, cases are `funny_0` and `prediction_0` … `prediction_7`:

- `detector_inputs.npz`: allowed geometry arrays, event IDs, fit/test indices. Efficient cached inputs for reproducible tests.
- `baseline.npz`, pooled Gaussian/Student files, and early `full_8192` outputs: early scattering-power stage.
- `inverse_radiation_length.npz`: external baseline and failed marginal candidate in rho units; inspect keys.
- **`pooled_likelihood.npz`**, key **`inverse_radiation_length_cm_inv`**: recommended full-sample pooled rho map.
- `evaluation_truth.npz`: evaluation density, generally **no material label map**; cannot identify Ag/Pb/U labels directly from this schema without separate truth.
- `pooled_cross_section.png`: recommended rho comparison. Earlier cross sections correspond to earlier algorithms/units.
- Aggregate `final_comparison.json` / `final_summary.json`: final pooled comparison.
- `scores`, `localization_scores`, inverse-score and spectrum-score artifacts belong to distinct study stages; read their quantity/method labels before quoting them.
- Protocol/quadrature/integrity artifacts record fixed choices and code snapshots. Later additions to experimental helpers mean an old hash is a historical snapshot, not necessarily the current hash of every experimental file.

The held-out objective uses geometric paths estimated from the held-out track's measured endpoints/PoCA. It is a conditional plug-in score under that path model, not an unrestricted generative score for arbitrary unseen detector observations.

## 14. External spectrum across all three methods

Implementation: `experiments/detector_only_likelihood/all_methods_external.py`.

- `_effective_momentum(tracks)` derives `p_eff(theta)` from the fixed prior using the **physical incoming** direction.
- `poca_external`: same exposure-normalized PoCA, sigma **6.25 cm**, but per-direction population effective momentum in the angular signal.
- `ExternalSpectrumMuNet`: overrides input features at column index 12 with standardized `p_eff(theta)`. Frozen weights and all other inference steps remain unchanged. This is a distribution-shift experiment; it is not proof the network was trained for this policy.
- Recommended likelihood retains sigma **3.125 cm**. Do not accidentally give it PoCA's 6.25 cm width while claiming the saved benchmark.

`run_all_methods_external.py` uses Funny 0 and main 0, seed-0 8192 selection. It freshly computes PoCA and MuNet but **reuses cached pooled-likelihood arrays**. Selection matches by the driver's conventions, but a new automatic four-array fingerprint assertion is not present. Verify source/IDs if adapting it to another sample.

A metadata trap: `ExternalSpectrumMuNet` inherits a scalar `momentum_estimate_mev` entry from the production constructor while overriding actual features. Read `momentum_feature_policy` and the override implementation; the inherited scalar does not describe the external feature actually used.

Saved results: `results/all_methods_external/{funny_0,prediction_0}/predictions.npz`, keys `munet`, `poca`, `likelihood`; separate `truth_evaluation_only.npz`; metadata/protocol/score files.

Full-grid MSE comparisons:

```text
                         Funny 0        main 0
MuNet fixed 1743          0.0893009      0.0482336
MuNet external spectrum   0.0965871      0.0470136
PoCA fixed 1743           0.1180004      0.0643504
PoCA external spectrum    0.4768872      0.1523348
likelihood fixed/raw      0.5409965      0.3165231
likelihood fixed/pooled   0.2337021      0.0783573
likelihood external/pool  0.2172217      0.0750294
```

The matched likelihood comparison shows that **most of the improvement comes from pooling**, with a smaller additional improvement from changing the prior on these two examples. Comparing external pooled likelihood only against fixed unpooled likelihood confounds two changes. The external prior does not improve every method or every case; do not claim it does.

## 15. Configuration and command-line workflow

### Defaults and overrides

`cli.DEFAULT_CONFIG` resolves to `src/muon_tomo/configs/default.yaml`. Root `configs/default.yaml` must stay synchronized if defaults are edited. `_config` loads the chosen YAML as a whole; it does not merge a partial custom YAML with defaults.

Important sections:

```yaml
physics:
  highland_constant_mev: 15.0
  reference_momentum_mev: 3000.0
  classical_momentum_policy: mean
  average_momentum_mev: 1743.4601815051958
munet:
  dosage: 1024
  momentum_estimate_mev: 1743.4601815051958
poca:
  minimum_coverage: 1
  grid_size: 32
  pooling_sigma_cm: 6.25
likelihood:
  iterations: 10
  initial_inverse_radiation_length_cm_inv: 0.00004
  minimum_coverage: 1
  epsilon: 1.0e-9
  update_statistic: median
  angular_resolution_rad: 1.0e-6
  position_resolution_cm: 1.0e-3
  maximum_inverse_radiation_length_cm_inv: 20.0
```

The `--average-momentum-mev` CLI flag changes **classical** methods, not MuNet's separately configured momentum feature. To change the scalar for all three in a production experiment, supply an explicit complete configuration with both settings. Do not assume one flag updates both.

### Standard production reconstruction recipe

This reproduces the **legacy fixed-prior production pipeline**, not the external detector-only method or mixed current-best bundle. Use a new output directory for a new experiment and omit `--calibration`.

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/muon-mpl .venv/bin/python -m muon_tomo.cli reconstruct \
  dataset/voxels_prediction_funny.tfrecord \
  --record-index 0 --limit 8192 --seed 0 --method all \
  --output-dir outputs/NEW-RUN
```

Additional options include `--format auto|tfrecord|root`, `--config`, `--weights`, `--dosage 512|1024`, `--max-chunks`, `--iterations`, `--likelihood-mode full|angular`, `--likelihood-update mean|median`, and `--classical-momentum-policy mean|per-track`.

Do not use `per-track` for the strict detector-only request. It exists for explicitly labeled simulation diagnostics.

Reconstruction writes canonical truth where available, one result per requested method, and a manifest. It does **not** automatically produce the user's required cross section or 3D viewer; run the plotting steps.

### Cross section from saved canonical files

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/muon-mpl .venv/bin/python -m muon_tomo.cli plot \
  --truth outputs/NEW-RUN/record_0_truth.npz \
  outputs/NEW-RUN/record_0_munet.npz \
  outputs/NEW-RUN/record_0_poca.npz \
  outputs/NEW-RUN/record_0_likelihood.npz \
  --projection-axis z --center-index 32 --slab-layers 7 \
  --output outputs/NEW-RUN/cross_section.png \
  --metrics-output outputs/NEW-RUN/plot_metrics.json
```

### 3D with the user's display settings

```sh
.venv/bin/python -m muon_tomo.cli plot3d \
  --truth outputs/NEW-RUN/record_0_truth.npz \
  outputs/NEW-RUN/record_0_munet.npz \
  outputs/NEW-RUN/record_0_poca.npz \
  outputs/NEW-RUN/record_0_likelihood.npz \
  --threshold-cm-inv 0.8000000715255737 --max-cubes 262144 --opacity 0.6 \
  --output outputs/NEW-RUN/viewer.html \
  --metrics-output outputs/NEW-RUN/viewer_metadata.json
```

No `--surface-only`. The threshold is the next float32 value after 0.8 because the selector internally uses `>=`; for canonical float32 arrays this implements strict `>0.8`. For noncanonical float64 arrays, implement/check strict comparison directly rather than assuming the same epsilon is exact.

### Compare and validate

`compare --truth TRUTH RESULT... --output METRICS` computes saved-result metrics without inference.

`validate --truth TRUTH RESULT... --output REPORT` requires exactly MuNet/PoCA/likelihood, identical canonical geometry, finite density, one nonmissing identical selected-track fingerprint, and both classical methods declaring shared `mean` momentum with equal scalar values.

Therefore current validation **does not support the mixed/current-best or external-prior bundle as-is**. Do not forge policy/fingerprint metadata to make it pass. It also checks truth geometry, not that truth belongs to the same input record; verify source/record and label content separately.

`truth INPUT --record-index N --output PATH` exports truth, but decoding still traverses the needed TFRecord stream. `inspect INPUT` can scan the full TFRecord; see section 5.

### Historical commands to avoid by default

`calibrate` fits against Funny truth and `--calibration` can apply it during reconstruction, plotting, or stability evaluation. Those options are still implemented for historical reproducibility, but violate the present raw-output workflow if used silently. Some CLI help text still says “calibrated” even when no calibration is supplied; do not treat that wording as an instruction to calibrate.

## 16. Plotting, 3D viewers, and interpretation

### Cross sections (`plotting.py`)

- Standard four panels: truth, MuNet, PoCA, likelihood.
- Default is the mean of seven z layers **29–35 inclusive**, centered at 32, i.e. `[:,:,29:36]`.
- This is a slab average, not one exact center slice and not a maximum-intensity projection.
- `--projection-axis x|y|z` changes the averaged axis; `--full-depth` averages all 64 layers.
- The appropriate array transpose is used for display so x/y orientation matches the canonical volume. Current physical extents are in cm; some historical screenshots used voxel indices.
- Shared color scaling is a display choice. Truth-derived color limits or material labels in evaluation figures do not enter reconstruction, but can visually saturate larger predictions.
- MSE annotations use the entire 64³ array, not just the displayed slab.

The plot's momentum-policy caption is not a universal metadata interpreter. It recognizes one specific external likelihood policy string; other branches assume a scalar from PoCA. Experimental metadata with missing/`None` scalar fields can cause misleading captions or conversion errors. Inspect the actual result policy and fix display handling explicitly if needed; do not change the arrays to fit a caption.

### 3D (`plotting3d.py`)

- Plotly mesh cubes, eight vertices and twelve triangles per voxel, with grouped opacity/color bands and a method selector.
- Rendering a solid voxel object naturally occludes interior cubes from the outside; this alone is not proof the data contains only a shell.
- `surface_only` is an optional geometry filter; default is false. The user's request is false plus no cap.
- Source CLI defaults remain threshold **0.7** and maximum **12,000 cubes**. They do not reflect the latest user preference without flags.
- The cap samples displayed candidates deterministically with RNG seed 0; it is not an intrinsic model output size or a top-density selection.
- At 64³, `max_cubes=262144` removes any candidate count cap.
- `_select_voxels`/viewer metadata distinguish qualifying voxel count from displayed count. Check both when diagnosing “always exactly 12000.”
- The strict current cut is prediction `>0.8`; this is solely a view cut, never a reconstruction mask or MSE cut.

Current-best Funny viewer has all qualifying cubes displayed: truth **35,760**, MuNet **20,122**, PoCA **28,864**, likelihood **3,888**.

A low-density object can legitimately have no visible truth voxels under this cut. Main record 3's truth maximum is around 0.702, below 0.8. Empty display there does not imply missing data.

Open an existing HTML through the Codex browser/file panel when useful; no server is necessary. Link local deliverables with absolute paths in chat. Do not regenerate inference just to open or replot a saved volume.

### Historical fill postprocess

`postprocess.py` implements a synchronous, single-pass six-face-neighbor fill of empty voxels. Neighbor averages include zero neighbors, use available neighbors at boundaries, and preserve existing nonempty values. This was used in historical screenshot matching (`historical-fill-viewer.html`) and is **not part of the current standard raw prediction**. A matching historical figure can depend on this extra transform.

## 17. Evaluation and material-region statistics

### Full-grid MSE versus other metrics

`compare.compare_to_truth` uses **all 262,144 voxels** for `mse_full_grid_cm_inv_squared`, including air and uncovered cells.

Other reported metrics such as MAE, RMSE, bias, and min/max use the result's **validity mask**. Consequently the field labeled RMSE is not necessarily the square root of the full-grid MSE. State the support when comparing metrics; use an explicit full-grid calculation if that is the intended statistic.

Funny 0's all-zero prediction MSE is approximately **0.29614525234**. Since most voxels are air, low global MSE can coexist with severe underestimation inside metal. Include region bias, localization, and/or nonair error when judging scientific quality; do not optimize only a single full-volume score.

### Latest requested material statistics

Saved in `outputs/funny-current-best/material_region_statistics.json`.
These include **all true voxels of each material in the full 3D volume**, not just the seven-layer cross section, visible voxels, eroded interiors, or result-validity mask.

Mean ± population standard deviation (`ddof=0`), all in `cm^-1`:

```text
method       silver                  lead                    uranium
MuNet        0.815581 ± 0.318395      0.882273 ± 0.322271      1.110434 ± 0.344387
PoCA         1.224401 ± 0.691526      1.452947 ± 0.634007      2.464576 ± 1.102694
Likelihood   0.306775 ± 0.214575      0.424057 ± 0.308266      0.563607 ± 0.250808
truth        1.170569                 1.781790                 3.158333
```

These standard deviations describe spatial variation of predictions, not standard errors of the mean, repeated-simulation uncertainty, or posterior uncertainty. Truth-region SD should mathematically be zero for each homogeneous label. Tiny approximately `1e-7` values in saved float32 statistics are accumulation roundoff; compute in float64 for future summaries.

`calibration.material_statistics` defaults to eroded material masks and valid-result support and also returns medians. It is a different statistic and should not silently replace this user's full-region request.

Example evaluation-only calculation with standard NPZs:

```python
import numpy as np

with np.load("outputs/funny-current-best/record_0_truth.npz", allow_pickle=False) as f:
    labels = f["material_labels"]
for method in ("munet", "poca", "likelihood"):
    with np.load(f"outputs/funny-current-best/record_0_{method}.npz", allow_pickle=False) as f:
        prediction = f["inverse_radiation_length_cm_inv"].astype(np.float64)
    for material, label in (("silver", 17), ("lead", 19), ("uranium", 21)):
        values = prediction[labels == label]
        print(method, material, values.size, values.mean(), values.std(ddof=0))
```

Truth is used here to define evaluation regions only. Do not use these means to renormalize the predictions afterward.

## 18. Saved results and experiment history

### Prefer these for current work

- **`outputs/poca-presentation/deliverables/PoCA_improvement.pptx`**: 14-slide PoCA follow-up in the user's 5 August 2026 deck template, created 9 September 2026. Funny 0, 8192 tracks, fixed 1743 prior. Holds production full likelihood at MSE 0.5409965 and MuNet at 0.0893009 while comparing previous corrected 32³ PoCA (110.2939) with exposure-pooled PoCA (0.1180). Includes motivation, equations, orthogonal cross sections, raw-baseline caveats, and speaker-note provenance. Private rebuild/evidence/preview files are under `outputs/poca-presentation/.build/`; this deck intentionally does not use the later experimental pooled likelihood.
- **`outputs/funny-current-best/`**: latest user-facing mixed-prior Funny 0 result, plots/viewer/statistics; limitations in section 2.
- **`outputs/detector-only-comparison/funny-fixed-1743/`**: useful production fixed-prior baseline. Despite the parent directory's name, 1743 has simulation-derived population provenance. Funny MSE MuNet 0.0893009, PoCA 0.1180004, unpooled likelihood 0.5409965.
- **`outputs/poca-exposure-study/`**: experiment/finalization scripts, README, final PoCA arrays for Funny 0/1, user-facing plots/viewer, pooling sweep history.
- **`experiments/detector_only_likelihood/report.html`** and `results/final_*`: latest isolated likelihood research report and pooled-method results.
- **`experiments/detector_only_likelihood/results/all_methods_external/`**: shared external-spectrum comparison and fixed-prior matched pooling control.
- **`outputs/random-prediction/record-0-exposure/`**: main record 0, production fixed-prior MuNet/PoCA/likelihood MSE approximately 0.048233642 / 0.064350425 / 0.316523123.
- **`outputs/random-prediction/record-3-exposure/`**: main record 3, approximately 0.02451264 / 0.05649019 / 0.35647663; low truth density makes the >0.8 viewer mostly/entirely empty for truth.

### Historical directories requiring explicit interpretation

- `outputs/funny-munet-fixed-momentum/`: earlier raw fixed-feature MuNet; Funny MSE about 0.0893678407.
- `outputs/funny-poca-32-corrected/`: fixed length aggregation but **old conditional PoCA**, MSE about 110.2939; “corrected” does not mean current estimator.
- `outputs/funny-poca-32/`: initial coarse-grid defect and some historical calibrated artifacts; not a baseline for current raw methods.
- `outputs/funny-after-side-exit-8192/`: side-exit-corrected full likelihood about 0.54099653, but old PoCA about 20.1327 and older MuNet policy. Do not assume all three files here use today's settings.
- `outputs/funny-after-side-exit-8192-pertrack/`: diagnostic simulation-momentum run, not allowed detector-only input policy.
- `outputs/funny-fixed-momentum-raw/`: older bundle; some arrays correspond to the initial coarse PoCA bug. The word “raw” alone does not establish correctness.
- `outputs/mismatch-check/angular-v2/`: historical angular/fill reproduction, including `historical-fill-viewer.html`. Old screenshot values MuNet 0.1103 / PoCA 0.2987 / angular likelihood 0.3064 belong to different processing/policies and are not current raw regression targets.
- `outputs/likelihood-audit/`: `REPORT.md`, `audit.py`, `algebra_check.py`, `controlled_variants.py`, `path_audit.py`, `coarse_test.py`, `angular_original_test.py`, and supporting JSON/NPZ results. Best source for why transport/model changes were investigated.
- `outputs/local-check/`: original local environment/data inspection, early plots, and historical calibration work; not current reconstruction defaults.
- `outputs/random-prediction/record-5-current/`, `record-6/`, `record-7/` and related subdirectories: earlier model/policy/viewer states. Main record 7 was used for 8192/all-44272/top-bottom-only/angular comparisons.

### Historical mistake chain to remember

1. Original screenshot and local output differed due to a combination of spatial/processing conventions, method versions, display cuts, and calibration, not simply missing plotted cubes.
2. Earlier outputs were adjusted with truth-based calibration. The user explicitly rejected that; do not recreate those values by refitting scale/intercept.
3. Side-exit covariance had a concrete downstream-distance bug; current full likelihood includes the fix.
4. Coarsening PoCA introduced a length-aggregation bug, later fixed. Even the corrected conditional estimator remained poor.
5. Exposure normalization plus spatial pooling greatly reduced PoCA MSE, but pooling choices were evaluated against Funny truth.
6. Changing the MuNet momentum feature materially changed its output. Keep feature policy, dosage, weights, selected IDs, and calibration status visible in comparisons.
7. Exactly 12,000 displayed MuNet voxels came from the viewer cap, not model sparsity. The user requested strict >0.8 with all qualifying cubes.
8. The later detector-only likelihood experiment improved chiefly via exposure-weighted spatial regularization. External-prior changes and regularization must be disentangled in comparisons.

## 19. Calibration, leakage, and unresolved scientific limitations

### Historical calibration implementation

`src/muon_tomo/calibration.py` still provides fitting, loading, application, and statistics. The old workflow fits Funny record 0 and checks Funny record 1, using material regions, default erosion of two voxels, background inclusion, and affine slope/intercept. It uses method-specific region summaries, weighted fitting, nonnegative behavior, and truth-derived clipping bounds.

Historical configuration files include:

- `configs/funny-v2-spatial.calibration.json`
- `configs/funny-v3-mean-momentum.calibration.json`
- `configs/funny-v4-robust-mlsd.calibration.json`

These exist for historical reproducibility. **They are not authorized defaults for present predictions.** Passing `--calibration` to reconstruction or plotting changes the values used/reported; it is not a harmless display setting.

### Three distinct truth pathways

1. **Direct inference leakage:** using labels, per-track simulated momentum, or truth masks to compute the prediction. Prohibited for detector-only work.
2. **Prior/calibration provenance:** e.g. fixed 1743 derived from Funny momentum or material-fitted affine scales. Even if no truth is read during a later run, the predictor can carry truth-derived information.
3. **Development selection:** choosing smoothing widths or the best approach based on benchmark truth. It is legitimate to evaluate research, but those cases are then development data, not blind tests.

The current experimental four-array function and external prior address the first two more directly than the production frozen-1743 pipeline. They do not undo the fact that the pooled approach was chosen after inspecting development-panel results. Future generalization claims need truly held-out objects and fixed settings.

### Remaining limitations / potential future work, not already implemented fixes

- Realistic accepted-track momentum distribution may differ from open-sky prior due to target attenuation, energy loss, angle acceptance, triggering, and low-energy cutoff. This affects absolute `1/X0` identifiability from position/direction alone.
- MuNet was not retrained for the external direction-conditioned input policy. Its training distribution needs a separate provenance investigation before deciding on retraining.
- PoCA approximates distributed scattering by one point and can blur/mislocalize density. Exposure pooling reduces variance at the cost of spatial resolution.
- Likelihood median/caps/finite iterations/path approximation are heuristic choices; better MSE after pooling does not establish exact statistical consistency.
- Full-grid MSE favors background accuracy in sparse objects. Current-best likelihood substantially underestimates metal-region density despite beating the zero predictor globally.
- Cached experimental reuse lacks a universal detector-only fingerprint/schema contract.
- Standard validation rejects experimental/mixed policies and does not ensure the truth file is from the same record.
- Plot captions can misread experimental metadata; current-best likelihood coverage is placeholder data.
- The original training momentum policy and all external original-source file versions have not been independently revalidated in this summary task.

Do not “fix” these by sample-specific scalar fitting, selecting a favorable record, hiding low predictions, or editing metadata to conceal incompatible inputs.

## 20. Tests, packaging, and validation status

Fresh verification during this documentation task, 2026-09-09:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/muon-mpl \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/pytest -q tests experiments/detector_only_likelihood/test_method.py
```

Result: **49 passed, 3 skipped**, about 0.82 seconds. This comprises **44 portable production tests and 5 experimental tests**. Three real-input integration tests were skipped because the corresponding environment variables were not set. This is not a claim that fresh full-data MuNet/ROOT inference or all historical experiments were rerun during the documentation task.

### Test map

- `test_geometry.py`: canonical bounds/indexing/clipping/tracing.
- `test_spatial_contract.py`: MuNet coordinate normalization, axis landmarks, spatial checks/fingerprints.
- `test_reconstruction.py`: reconstruction contracts; exposure behavior including rays without valid PoCA; constant-field pooling behavior; fixed-vs-per-track momentum handling; full/angular output contracts.
- `test_compare.py`: full-grid scoring behavior.
- `test_units.py`: physical quantity conversion and labels.
- `test_result_schema.py`: result save/load contract.
- `test_calibration.py`: historical calibration functionality, not permission to apply it.
- `test_postprocess.py`: synchronous neighbor fill.
- `test_stability.py`: detector-error perturbation invariants.
- `test_plotting.py`: slab/projection behavior.
- `test_plotting3d.py`: cube geometry, surface filtering, display cap, opacity behavior.
- `test_packaging.py`: packaged default configuration / root-copy consistency.
- `test_munet_integration.py`: packaged weights and optional real Funny inference.
- `test_tfrecord_integration.py`, `test_root_integration.py`: optional real input checks.
- Experimental `test_method.py`: analytic/numerical Gaussian/Student gradient agreement, direction-magnitude invariance of detector inputs, known synthetic uniform scattering-power recovery, spectrum gradient, exposure-pool identity/constant behavior.

Opt-in environment variables are **`FUNNY_TFRECORD`** and **`GEANT4_ROOT_SAMPLE`**. Read the specific test before claiming what an opt-in run verifies. No dedicated newly added analytic side-exit regression was identified in the portable test list during this inspection; the saved audit contains related checks.

CI runs portable tests on Ubuntu Python 3.11/3.13, lint/format checks over `src` and `tests`, plus build/twine package checks. Current local `ruff`, `build`, and `twine` were absent, so **fresh lint/build success was not established** here. Do not cite the old `VALIDATION.md` test count as today's result.

### Stability workflow

`stability.py` writes an errored TFRecord with perturbed detector positions/directions and preserved label payload; CLI compares clean/errored reconstructions using fixed seeds. Defaults include 8192 tracks, error and MuNet seed 20260803, position sigma 0.29 cm, angular sigma 0.5 degrees, and 10 likelihood iterations.

A small QA weakness exists in the current truth-preservation report: the `np.array_equal(labels, labels)` expression is tautological rather than an independent comparison of saved output labels. Do not treat that flag alone as proof of round-trip truth preservation. This documentation task does not modify it.

## 21. External references, licensing, and preservation

Original repositories:

- MuonTruck: <https://github.com/kfung010/muonTruck>
- MuNet source: <https://github.com/jedlimlx/Muon-Tomography-AI>

`NOTICE` records a MuonTruck source revision **`8393853a47e8654552a62d1886fb6766d637782f`**, with local uncommitted changes at import. Do not equate this integrated tree with a pristine upstream revision.

External local paths verified to exist during this inspection:

- `/Users/longsangchan/Muon/muonTruck/recon/`
- `/Users/longsangchan/Library/Mobile Documents/com~apple~CloudDocs/CUHK Ph.D/Muon Tomography/muonTruck/recon/`
- `/Users/longsangchan/Library/Mobile Documents/com~apple~CloudDocs/CUHK Ph.D/Muon Tomography/Statistical_Reconstruction_for_Cosmic_Ra.pdf`

The two `recon` directories need not be identical. If a new comparison is requested, use the exact user-specified path and compare the relevant source, rather than assuming both external copies share a version. Historical audits reference original `utils_likelihood.py` / `utils_poca.py` and generator behavior. External source is reference material; do not modify it as part of an internal fix without authorization.

Paper/source references used by the research artifacts:

- Statistical reconstruction paper: <https://www.hep.phy.cam.ac.uk/~hommels/CosmicConcrete_dir/MuonPapersForBart/Paper_IEEE_16_8_Aug2007_StatImageReco.pdf>
- Reyna spectrum: <https://arxiv.org/pdf/hep-ph/0604145>

Historical provenance records include Funny dataset SHA256
`65c160c4e38ac8477b1fff494ed55a7f883e732f3c5c21d36237c83fbd205e00`.
This is a recorded historical hash, not a fresh full-input rehash in this documentation task. Prefer existing verified hashes when the input is unchanged; do not repeatedly hash the 50 GiB main dataset without need.

License handling:

- Integration code is under the repository's Apache-2.0 terms.
- Vendored MuNet has Apache-2.0 provenance.
- `NOTICE` explicitly records missing upstream license information for MuonTruck-derived material. Do not claim every derived component has a fully resolved Apache license merely because root `LICENSE` exists.
- Read `NOTICE`, `LICENSE`, `LICENSES/Apache-2.0.txt`, and `third_party/` provenance before redistribution work.

## 22. Recipes and future update checklist

### Resume work without re-exploring everything

1. Read sections 1–3 and the relevant algorithm/recipe section.
2. Identify dataset **and** record, requested momentum policy, and whether the user wants production or experimental behavior.
3. Check the specific source/defaults and saved metadata; prefer existing arrays for plotting/statistics.
4. For a new prediction, preserve selection IDs and settings before evaluating truth.
5. Produce both requested visual artifacts and raw full-grid metrics; report material bias if making a performance claim.
6. Add the new artifact path and any changed defaults/provenance to this summary, rather than appending an unstructured turn-by-turn transcript.

### Detector-only recommended likelihood from cached arrays

The experimental files use sibling imports (`from method import ...`), so put the experimental directory on the module search path or run from that directory. Example structure, with input arrays already loaded:

```python
import sys
from pathlib import Path

root = Path("/Users/longsangchan/Muon/muon-tomography-unified-main")
sys.path.insert(0, str(root / "experiments/detector_only_likelihood"))
from exposure_pooled_likelihood import reconstruct_inverse_radiation_length

# Each position is in cm; each direction is normalized by the detector wrapper.
# Preserve the repository endpoint convention and the selected event ordering.
density, metadata = reconstruct_inverse_radiation_length(
    entry_position_cm, entry_direction, exit_position_cm, exit_direction
)
```

Do not add truth or momentum arguments. Save the prediction and inference metadata before opening labels for evaluation. The returned tuple is not automatically a fully populated `ReconstructionResult`; if exporting for the standard viewer, record real coverage/fingerprints or explicitly mark missing fields, not fabricated physical statistics.

### Sanity checks before comparing outputs

- Correct source TFRecord and record index; Funny vs main `record_0` is unambiguous.
- Correct material-label histogram and orientation for the evaluation truth.
- All arrays canonical 64³ xyz, physical bounds [-50,50] cm, density in `cm^-1`.
- Same selected IDs/geometry and compatible dosage/chunk truncation.
- Momentum policy matches the claim; scalar 1743 provenance is disclosed if used.
- No calibration metadata or hidden plot-time calibration for a raw-output request.
- Likelihood mode/iterations/update/cap/pooling and PoCA grid/pooling match the cited baseline.
- Full-grid MSE calculated before threshold; viewer candidate/display counts are separate.
- If comparing only one method change, hold all other methods and truth fixed rather than recomputing them with accidental new settings.
- If matching external vs fixed likelihood, compare **both pooled** as well as the raw baseline.

### Update this document when behavior changes

Record the date, affected files/defaults, exact run/artifact path, selected inputs, unit/policy changes, tests actually run, and whether any old result is superseded. Remove or revise stale “current” statements instead of leaving conflicting versions in the quick reference. Keep historical results labeled as historical.

Avoid embedding giant logs or arrays. Use a short result and an exact path to its manifest/report. Preserve clear distinctions among observed facts, historical records, implementation behavior, and hypotheses. This is what allows the next action to start from a few targeted reads rather than a repository-wide search.
