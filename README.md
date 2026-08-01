# Muon Tomography Unified

A standalone, inference-focused repository that runs three muon-scattering
tomography methods through one data model, one physical target, and one fixed
voxel geometry:

- frozen **Mu-Net**
- fixed-average-momentum **PoCA**
- **likelihood/MLEM**, including the muonTruck 2×2 covariance update

The primary output of every method and the simulation truth is

\[
\rho_X = \frac{1}{X_0}\quad[\mathrm{cm}^{-1}],
\]

where \(X_0\) is radiation length in centimetres.

This repository was created additively. It does not modify the source
`Muon-Tomography-AI` project, the source `muonTruck` project, their ROOT files,
or their TFRecords.

## Current validated status

The following workflows have been exercised successfully on the local samples:

| Workflow | Status |
|---|---:|
| Parse both records of `voxels_prediction_funny.tfrecord` | passed |
| Map truth labels to a finite `64×64×64` \(1/X_0\) volume | passed |
| Parse the `cosmic_leadBox_200000.root` Geant4 hit tree | passed |
| Frozen Mu-Net inference from TFRecord | passed |
| PoCA reconstruction from TFRecord | passed |
| Likelihood reconstruction from TFRecord | passed |
| All three methods from Geant4 ROOT | passed |
| Same shape, bounds, axes, and voxel size across methods | passed |
| Fit on funny record 0 and evaluate on funny record 1 | passed |
| PoCA/likelihood ignore every per-muon momentum magnitude | passed |
| PoCA/likelihood use the same configured 3 GeV population prior | passed |
| Reproducible 2×2 truth/MuNet/PoCA/likelihood slab plot | passed |
| Unit and synthetic reconstruction tests | passed |

The committed code does not train or fine-tune Mu-Net. The copied weights are
loaded read-only at inference time.

## Hard geometry contract

Every method uses exactly the same `GridGeometry` object:

| Property | Default and required value |
|---|---|
| Shape | `(64, 64, 64)` |
| Axis order | `xyz` |
| Array convention | `volume[x, y, z]` |
| Lower bound | `(-50, -50, -50)` cm |
| Upper bound | `(50, 50, 50)` cm |
| Voxel size | `(1.5625, 1.5625, 1.5625)` cm |

These bounds are fixed by the frozen Mu-Net transform
`normalized = position_cm / 100 + 0.5`: normalized coordinates 0 and 1 are
exactly -50 and +50 cm. The constructor rejects any other shape, bounds, or axis
order. Detector track reference planes remain at -100 and +100 cm and are
independent of the target voxel cube. Result validation rejects files that do
not share identical geometry or the exact same selected-track fingerprint.

## Repository layout

```text
muon-tomography-unified/
├── calibration/                 generated/versioned calibration files
├── configs/default.yaml         common geometry and method defaults
├── src/muon_tomo/
│   ├── _vendor/munet/           frozen inference architecture and weights
│   ├── io/                      TFRecord and Geant4 ROOT adapters
│   ├── reconstruction/          Mu-Net, PoCA, and likelihood
│   ├── calibration.py           funny material-region calibration
│   ├── cli.py                   unified command-line workflow
│   ├── constants.py             material X0 table and conversions
│   ├── geometry.py              mandatory 64³ geometry and ray tracing
│   └── types.py                 shared tracks, truth, and result schema
├── tests/                       unit, synthetic, and integration tests
├── third_party/                 copied license information
├── NOTICE                       source provenance
└── pyproject.toml               package and dependency definition
```

Generated reconstructions are written below `outputs/`, which is ignored by
Git. No command writes into either source repository or into the dataset
directory.

## Installation

Python 3.11–3.13 is supported. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
```

The `all` extra installs TensorFlow, ROOT I/O, plotting, and testing packages.
Smaller installations are possible:

```bash
python -m pip install -e .          # classical methods and ROOT I/O
python -m pip install -e '.[munet]' # add TensorFlow/Mu-Net
python -m pip install -e '.[test]'  # add tests
```

Confirm the command is installed:

```bash
muon-tomo --help
```

## Input data

### TFRecord `x`/`y` format

The adapter reads GZIP-compressed `tf.train.Example` records with two serialized
tensors:

- `x`: floating-point track table shaped `(N, >=12)`
- `y`: integer material-label volume shaped exactly `(64,64,64)`

The interpreted `x` columns are:

| Columns | Meaning | Stored unit |
|---|---|---|
| `0:3` | entry position | mm |
| `3:6` | entry momentum vector | MeV |
| `6:9` | exit position | mm |
| `9:12` | exit direction | unit vector |
| `12:` | retained source extras, not required by the common table | source-defined |

Positions are converted to centimetres and directions are normalized. Momentum
is the norm of columns `3:6`.

Inspect the funny sample without changing it:

```bash
FUNNY=/Users/nelsonchan/Muon_Tomo/Muon-Tomography-AI/dataset/voxels_prediction_funny.tfrecord
muon-tomo inspect "$FUNNY"
```

The current sample contains two records with 44,240 and 44,323 tracks. Both
truth volumes contain material labels `0`, `17`, `19`, and `21`.

### Geant4 ROOT hit format

The ROOT adapter reads the `hits` tree branches:

```text
eventNumber
hitPositionX_truth  hitPositionY_truth  hitPositionZ_truth
hitMomentumX_truth  hitMomentumY_truth  hitMomentumZ_truth
```

For every event with at least two hits above and two hits below the cargo, the
adapter fits two 3-D lines with SVD and intersects them with fixed detector
reference planes at -100 and +100 cm. It stores the lower point first and upper
point second, matching the TFRecord ordering, and creates the same `TrackTable`
used by all methods. Detector planes are deliberately independent of the
`[-50,+50]` cm target cube. The muonTruck unit conventions are retained:

- stored hit position to centimetres: `× 0.1`
- stored momentum to MeV: `× 1000`

Inspect a ROOT file:

```bash
ROOT_SAMPLE=/Users/nelsonchan/Muon_Tomo/Muon-Tomography-AI/cosmic_leadBox_200000.root
muon-tomo inspect "$ROOT_SAMPLE" --limit 1024
```

## Truth definition

Material labels are mapped through the Mu-Net material table. Relevant funny
values are:

| Label | Material | \(X_0\) (cm) | \(1/X_0\) (cm⁻¹) |
|---:|---|---:|---:|
| 0 | air | ∞ | 0 |
| 17 | silver | 0.854286 | 1.170569 |
| 19 | lead | 0.561233 | 1.781790 |
| 21 | uranium | 0.316623 | 3.158333 |

Export canonical truth:

```bash
muon-tomo truth "$FUNNY" \
  --record-index 0 \
  --output outputs/funny/record_0_truth.npz
```

Air is stored as zero inverse radiation length. Derived radiation length is
infinite wherever \(\rho_X=0\).

## Physics and unit conversion

The leading multiple-scattering relationship is

\[
\operatorname{Var}(\theta) \simeq
\left(\frac{k}{p}\right)^2\frac{L}{X_0}.
\]

This repository uses the existing project convention

```text
k = 15 MeV
p_ref = 3000 MeV
```

and defines reference scattering density by

\[
\lambda_{\mathrm{ref}}
=\left(\frac{k}{p_{\mathrm{ref}}}\right)^2\rho_X.
\]

When expressed in mrad²/cm:

\[
\lambda_{\mathrm{ref}}[\mathrm{mrad}^2/\mathrm{cm}]
=25\,\rho_X[\mathrm{cm}^{-1}],
\]

so \(\rho_X=\lambda_{\mathrm{ref}}/25\). The classical default is a fixed
population-average prior \(\bar p=3000\) MeV. PoCA and likelihood assign this
same scalar to every selected event; neither method reads an event's simulated
momentum magnitude. The prior is configured independently of each record, so
record-level Geant4 truth cannot leak into reconstruction. Constants and the
information policy are recorded in configuration and result metadata.

## Reconstruction methods

### Frozen Mu-Net

Mu-Net uses the copied `model_512.weights.h5` or `model_1024.weights.h5`. The
adapter reproduces the verified 14-feature inference path:

- normalized entry position
- unit entry direction
- normalized exit position
- unit exit direction
- deterministic zero momentum-estimate channel
- direction-difference magnitude

Predictions from complete chunks are averaged. The native learned output is
preserved as `raw_estimator`; negative values are clipped only in the physical
primary volume. No optimizer, loss, training dataset, `fit`, or weight-writing
operation exists in the supported workflow.

### PoCA

For each event, PoCA computes the midpoint of closest approach between the
incoming and outgoing lines. Under the default hardware-compatible information
policy it deposits

\[
\widehat{\rho}_{X,i}=
\frac{\bar p^2}{k^2}
\frac{(\Delta\theta_{x,i}^2+\Delta\theta_{y,i}^2)/2}{L_{i,\mathrm{voxel}}}
\]

in the PoCA voxel. Values are averaged over events assigned to that voxel.
`occupancy_count` is the number of localized events and is never presented as
the reconstruction itself. In the primary prediction, each zero-occupancy
voxel is filled by one synchronous mean of its in-bounds six face-connected
neighbors, both before and after calibration. Measured voxels are unchanged,
while `valid_mask` separately records whether a voxel met the configured
coverage threshold.

### Likelihood/MLEM

The `full` mode ports the muonTruck covariance MLEM structure. For every voxel
segment with length \(L\) and remaining distance \(T\), it constructs

\[
W=\begin{bmatrix}
L & L^2/2+LT\\
L^2/2+LT & L^3/3+L^2T+LT^2
\end{bmatrix}.
\]

Track covariance is assembled sparsely over crossed voxels and updated using
both projected angular and displacement observations. Internal reference
scattering density is converted to canonical \(1/X_0\). Every event covariance
uses the same configured \(\bar p\), not its hidden Geant4 momentum.

The optional `angular` mode is a faster path-integral MLEM using momentum-
normalized angular observations. It has the same result contract but is a
different estimator; a calibration must only be applied to the mode that
created it. Its native estimator is already in cm\(^{-1}\), so likelihood uses
an identity calibration rather than an unstable material-median affine fit.
Zero-occupancy voxels use the same single-pass six-face-neighbor fill as PoCA;
the coverage mask is retained separately.

## Run all methods on `funny`

The following reproduces the validated same-track workflow. Each method sees
the same deterministically selected 8,192 events and the same geometry:

```bash
mkdir -p outputs/funny

muon-tomo reconstruct "$FUNNY" \
  --record-index 0 \
  --method all \
  --limit 8192 \
  --seed 0 \
  --max-chunks 8 \
  --iterations 10 \
  --likelihood-mode angular \
  --classical-momentum-policy mean \
  --average-momentum-mev 3000 \
  --output-dir outputs/funny

muon-tomo reconstruct "$FUNNY" \
  --record-index 1 \
  --method all \
  --limit 8192 \
  --seed 0 \
  --max-chunks 8 \
  --iterations 10 \
  --likelihood-mode angular \
  --classical-momentum-policy mean \
  --average-momentum-mev 3000 \
  --output-dir outputs/funny
```

When Mu-Net is included, the selected tracks are trimmed once to a whole number
of Mu-Net dosage chunks and that exact ordered selection is passed to all three
methods. A SHA-256 fingerprint over event IDs, positions, directions, and
momenta is stored in every result and must match during validation. Omit
`--limit` and `--max-chunks` to use the largest common complete-chunk selection.

For full covariance likelihood, replace `angular` with `full`. Refit calibration
for that mode.

`mean` means a fixed population prior, not the arithmetic mean of hidden
momenta in the selected simulation record. `per-track` remains available only
to reproduce earlier studies and is rejected by the aligned-comparison
validator.

## Calibrate with `funny`

The current `funny-v3-mean-momentum` protocol fits record 0 and evaluates
record 1 under the fixed-average policy:

```bash
muon-tomo calibrate \
  --dataset "$FUNNY" \
  --results-dir outputs/funny \
  --fit-records 0 \
  --validate-records 1 \
  --calibration-id funny-v3-mean-momentum \
  --output configs/funny-v3-mean-momentum.calibration.json
```

For each method the calibration code:

1. maps material labels to \(1/X_0\);
2. erodes non-air material masks by the configured two voxels;
3. intersects masks with the method's `valid_mask`;
4. measures robust material-region native values;
5. fits a versioned, nonnegative-slope affine post-processing map for Mu-Net
   and PoCA, while retaining identity calibration for likelihood because its
   native estimator is already in cm\(^{-1}\);
6. records fit regions and record-1 metrics in the JSON report.

This is post-processing only. Mu-Net weights are not changed.

Apply a calibration during reconstruction:

```bash
muon-tomo reconstruct "$FUNNY" \
  --record-index 1 \
  --method all \
  --likelihood-mode angular \
  --classical-momentum-policy mean \
  --average-momentum-mev 3000 \
  --calibration configs/funny-v3-mean-momentum.calibration.json \
  --output-dir outputs/funny_calibrated
```

## Plot the aligned comparison

Generate the four 64×64 panels as a seven-layer mean slab around `z=32`:

```bash
muon-tomo plot \
  --truth outputs/funny/record_0_truth.npz \
  outputs/funny/record_0_munet.npz \
  outputs/funny/record_0_poca.npz \
  outputs/funny/record_0_likelihood.npz \
  --calibration configs/funny-v3-mean-momentum.calibration.json \
  --center-index 32 \
  --slab-layers 7 \
  --output outputs/figures/funny_record0_mean_momentum.png \
  --metrics-output outputs/funny/record_0_plot_metrics.json
```

All panels use the same physical colour scale. Each title reports MSE over the
complete aligned `64³` volume, not over the displayed slab. The truth panel
labels the silver, lead, and uranium \(1/X_0\) values.

## Run all methods on Geant4 ROOT

```bash
muon-tomo reconstruct "$ROOT_SAMPLE" \
  --method all \
  --limit 1024 \
  --seed 0 \
  --max-chunks 1 \
  --iterations 2 \
  --likelihood-mode full \
  --classical-momentum-policy mean \
  --average-momentum-mev 3000 \
  --output-dir outputs/root_smoke
```

For a production run, omit the track/chunk limits and increase likelihood
iterations according to convergence. A ROOT input has no embedded voxel truth;
comparison requires a separately exported truth volume using the identical
geometry.

## Output schema

Every method writes a compressed NPZ with these keys:

| Key | Shape | Meaning |
|---|---|---|
| `inverse_radiation_length_cm_inv` | `64³` | primary calibrated physical result |
| `radiation_length_cm` | `64³` | derived \(X_0\), `inf` where \(\rho_X=0\) |
| `scattering_density_ref_mrad2_per_cm` | `64³` | derived reference scattering density |
| `raw_estimator` | `64³` | untouched method-native volume |
| `occupancy_count` | `64³` | method-specific coverage, never density |
| `valid_mask` | `64³` | voxels valid for quantitative comparison |
| `uncertainty` | `64³` | optional method uncertainty |
| `metadata_json` | scalar | units, geometry, method, source, and settings |

Truth NPZ files contain:

- `material_labels`
- `inverse_radiation_length_cm_inv`
- `metadata_json`

## Validate and compare

Validate that truth and a three-method set share the exact grid, orientation,
scale, ordered selected-track fingerprint, and shared classical momentum
policy/value:

```bash
muon-tomo validate \
  outputs/funny/record_0_munet.npz \
  outputs/funny/record_0_poca.npz \
  outputs/funny/record_0_likelihood.npz \
  --truth outputs/funny/record_0_truth.npz \
  --output outputs/funny/record_0_validation.json
```

Compare results in the same physical units:

```bash
muon-tomo compare \
  --truth outputs/funny/record_0_truth.npz \
  outputs/funny/record_0_munet.npz \
  outputs/funny/record_0_poca.npz \
  outputs/funny/record_0_likelihood.npz \
  --output outputs/funny/record_0_metrics.json
```

Every comparison reports MSE over the complete aligned `64³` grid, after the
six-face-neighbor fill for zero-occupancy PoCA and likelihood voxels. It also
reports coverage-conditioned MAE, RMSE, bias, coverage, and value ranges in
cm⁻¹. Cross-section figures always annotate the full-volume `64³` MSE, never a
slice-only MSE.

## Tests

Run the portable suite:

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 pytest -q
```

Run the real TFRecord integration test:

```bash
FUNNY_TFRECORD="$FUNNY" \
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 \
pytest -q tests/test_tfrecord_integration.py
```

Run the real ROOT integration test:

```bash
GEANT4_ROOT_SAMPLE="$ROOT_SAMPLE" \
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 \
pytest -q tests/test_root_integration.py
```

Tests cover the fixed `64³`/±50 cm contract, Mu-Net normalization equivalence,
positive xyz orientation without swaps or reflections, physical landmark
mapping, exact selected-track fingerprints, segment-length accounting, unit
conversions, material truth, result round-trips, both likelihood modes, PoCA
count separation, TFRecord parsing, and ROOT detector-plane ordering.

## Configuration

[`configs/default.yaml`](configs/default.yaml) is the single default source for:

- geometry bounds, shape, and axis order;
- Highland constant and reference momentum;
- fixed population-average momentum and classical information policy;
- frozen Mu-Net dosage and feature normalization;
- PoCA coverage threshold;
- likelihood initialization, iteration count, and numerical epsilon;
- calibration mask erosion and intercept policy.

Pass an alternative read-only configuration with `--config PATH`. Any geometry
with a shape other than `64³` is rejected.

## Reproducibility

Each result records:

- absolute input path and format;
- TFRecord record index or ROOT tree;
- deterministic selection limit and seed;
- method and algorithm mode;
- frozen weight path and dosage for Mu-Net;
- likelihood iteration and convergence history;
- whether per-muon momentum was used and the assumed population-average value;
- geometry and units;
- calibration parameters, if applied.

The source data is never copied into generated output and never rewritten.

## Troubleshooting

### TensorFlow prints XLA or retracing messages

These messages come from the copied Mu-Net architecture and do not imply that
weights are being trained. Inference calls always use `training=False`.

### Mu-Net says there are not enough tracks

The frozen models require at least one complete chunk of 512 or 1024 tracks.
Select the matching dosage or provide more events.

### A calibration produces unexpected values

Confirm that reconstruction and calibration used the same likelihood mode,
geometry, event selection, material table, and units. Inspect `raw_estimator`
and `metadata_json` before refitting.

### A voxel is zero

Check `valid_mask` and `occupancy_count`. Zero in an unobserved voxel is storage,
not evidence that the material is air.

### ROOT branches are rejected

Use `muon-tomo inspect FILE.root`. The adapter reports missing branches and the
selected tree name explicitly.

## Provenance and licensing

Mu-Net architecture code and frozen weights were copied from the local
`Muon-Tomography-AI` project. That project is Apache-2.0 licensed; its complete
license is retained in `third_party/Muon-Tomography-AI-LICENSE`.

PoCA and likelihood were adapted from the local `muonTruck` reconstruction at
revision `8393853a47e8654552a62d1886fb6766d637782f`, including relevant local
research changes. No license file was found there, so redistribution beyond the
owner's environment requires confirmation of the applicable license.

See [`NOTICE`](NOTICE) for details.
