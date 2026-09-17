# Muon Tomography Unified

Muon tomography uses the deflection of naturally occurring cosmic muons to
infer the distribution of material inside an object. This repository provides
one reproducible framework for running and comparing three reconstruction
methods on the same tracks, geometry, physical target, and evaluation metrics:

- **MuNet** — frozen neural-network inference;
- **PoCA** — point-of-closest-approach reconstruction;
- **Likelihood / MLEM** — iterative maximum-likelihood reconstruction.

The project is inference-focused. It does not train or fine-tune MuNet, and it
does not modify the source `Muon-Tomography-AI` repository, the source
`muonTruck` repository, or the input datasets.

## What this project provides

The command-line workflow can:

- read TensorFlow TFRecord samples and Geant4 ROOT hit data;
- convert both input types to one common track-table representation;
- reconstruct a common `64 × 64 × 64` volume with MuNet, PoCA, and MLEM;
- express prediction and simulation truth in inverse radiation length,
  \(1/X_0\), in `cm⁻¹`;
- calibrate method outputs to the common target;
- compare truth and predictions using full-volume metrics;
- create top, side, and full-depth 2D projections;
- create a self-contained, interactive Plotly 3D voxel viewer;
- run paired clean-versus-errored detector-stability tests.

The central goal is a fair comparison: every method uses the same selected
tracks and the same reconstruction geometry, while method-specific outputs are
stored alongside their coverage and metadata.

## Reconstruction pipeline

```text
TFRecord or Geant4 ROOT input
            │
            ▼
Common TrackTable + truth volume
            │
     ┌──────┼──────┐
     ▼      ▼      ▼
   MuNet   PoCA   Likelihood/MLEM
     │      │      │
     └──────┼──────┘
            ▼
Calibration, validation, full-grid metrics,
2D projections, and interactive 3D viewer
```

## Common geometry and physical target

All reconstructions use one mandatory geometry:

| Property | Value |
|---|---|
| Array convention | `volume[x, y, z]` |
| Grid shape | `64 × 64 × 64` |
| Axis order | `xyz` |
| Lower bound | `(-50, -50, -50)` cm |
| Upper bound | `(50, 50, 50)` cm |
| Voxel size | `(1.5625, 1.5625, 1.5625)` cm |
| Detector reference planes | `z = -100 cm` and `z = +100 cm` |

The primary quantity for both truth and reconstruction is

\[
\rho_X = \frac{1}{X_0}\quad[\mathrm{cm}^{-1}],
\]

where \(X_0\) is material radiation length. Air is represented by zero.

For the Funny sample, the truth materials of interest are:

| Label | Material | \(1/X_0\) (`cm⁻¹`) |
|---:|---|---:|
| 0 | Air | 0.000000 |
| 17 | Silver | 1.170569 |
| 19 | Lead | 1.781790 |
| 21 | Uranium | 3.158333 |

## Methods

### MuNet

MuNet is run with copied, frozen weights. The supported workflow performs
inference only; it does not call a training routine or write new weights.

The adapter reproduces the verified 14-feature inference input, including the
frozen model's zero momentum-estimate feature. MuNet processes complete dosage
chunks and averages their predictions.

### PoCA

PoCA computes a point of closest approach between incoming and outgoing fitted
track lines, then deposits a scattering estimate in the corresponding voxel.
`occupancy_count` records the number of localized events and is not itself the
physical reconstruction. Zero-occupancy voxels remain zero.

### Likelihood / MLEM

The `full` likelihood mode implements the muonTruck-style `2 × 2` covariance
update using projected angular and displacement observations. It stores a
voxelized inverse-radiation-length estimate, coverage information, convergence
history, and configuration metadata. The validated default uses ten iterations
and a voxel-wise median update statistic.

### Momentum information policy

For the detector-realistic comparison, PoCA and likelihood do not use each
event's simulated momentum magnitude. Both use the same fixed effective
population momentum, recorded in output metadata. MuNet uses its verified
zero-momentum feature. This policy is intended for a detector setup in which
track positions and directions are measured but momentum is not.

## Input formats

### TensorFlow TFRecord

The TFRecord adapter reads GZIP-compressed `tf.train.Example` records with:

| Field | Content |
|---|---|
| `x` | Floating-point track table shaped `(N, >=12)` |
| `y` | Integer material-label volume shaped `(64, 64, 64)` |

The required track columns are:

| Columns | Meaning | Stored unit |
|---|---|---|
| `0:3` | Entry position | mm |
| `3:6` | Entry momentum vector | MeV |
| `6:9` | Exit position | mm |
| `9:12` | Exit direction | unit vector |

Positions are converted to centimetres and directions are normalized in the
common track table.

### Geant4 ROOT

The ROOT adapter reads the `hits` tree and requires event number, hit position,
and hit momentum branches. It fits upper and lower 3D track segments and
intersects those lines with the fixed detector reference planes. The resulting
records use the same common track-table representation as TFRecord input.

A ROOT hit file does not itself contain the required voxel truth volume; a
truth comparison therefore requires a separate truth NPZ with the identical
geometry.

## Installation

Python 3.11–3.13 is supported.

```bash
git clone <repository-url>
cd muon-tomography-unified

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
```

Confirm that the CLI is available:

```bash
muon-tomo --help
```

Smaller installations are also supported:

```bash
python -m pip install -e .          # classical methods and ROOT I/O
python -m pip install -e '.[munet]' # add TensorFlow and MuNet
python -m pip install -e '.[test]'  # add test dependencies
```

## Quick start

Input datasets are intentionally not included in this repository. Set paths to
your local copies:

```bash
export FUNNY_TFRECORD=/path/to/voxels_prediction_funny.tfrecord
export ROOT_SAMPLE=/path/to/cosmic_leadBox_200000.root
```

Inspect the TFRecord:

```bash
muon-tomo inspect "$FUNNY_TFRECORD"
```

Run all methods on Funny record 0 with a deterministic 8,192-track selection:

```bash
muon-tomo reconstruct "$FUNNY_TFRECORD" \
  --record-index 0 \
  --method all \
  --limit 8192 \
  --seed 0 \
  --max-chunks 8 \
  --iterations 10 \
  --likelihood-mode full \
  --classical-momentum-policy mean \
  --average-momentum-mev 1743.4601815051958 \
  --output-dir outputs/funny
```

Validate that all reconstruction outputs share the selected tracks and grid:

```bash
muon-tomo validate \
  outputs/funny/record_0_munet.npz \
  outputs/funny/record_0_poca.npz \
  outputs/funny/record_0_likelihood.npz \
  --truth outputs/funny/record_0_truth.npz \
  --output outputs/funny/record_0_validation.json
```

Compute full-grid metrics:

```bash
muon-tomo compare \
  --truth outputs/funny/record_0_truth.npz \
  outputs/funny/record_0_munet.npz \
  outputs/funny/record_0_poca.npz \
  outputs/funny/record_0_likelihood.npz \
  --output outputs/funny/record_0_metrics.json
```

## Calibration and evaluation

The repository supports a versioned affine calibration from method-native
output to \(1/X_0\). Calibration is post-processing only: it does not retrain
MuNet or alter frozen weights.

The documented Funny protocol fits calibration on record 0 and evaluates it on
record 1:

```bash
muon-tomo calibrate \
  --dataset "$FUNNY_TFRECORD" \
  --results-dir outputs/funny \
  --fit-records 0 \
  --validate-records 1 \
  --calibration-id funny-v4-robust-mlsd \
  --output configs/funny-v4-robust-mlsd.calibration.json
```

Comparisons report MSE over the complete aligned `64³` volume. A displayed
slice or slab never changes the evaluation scope.

## Stability test

The stability workflow creates one standalone, reproducible errored TFRecord,
then compares clean and errored predictions using the same truth, configuration,
and calibration.

The documented error model is:

| Variable | Perturbation |
|---|---|
| Entry/exit `x, y` positions | Independent Gaussian error, \(\sigma=0.29\) cm |
| Entry/exit `z` positions | Unchanged |
| Entry/exit directions | Gaussian angular error, \(\sigma=0.5^\circ\), then renormalized |
| Momentum magnitudes | Unchanged in the derived record |

Example:

```bash
muon-tomo stability \
  --dataset "$FUNNY_TFRECORD" \
  --record-index 0 \
  --limit 8192 \
  --position-sigma-cm 0.29 \
  --angular-sigma-deg 0.5 \
  --error-seed 20260803 \
  --munet-seed 20260803 \
  --iterations 10 \
  --calibration configs/funny-v4-robust-mlsd.calibration.json \
  --output-dir outputs/stability/funny_record0_8192
```

The output directory includes a `stability_mse_comparison.json` report.

## Plots and interactive 3D viewer

### 2D comparison plot

Create a four-panel top view—truth, MuNet, PoCA, and likelihood—as a seven
layer average around `z = 32`:

```bash
muon-tomo plot \
  --truth outputs/funny/record_0_truth.npz \
  outputs/funny/record_0_munet.npz \
  outputs/funny/record_0_poca.npz \
  outputs/funny/record_0_likelihood.npz \
  --calibration configs/funny-v4-robust-mlsd.calibration.json \
  --center-index 32 \
  --slab-layers 7 \
  --projection-axis z \
  --output outputs/figures/funny_record0_top_view.png
```

The truth panel includes the material values for the relevant Funny sample.
The plotter also supports a high-contrast material key for air, silver, lead,
and uranium. All 2-D reconstruction comparison panels use a fixed physical color
scale of **0–3.2 cm⁻¹**, including slab and full-depth views. Values above 3.2
saturate visually; this never clips the reconstruction arrays or MSE calculation.
Panel titles report the full-grid MSE, not slice-only MSE.

Use one of the following options for full-depth views:

```text
--full-depth --projection-axis z  # x–y top view
--full-depth --projection-axis x  # y–z side view
--full-depth --projection-axis y  # x–z side view
```

### Interactive 3D viewer

Generate a self-contained HTML viewer:

```bash
muon-tomo plot3d \
  --truth outputs/funny/record_0_truth.npz \
  outputs/funny/record_0_munet.npz \
  outputs/funny/record_0_poca.npz \
  outputs/funny/record_0_likelihood.npz \
  --calibration configs/funny-v4-robust-mlsd.calibration.json \
  --threshold-cm-inv 0.7 \
  --opacity 0.60 \
  --max-cubes 12000 \
  --output outputs/figures/funny_record0_3d_voxels.html
```

Open the result on macOS:

```bash
open outputs/figures/funny_record0_3d_voxels.html
```

Use the dropdown to select truth or a reconstruction, drag to rotate, scroll
to zoom, and hover to inspect voxel values. By default, interior voxels are
included, values below `0.7 cm⁻¹` are transparent, and opacity is capped at
`0.60`. Use `--surface-only` to reduce the number of displayed cubes.

## Output format

Each reconstruction is stored as a compressed NPZ file:

| Key | Shape | Meaning |
|---|---|---|
| `inverse_radiation_length_cm_inv` | `64³` | Primary physical reconstruction |
| `radiation_length_cm` | `64³` | Derived \(X_0\); infinite where prediction is zero |
| `scattering_density_ref_mrad2_per_cm` | `64³` | Derived reference scattering density |
| `raw_estimator` | `64³` | Method-native output before calibration |
| `occupancy_count` | `64³` | Method-specific coverage count |
| `valid_mask` | `64³` | Quantitative-comparison mask |
| `uncertainty` | `64³`, optional | Method uncertainty, if available |
| `metadata_json` | scalar | Geometry, units, settings, source, and track fingerprint |

Truth NPZ files contain `material_labels`,
`inverse_radiation_length_cm_inv`, and `metadata_json`.

## Validation and testing

The validator checks shared geometry, axis order, selected-track fingerprint,
and the declared shared classical momentum policy.

Run the portable test suite:

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 pytest -q
```

Optional integration tests require local input data:

```bash
FUNNY_TFRECORD="$FUNNY_TFRECORD" pytest -q tests/test_tfrecord_integration.py
GEANT4_ROOT_SAMPLE="$ROOT_SAMPLE" pytest -q tests/test_root_integration.py
```

The tests cover geometry, units, truth mapping, reconstruction result schema,
track selection, plotting, the 3D viewer, stability data, TFRecord parsing,
and ROOT adaptation.

## Known limitation: likelihood/MLEM

The likelihood/MLEM workflow runs successfully and produces schema-valid,
geometry-aligned output. In the recorded Funny comparisons, however, its
full-grid MSE is higher than the corresponding MuNet and PoCA results, and it
is not currently presented as a trusted quantitative material-density
reconstruction.

The repository retains likelihood/MLEM because it is part of the unified
comparison and stability workflow. This README intentionally does not assign an
unverified explanation for the observed performance.

## Repository layout

```text
muon-tomography-unified/
├── configs/              shared settings and versioned calibration files
├── LICENSES/             complete reusable licence texts
├── src/muon_tomo/
│   ├── _vendor/munet/    frozen MuNet architecture and weights
│   ├── configs/          packaged runtime default configuration
│   ├── io/               TFRecord and ROOT adapters
│   ├── reconstruction/   MuNet, PoCA, and likelihood implementations
│   ├── plotting.py       2D comparison plots
│   ├── plotting3d.py     self-contained Plotly viewer
│   ├── stability.py      fixed detector-error workflow
│   └── cli.py            unified command-line interface
├── tests/                unit, synthetic, and integration tests
├── third_party/          provenance and license records
├── CONTRIBUTING.md       development and review requirements
├── VALIDATION.md         recorded validation notes
├── NOTICE                source attribution
└── pyproject.toml        package definition
```

Generated files are written under `outputs/`, which is excluded from Git.

## Data, reproducibility, and provenance

- Input TFRecords, ROOT files, and generated outputs are not distributed with
  the repository.
- MuNet weights are stored in the repository and their checksums are recorded
  under `third_party/`.
- Every reconstruction records geometry, units, configuration, and a selected
  track fingerprint in metadata.
- The source projects and copied components are documented in `NOTICE` and
  `third_party/`.

Before publishing a fork, do not add datasets or generated outputs unless you
have explicit permission to redistribute them.

## License and acknowledgements

This repository contains components under different licensing conditions. The
original unified integration code and the vendored MuNet implementation are
available under Apache-2.0. The licence for the adapted muonTruck material has
not been confirmed, so the complete repository must not be redistributed
publicly until permission or a compatible upstream licence is established.
See [LICENSE](LICENSE) and [NOTICE](NOTICE) for the exact component boundaries.

MuNet code and frozen-weight provenance are documented in
[NOTICE](NOTICE) and [third_party](third_party/). The likelihood implementation
adapts the established muonTruck covariance reconstruction structure within the
common geometry and result schema used here.

### Upstream method attribution

This project uses the MuNet architecture and frozen weights from the original
MuNet work. It also adapts the PoCA and likelihood implementation from the
original muonTruck research source. Please acknowledge both upstream projects
when reporting results obtained with this repository.

**MuNet paper**

> Lim, Li Xin Jed, and Ziming Qiu. “μ-Net: ConvNext-Based U-Nets for Cosmic
> Muon Tomography.” *arXiv preprint* arXiv:2312.17265 (2023).
> [https://arxiv.org/abs/2312.17265](https://arxiv.org/abs/2312.17265)

The associated source implementation is
[jedlimlx/Muon-Tomography-AI](https://github.com/jedlimlx/Muon-Tomography-AI),
released under Apache-2.0. This unified repository does not claim authorship of
MuNet; it integrates the supplied frozen model into a common comparison
workflow.

**PoCA and likelihood / MLEM source**

> K. Fung. *muonTruck* research code, `recon/generate_tfrecord.py`.
> GitHub repository. Accessed 24 August 2026.
> [https://github.com/kfung010/muonTruck](https://github.com/kfung010/muonTruck)

The PoCA and likelihood components in this repository are adapted from that
source. This is a software citation rather than a paper citation; if the
muonTruck authors provide a preferred scholarly reference, use it in addition
to the source-code citation above.

GitHub and compatible reference managers can also read the repository's
[CITATION.cff](CITATION.cff). Development and review instructions are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Custom objects and simulated datasets

Create material grids, run the bundled headless Geant4 backend, and export the
existing GZIP TFRecord x/y format using `create-voxels`, `build-simulator`,
`generate-dataset`, and `pack-simulation`. See [custom simulation](docs/custom_simulation.md)
for example scenes, build instructions, source settings, provenance and physical-model limits.
The simulator is optional; existing reconstruction workflows need no Geant4 installation.
