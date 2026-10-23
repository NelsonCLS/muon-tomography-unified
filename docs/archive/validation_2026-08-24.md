# Validation record

This file records checks performed on the unified repository. It distinguishes
software-release validation from reconstruction results already recorded in
versioned configuration files.

## Release audit: 24 August 2026

The release audit used Python 3.13 and read the original TFRecord, Geant4 ROOT
file, and MuNet weights without modifying them.

The following checks passed:

- Ruff linting and formatting checks for first-party source and tests;
- 45 automated tests, including the real Funny TFRecord adapter, Geant4 ROOT
  adapter, frozen MuNet inference, geometry, unit, reconstruction, stability,
  plotting, and result-schema tests;
- source-distribution and wheel builds;
- Twine package-metadata validation;
- an unpacked-wheel smoke test confirming that the command-line code loads its
  packaged default configuration rather than relying on the source tree;
- Citation File Format 1.2 validation for `CITATION.cff`;
- an installed-environment dependency audit with no known vulnerabilities
  reported at the time of the check;
- Git whitespace validation and a current-tree scan for committed credentials
  and machine-specific absolute paths.

The real-input hashes matched `third_party/VALIDATED_INPUTS.sha256`:

```text
65c160c4e38ac8477b1fff494ed55a7f883e732f3c5c21d36237c83fbd205e00  voxels_prediction_funny.tfrecord
ae7fe70af88ed461a51b3723522ea50f39b4e76e1a61b61b5b181edcfbced9ac  cosmic_leadBox_200000.root
```

The frozen weights matched `third_party/MUNET_WEIGHTS.sha256`:

```text
6fc8764ddb899c969329f5c44f649da8872fc9dd669a8a5108485e7954ce67ec  model_512.weights.h5
d8383322563156697d93dacf1ffaa5caf0588a6c69096462908d9a6aba6e28d5  model_1024.weights.h5
```

TensorFlow emitted dependency deprecation warnings during MuNet inference on
Python 3.13; inference completed and its integration test passed.

## Spatial contract

Every reconstruction uses this fixed geometry:

```text
shape:       [64, 64, 64]
indexing:    volume[x, y, z]
axis order:  xyz; +x, +y, and +z increase with array index
bounds:      [-50, -50, -50] to [50, 50, 50] cm
voxel size:  [1.5625, 1.5625, 1.5625] cm
```

The frozen MuNet input transform is
`normalized = position_cm / 100 + 0.5`, which maps the reconstruction bounds to
`[0, 1]³`. Detector reference planes remain at `z = -100 cm` and `z = +100 cm`;
they define track endpoints rather than reconstruction bounds.

Automated spatial tests verify the MuNet normalization, physical voxel-index
mapping, positive axis orientation, closest-approach landmark mapping, and
selected-track fingerprints.

## Recorded Funny calibration result

`configs/funny-v4-robust-mlsd.calibration.json` records calibration fitted on
Funny record 0 and evaluated on record 1. Its held-out full-grid `64³` MSE
values, in `cm⁻²`, are:

```text
MuNet:       0.103476783
PoCA:        0.289265049
Likelihood:  0.364103652
```

These values apply only to the recorded sample, selection, estimator settings,
momentum policy, and calibration. They are not a general performance claim.

Likelihood/MLEM produces finite, schema-valid, geometry-aligned output. Its
recorded held-out MSE is higher than the corresponding MuNet and PoCA values,
so it is not presented as a trusted quantitative density reconstruction in the
current workflow. No unverified cause is assigned here.

## Geant4 ROOT input

The real ROOT integration test fits upper and lower hit segments, maps them to
the common track representation at the detector reference planes, and verifies
finite positive momentum values and the expected direction ordering. A ROOT
hit file has no embedded `64³` truth volume, so the adapter test does not claim
a truth-comparison result.

## Data immutability

The source repositories, TFRecord, ROOT file, and generated research outputs
were not modified by the release audit. Repository tests treat real inputs as
read-only and write temporary products only to test or output locations.
