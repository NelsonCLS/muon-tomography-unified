# Validation record

Validation was performed in an isolated Python 3.13 environment created from
`pyproject.toml`.

## Automated suite

```text
9 passed, 2 integration tests skipped when sample paths were not supplied
```

The skipped tests were then executed explicitly with their local sample paths:

```text
funny TFRecord integration: 1 passed
Geant4 ROOT integration:    1 passed
```

## End-to-end TFRecord

Both funny records were reconstructed with the same deterministic 8,192-track
selection for Mu-Net, PoCA, and angular likelihood. Each result contained a
finite `(64,64,64)` `inverse_radiation_length_cm_inv` volume.

The formal validator returned `status: success` for both records with:

```text
shape:       [64, 64, 64]
axis order:  xyz
bounds:      [-100, -100, -100] to [100, 100, 100] cm
voxel size:  [3.125, 3.125, 3.125] cm
methods:     likelihood, munet, poca
```

Calibration `funny-v1` was fitted on record 0 and its report includes record 1
metrics. This operation changed no Mu-Net weight and no source dataset.

## End-to-end Geant4 ROOT

The local `cosmic_leadBox_200000.root` hit tree was converted to the common
track table. Frozen Mu-Net, PoCA, and full covariance likelihood all completed.
The same formal `64³` geometry validator returned `status: success`.

## Immutability

All new source, environments, calibration reports, and outputs were created
under `muon-tomography-unified`. The existing `Muon-Tomography-AI`, `muonTruck`,
TFRecord, and ROOT inputs were read only.
