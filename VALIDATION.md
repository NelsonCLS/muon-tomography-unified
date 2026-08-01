# Validation record

Validation was performed in an isolated Python 3.13 environment created from
`pyproject.toml`. The source TFRecord, Geant4 ROOT file, source repositories,
and frozen Mu-Net weights were read only.

## Corrected spatial contract

The frozen Mu-Net input transform is
`normalized = position_cm / 100 + 0.5`. Therefore its normalized output domain
`[0,1]³` is physically `[-50,+50]³` cm. Earlier classical reconstructions used
`[-100,+100]³` cm; that factor-of-two scale error made PoCA and likelihood
objects appear smaller than truth. The corrected mandatory geometry is:

```text
shape:       [64, 64, 64]
indexing:    volume[x,y,z]
axis order:  xyz; +x, +y, and +z increase with array index
bounds:      [-50, -50, -50] to [50, 50, 50] cm
voxel size:  [1.5625, 1.5625, 1.5625] cm
```

Detector reference planes remain at z=-100 cm and z=+100 cm. They define track
endpoints, not voxel-cube bounds. The ROOT adapter stores the lower point first
and upper point second, matching the TFRecord and frozen Mu-Net feature order.

## Automated suite

The complete suite, including both real-data integration tests, passed:

```text
27 passed
```

Spatial tests prove Mu-Net normalized coordinates and physical voxel indices
are equivalent, positive xyz axes are neither swapped nor reflected, a known
physical closest-approach landmark maps to its expected voxel, and changing the
track count or ordering changes the selection fingerprint.

## End-to-end TFRecord

Both funny records were reconstructed with the same exact ordered 8,192-track
selection for Mu-Net, PoCA, and angular likelihood. Each method produced a
finite `(64,64,64)` `inverse_radiation_length_cm_inv` volume. Formal validation
against exported truth returned `status: success`,
`truth_geometry_matches: true`, and these per-record fingerprints:

```text
record 0: df466b5ec169d2dbf83ff03afafb0f46b7e4b2f432e81857b0f21f28444658d5
record 1: 1eb75ea74113024ded3fb38eb6f9a6ef882ed2b654170aa1e15bbd3fcc1c1608
```

On record 0, all methods received those same 8,192 events. Under the corrected
target cube, 1,648 events had an in-grid PoCA point and 2,744 straight chords
crossed the cube for likelihood. Mu-Net consumes all 8,192 as required by its
frozen inference path. These differences are algorithm-specific acceptance,
not different input selections or coordinate systems.

Calibration `funny-v2-spatial` was fitted on record 0 and evaluated on record
1. Held-out full-grid `(64³)` MSE in cm⁻² is:

```text
Mu-Net:      0.103424883
PoCA:        0.329745306
Likelihood:  0.309302788
```

Likelihood retains its native cm⁻¹ identity calibration. On the corrected
geometry, the attempted material-median affine map amplified likelihood by
7.39× and worsened held-out MSE from 0.309 to 6.866, so it was rejected.

## End-to-end Geant4 ROOT

The local `cosmic_leadBox_200000.root` hit tree was converted to the common
track table. Frozen Mu-Net, PoCA, and full-covariance likelihood all completed
on the corrected geometry. Formal validation returned `status: success` and a
shared ordered-track fingerprint:

```text
b256b9846f145046400d8edbf67a961ae9b52692466449ca39d4024f76d1da86
```

No truth geometry was asserted for this ROOT smoke test because the input has
no embedded voxel truth (`truth_geometry_checked: false`).

## Interpretation boundary

The geometry, scale, orientation, position, and exact method inputs are now
verified consistently. This does not make the three estimators equally
resolved. At 8,192 tracks PoCA remains sparse, and likelihood remains broad and
noisy—especially along z—because of their coverage and inverse-problem limits.
Those residual differences should not be interpreted as a remaining spatial
coordinate mismatch.

## Immutability

All new source, environments, calibration reports, and outputs were created
under `muon-tomography-unified`. The existing `Muon-Tomography-AI`, `muonTruck`,
TFRecord, and ROOT inputs were not modified.
