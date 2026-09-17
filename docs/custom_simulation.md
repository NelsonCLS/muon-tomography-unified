# Custom objects and Geant4 datasets

The unified package now contains voxel creation, a headless Geant4 simulation,
and an exporter for the same GZIP TFRecord `x`/`y` format as the existing datasets.
No sibling checkout or hard-coded home directory is required.

## Quick start

Install this project's `munet` extra for TensorFlow. Building the simulator also
requires CMake, a C++ compiler, Geant4 and its physics data. Activate the Geant4
installation's environment first. No Geant4 dependency is needed for normal
reconstruction, voxel creation, or packing an existing simulation CSV.

From the repository root:

```bash
muon-tomo build-simulator --build-dir build/geant4
muon-tomo generate-dataset \
  --scene configs/scenes/water_lead_boxes.yaml configs/scenes/mixed_shapes.yaml \
  --source-config configs/scenes/source.yaml \
  --events 20000 --seed 12345 \
  --executable build/geant4/muon_tomo_sim \
  --output dataset/generated_custom/examples.tfrecord
muon-tomo inspect dataset/generated_custom/examples.tfrecord
muon-tomo reconstruct dataset/generated_custom/examples.tfrecord \
  --record-index 0 --method poca --output-dir outputs/custom-example
```

If CMake cannot find Geant4, supply
`--geant4-dir /path/to/lib/cmake/Geant4`. Use `--cxx /path/to/compiler` when the
Geant4 installation requires a matching compiler. On Windows supply the built
`muon_tomo_sim.exe` path to `--executable`.

`--events` counts **generated primaries per record**, not accepted tracks or a
measurement duration. Every surviving primary muon entering the lower detector
is retained, including tracks missing the target cube. Attenuation and detector
acceptance determine the number of rows. `--records-per-scene 2` generates two
independent samples of each scene with consecutive seeds. Dataset records are
ordered by scene, then sample. Existing output files/run directories are refused.

The TFRecord has a `.tfrecord.json` manifest. The adjacent `examples_runs/`
directory retains exact scene definitions, voxel files, CSV tracks, source macros,
Geant4 logs, seeds, source/executable hashes and a per-run manifest. No labels are
used to select tracks. Source parameters and geometry are recorded separately from
the fixed TFRecord schema. Generation failures return a nonzero status and retain
logs. A failed packing operation does not publish a partial TFRecord.

## Define a custom shape

All lengths are centimetres. The target is `[-50,50]` on each axis, with 64 cells
per side (1.5625 cm). Material labels use the existing 22-entry table: air=0,
water=3, silver=17, lead=19 and uranium=21, for example.

```yaml
version: 1
name: lead_shell_with_water_core
background: air
shapes:
  - type: hollow_box
    material: lead
    size_cm: [50, 50, 50]
    wall_cm: 4
    center_cm: [0, 0, 0]
  - type: ellipsoid
    material: water
    radii_cm: [10, 12, 15]
    center_cm: [0, 0, 0]
    rotation_deg: [0, 30, 0]
```

Supported shapes:

- `box`: `size_cm` (three full side lengths).
- `hollow_box`: `size_cm` and `wall_cm`.
- `sphere`: `radius_cm`.
- `ellipsoid`: `radii_cm` (three semiaxes).
- `cylinder`: `radius_cm`, `height_cm`, optional `inner_radius_cm` for a tube.
  Its unrotated axis is z.

`center_cm` and `rotation_deg` default to zero. Rotations are active rotations
about the shape centre, applied x then y then z (`Rz @ Ry @ Rx`). Shapes are
sampled at voxel centres and cropped to the target cube. Features smaller than a
voxel may disappear; the simulation uses the resulting voxelized solid, not an
analytic surface. Empty sampled shapes are rejected. Overlaps with earlier
painted shapes raise an error unless that shape specifies `overlap: replace`.
Painting `material: air` with replacement carves a cavity. A hollow box only
paints its walls; it leaves its interior unchanged.

```bash
muon-tomo create-voxels configs/scenes/mixed_shapes.yaml --output outputs/object.npz
muon-tomo create-voxels configs/scenes/mixed_shapes.yaml --output outputs/object.txt
```

For arbitrary shapes, Python callers can construct any integer array of shape
`(64,64,64)` and call `muon_tomo.simulation.save_voxels`. The low-level
`run_simulation` API accepts `.npy`, `.npz` (`material_labels` key), and `.txt`.
Flat text uses NumPy C order: z varies fastest, then y, then x. Noninteger values,
wrong sizes and unknown material IDs are rejected. There is no CAD/STL importer.

## Simulation model and limits

The backend uses Geant4's `FTFP_BERT` physics list. Object voxels and surrounding
air use the material IDs defined in the current Mu-Net dataset contract. Unlike
the old generator, cell centres are placed at `-50 + (index+0.5)*1.5625 cm` so
simulated solids and reconstruction truth share the same grid.

Incoming states are defined at z=+100 cm. Outgoing states are recorded at the
entrance to an ideal 2 m x 2 m detector plane at z=-100 cm. Only the primary
`mu-` is recorded, once per event. This is **ideal tracking geometry**, not an RPC
hit/coincidence/electronics model. It does not simulate RPC efficiency, strip
readout, timing, or fitted-track resolution. Detector-error studies remain a
separate workflow. It is not the 1 m² RPC apparatus by default.

The supplied source uses a uniform 2 m x 2 m launch square, Geant4 GPS cosine
angular distribution truncated at 45 degrees, and a kinetic-energy power law
with exponent -2.7 between 1 and 200 GeV. These are explicit simulation settings,
not a calibrated cosmic spectrum. `configs/scenes/source.yaml` exposes the source
parameters. `angular_mode: vertical` and `energy_mode: mono` allow controls.
The power-law exponent uses `/gps/ene/alpha`; the old script's `biasAlpha` did not
specify the physical energy law. The new simulator also omits the legacy
straight-ray rejection, which could reject a ray before scattering carried it
into acceptance. Therefore **format compatibility does not imply the same
training distribution as existing files**. Record these differences in studies.

`y` retains the existing nominal material-to-1/X0 mapping so all current loaders
agree. Geant4 computes transport from its own material database. The simulation
log reports Geant4 radiation lengths for comparison; they are not silently used
to redefine the canonical labels or calibrate predictions.

## TFRecord format and existing CSV conversion

Each `tf.train.Example` has exactly two byte features, `x` and `y`, encoded using
`tf.io.serialize_tensor`:

- `x`: float64 `(N,13)` from new simulations. Columns 0:3 are the **lower/outgoing
  position in mm**; 3:6 the **outgoing momentum in MeV/c**; 6:9 the **upper/incoming
  position in mm**; 9:12 the **incoming unit direction**; column 12 is **initial
  kinetic energy in MeV**. This lower-first ordering matches current datasets,
  despite the historical TrackTable field names.
- `y`: int32 `(64,64,64)`, indexed `[x,y,z]`, using the existing material IDs.
- Compression: GZIP, with one object/sample per record.

```bash
muon-tomo pack-simulation \
  --csv run1/tracks.csv run2/tracks.csv \
  --voxels run1/voxels.txt run2/voxels.txt \
  --output dataset/generated_custom/imported.tfrecord
```

The converter accepts the new named CSV header or legacy numeric-only 12/13-column
CSV files. It never discards the first numeric row as a header and never reorders
columns. Pair the correct truth with each CSV; use run manifests to establish
provenance for external files. Twelve-column legacy data stays twelve-column.
No per-track energy/momentum is invented.

## Validation

```bash
pytest -q tests/test_simulation.py
MUON_TOMO_SIMULATOR="$PWD/build/geant4/muon_tomo_sim" pytest -q tests/test_simulation.py
```

Tests cover physical voxel alignment, rotation and cavities, label/file validation,
CSV first-row preservation, TFRecord dtype/order/unit round trips with the current
loader, seeded native simulation and an end-to-end custom-scene smoke test.
