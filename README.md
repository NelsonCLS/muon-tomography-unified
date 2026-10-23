# Muon Tomography Unified

A reproducible framework for comparing **MuNet**, **PoCA**, and **likelihood**
reconstruction on the same selected muon tracks. It reads TFRecord, Geant4
ROOT, or detector-only NPZ inputs and produces inverse radiation length `1/X0` in `cm^-1` on a common
64 × 64 × 64 grid. A headless Geant4 simulator creates custom object datasets.

Start with [the project map](project_summary.md) for task-to-file navigation.
Scientific assumptions are explicit in [scientific contracts](docs/scientific_contracts.md).

## Install

Python 3.11–3.13 is supported. Clone and install from source:

```sh
git clone https://github.com/NelsonCLS/muon-tomography-unified.git
cd muon-tomography-unified
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]'
muon-tomo --help
```

`pip install -e .` provides classical reconstruction and ROOT I/O. TFRecord
reading requires `.[tfrecord]`; MuNet requires `.[munet]`; plots require `.[plot]`.
`.[all]` includes these and test dependencies. Geant4/CMake are separate native
requirements only for running the simulator. See [reference environments](environments/README.md).

## Try it without downloading a dataset

The repository includes code, frozen MuNet weights and example configurations.
**TFRecord/ROOT datasets and historical results are not included or automatically
downloaded.** Catalogue entries describe local research files; they do not make
those files available in a fresh checkout.

```sh
python examples/create_detector_sample.py --output data/derived/demo-tracks.npz
muon-tomo predict --input data/derived/demo-tracks.npz --method poca --no-truth --open
```

This deterministic toy sample demonstrates the file-to-plot workflow, not detector
physics or reconstruction accuracy. It contains only positions and directions,
with no truth labels or per-track momentum. For a smaller installation this demo
needs only `pip install -e '.[plot]'`; TensorFlow and Geant4 are unnecessary.
Use `--method all` with the full installation to exercise all three methods.

## Prediction on your dataset

Place your compatible Funny dataset at the path below, or substitute your own
TFRecord, ROOT or detector NPZ path. See the [data contract](docs/scientific_contracts.md)
and [detector example](examples/README.md) for units and endpoint conventions.

```sh
muon-tomo predict \
  --input dataset/voxels_prediction_funny.tfrecord \
  --record-index 0 --limit 8192 --seed 0 --method all --open
```

This creates a unique run under `runs/` with predictions, selected track IDs,
resolved settings, provenance, a cross-sectional plot, and an interactive 3D viewer.
Truth and metrics are included when requested and available. `--no-truth` skips
truth loading entirely. `--open` opens the saved visual artifacts after completion;
omit it on a headless computer.

If you have the local datasets at the catalogue paths, you may also use their IDs:

```sh
muon-tomo dataset list
muon-tomo predict --input legacy-funny --record-index 0 --method all
muon-tomo run list
```

Register a new local file without copying it:

```sh
muon-tomo dataset register /path/to/new.tfrecord --id my-new-sample
```

Local inputs under `dataset/` and `dataset_target_crossing/` and outputs under
`outputs/` are preserved and excluded from distribution. See [storage and provenance](docs/storage.md).

## Scientific defaults

- **MuNet:** frozen weights, 1024-track chunks, averaged for a larger selected dose.
- **PoCA:** a 32³ exposure-normalized estimate with separate Gaussian pooling of
  scattering signal and exposure (σ = 6.25 cm), expanded back to 64³.
- **Likelihood:** production full angular/displacement covariance, median update,
  ten iterations. Experimental candidates are separate and opt-in.
- Normal predictions are **not fitted or calibrated against object truth**.
- The historical fixed 1743.4601815 MeV/c prior is an **inverse-square effective
  momentum derived from Funny 0 simulation**, not an arithmetic mean or an
  independently measured cosmic spectrum. Changing this is a scientific change.
- Full-grid MSE uses all 262,144 voxels before plotting cuts. The standard viewer
  shows all voxels strictly above 0.8 cm⁻¹, including interiors.

The current methods retain the 64³ geometry for compatibility with pretrained
MuNet. It is not a claim of 1.56 cm physical reconstruction resolution. Detector
acceptance and track quality determine acquisition time; a selected count is not time.

## More workflows

- [Commands, existing artifacts, and lower-level workflows](docs/workflows.md)
- [Custom objects and simulator](docs/custom_simulation.md)
- [Architecture and extension points](docs/architecture.md)
- [Tests, CI, and numerical parity](docs/testing.md)
- [Experimental likelihood status and historical recipes](docs/research.md)
- [Contribution guidance](CONTRIBUTING.md)
- [References](docs/references.md) and [validation records](VALIDATION.md)

The package is research software. Upstream model/source provenance and the mixed
licensing situation are recorded in [LICENSE](LICENSE), [NOTICE](NOTICE), and
[third_party](third_party/). Preserve those records when redistributing.
