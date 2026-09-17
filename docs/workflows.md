# Common workflows

Activate the environment installed from this checkout. `python -m muon_tomo.cli`
is equivalent to the `muon-tomo` entry point. Set `MPLBACKEND=Agg` on headless
machines. List command options with `muon-tomo COMMAND --help`.

## Complete prediction

```sh
muon-tomo predict --input legacy-funny --record-index 0 --limit 8192 --seed 0 --method all --open
```

The initial catalogue names include `legacy-funny`, `legacy-main`,
`legacy-validation`, and their `-target-crossing` derivatives. A file path works
without registration. Methods are `all`, `munet`, `poca`, or `likelihood`.

Options include `--config override.yaml`, `--runs-dir /path/to/runs`,
`--catalog local-catalogue.json`, and `--no-truth`. Every invocation creates a new
run; no old result is overwritten or silently adopted as a cache. The default
8192 limit and seed 0 are reproducible only together with source identity, selected
rows, filtering, and chunk policy recorded in the run.

For real detector input, the versioned detector NPZ format, or an x-only TFRecord:

```sh
muon-tomo predict --input /path/to/detector.npz --method poca --no-truth
```

The output still includes the available method's plot and viewer. No truth MSE is
claimed. ROOT and detector NPZ input have no embedded truth. The frozen production scalar remains
historical simulation-derived prior information; detector-only input does not
make that prior externally validated.

## Configuration overrides

A YAML override changes only specified settings; unknown keys fail early:

```yaml
poca:
  grid_size: 64
  pooling_sigma_cm: 3.125
```

This changes scientific settings and is recorded as a distinct run, not presented
as the default PoCA. `physics.effective_momentum_mev` is the clearer alias for the
historical `average_momentum_mev` setting; conflicting simultaneous values fail.
The managed `predict` workflow fixes `munet.inference_seed` (default 0) because
the frozen network includes stochastic track placement even at inference. The
legacy `reconstruct` command retains its previous behavior. This seed is separate
from the input-selection `--seed`; both are recorded.

MuNet's assumed feature is a separate configuration field; do not accidentally
claim a common prior if changing only one method's settings.

## Replot saved arrays without reconstruction

Use a completed run directory in place of `RUN` below:

```sh
muon-tomo plot --truth RUN/evaluation/truth.npz RUN/predictions/munet.npz RUN/predictions/poca.npz RUN/predictions/likelihood.npz --output /tmp/cross_section.png
muon-tomo plot3d --truth RUN/evaluation/truth.npz RUN/predictions/munet.npz RUN/predictions/poca.npz RUN/predictions/likelihood.npz --threshold-cm-inv 0.8 --output /tmp/viewer.html
```

Do not pass historical calibration files. The new `predict` workflow explicitly
uses strict > 0.8, no display cap, and interior voxels. Consult `plot3d --help` for
legacy display controls rather than inferring old viewer settings from its image.
Do not write replacement figures into a completed managed run; choose a new path.

## Lower-level and simulation commands

Legacy `inspect`, `truth`, `reconstruct`, `compare`, `validate`, `plot`, `plot3d`, and
`stability` commands remain available for existing scripts. Historical `calibrate`
also remains available solely for deliberate reproduction, never normal prediction.
The legacy output naming/layout is unchanged. Existing result files now refuse
implicit replacement; choose a fresh output directory when rerunning a legacy
reconstruction. `plot` and `plot3d` now accept one
or more result files with optional `--truth`, so detector-only and single-method
results can be displayed directly. `run list --full` includes full provenance; the
default run list is compact.

For custom objects, follow [the simulator guide](custom_simulation.md). Generated
primary count, detector-accepted count, target-crossing count, and selected inference
count are different quantities. `generate-dataset` does not establish acquisition
time merely from the number of simulated primaries.

## Continuing experimental studies

```sh
python experiments/likelihood_physics_study/run.py extract --results-dir runs/physics-study-new
python experiments/likelihood_physics_study/run.py fit --results-dir runs/physics-study-new
python experiments/likelihood_physics_study/run.py evaluate --results-dir runs/physics-study-new
```

This is an explicit potentially long research workload. The fixed study still uses
its documented existing development detector inputs and validation TFRecord.
The directory format is its research protocol, not the standard prediction run
schema. Old `results/` cannot be reused after source/configuration changes; the
runner explains the mismatch and preserves the old results.
