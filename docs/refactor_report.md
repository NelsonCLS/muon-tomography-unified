# Refactor report — 18 September 2026

The readability, modularization, storage and workflow refactor is implemented.
The three production estimators retain their numerical calculations and frozen
weights. Existing datasets and historical results stay in place.

## What improved

- **Navigation:** `project_summary.md` is now a short task map. `AGENTS.md` records
  scientific invariants and working conventions. Separate guides explain science,
  architecture, storage, commands, tests and research. Previous long narratives
  are preserved in `docs/archive/`.
- **Modules:** CLI/configuration, input adapters, selections, momentum policies,
  reconstruction factories, evaluation, visualization, run storage and research
  now have separate boundaries. Legacy imports and commands remain available.
  Registering another method does not require a new CLI execution branch.
- **Data contracts:** measured detector positions/directions have an explicit
  momentum-free interface. TFRecord truth decoding is separate from inference;
  ROOT can use position branches only; versioned detector NPZ is supported.
  Stale compatibility caches, invalid IDs and malformed lineage are rejected.
- **Storage:** 13 datasets are registered in place, with parent relationships and
  existing row maps. Historical artifacts have an inventory. New predictions get
  unique run directories containing resolved settings, selection IDs, source and
  checkpoint identities, raw arrays, optional evaluation, PNG and HTML.
- **Scientific traceability:** output remains raw `1/X0` in `cm^-1`; MSE remains
  `cm^-2`. Normal prediction cannot apply truth calibration or per-track simulated
  momentum. Prior provenance is explicit: the preserved 1743.4601815 MeV/c scalar
  is historical simulation-derived effective momentum, not an external spectrum.
- **Plots:** any number of methods, optional truth, duplicate display names and
  non-cubic core geometry are supported. The default viewer includes every voxel
  strictly above 0.8 cm^-1, including interiors. Current solvers still explicitly
  require the original 64³ geometry.
- **Research and maintenance:** numerical experimental helpers are importable from
  `muon_tomo.research`; old module paths remain wrappers. Historical recipes are
  archived with hashes. The physics study refuses stale/unverifiable cached fits.
  CI, package contents, environment records and a repeatable workflow harness are
  included.

## Usage changes

From the repository, the shortest complete prediction is:

```sh
.venv/bin/muon-tomo predict --input legacy-funny --open
```

Its defaults are record 0, 8192 requested tracks, selection seed 0, all three
production methods and MuNet inference seed 0. The explicit equivalent is:

```sh
.venv/bin/muon-tomo predict --input dataset/voxels_prediction_funny.tfrecord --record-index 0 --limit 8192 --seed 0 --method all --open
```

Results are under `runs/<unique-id>/`: raw arrays in `predictions/`, truth/metrics
in `evaluation/`, cross section/viewer in `figures/`. `--runs-dir` selects another
location. `--method poca` selects one method; `--no-truth` prevents label loading.
Use a file path or a catalogue ID. Register/list with `dataset register` / `dataset
list`; inspect run manifests with `run list`.

Existing `reconstruct`, `compare`, `validate`, `plot`, `plot3d`, `stability`, and
simulation commands still work. Existing output files now refuse implicit
replacement: use a new output directory. Configuration files can be partial
overrides; unknown keys and inconsistent settings fail early. Explicit custom
checkpoint paths are resolved relative to the configuration file.

MuNet already contained random placement of unscattered tracks during inference.
The new managed workflow fixes and records its seed. The model and its calculations
are unchanged; legacy `reconstruct` keeps its earlier random behavior.

## Review and verification

- **147 tests passed, no skips or failures**, including real TFRecord/MuNet and
  ROOT integration, native seeded Geant4, synthetic research gradients/recovery,
  contracts, storage, selection lineage, cache invalidation and visualization.
- **21 public command workflows passed**: creation/packing, inspection/truth export,
  all-method reconstruction, comparison/validation, saved plots/viewers, single
  method without truth, detector-error stability, catalogue operations, managed
  predictions, real Funny 0, native generation and generated-data prediction.
- Additional complete detector-NPZ and position-only ROOT predictions passed.
  Filtering retained 1024/1024 control tracks and rejected all ten injected misses
  from a 1034-track input, preserving labels and the original source.
- **Numerical parity:** on original Funny 0, target-filtered Funny 0 and a generated
  lead sample, each with 8192 selected tracks, PoCA/likelihood density, raw estimator,
  occupancy and masks are byte-identical to the preserved baseline. MuNet primary
  density is byte-identical with the same inference seed. Model file hashes match.
- All three reconstruction function ASTs match the baseline apart from an explicit
  equal-length iteration assertion. No calibration or physics tuning was introduced.
- Ruff lint/format passed. Wheel and source distribution build/validation passed;
  the final installed wheel completed all three methods on Funny 0 with 8192 tracks
  outside the checkout, reproducing the baseline primary arrays byte-for-byte.
- Independent module reviews checked adapters, configuration, workflow boundaries,
  plotting, cache behavior, packaging and portability. Review fixes include stale
  detector-cache rejection, scalar prior validation, empty-coverage metrics,
  arbitrary-grid plotting defaults, checkpoint consistency and the Git case rename.

The ten test warnings originate in the existing TensorFlow/gast dependency.
Remote CI, Linux/CUDA execution and full large-dataset scans were not run locally.
The source spectrum and reconstruction quality have not been retuned. Environment
files record the tested setup; they are not a solved lock for every platform.

## Evidence and reproduction

- [Structured verification](../benchmarks/refactor_validation_2026-09-18.json)
- [Packaging evidence](../benchmarks/package_smoke_2026-09-18.json)
- [Workflow harness](../benchmarks/verify_workflows.py)
- [Command evidence and generated artifacts](../outputs/refactor-validation/cli-workflows/)
- [Final test output](../outputs/refactor-validation/final_tests.txt)

Baseline source is preserved in local Git commit `30c67e2`; a temporary complete
snapshot was also retained before edits. Large source files and historical outputs
were not moved or rehashed. All 13 registered datasets retain their recorded file
sizes and modification times. This preservation check is not a claim that every
byte of every large dataset was reread.
