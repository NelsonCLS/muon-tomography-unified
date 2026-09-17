# Architecture and extension guide

The refactor separates orchestration and storage from numerical reconstruction.
Existing numerical methods and saved NPZ formats stay compatible.

```text
CLI / config
    -> dataset catalogue -> detector-track loading -> selection/provenance
    -> prediction workflow -> method registry -> reconstruction methods
    -> run storage -> raw predictions
    -> evaluation (optional truth) -> plots/viewer

simulation -> generated dataset + source/scene manifest -> same loading path
research helpers -> shared geometry + explicit experimental solver
```

## Boundaries

- `cli.py` preserves the public command entry point and legacy commands.
- `config.py` resolves and validates settings. Defaults live in packaged
  `configs/default.yaml`; the repository copy is checked for equality.
- `data/` owns dataset identity, detector-only reads, and row-selection provenance.
- `artifacts/` owns new run directories and manifests. Methods do not choose paths.
- `workflows/` coordinates loading, identical selections, reconstruction, and
  artifact production. Truth is loaded for evaluation separately from inference.
- `reconstruction/registry.py` is the method-selection/factory boundary. The three
  existing implementation files retain their mathematical behavior.
- `geometry.py` provides generic grid validation/indexing/tracing; production
  reconstruction constructors explicitly enforce the canonical MuNet-compatible
  grid. `reconstruction/paths.py` owns shared path geometry, with `common.py`
  preserving old import paths.
- `types.py` retains legacy track/truth/result structures and NPZ compatibility.
- `evaluation/metrics.py` owns scoring; `compare.py` is its compatibility import.
- `visualization/` owns shared panels, sections, and voxel viewers; `plotting.py`
  and `plotting3d.py` preserve old imports. Display transforms must not mutate
  predictions or define scoring support.
- `simulation/` builds objects, invokes the Geant4 executable, and writes records.
- `research/` is an importable namespace for explicit experiments; production
  must never choose it implicitly. Runners and report logic remain `experiments/`.
- `_vendor/` and its frozen weights are preserved upstream code/resources.

## Add a reconstruction method

1. Implement the solver under `reconstruction/` for a supported method, or under
   `research/` while experimental. Take track inputs and explicit configuration;
   do not read dataset files, truth, global mutable settings, or output paths.
2. Return the existing `ReconstructionResult` contract:64³ density, raw estimator,
   occupancy/coverage diagnostics, validity mask, geometry, metadata.
3. Declare units, algorithm/version, momentum prior and provenance, regularization,
   used-track count, and limitations. A new scattering-power quantity must not be
   mislabeled `1/X0`.
4. Add the method to the registry with a factory and explicit production/experimental
   status. `all` selects production entries only; experimental IDs are opt-in. Keep
   orchestration generic; do not add an independent dataset sampler per method.
5. Add analytical/contract tests and a fixed-selection comparison. Changing physics
   is separately reviewed from code movement. Update scientific/configuration docs.

The grid core can represent other shapes/bounds; existing reconstruction methods
explicitly retain canonical 64³ constraints. This building block does not imply the
frozen MuNet model or existing algorithms have been validated on arbitrary grids.
`physics/` makes momentum-prior identities explicit without changing defaults.

## Add inputs or storage

Convert an external format at the adapter boundary, retaining units, endpoint
conventions, original row IDs, and source metadata. Resolve data through the
catalogue without copying large raw files. Large-file indexing/conversion is an
explicit operation; ordinary catalogue/list commands must remain metadata-only.

Use unique run IDs, a resolved config snapshot, and a status manifest. A failed run
keeps diagnostic metadata and is not relabeled complete. New machine-local paths
belong in catalogue/local settings, never numerical modules.
