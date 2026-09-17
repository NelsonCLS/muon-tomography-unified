# Numerical regression and scientific benchmarks

Unit tests answer whether contracts/math work. A refactor benchmark answers
whether the same selected events, configuration and weights produce the same
arrays. A scientific benchmark answers whether a changed method works better on
predeclared objects. Keep these questions separate.

The current refactor baseline is the source snapshot/commit recorded in its report.
Representative parity cases should include original and target-filtered Funny 0,
a sparse/weak object, and a synthetic asymmetric geometry. Compare array shape,
units, selected IDs/order, density/occupancy/masks, diagnostics and metadata. Use
exact equality for discrete arrays and stated tolerances for floating point.
Do not accept equal MSE alone as evidence of unchanged predictions.

`legacy_reports.json` is a small path index of existing scientific reports; it is
not a new benchmark execution or a fresh verification of their numerical claims.
Actual large outputs stay at their original locations. New benchmark runs belong
under `runs/` (or a configured external location), with immutable provenance.

No automatic promotion of an experimental method occurs when a score improves.
Holdout evaluation must use a protocol fixed before looking at object truth.

## Run the public workflow check

```sh
python benchmarks/verify_workflows.py --output-dir /tmp/muon-workflows-new
```

A new directory is required; command logs and `workflow_report.json` are preserved.
Optional `--dataset` and `--simulator` arguments exercise existing real inputs and
native simulation. The initial refactor evidence remains under
`outputs/refactor-validation/`; packaging smoke has its separate
`package_smoke_2026-09-18.json` record.
