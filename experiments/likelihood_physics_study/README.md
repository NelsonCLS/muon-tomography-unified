# Physics-first likelihood study

Historical reports, figures and `results/` below are local artifacts and are not
included in a fresh checkout. The tracked [text report](REPORT.md) preserves the
findings; these experimental scores do not describe current production defaults.

Open **[report.html](report.html)** for the full physical explanation, all method scores, selectable reconstruction images and convergence checks. **[REPORT.md](REPORT.md)** contains the complete textual report. The scientific comparison plot is available as [PNG](results/validation_comparison.png) and [PDF](results/validation_comparison.pdf).

## Main result

Eleven fixed variants were tested on three familiar development objects and six additional validation objects from the original unfiltered dataset. Each object uses 8,192 selected tracks, with a separate 6,144/2,048 training/reserved split for conditional track scoring.

On the six additional objects, equal-object mean full-volume MSE is:

- Existing median solver, 10 steps, common external prior: **1.236006 cm⁻²**.
- Existing exposure pooling, 10 steps: **0.261319 cm⁻²**.
- Incoming-transverse coordinates, median update, 40 steps, exposure pooling: **0.248528 cm⁻²**.

Pooling supplies the largest improvement. The transverse 40-step variant improves on the pooled control by 4.89% on average, with wins on only 3/6 additional objects. It improves material recovery and localization but increases air leakage relative to Pool10. Direct Gaussian fits, the fixed quadratic/Huber penalties, Student–Huber and angular-only fitting do not beat Pool10 on mean validation MSE.

This is an **experimental option**, not a production replacement. All production Python sources and configuration were left unchanged. Comparisons use the same external Reyna population spectrum and C=15 MeV; the production configuration's different 1743.460 MeV/c scalar is not the baseline scale used here. No label-derived rescaling is applied.

## Use the candidate

After installing the repository with `pip install -e ".[all]"`:

```python
from muon_tomo.research.likelihood_physics_study.core import reconstruct_local_pooled

rho_cm_inv, metadata = reconstruct_local_pooled(
    entry_position_cm,
    entry_direction,
    exit_position_cm,
    exit_direction,
    iterations=40,
)
```

Each argument is a finite `(N, 3)` array. Directions are normalized; their original magnitudes are discarded. Track order is resolved with the existing common geometry code. The return volume is `(64, 64, 64)`, indexed `[x, y, z]`, and covers `[-50, 50] cm` on each axis. A 32³ pooling grid and 3.125 cm Gaussian sigma limit the effective spatial resolution. Absolute 1/X₀ is conditional on the external spectrum and approximate path model. No measured momentum or truth argument exists.

For the exact saved result, load `results/<case>/full_maps.npz` and select `local_pool40`. The `local_pool10` output is the lower-background alternative. `training_maps.npz` contains fits that exclude the reserved event rows; it must not be confused with the full-map image evaluation.

## Reproduce

Use the existing repository environment. From the repository root:

```sh
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=/tmp/muon-mpl
export MPLBACKEND=Agg
.venv/bin/python -m pytest -q experiments/likelihood_physics_study/test_core.py
.venv/bin/python experiments/likelihood_physics_study/run.py extract
.venv/bin/python experiments/likelihood_physics_study/run.py fit
.venv/bin/python experiments/likelihood_physics_study/convergence.py
.venv/bin/python experiments/likelihood_physics_study/run.py evaluate
.venv/bin/python experiments/likelihood_physics_study/report.py
```

Completed case fits are reused only when their provenance matches. The commands above describe the historical default layout. For isolated `run.py` recomputation, pass `--results-dir /path/to/new-results` before the stage name; inspect the other report/convergence runners before combining them with a custom directory. Preserve old results. Dependencies include NumPy, SciPy, TensorFlow for extraction/evaluation, Matplotlib and markdown-it-py for report generation.

Development cases reuse detector-only inputs and evaluation truth under `experiments/detector_only_likelihood/results`. The six additional objects are extracted from `dataset/voxels_prediction_val.tfrecord`. The predecessor's `method.py` and `spectrum_method.py` supply its verified track container, path moments and external spectrum. These dependencies are explicit; this experiment is not a standalone package detached from the unified repository.

`run.py` does not decode the additional validation labels during fitting. Extraction stores serialized labels separately for the evaluation stage. All main maps are saved before `evaluate` loads labels. Candidate settings and additional validation indices are recorded in `results/protocol.json`; the longer-optimizer check was fixed separately before evaluation in `results/convergence_protocol.json`.

## Evidence and limitations

- `results/metrics.json`: every method and object, with image and reserved-track metrics.
- `results/summary.json`: equal-object group means and exploratory object-bootstrap intervals.
- `results/*/fit_complete.json`: exact usable track counts, coverage, optimizer histories and flags.
- `results/*/reserved_scores.npz`: per-event conditional scores and the selected reserved event indices.
- `results/convergence_evaluation.json`: extra-budget results, including optimizer convergence flags.
- `results/integrity.json`: production-source integrity check.
- `results/source_at_run`: sources matching the initial protocol hashes, retained for audit. The later convenience wrapper does not alter the kernels used in fitting.
- `results/final_source_sha256.json`: delivered source checksums.

The main direct optimizers hit their 160-step limit. Additional 1,000-step checks do not rescue their image accuracy on the two tested objects, but global convergence is not established. The six validation objects share the existing simulation family. Momentum acceptance, energy loss, true internal paths and correlated detector errors remain uncalibrated. Conditional reserved-track scores do not establish unconditional predictive performance. See the report for the full derivation and the material-versus-background tradeoff.
