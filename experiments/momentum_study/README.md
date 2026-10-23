# Momentum-policy study (protocol frozen 2026-10-20)

Compare frozen MuNet and production exposure-normalized PoCA, changing only the
momentum supplied to each method. No training, truth-fitted rescaling, label mask,
source-data edits, or production-default changes. This is an explicit simulation
oracle study as well as a study of reusable population priors.

## Fixed design

- Twelve preselected objects: generated centred 25 cm lead cube; filtered legacy
  Funny 0, Funny 1 and Custom 0; validation records 17, 53, 101, 257, 503, 997;
  test records 37 and 211. Selection is by record ID before scoring, not by success.
- 8,192 target-crossing rows per object, sampled without replacement with NumPy
  seed 0, restored to source order. Save exact row IDs and selected-record hashes.
- Reference: all accepted tracks in filtered training records 0–31, independent
  of the evaluation records. Fit only momentum/angle statistics, never labels.
- Primary momentum truth is the outgoing detector momentum, matching the legacy
  TFRecord feature used by the frozen model. Initial/source momentum and
  sqrt(p_in*p_out) are additional oracle diagnostics for energy loss; neither is
  a measurement or the exact momentum integral along the scattering path.
- 15 policies: historical fixed 1743.4601815 MeV/c; reference arithmetic mean,
  geometric mean, median, inverse-square effective mean; reference zenith-bin
  arithmetic and inverse-square means; per-event outgoing, initial and geometric
  endpoint truth; object-specific arithmetic, geometric, inverse-square means;
  object-specific zenith-bin arithmetic and inverse-square means.
- Arithmetic = E[p]; geometric = exp(E[log p]); inverse-square effective =
  E[p^-2]^-1/2. All reference/object averages use outgoing momentum.
- Zenith is the physical incoming track angle, using the shared endpoint-order
  adapter. Fixed edges: 0,5,...,45,90 degrees. A bin with fewer than 100 reference
  tracks (20 object tracks) falls back to the corresponding global statistic.
  No fitted angular law, angular interpolation, or material conditioning.
- PoCA: 32³, Gaussian sigma 6.25 cm, Highland constant 15 MeV, pooled signal /
  pooled exposure, expanded to canonical 64³. Cache momentum-independent geometry
  only; verify full-array parity with production PoCA for fixed and per-track p.
- MuNet: frozen 1024-dosage weights, eight chunks averaged, training=False. Change
  only feature column 12 with existing training normalization (5585.2666,
  13839.263 MeV/c). Reuse the loaded model. Reset TensorFlow seed 0 immediately
  before each policy's eight prediction calls, after model construction, so
  unscattered-track random placements are paired across policies. This controlled
  RNG initialization may differ from the managed command's pre-build seed 0.
  Verify cached inference against an independently initialized model under this
  same post-build seed convention; test constant-feature parity with production.

## Evaluation and uncertainty

Raw predictions precede truth evaluation. Save all arrays, assigned momenta,
policy definitions, reference bin counts, software/model/code identities, timings,
full-grid MSE, object-only MSE, background MSE, object bias and material means,
background false-positive fraction >0.8 cm^-1, and all-zero prediction baselines.
Use unweighted macro-averages across objects; never pool all voxels across objects.
Report eleven legacy-source objects and the new-source lead cube separately.
For legacy paired differences, bootstrap objects (10,000 draws, seed 20261020)
and report percentile 95% intervals; these are exploratory and conditional on
this small, nonrandom object set, not coverage guarantees for future objects.
No selection/tuning and no multiple-comparison significance claims. One track
subset and one inference seed per object; no independent exposure repeats.

All reconstruction cross sections saturate at 3.2 cm^-1, with identical physical
coordinates. Save complete numbers as CSV/JSON and representative comparisons.
The new lead source differs from the legacy training simulator/spectrum; report
it as a domain-shift diagnostic, not a pooled estimate of deployment accuracy.
Historical fixed momentum came from Funny 0, so Funny 0 is not independent of
that baseline. Reference priors are simulation-derived, not externally measured.

## Run

From the installed repository environment (`pip install -e '.[all]'`). This
study needs the local training/evaluation datasets named in the fixed design;
those datasets and historical result directories are not distributed with GitHub.
The dataset-free quickstart in the root README is the portable software demo.

Run from the repository root:

```
python experiments/momentum_study/run.py --root . --output outputs/momentum-study/new-run
python experiments/momentum_study/report.py --run outputs/momentum-study/new-run
```

The runner requires a new output directory. It scans each needed compressed
source once, caching only requested records and hashes rather than hashing huge
whole files. Production defaults and source data remain unchanged.

## Schema validation correction before completing evaluation

The interrupted `20261020-v1` run incorrectly treated columns 9:12 as a momentum
vector for two optional oracle diagnostics. They are unit directions. V1 is
invalid and must not be used as a completed comparison. `20261020-v3` converts
initial kinetic energy T in column 12 to p=sqrt(T*(T+2m_mu)), with m_mu=105.6583755
MeV/c². It is source/vertex momentum, not measured upper-plane momentum. The
correction changes only the two extra initial-energy oracle inputs; primary
outgoing truth, means, object IDs, reference records and all reconstruction
settings are unchanged. A regression test rejects the erroneous direction norm.

Custom 0 has the older 12-column schema without initial kinetic energy. Its two
initial-energy oracle policies are unavailable and are omitted, never imputed.
All 13 primary policies run on all 12 objects; two optional source-energy policies
run on 11 objects: 356 reconstructions total. Summary comparisons for those two
use only the same available objects in both policy and baseline. The v2 input
validation correctly stopped before inference on encountering this missing field.

## 2026-10-23: 10th percentile extension

`ref_percentile10` assigns the global 10th percentile of the independent
reference outgoing momenta (NumPy linear interpolation) to every evaluation track.
It uses no per-object or per-event truth. The original 15 policies remain,
bringing the study to 16 policies and 380 reconstructions across the same objects.
The percentile is an exploratory low-momentum comparison, not a claim of an
unbiased scattering estimator or a newly tuned production default.

Reuse saved selected inputs and original reconstructions without rescanning sources:

```sh
python experiments/momentum_study/extend_percentile10.py --base-run outputs/momentum-study/20261020-v3 --output outputs/momentum-study/20261023-percentile10
python experiments/momentum_study/report.py --run outputs/momentum-study/20261023-percentile10
```
