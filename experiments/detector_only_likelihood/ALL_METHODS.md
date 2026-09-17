# All-method external-spectrum comparison

This is an experimental comparison. Production code and Mu-Net weights are unchanged.

Every method uses the same fixed Reyna sea-level spectrum, conditioned only on the measured incoming direction. The spectrum is integrated over the unobserved momentum and supplies an expected inverse-square momentum coefficient. No simulated momentum magnitude, material label, truth mask, truth-derived scale, or per-object tuning is passed into reconstruction.

| case | PoCA MSE | likelihood MSE | MuNet MSE | empty-volume MSE |
| --- | ---: | ---: | ---: | ---: |
| Funny 0 | 0.47689 | **0.21722** | 0.09659 | 0.29615 |
| main record 0 | 0.15233 | **0.07503** | 0.04701 | 0.11226 |

All values are full 64³ MSE in ((\mathrm{cm}^{-1})^2). The likelihood column is the saved exposure-pooled likelihood experiment. PoCA uses the same exposure-normalized PoCA estimator and its fixed 6.25 cm pool, with the fixed external spectrum replacing its single momentum prior. MuNet uses the external direction-conditioned effective momentum as its 14th input feature while keeping the original weights unchanged.

The MuNet number is not an apples-to-apples validated improvement: the weights were trained with a different momentum-feature policy. A principled all-method comparison requires retraining MuNet with the same external-spectrum feature policy, or retraining without momentum as an input. The experimental MuNet result is useful as a sensitivity check only.

For likelihood, applying the same 3.125 cm exposure pooling to both momentum policies gives a cleaner comparison: Funny 0 is 0.23370 with frozen 1,743 MeV/c and 0.21722 with the external spectrum; main record 0 is 0.07836 and 0.07503, respectively. Most of the likelihood gain therefore comes from pooling, with a smaller additional gain from replacing the frozen momentum prior.

The saved arrays and metadata are in `results/all_methods_external/`. `predictions.npz` contains raw (1/X_0) volumes; `truth_evaluation_only.npz` is written separately after all predictions are saved. The production likelihood source hash is recorded in `protocol.json`.
