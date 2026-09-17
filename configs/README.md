# Configuration

`default.yaml` mirrors the packaged `src/muon_tomo/configs/default.yaml`; a test
checks equality. `muon_tomo.config` owns typed configuration validation and merging.
New run workflows save the fully resolved configuration alongside predictions.

Current production settings preserve the historical fixed inverse-square effective
momentum 1743.4601815051958 MeV/c; this was derived from Funny 0 simulation. PoCA
uses 32³ and σ=6.25 cm. MuNet uses frozen 1024-dosage weights and the assumed scalar
feature. Likelihood stays the full-covariance median ten-iteration implementation.

`scenes/` contains object/source inputs for the custom simulator; source
kinetic-energy spectrum and reconstruction momentum prior are distinct settings.
See `docs/custom_simulation.md` and `docs/scientific_contracts.md`.

The three `funny-*.calibration.json` files are **historical truth-fitted artifacts**.
They are retained for provenance, not applied by normal prediction. They are
method/version/momentum-policy specific; do not use them to reproduce attractive
old MSE values. New comparison outputs remain uncalibrated against truth.

`munet.inference_seed: 0` controls stochastic placement in the new managed
`predict` workflow; the legacy `reconstruct` command keeps its previous behavior.
This is distinct from the track selection seed.
