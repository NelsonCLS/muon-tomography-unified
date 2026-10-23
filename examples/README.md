# Dataset-free detector example

Run from the repository root after installing `.[plot]`:

```sh
python examples/create_detector_sample.py --output data/derived/demo-tracks.npz
muon-tomo predict --input data/derived/demo-tracks.npz --method poca --no-truth --open
```

The writer refuses to overwrite existing files. Reuse that input on subsequent
predictions; each prediction gets a new directory under `runs/`. Omit `--open` on
a server. The CLI prints the saved run location, including cross-section PNG and
interactive HTML viewer. The standard >0.8 viewer cut may hide weak predictions;
this does not remove voxels from the saved arrays.

`DetectorTracks` uses physical incoming then outgoing positions in centimetres,
and flight-oriented directions (normalized by the constructor). The NPZ writer
stores only these four arrays, integer event IDs and schema metadata. It never
stores simulation momentum or truth. See `src/muon_tomo/io/detector.py` for the
interchange schema. Legacy TFRecord endpoint order differs; use its adapter.

The sample uses 8,192 vertical incoming rays and random outgoing deflections at
z=0, with endpoint planes at +/-100 cm. It is an intentionally simple software
fixture, not a Geant4 sample, material model, realistic dose or accuracy benchmark.
It cannot produce a truth panel or MSE because no truth volume exists.
