# Historical recipe source archive

These scripts previously lived beside generated results or datasets. They are
copied byte-for-byte into versionable source storage; the original files remain.
`index.json` records every original location and SHA-256.

**Reference only: do not run the copies blindly.** Some execute work at import,
contain absolute machine paths, locate data relative to their original directory,
apply historical calibration, or overwrite their original report. Their original
assumptions have deliberately not been reinterpreted during structural refactoring.
Use `muon-tomo predict` for new standard runs. Port an old recipe into an explicit,
parameterized experiment before repeating it, keep its protocol and selection IDs,
and save outputs in a fresh run directory.

The useful families are PoCA exposure/grid-width ablations, track acceptance
analysis, filtered/original comparisons, likelihood algebra/path audits, the
stopped likelihood development candidate, and custom lead-block comparisons.
