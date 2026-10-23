"""Create a deterministic detector-only toy input, not a physical simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from muon_tomo.io.detector import save_detector_npz
from muon_tomo.types import DetectorTracks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    n = 8192
    xy = rng.uniform(-40.0, 40.0, (n, 2))
    incoming = np.tile([0.0, 0.0, -1.0], (n, 1))
    outgoing = np.column_stack((rng.normal(0.0, 0.015, (n, 2)), -np.ones(n)))
    # Broken straight paths meet at z=0; detector reference planes are +/-100 cm.
    entry = np.column_stack((xy, np.full(n, 100.0)))
    exit_position = np.column_stack((xy + 100.0 * outgoing[:, :2], np.full(n, -100.0)))
    tracks = DetectorTracks(
        entry,
        incoming,
        exit_position,
        outgoing,
        source="synthetic-detector-demo-seed-0",
        metadata={"purpose": "software demo only; not a physical scattering simulation"},
    )
    print(save_detector_npz(tracks, args.output))


if __name__ == "__main__":
    main()
