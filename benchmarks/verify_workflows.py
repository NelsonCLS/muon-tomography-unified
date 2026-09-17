"""Exercise public commands on a small synthetic dataset; preserve all evidence.

Run with the installed project Python. --output-dir must not exist. Optional real
input and Geant4 executable checks never modify their source datasets or binary.
This validates software workflows, not reconstruction accuracy or acquisition dose.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, help="optional real TFRecord, record 0")
    parser.add_argument("--simulator", type=Path, help="optional existing Geant4 executable")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "MPLCONFIGDIR": str(output / "matplotlib-cache"),
    }
    records = []

    def run(name: str, *arguments: object):
        command = [sys.executable, "-m", "muon_tomo.cli", *map(str, arguments)]
        start = time.monotonic()
        process = subprocess.run(
            command, capture_output=True, text=True, env=env, check=False, timeout=300
        )
        (output / f"{name}.log").write_text(process.stdout + process.stderr)
        records.append(
            {
                "workflow": name,
                "command": command,
                "seconds": round(time.monotonic() - start, 3),
                "exit_code": process.returncode,
            }
        )
        (output / "workflow_report.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"{name}: {'PASS' if process.returncode == 0 else 'FAIL'}", flush=True)
        if process.returncode:
            raise RuntimeError(f"{name} failed; see {output / (name + '.log')}")
        return process.stdout

    rng = np.random.default_rng(927)
    n = 1024
    x = np.zeros((n, 13), dtype=np.float64)
    upper_xy = rng.uniform(-40, 40, (n, 2))
    incoming = np.tile([0.0, 0.0, -1.0], (n, 1))
    outgoing = np.column_stack([rng.normal(0, 0.015, (n, 2)), -np.ones(n)])
    outgoing /= np.linalg.norm(outgoing, axis=1)[:, None]
    x[:, :2] = (upper_xy + 100 * outgoing[:, :2] / -outgoing[:, 2, None]) * 10
    x[:, 2] = -1000
    x[:, 3:6] = outgoing * 3000
    x[:, 6:8] = upper_xy * 10
    x[:, 8] = 1000
    x[:, 9:12] = incoming
    x[:, 12] = 3000
    csv = output / "tracks.csv"
    np.savetxt(csv, x, delimiter=",", fmt="%.17g")
    scene = output / "scene.json"
    scene.write_text(
        json.dumps(
            {
                "name": "asymmetric-lead",
                "shapes": [
                    {
                        "type": "box",
                        "material": "lead",
                        "size_cm": [25, 12.5, 6.25],
                        "center_cm": [12.5, -12.5, 0],
                    }
                ],
            }
        )
    )
    voxels, dataset, truth = (
        output / "voxels.npz",
        output / "synthetic.tfrecord",
        output / "truth.npz",
    )
    run("help", "--help")
    run("create_voxels", "create-voxels", scene, "--output", voxels)
    run("pack_simulation", "pack-simulation", "--csv", csv, "--voxels", voxels, "--output", dataset)
    run("inspect", "inspect", dataset, "--output", output / "inspect.json")
    run("truth", "truth", dataset, "--output", truth)
    legacy = output / "legacy"
    run(
        "reconstruct_all",
        "reconstruct",
        dataset,
        "--method",
        "all",
        "--limit",
        n,
        "--output-dir",
        legacy,
    )
    results = [legacy / f"record_0_{method}.npz" for method in ("munet", "poca", "likelihood")]
    run("compare", "compare", "--truth", truth, *results, "--output", output / "comparison.json")
    run("validate", "validate", "--truth", truth, *results, "--output", output / "validation.json")
    run("plot", "plot", "--truth", truth, *results, "--output", output / "section.png")
    run("plot3d", "plot3d", "--truth", truth, *results, "--output", output / "viewer.html")
    run("single_plot_no_truth", "plot", results[1], "--output", output / "poca.png")
    run("single_viewer_no_truth", "plot3d", results[1], "--output", output / "poca.html")
    run(
        "stability",
        "stability",
        "--dataset",
        dataset,
        "--limit",
        n,
        "--iterations",
        1,
        "--output-dir",
        output / "stability",
    )
    catalog = output / "datasets.json"
    run(
        "dataset_register",
        "dataset",
        "register",
        dataset,
        "--id",
        "smoke-synthetic",
        "--kind",
        "simulated",
        "--catalog",
        catalog,
    )
    run("dataset_list", "dataset", "list", "--catalog", catalog)
    run(
        "predict_all",
        "predict",
        "--input",
        "smoke-synthetic",
        "--limit",
        n,
        "--catalog",
        catalog,
        "--runs-dir",
        output / "runs",
    )
    run(
        "predict_no_truth",
        "predict",
        "--input",
        dataset,
        "--method",
        "poca",
        "--no-truth",
        "--runs-dir",
        output / "runs",
    )
    run("run_list", "run", "list", "--runs-dir", output / "runs")
    if args.dataset:
        run(
            "predict_real",
            "predict",
            "--input",
            args.dataset.resolve(),
            "--limit",
            8192,
            "--runs-dir",
            output / "real-runs",
        )
    if args.simulator:
        source = output / "vertical.json"
        source.write_text(
            json.dumps(
                {"angular_mode": "vertical", "energy_mode": "mono", "source_half_width_cm": 5}
            )
        )
        generated = output / "generated.tfrecord"
        run(
            "generate_dataset",
            "generate-dataset",
            "--scene",
            scene,
            "--output",
            generated,
            "--source-config",
            source,
            "--executable",
            args.simulator.resolve(),
            "--events",
            32,
            "--seed",
            47,
        )
        run(
            "predict_generated",
            "predict",
            "--input",
            generated,
            "--method",
            "poca",
            "--runs-dir",
            output / "generated-runs",
        )
    from muon_tomo.artifacts.runs import list_runs

    managed = list_runs(output / "runs")
    if len(managed) != 2 or any(row["status"] != "completed" for row in managed):
        raise RuntimeError("managed prediction manifests are not complete")
    print(f"Validated {len(records)} public command workflows; evidence in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
