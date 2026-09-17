"""Optional simulation CLI registration; native builds are explicit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .objects import create_voxels, load_voxels, read_scene, save_voxels
from .pipeline import SourceConfig, build_simulator, generate_dataset, sha256
from .records import read_simulation_csv, write_records


def command_voxels(args: argparse.Namespace) -> None:
    scene = read_scene(args.scene)
    labels = create_voxels(scene)
    path = save_voxels(labels, args.output, metadata={"scene": scene})
    print(f"Created {path}: 64 x 64 x 64 material labels")


def command_build(args: argparse.Namespace) -> None:
    print(build_simulator(args.build_dir, geant4_dir=args.geant4_dir, cxx=args.cxx, jobs=args.jobs))


def command_generate(args: argparse.Namespace) -> None:
    manifest = generate_dataset(
        args.scene,
        args.output,
        executable=args.executable,
        events=args.events,
        seed=args.seed,
        records_per_scene=args.records_per_scene,
        source=SourceConfig.from_file(args.source_config),
        work_dir=args.work_dir,
    )
    print(
        json.dumps(
            {
                "output": manifest["tfrecord"],
                "records": [
                    {
                        k: row[k]
                        for k in (
                            "record_index",
                            "scene_name",
                            "generated_primaries",
                            "accepted_tracks",
                            "seed",
                        )
                    }
                    for row in manifest["records"]
                ],
            },
            indent=2,
        )
    )


def command_pack(args: argparse.Namespace) -> None:
    if len(args.csv) != len(args.voxels):
        raise ValueError("Supply one --voxels file per --csv file")
    target = Path(args.output)
    manifest_path = target.with_suffix(target.suffix + ".json")
    if target.exists() or manifest_path.exists():
        raise FileExistsError(f"Output dataset or manifest already exists: {target}")
    pairs = list(zip(args.csv, args.voxels, strict=True))
    records = write_records(
        target, ((read_simulation_csv(csv), load_voxels(voxels)) for csv, voxels in pairs)
    )
    report = {
        "compression": "GZIP",
        "x_dtype": "float64",
        "y_dtype": "int32",
        "records": records,
        "output": str(target.resolve()),
        "sha256": sha256(target),
        "sources": [
            {
                "csv": str(Path(c).resolve()),
                "csv_sha256": sha256(c),
                "voxels": str(Path(v).resolve()),
                "voxels_sha256": sha256(v),
            }
            for c, v in pairs
        ],
    }
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def add_simulation_commands(sub) -> None:
    voxels = sub.add_parser("create-voxels", help="create a canonical material grid from a scene")
    voxels.add_argument("scene", help="YAML/JSON scene with shapes and materials")
    voxels.add_argument("--output", required=True, help=".npz, .npy or flat .txt")
    voxels.set_defaults(func=command_voxels)

    build = sub.add_parser("build-simulator", help="build the bundled headless Geant4 simulation")
    build.add_argument("--build-dir", default="build/geant4")
    build.add_argument("--geant4-dir", help="directory containing Geant4Config.cmake")
    build.add_argument("--cxx", help="optional C++ compiler path")
    build.add_argument("--jobs", type=int, default=2)
    build.set_defaults(func=command_build)

    generate = sub.add_parser(
        "generate-dataset", help="simulate custom scenes and write GZIP TFRecords"
    )
    generate.add_argument("--scene", nargs="+", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--executable", default="build/geant4/muon_tomo_sim")
    generate.add_argument(
        "--events", type=int, default=20000, help="generated primaries per record"
    )
    generate.add_argument("--seed", type=int, default=12345)
    generate.add_argument("--records-per-scene", type=int, default=1)
    generate.add_argument("--source-config", help="YAML/JSON source settings")
    generate.add_argument(
        "--work-dir", help="new directory retaining simulation CSVs, grids and logs"
    )
    generate.set_defaults(func=command_generate)

    pack = sub.add_parser(
        "pack-simulation", help="pack existing CSV/voxel pairs into the x/y TFRecord schema"
    )
    pack.add_argument("--csv", nargs="+", required=True)
    pack.add_argument("--voxels", nargs="+", required=True)
    pack.add_argument("--output", required=True)
    pack.set_defaults(func=command_pack)
