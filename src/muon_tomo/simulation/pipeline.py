"""Build and run the bundled optional Geant4 backend, then export x/y records."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..constants import GRID_LOWER_CM, GRID_SHAPE, GRID_UPPER_CM, MATERIAL_NAMES
from .objects import create_voxels, load_voxels, read_scene, save_voxels
from .records import read_simulation_csv, write_records

BACKEND = Path(__file__).with_name("geant4")
MAX_SEED = 2147483646


@dataclass(frozen=True)
class SourceConfig:
    """Explicit source model; not a calibrated cosmic spectrum or acquisition time."""

    angular_mode: str = "cosine"
    max_zenith_deg: float = 45.0
    source_half_width_cm: float = 100.0
    energy_mode: str = "power_law"
    energy_min_gev: float = 1.0
    energy_max_gev: float = 200.0
    energy_power: float = -2.7
    mono_energy_gev: float = 3.0

    def validate(self) -> None:
        if self.angular_mode not in ("cosine", "vertical"):
            raise ValueError("angular_mode must be cosine or vertical")
        if self.energy_mode not in ("power_law", "mono"):
            raise ValueError("energy_mode must be power_law or mono")
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name not in ("angular_mode", "energy_mode") and not np.isfinite(value):
                raise ValueError(f"Source {field.name} must be finite")
        if not 0 < self.max_zenith_deg < 90:
            raise ValueError("max_zenith_deg must be strictly between 0 and 90")
        if self.source_half_width_cm <= 0:
            raise ValueError("source_half_width_cm must be positive")
        if not 0 < self.energy_min_gev < self.energy_max_gev:
            raise ValueError("Require 0 < energy_min_gev < energy_max_gev")
        if self.mono_energy_gev <= 0:
            raise ValueError("mono_energy_gev must be positive")

    @classmethod
    def from_file(cls, path: str | Path | None) -> SourceConfig:
        if path is None:
            return cls()
        values = yaml.safe_load(Path(path).read_text())
        if not isinstance(values, dict):
            raise ValueError("Source configuration must be a mapping")
        unknown = values.keys() - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown source settings: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    def macro(self) -> str:
        self.validate()
        lines = [
            "/control/verbose 0",
            "/run/verbose 0",
            "/event/verbose 0",
            "/tracking/verbose 0",
            "/gps/verbose 0",
            "/gps/particle mu-",
            "/gps/pos/type Plane",
            "/gps/pos/shape Square",
            "/gps/pos/centre 0 0 100 cm",
            f"/gps/pos/halfx {self.source_half_width_cm:.17g} cm",
            f"/gps/pos/halfy {self.source_half_width_cm:.17g} cm",
        ]
        if self.angular_mode == "vertical":
            lines += ["/gps/direction 0 0 -1"]
        else:
            lines += [
                "/gps/ang/type cos",
                "/gps/ang/mintheta 0 deg",
                f"/gps/ang/maxtheta {self.max_zenith_deg:.17g} deg",
            ]
        if self.energy_mode == "mono":
            lines += ["/gps/ene/type Mono", f"/gps/ene/mono {self.mono_energy_gev:.17g} GeV"]
        else:
            lines += [
                "/gps/ene/type Pow",
                f"/gps/ene/min {self.energy_min_gev:.17g} GeV",
                f"/gps/ene/max {self.energy_max_gev:.17g} GeV",
                f"/gps/ene/alpha {self.energy_power:.17g}",
            ]
        return "\n".join(lines) + "\n"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], log: Path, *, cwd: Path | None = None) -> None:
    with log.open("w") as handle:
        try:
            result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
        except OSError as exc:
            raise RuntimeError(f"Could not run {command[0]}: {exc}; log: {log}") from exc
    if result.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-18:])
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}\n{tail}")


def build_simulator(
    build_dir: str | Path,
    *,
    geant4_dir: str | Path | None = None,
    cxx: str | Path | None = None,
    jobs: int = 2,
) -> Path:
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("CMake is required to build the Geant4 simulator")
    if jobs < 1:
        raise ValueError("jobs must be positive")
    directory = Path(build_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    command = [cmake, "-S", str(BACKEND), "-B", str(directory), "-DCMAKE_BUILD_TYPE=Release"]
    if geant4_dir:
        command.append(f"-DGeant4_DIR={Path(geant4_dir).resolve()}")
    if cxx:
        command.append(f"-DCMAKE_CXX_COMPILER={cxx}")
    _run(command, directory / "configure.log")
    _run(
        [cmake, "--build", str(directory), "--config", "Release", "--parallel", str(jobs)],
        directory / "build.log",
    )
    for candidate in (
        directory / "muon_tomo_sim",
        directory / "Release/muon_tomo_sim.exe",
        directory / "muon_tomo_sim.exe",
    ):
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"Build succeeded but no muon_tomo_sim executable found in {directory}")


def run_simulation(
    executable: str | Path,
    voxels: str | Path,
    output_dir: str | Path,
    *,
    events: int,
    seed: int,
    source: SourceConfig | None = None,
) -> dict:
    if not isinstance(events, int) or not 0 < events <= MAX_SEED:
        raise ValueError("events must be an integer in 1..2147483646")
    if not isinstance(seed, int) or not 0 < seed <= MAX_SEED:
        raise ValueError("seed must be an integer in 1..2147483646")
    executable = Path(executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Build simulator first; executable not found: {executable}")
    config = source or SourceConfig()
    config.validate()
    labels = load_voxels(voxels)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    voxel_file = save_voxels(labels, directory / "voxels.txt")
    macro = directory / "source.mac"
    macro.write_text(config.macro())
    csv_file = directory / "tracks.csv"
    log = directory / "simulation.log"
    command = [
        str(executable),
        str(voxel_file),
        str(csv_file),
        str(macro),
        str(events),
        str(seed),
        str(config.source_half_width_cm * 10),
    ]
    _run(command, log, cwd=directory)
    tracks = read_simulation_csv(csv_file)
    if not np.allclose(tracks[:, 2], -1000, rtol=0, atol=1e-4):
        raise ValueError("Simulation outgoing tracks do not lie at z=-1000 mm")
    if not np.allclose(tracks[:, 8], 1000, rtol=0, atol=1e-4):
        raise ValueError("Simulation incoming tracks do not lie at z=+1000 mm")
    if len(tracks) > events:
        raise ValueError("Simulation wrote more accepted primary tracks than generated events")
    version = subprocess.run(
        [str(executable), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "generated_primaries": events,
        "accepted_tracks": len(tracks),
        "seed": seed,
        "source": dataclasses.asdict(config),
        "csv": str(csv_file),
        "voxels": str(voxel_file),
        "csv_sha256": sha256(csv_file),
        "voxel_file_sha256": sha256(voxel_file),
        "executable": str(executable),
        "executable_sha256": sha256(executable),
        "simulator_version": version,
        "physics_list": "FTFP_BERT",
        "backend_sources": {p.name: sha256(p) for p in sorted(BACKEND.iterdir()) if p.is_file()},
        "detector": {
            "type": "ideal planes; no RPC response model",
            "z_cm": [100, -100],
            "full_width_cm": 200,
            "track_selection": "primary mu- reaching lower plane",
        },
        "geometry": {
            "shape": list(GRID_SHAPE),
            "axis_order": "xyz",
            "lower_cm": GRID_LOWER_CM,
            "upper_cm": GRID_UPPER_CM,
            "placement": "voxel centres at lower + (index + 0.5) * cell width",
        },
        "g4_environment": {k: v for k, v in os.environ.items() if k.startswith("G4")},
        "platform": platform.platform(),
        "command": command,
    }
    (directory / "simulation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def generate_dataset(
    scenes: list[str | Path],
    output: str | Path,
    *,
    executable: str | Path,
    events: int = 20000,
    seed: int = 12345,
    records_per_scene: int = 1,
    source: SourceConfig | None = None,
    work_dir: str | Path | None = None,
) -> dict:
    """One record per scene/sample, with reproducible seeds and retained CSV/voxel evidence."""
    if not scenes or records_per_scene < 1:
        raise ValueError("Supply scenes and a positive records_per_scene")
    if not 0 < seed <= MAX_SEED - len(scenes) * records_per_scene + 1:
        raise ValueError("The requested seed range is outside 1..2147483646")
    target = Path(output).resolve()
    manifest_path = target.with_suffix(target.suffix + ".json")
    if target.exists() or manifest_path.exists():
        raise FileExistsError(f"Output dataset or manifest already exists: {target}")
    # Check optional TFRecord dependency before an expensive simulation starts.
    from ..io.tfrecord import _tensorflow

    _tensorflow()
    definitions = [read_scene(path) for path in scenes]
    arrays = [create_voxels(scene) for scene in definitions]
    config = source or SourceConfig()
    config.validate()
    directory = Path(work_dir).resolve() if work_dir else target.parent / (target.stem + "_runs")
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    for scene_index, (path, definition, labels) in enumerate(
        zip(scenes, definitions, arrays, strict=True)
    ):
        scene_path = directory / f"scene_{scene_index:03d}.json"
        scene_path.write_text(json.dumps(definition, indent=2) + "\n")
        grid_path = save_voxels(
            labels,
            directory / f"scene_{scene_index:03d}.npz",
            metadata={"source_scene": str(Path(path).resolve())},
        )
        for sample in range(records_per_scene):
            index = len(records)
            report = run_simulation(
                executable,
                grid_path,
                directory / f"record_{index:03d}",
                events=events,
                seed=seed + index,
                source=config,
            )
            report.update(
                {
                    "scene_index": scene_index,
                    "sample_index": sample,
                    "scene_name": definition.get("name", Path(path).stem),
                    "scene_file": str(scene_path),
                    "scene_sha256": sha256(scene_path),
                }
            )
            records.append(report)
    summaries = write_records(
        target,
        ((read_simulation_csv(record["csv"]), load_voxels(record["voxels"])) for record in records),
    )
    manifest = {
        "schema": "muon_tomo_custom_generation_v1",
        "tfrecord": str(target),
        "sha256": sha256(target),
        "compression": "GZIP",
        "x_dtype": "float64",
        "x_columns": "out position mm, out momentum MeV/c, in position mm, in unit direction, initial kinetic energy MeV",
        "y_dtype": "int32",
        "y_shape": list(GRID_SHAPE),
        "material_names": MATERIAL_NAMES,
        "truth_mapping": "existing Mu-Net material table (not a new Geant4 X0 calibration)",
        "physical_model_note": "Ideal detector acceptance and configured source; not RPC electronics or a measured exposure-time model. File format compatibility does not establish identical training distribution.",
        "records": [
            dict(report, **summary) for report, summary in zip(records, summaries, strict=True)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
