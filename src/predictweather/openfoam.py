from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from predictweather.windninja import expected_windninja_ascii_paths


@dataclass
class OpenFoamRunError(RuntimeError):
    stage: str
    message: str
    command: list[str]
    output_dir: Path
    stdout: str = ""
    stderr: str = ""
    missing_outputs: list[str] | None = None
    details: dict | None = None

    def __post_init__(self) -> None:
        summary = f"{self.stage}: {self.message}"
        if self.missing_outputs:
            summary += f" Missing outputs: {', '.join(self.missing_outputs)}."
        super().__init__(summary)


def _split_command(command_text: str) -> list[str]:
    if os.name != "nt":
        return shlex.split(command_text)

    return [part.strip("\"'") for part in shlex.split(command_text, posix=False)]


def _runner_command() -> list[str]:
    runner = os.environ.get("PONDWIND_OPENFOAM_RUNNER", "").strip()
    if not runner:
        return []

    command = _split_command(runner)
    if len(command) == 1 and command[0].lower().endswith(".ps1"):
        return ["powershell", "-ExecutionPolicy", "Bypass", "-File", command[0]]
    return command


def _copy_first_existing(candidates: list[Path], destination: Path) -> None:
    if destination.exists():
        return
    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, destination)
            return


def _normalize_outputs(output_dir: Path, expected_outputs: dict[str, Path]) -> None:
    aliases = {
        "speed": [output_dir / "speed.asc", output_dir / "wind_speed.asc", output_dir / "velocity.asc"],
        "direction": [output_dir / "direction.asc", output_dir / "wind_direction.asc", output_dir / "angle.asc"],
        "u": [output_dir / "u.asc", output_dir / "U.asc", output_dir / "wind_u.asc"],
        "v": [output_dir / "v.asc", output_dir / "V.asc", output_dir / "wind_v.asc"],
    }
    for key, candidates in aliases.items():
        _copy_first_existing(candidates, expected_outputs[key])


def run_openfoam_domain_average(
    elevation_tif: Path,
    output_dir: Path,
    wind_speed_mps: float,
    wind_direction_deg: float,
    mesh_resolution_m: float,
    timeout_seconds: int = 1800,
) -> dict:
    command_prefix = _runner_command()
    if not command_prefix:
        raise OpenFoamRunError(
            stage="openfoam_setup",
            message=(
                "OpenFOAM solver is experimental and requires PONDWIND_OPENFOAM_RUNNER "
                "to point to a local runner script or command."
            ),
            command=[],
            output_dir=output_dir,
            details={
                "required_env_var": "PONDWIND_OPENFOAM_RUNNER",
                "expected_outputs": ["speed.asc or wind_speed.asc", "direction.asc or wind_direction.asc"],
            },
        )

    if not elevation_tif.exists():
        raise OpenFoamRunError(
            stage="openfoam_setup",
            message=f"Elevation input not found at {elevation_tif}",
            command=command_prefix,
            output_dir=output_dir,
        )

    if output_dir.exists():
        for existing in output_dir.iterdir():
            if existing.is_dir():
                shutil.rmtree(existing)
            else:
                existing.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_outputs = expected_windninja_ascii_paths(
        elevation_tif=elevation_tif,
        wind_speed_mps=wind_speed_mps,
        wind_direction_deg=wind_direction_deg,
        mesh_resolution_m=mesh_resolution_m,
        output_dir=output_dir,
    )
    case_dir = output_dir / "case"
    request_path = output_dir / "openfoam_request.json"
    request = {
        "solver": "openfoam",
        "api_version": 1,
        "elevation_tif": str(elevation_tif),
        "output_dir": str(output_dir),
        "case_dir": str(case_dir),
        "wind_speed_mps": float(wind_speed_mps),
        "wind_direction_deg": float(wind_direction_deg),
        "mesh_resolution_m": float(mesh_resolution_m),
        "expected_outputs": {key: str(value) for key, value in expected_outputs.items()},
    }
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

    command = [
        *command_prefix,
        "--request-json",
        str(request_path),
        "--elevation-file",
        str(elevation_tif),
        "--output-dir",
        str(output_dir),
        "--case-dir",
        str(case_dir),
        "--speed-mps",
        f"{wind_speed_mps:.6f}",
        "--direction-deg",
        f"{wind_direction_deg:.6f}",
        "--mesh-resolution-m",
        f"{mesh_resolution_m:.0f}",
        "--speed-output",
        str(expected_outputs["speed"]),
        "--direction-output",
        str(expected_outputs["direction"]),
        "--u-output",
        str(expected_outputs["u"]),
        "--v-output",
        str(expected_outputs["v"]),
    ]

    try:
        run_kwargs = {
            "capture_output": True,
            "text": True,
            "check": True,
            "timeout": timeout_seconds,
        }
        if os.name == "nt":
            run_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        completed = subprocess.run(command, **run_kwargs)
    except subprocess.TimeoutExpired as exc:
        raise OpenFoamRunError(
            stage="openfoam_timeout",
            message=f"OpenFOAM runner exceeded timeout of {timeout_seconds} seconds",
            command=command,
            output_dir=output_dir,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise OpenFoamRunError(
            stage="openfoam_subprocess",
            message=f"OpenFOAM runner exited with code {exc.returncode}",
            command=command,
            output_dir=output_dir,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        ) from exc

    _normalize_outputs(output_dir, expected_outputs)
    required_keys = ["speed", "direction"]
    missing_outputs = [str(expected_outputs[key]) for key in required_keys if not expected_outputs[key].exists()]
    if missing_outputs:
        raise OpenFoamRunError(
            stage="openfoam_outputs",
            message="OpenFOAM runner completed but did not produce required ASCII outputs",
            command=command,
            output_dir=output_dir,
            stdout=completed.stdout,
            stderr=completed.stderr,
            missing_outputs=missing_outputs,
            details={"request_json": str(request_path)},
        )

    return {
        "solver": "openfoam",
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_dir": str(output_dir),
        "request_json": str(request_path),
        "expected_outputs": {key: str(value) for key, value in expected_outputs.items()},
    }
