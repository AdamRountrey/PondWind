from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from predictweather.windninja import _read_aaigrid, expected_windninja_ascii_paths


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


def _is_wsl_terrain_runner(command: list[str]) -> bool:
    command_text = " ".join(command).lower()
    return "--openfoam-wsl-runner" in command_text or "openfoam_wsl_terrain_runner.py" in command_text


def _openfoam_wsl_preflight(command: list[str], timeout_seconds: int = 20) -> dict:
    if not _is_wsl_terrain_runner(command):
        return {
            "available": True,
            "mode": "custom_runner",
            "message": "Custom OpenFOAM runner configured; WSL preflight was not required.",
        }
    if os.name != "nt":
        return {
            "available": True,
            "mode": "non_windows_runner",
            "message": "Non-Windows runner configured; WSL preflight was not required.",
        }
    wsl_path = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl_path:
        return {
            "available": False,
            "mode": "wsl_openfoam13",
            "message": "WSL is not installed or wsl.exe is not on PATH.",
            "install_hint": "PondWind works without CFD. Experimental CFD requires WSL2 Ubuntu with OpenFOAM 13.",
        }
    probe_command = [
        wsl_path,
        "bash",
        "-lc",
        "source /opt/openfoam13/etc/bashrc >/dev/null 2>&1 && command -v foamRun >/dev/null && command -v blockMesh >/dev/null && command -v checkMesh >/dev/null",
    ]
    try:
        completed = subprocess.run(
            probe_command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "available": False,
            "mode": "wsl_openfoam13",
            "message": f"WSL/OpenFOAM preflight timed out after {timeout_seconds} seconds.",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "install_hint": "PondWind works without CFD. Experimental CFD requires WSL2 Ubuntu with OpenFOAM 13.",
        }
    except OSError as exc:
        return {
            "available": False,
            "mode": "wsl_openfoam13",
            "message": f"Unable to run WSL preflight: {exc}",
            "install_hint": "PondWind works without CFD. Experimental CFD requires WSL2 Ubuntu with OpenFOAM 13.",
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "mode": "wsl_openfoam13",
            "message": "WSL was found, but OpenFOAM 13 commands were not available.",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "install_hint": "Install OpenFOAM 13 in WSL Ubuntu, or use WindNinja-only reports.",
        }
    return {
        "available": True,
        "mode": "wsl_openfoam13",
        "message": "WSL OpenFOAM 13 preflight passed.",
    }


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


def _finite_output_counts(expected_outputs: dict[str, Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, path in expected_outputs.items():
        if not path.exists():
            continue
        data, _ = _read_aaigrid(path)
        counts[key] = int(np.isfinite(data).sum())
    return counts


def _grid_stats(data: np.ndarray) -> dict[str, float | int]:
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return {
            "finite_count": 0,
            "min": float("nan"),
            "p05": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
        }
    return {
        "finite_count": int(valid.size),
        "min": float(valid.min()),
        "p05": float(np.percentile(valid, 5)),
        "median": float(np.median(valid)),
        "p95": float(np.percentile(valid, 95)),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
    }


def _speed_output_stats(speed_path: Path) -> dict[str, float | int]:
    data, _ = _read_aaigrid(speed_path)
    return _grid_stats(data)


def _default_max_horizontal_cells(mesh_resolution_m: float) -> int:
    base_cells = 12000
    base_resolution_m = 30.0
    requested_resolution = max(float(mesh_resolution_m), 1.0)
    scaled_cells = int(math.ceil(base_cells * (base_resolution_m / requested_resolution) ** 2))
    return max(base_cells, min(scaled_cells, 120000))


def _uv_from_speed_direction_grid(speed_mps: np.ndarray, direction_from_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction_to_rad = np.deg2rad((direction_from_deg + 180.0) % 360.0)
    return speed_mps * np.sin(direction_to_rad), speed_mps * np.cos(direction_to_rad)


def _circular_abs_diff_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    return np.abs(((a_deg - b_deg + 180.0) % 360.0) - 180.0)


def _read_wind_components(paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    speed, _ = _read_aaigrid(paths["speed"])
    direction, _ = _read_aaigrid(paths["direction"])
    if paths.get("u") is not None and paths["u"].exists() and paths.get("v") is not None and paths["v"].exists():
        u, _ = _read_aaigrid(paths["u"])
        v, _ = _read_aaigrid(paths["v"])
    else:
        u, v = _uv_from_speed_direction_grid(speed, direction)
    return speed.astype(np.float32), direction.astype(np.float32), u.astype(np.float32), v.astype(np.float32)


def _transform_from_aaigrid_header(header: dict[str, float]):
    return from_origin(
        float(header["xllcorner"]),
        float(header["yllcorner"]) + float(header["nrows"]) * float(header["cellsize"]),
        float(header["cellsize"]),
        float(header["cellsize"]),
    )


def _reproject_grid_to_header(
    data: np.ndarray,
    *,
    src_header: dict[str, float],
    dst_header: dict[str, float],
    resampling: Resampling,
    src_nodata: float,
    dst_nodata: float,
) -> np.ndarray:
    source = data.astype(np.float32).copy()
    if np.isfinite(src_nodata):
        source[~np.isfinite(source)] = src_nodata
    destination = np.full((int(dst_header["nrows"]), int(dst_header["ncols"])), dst_nodata, dtype=np.float32)
    reproject(
        source=source,
        destination=destination,
        src_transform=_transform_from_aaigrid_header(src_header),
        src_crs="EPSG:26917",
        dst_transform=_transform_from_aaigrid_header(dst_header),
        dst_crs="EPSG:26917",
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    destination[destination == dst_nodata] = np.nan
    return destination.astype(np.float32)


def _comparison_subset_metrics(
    *,
    windninja_speed: np.ndarray,
    windninja_direction: np.ndarray,
    windninja_u: np.ndarray,
    windninja_v: np.ndarray,
    openfoam_speed: np.ndarray,
    openfoam_direction: np.ndarray,
    openfoam_u: np.ndarray,
    openfoam_v: np.ndarray,
    subset_mask: np.ndarray,
) -> dict[str, float | int | dict]:
    valid = (
        subset_mask
        & np.isfinite(windninja_speed)
        & np.isfinite(openfoam_speed)
        & np.isfinite(windninja_direction)
        & np.isfinite(openfoam_direction)
        & np.isfinite(windninja_u)
        & np.isfinite(windninja_v)
        & np.isfinite(openfoam_u)
        & np.isfinite(openfoam_v)
    )
    if not np.any(valid):
        return {"finite_pair_count": 0}

    speed_delta = openfoam_speed[valid] - windninja_speed[valid]
    vector_delta_sq = (openfoam_u[valid] - windninja_u[valid]) ** 2 + (openfoam_v[valid] - windninja_v[valid]) ** 2
    direction_error = _circular_abs_diff_deg(openfoam_direction[valid], windninja_direction[valid])
    ratio_valid = valid & (windninja_speed > 0.25)
    ratios = openfoam_speed[ratio_valid] / windninja_speed[ratio_valid]
    ratio_stats = _grid_stats(ratios.astype(np.float32)) if ratios.size else _grid_stats(np.array([], dtype=np.float32))
    return {
        "finite_pair_count": int(np.count_nonzero(valid)),
        "speed_bias_mps": float(np.mean(speed_delta)),
        "speed_mae_mps": float(np.mean(np.abs(speed_delta))),
        "speed_rmse_mps": float(np.sqrt(np.mean(speed_delta**2))),
        "vector_rmse_mps": float(np.sqrt(np.mean(vector_delta_sq))),
        "direction_abs_error_mean_deg": float(np.mean(direction_error)),
        "direction_abs_error_median_deg": float(np.median(direction_error)),
        "direction_abs_error_p95_deg": float(np.percentile(direction_error, 95)),
        "openfoam_speed_stats_mps": _grid_stats(openfoam_speed[valid]),
        "windninja_speed_stats_mps": _grid_stats(windninja_speed[valid]),
        "speed_ratio_openfoam_to_windninja": {
            "finite_count": ratio_stats["finite_count"],
            "p05": ratio_stats["p05"],
            "median": ratio_stats["median"],
            "p95": ratio_stats["p95"],
            "max": ratio_stats["max"],
        },
    }


def compare_wind_outputs(
    *,
    windninja_paths: dict[str, Path],
    openfoam_paths: dict[str, Path],
    water_mask: np.ndarray | None = None,
) -> dict:
    windninja_speed, windninja_header = _read_aaigrid(windninja_paths["speed"])
    windninja_direction, _ = _read_aaigrid(windninja_paths["direction"])
    if windninja_paths.get("u") is not None and windninja_paths["u"].exists() and windninja_paths.get("v") is not None and windninja_paths["v"].exists():
        windninja_u, _ = _read_aaigrid(windninja_paths["u"])
        windninja_v, _ = _read_aaigrid(windninja_paths["v"])
    else:
        windninja_u, windninja_v = _uv_from_speed_direction_grid(windninja_speed, windninja_direction)

    openfoam_speed, openfoam_header = _read_aaigrid(openfoam_paths["speed"])
    openfoam_direction, _ = _read_aaigrid(openfoam_paths["direction"])
    if openfoam_paths.get("u") is not None and openfoam_paths["u"].exists() and openfoam_paths.get("v") is not None and openfoam_paths["v"].exists():
        openfoam_u, _ = _read_aaigrid(openfoam_paths["u"])
        openfoam_v, _ = _read_aaigrid(openfoam_paths["v"])
    else:
        openfoam_u, openfoam_v = _uv_from_speed_direction_grid(openfoam_speed, openfoam_direction)

    if windninja_speed.shape != openfoam_speed.shape:
        openfoam_u = _reproject_grid_to_header(
            openfoam_u,
            src_header=openfoam_header,
            dst_header=windninja_header,
            resampling=Resampling.bilinear,
            src_nodata=-9999.0,
            dst_nodata=-9999.0,
        )
        openfoam_v = _reproject_grid_to_header(
            openfoam_v,
            src_header=openfoam_header,
            dst_header=windninja_header,
            resampling=Resampling.bilinear,
            src_nodata=-9999.0,
            dst_nodata=-9999.0,
        )
        openfoam_speed = np.hypot(openfoam_u, openfoam_v).astype(np.float32)
        openfoam_direction = ((270.0 - np.degrees(np.arctan2(openfoam_v, openfoam_u))) % 360.0).astype(np.float32)
        if water_mask is not None and water_mask.shape == tuple(int(openfoam_header[key]) for key in ("nrows", "ncols")):
            water_mask = _reproject_grid_to_header(
                water_mask.astype(np.float32),
                src_header=openfoam_header,
                dst_header=windninja_header,
                resampling=Resampling.nearest,
                src_nodata=-9999.0,
                dst_nodata=-9999.0,
            )
            water_mask = np.isfinite(water_mask) & (water_mask >= 0.5)

    full_mask = np.ones(windninja_speed.shape, dtype=bool)
    result = {
        "full_domain": _comparison_subset_metrics(
            windninja_speed=windninja_speed,
            windninja_direction=windninja_direction,
            windninja_u=windninja_u,
            windninja_v=windninja_v,
            openfoam_speed=openfoam_speed,
            openfoam_direction=openfoam_direction,
            openfoam_u=openfoam_u,
            openfoam_v=openfoam_v,
            subset_mask=full_mask,
        )
    }
    if water_mask is not None:
        if water_mask.shape != windninja_speed.shape:
            result["water_only"] = {
                "finite_pair_count": 0,
                "error": f"Water mask shape {water_mask.shape} does not match wind grid shape {windninja_speed.shape}.",
            }
        else:
            result["water_only"] = _comparison_subset_metrics(
                windninja_speed=windninja_speed,
                windninja_direction=windninja_direction,
                windninja_u=windninja_u,
                windninja_v=windninja_v,
                openfoam_speed=openfoam_speed,
                openfoam_direction=openfoam_direction,
                openfoam_u=openfoam_u,
                openfoam_v=openfoam_v,
                subset_mask=water_mask.astype(bool),
            )
    return result


def validate_wind_outputs(
    *,
    paths: dict[str, Path],
    boundary_wind_speed_mps: float,
    max_output_speed_mps: float | None = None,
) -> dict:
    required_keys = ["speed", "direction"]
    missing = [str(paths[key]) for key in required_keys if key not in paths or not paths[key].exists()]
    if missing:
        raise ValueError(f"Required wind ASCII outputs are missing: {missing}")

    speed, _ = _read_aaigrid(paths["speed"])
    direction, _ = _read_aaigrid(paths["direction"])
    speed_stats = _grid_stats(speed)
    direction_stats = _grid_stats(direction)
    if int(speed_stats["finite_count"]) == 0 or int(direction_stats["finite_count"]) == 0:
        raise ValueError(
            f"Wind outputs contain no finite values: speed={speed_stats['finite_count']}, direction={direction_stats['finite_count']}"
        )
    finite_speed = speed[np.isfinite(speed)]
    finite_direction = direction[np.isfinite(direction)]
    if np.any(finite_speed < 0.0):
        raise ValueError("Wind speed output contains negative values.")
    if np.any((finite_direction < 0.0) | (finite_direction > 360.0)):
        raise ValueError("Wind direction output contains values outside 0-360 degrees.")
    speed_limit = max(
        float(max_output_speed_mps if max_output_speed_mps is not None else os.environ.get("PONDWIND_OPENFOAM_MAX_OUTPUT_SPEED_MPS", "75")),
        float(boundary_wind_speed_mps) * 8.0,
    )
    if float(speed_stats["max"]) > speed_limit:
        raise ValueError(f"Wind output speed exceeded sanity limit: stats={speed_stats}, limit_mps={speed_limit:.3f}")
    return {
        "speed_stats_mps": speed_stats,
        "direction_stats_deg": direction_stats,
        "speed_limit_mps": speed_limit,
    }


def run_openfoam_domain_average(
    elevation_tif: Path,
    output_dir: Path,
    wind_speed_mps: float,
    wind_direction_deg: float,
    mesh_resolution_m: float,
    timeout_seconds: int = 3600,
) -> dict:
    command_prefix = _runner_command()
    if not command_prefix:
        raise OpenFoamRunError(
            stage="openfoam_unavailable",
            message=(
                "Experimental OpenFOAM CFD is not configured. WindNinja products can still be used without CFD."
            ),
            command=[],
            output_dir=output_dir,
            details={
                "skipped": True,
                "required_env_var": "PONDWIND_OPENFOAM_RUNNER",
                "install_hint": "PondWind works without CFD. Experimental CFD requires WSL2 Ubuntu with OpenFOAM 13.",
            },
        )

    availability = _openfoam_wsl_preflight(command_prefix)
    if not availability["available"]:
        raise OpenFoamRunError(
            stage="openfoam_unavailable",
            message=f"Experimental OpenFOAM CFD skipped: {availability['message']}",
            command=command_prefix,
            output_dir=output_dir,
            stdout=str(availability.get("stdout", "")),
            stderr=str(availability.get("stderr", "")),
            details={
                "skipped": True,
                "availability": availability,
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
    base_name = expected_outputs["speed"].name.removesuffix("_vel.asc")
    expected_outputs["turbulence_intensity"] = output_dir / f"{base_name}_ti_pct.asc"
    expected_outputs["tke"] = output_dir / f"{base_name}_tke.asc"
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
        "vertical_cells": int(os.environ.get("PONDWIND_OPENFOAM_VERTICAL_CELLS", "20")),
        "domain_height_m": float(os.environ.get("PONDWIND_OPENFOAM_DOMAIN_HEIGHT_M", "400")),
        "max_horizontal_cells": int(
            os.environ.get(
                "PONDWIND_OPENFOAM_MAX_HORIZONTAL_CELLS",
                str(_default_max_horizontal_cells(mesh_resolution_m)),
            )
        ),
        "roughness": {
            "water_z0_m": float(os.environ.get("PONDWIND_OPENFOAM_WATER_Z0_M", "0.0002")),
            "grass_z0_m": float(os.environ.get("PONDWIND_OPENFOAM_GRASS_Z0_M", "0.03")),
            "tree_z0_m": float(os.environ.get("PONDWIND_OPENFOAM_TREE_Z0_M", "0.3")),
        },
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
        "--vertical-cells",
        str(request["vertical_cells"]),
        "--domain-height-m",
        f"{request['domain_height_m']:.3f}",
        "--max-horizontal-cells",
        str(request["max_horizontal_cells"]),
        "--water-z0-m",
        f"{request['roughness']['water_z0_m']:.8f}",
        "--grass-z0-m",
        f"{request['roughness']['grass_z0_m']:.8f}",
        "--tree-z0-m",
        f"{request['roughness']['tree_z0_m']:.8f}",
        "--speed-output",
        str(expected_outputs["speed"]),
        "--direction-output",
        str(expected_outputs["direction"]),
        "--u-output",
        str(expected_outputs["u"]),
        "--v-output",
        str(expected_outputs["v"]),
        "--turbulence-intensity-output",
        str(expected_outputs["turbulence_intensity"]),
        "--tke-output",
        str(expected_outputs["tke"]),
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

    finite_counts = _finite_output_counts(expected_outputs)
    empty_outputs = [key for key in required_keys if finite_counts.get(key, 0) == 0]
    if empty_outputs:
        raise OpenFoamRunError(
            stage="openfoam_outputs",
            message="OpenFOAM runner completed but produced no finite values in required ASCII outputs",
            command=command,
            output_dir=output_dir,
            stdout=completed.stdout,
            stderr=completed.stderr,
            details={
                "request_json": str(request_path),
                "finite_counts": finite_counts,
                "empty_outputs": empty_outputs,
                "expected_outputs": {key: str(value) for key, value in expected_outputs.items()},
            },
        )

    try:
        validation = validate_wind_outputs(
            paths=expected_outputs,
            boundary_wind_speed_mps=wind_speed_mps,
        )
    except ValueError as exc:
        raise OpenFoamRunError(
            stage="openfoam_outputs",
            message=f"OpenFOAM runner completed but produced unusable wind outputs: {exc}",
            command=command,
            output_dir=output_dir,
            stdout=completed.stdout,
            stderr=completed.stderr,
            details={
                "request_json": str(request_path),
                "expected_outputs": {key: str(value) for key, value in expected_outputs.items()},
            },
        ) from exc

    runner_summary_path = output_dir / "openfoam_wsl_terrain_runner_summary.json"
    runner_summary = None
    if runner_summary_path.exists():
        try:
            runner_summary = json.loads(runner_summary_path.read_text(encoding="utf-8"))
        except Exception:
            runner_summary = {"summary_path": str(runner_summary_path), "parse_error": True}
    runner_kind = str((runner_summary or {}).get("runner") or "custom_runner_no_wsl_summary")
    is_scientific_cfd_candidate = runner_kind == "openfoam_wsl_terrain_runner"
    solver_mode = str(
        (runner_summary or {}).get("solver_mode")
        or ("steady_incompressible_rans_abl" if is_scientific_cfd_candidate else "custom_wind_grid_adapter")
    )

    return {
        "solver": "openfoam",
        "solver_mode": solver_mode,
        "runner_kind": runner_kind,
        "scientific_label": (
            "experimental_openfoam_neutral_abl_rans"
            if is_scientific_cfd_candidate
            else "custom_wind_grid_adapter_not_validated_cfd"
        ),
        "is_scientific_cfd_candidate": is_scientific_cfd_candidate,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_dir": str(output_dir),
        "request_json": str(request_path),
        "expected_outputs": {key: str(value) for key, value in expected_outputs.items()},
        "validation": validation,
        "runner_summary": runner_summary,
    }
