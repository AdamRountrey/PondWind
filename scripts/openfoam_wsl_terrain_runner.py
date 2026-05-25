from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import traceback
from pathlib import Path, PureWindowsPath

import numpy as np
import rasterio
from rasterio.enums import Resampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental WSL/OpenFOAM terrain runner for PondWind.")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--elevation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--speed-mps", type=float, required=True)
    parser.add_argument("--direction-deg", type=float, required=True)
    parser.add_argument("--mesh-resolution-m", type=float, required=True)
    parser.add_argument("--speed-output", required=True)
    parser.add_argument("--direction-output", required=True)
    parser.add_argument("--u-output", required=True)
    parser.add_argument("--v-output", required=True)
    parser.add_argument("--turbulence-intensity-output", default="")
    parser.add_argument("--tke-output", default="")
    parser.add_argument("--vertical-cells", type=int, default=int(os.environ.get("PONDWIND_OPENFOAM_VERTICAL_CELLS", "20")))
    parser.add_argument("--domain-height-m", type=float, default=float(os.environ.get("PONDWIND_OPENFOAM_DOMAIN_HEIGHT_M", "400")))
    parser.add_argument("--max-horizontal-cells", type=int, default=int(os.environ.get("PONDWIND_OPENFOAM_MAX_HORIZONTAL_CELLS", "12000")))
    parser.add_argument("--max-output-speed-mps", type=float, default=float(os.environ.get("PONDWIND_OPENFOAM_MAX_OUTPUT_SPEED_MPS", "75")))
    parser.add_argument("--water-z0-m", type=float, default=float(os.environ.get("PONDWIND_OPENFOAM_WATER_Z0_M", "0.0002")))
    parser.add_argument("--grass-z0-m", type=float, default=float(os.environ.get("PONDWIND_OPENFOAM_GRASS_Z0_M", "0.03")))
    parser.add_argument("--tree-z0-m", type=float, default=float(os.environ.get("PONDWIND_OPENFOAM_TREE_Z0_M", "0.3")))
    parser.add_argument("--reference-height-m", type=float, default=float(os.environ.get("PONDWIND_OPENFOAM_REFERENCE_HEIGHT_M", "10")))
    parser.add_argument("--enable-potential-init", action="store_true", help="Run potentialFoam initialization before the final RANS solve.")
    parser.add_argument("--skip-potential-init", action="store_true", help="Skip potentialFoam initialization before the final RANS solve.")
    return parser.parse_args()


def _foam_header(class_name: str, object_name: str) -> str:
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    object      {object_name};
}}
"""


def _uv_from_speed_direction(speed_mps: float, direction_from_deg: float) -> tuple[float, float]:
    direction_to_rad = math.radians((direction_from_deg + 180.0) % 360.0)
    return speed_mps * math.sin(direction_to_rad), speed_mps * math.cos(direction_to_rad)


def _speed_direction_from_uv(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.hypot(u, v).astype(np.float32)
    direction = ((270.0 - np.degrees(np.arctan2(v, u))) % 360.0).astype(np.float32)
    return speed, direction


def _windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = PureWindowsPath(resolved).drive
    if drive:
        letter = drive.rstrip(":").lower()
        parts = PureWindowsPath(resolved).parts[1:]
        return "/mnt/" + letter + "/" + "/".join(parts).replace("\\", "/")
    return str(resolved).replace("\\", "/")


def _wsl_shell_command(command: str) -> str:
    bashrc_override = os.environ.get("PONDWIND_OPENFOAM_BASHRC", "").strip()
    source_lines = ["set +e"]
    if bashrc_override:
        source_lines.append(f'if [ -f "{bashrc_override}" ]; then . "{bashrc_override}"; fi')
    source_lines.append('if [ -f "/opt/openfoam13/etc/bashrc" ]; then . "/opt/openfoam13/etc/bashrc"; fi')
    source_lines.append("set -e")
    return "\n".join([*source_lines, command])


def _run_wsl_status(command: str, timeout_seconds: int = 1800) -> tuple[int, str]:
    run_kwargs = {
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": timeout_seconds,
    }
    if os.name == "nt":
        run_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        ["wsl", "bash", "-lc", _wsl_shell_command(command)],
        **run_kwargs,
    )
    return int(completed.returncode), (completed.stdout or "") + (completed.stderr or "")


def _run_wsl(command: str, timeout_seconds: int = 1800) -> str:
    return_code, output = _run_wsl_status(command, timeout_seconds=timeout_seconds)
    if return_code != 0:
        raise RuntimeError(f"WSL/OpenFOAM command failed with code {return_code}: {command}\n{output}")
    return output


def _foam_command_exists(command: str) -> bool:
    try:
        _run_wsl(f"command -v {command} >/dev/null 2>&1", timeout_seconds=30)
        return True
    except Exception:
        return False


def _run_solver(case_dir: Path) -> str:
    case_arg = _windows_to_wsl_path(case_dir)
    if not _foam_command_exists("foamRun"):
        raise RuntimeError("OpenFOAM 13 foamRun was not found in WSL. Install/source OpenFOAM 13 before running experimental CFD.")
    return _run_wsl(f'foamRun -solver incompressibleFluid -case "{case_arg}"')


def _run_potential_solver(case_dir: Path) -> str:
    case_arg = _windows_to_wsl_path(case_dir)
    return _run_wsl(f'potentialFoam -case "{case_arg}" -initialiseUBCs -writep')


def _run_check_mesh(case_dir: Path) -> dict:
    case_arg = _windows_to_wsl_path(case_dir)
    return_code, output = _run_wsl_status(f'checkMesh -case "{case_arg}" -allGeometry -allTopology')
    failed_count = re.search(r"Failed\s+([1-9][0-9]*)", output)
    passed = return_code == 0 and "Mesh OK" in output and failed_count is None
    return {
        "passed": passed,
        "return_code": return_code,
        "mesh_ok": "Mesh OK" in output,
        "failed": failed_count is not None,
        "summary": _compact_log_tail(output, line_count=80),
    }


def _renumber_mesh(case_dir: Path) -> str:
    case_arg = _windows_to_wsl_path(case_dir)
    return _run_wsl(f'renumberMesh -case "{case_arg}" -overwrite')


def _run_sampler(case_dir: Path) -> str:
    case_arg = _windows_to_wsl_path(case_dir)
    commands = [
        f'foamPostProcess -case "{case_arg}" -latestTime -dict system/sampleDict',
        f'postProcess -case "{case_arg}" -latestTime -dict system/sampleDict',
    ]
    failures: list[str] = []
    for command in commands:
        try:
            return _run_wsl(command)
        except Exception as exc:
            failures.append(f"{command}\n{exc}")
    raise RuntimeError("OpenFOAM sampling failed for every supported command variant.\n\n" + "\n\n".join(failures))


def _write_aaigrid(path: Path, data: np.ndarray, *, xllcorner: float, yllcorner: float, cellsize: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nodata = -9999.0
    rows = np.where(np.isfinite(data), data, nodata)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"ncols {data.shape[1]}\n")
        handle.write(f"nrows {data.shape[0]}\n")
        handle.write(f"xllcorner {xllcorner:.6f}\n")
        handle.write(f"yllcorner {yllcorner:.6f}\n")
        handle.write(f"cellsize {cellsize:.6f}\n")
        handle.write(f"NODATA_value {nodata:.1f}\n")
        for row in rows:
            handle.write(" ".join(f"{value:.6f}" for value in row) + "\n")


def _aaigrid_has_finite(path: Path) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines()
    nodata = -9999.0
    for line in lines[:6]:
        parts = line.split()
        if len(parts) == 2 and parts[0].lower() == "nodata_value":
            nodata = float(parts[1])
            break
    rows = []
    for line in lines[6:]:
        if line.strip():
            rows.extend(float(item) for item in line.split())
    data = np.array(rows, dtype=np.float32)
    return bool(rows) and bool((np.isfinite(data) & (data != nodata)).any())


def _aaigrid_stats(path: Path) -> dict[str, float | int]:
    data = []
    nodata = -9999.0
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[:6]:
        parts = line.split()
        if len(parts) == 2 and parts[0].lower() == "nodata_value":
            nodata = float(parts[1])
            break
    for line in lines[6:]:
        if line.strip():
            data.extend(float(item) for item in line.split())
    array = np.array(data, dtype=np.float32)
    valid = array[np.isfinite(array) & (array != nodata)]
    if valid.size == 0:
        return {"finite_count": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "finite_count": int(valid.size),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
    }


def _compact_log_tail(text: str, line_count: int = 60) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").splitlines() if line.strip()]
    return "\n".join(lines[-line_count:])


def _validate_speed_output(args: argparse.Namespace) -> None:
    stats = _aaigrid_stats(Path(args.speed_output))
    if int(stats["finite_count"]) == 0:
        raise RuntimeError("OpenFOAM output contained no finite wind speed values.")
    speed_limit = max(float(args.max_output_speed_mps), float(args.speed_mps) * 8.0)
    if float(stats["max"]) > speed_limit:
        raise RuntimeError(f"OpenFOAM output speed exceeded sanity limit: stats={stats}, limit_mps={speed_limit:.3f}")


def _derive_roughness(
    src: rasterio.io.DatasetReader,
    *,
    ny: int,
    nx: int,
    water_z0_m: float,
    grass_z0_m: float,
    tree_z0_m: float,
) -> tuple[np.ndarray, dict]:
    if src.count < 4:
        roughness = np.full((ny, nx), grass_z0_m, dtype=np.float32)
        return roughness, {
            "z0_m": float(grass_z0_m),
            "source": "default_grass",
            "water_fraction": 0.0,
            "tree_fraction": 0.0,
            "min_z0_m": float(grass_z0_m),
            "max_z0_m": float(grass_z0_m),
            "mean_z0_m": float(grass_z0_m),
            "median_z0_m": float(grass_z0_m),
        }

    fuel_model = src.read(4, out_shape=(ny, nx), resampling=Resampling.nearest, masked=True).astype(np.float32).filled(np.nan)
    fuel_model_codes = np.where(np.isfinite(fuel_model), np.round(fuel_model), -9999).astype(np.int16)
    water_mask = fuel_model_codes == 98
    tree_mask = np.zeros((ny, nx), dtype=bool)
    canopy_cover = None
    canopy_height_m = None
    if src.count >= 6:
        canopy_cover = src.read(5, out_shape=(ny, nx), resampling=Resampling.bilinear, masked=True).astype(np.float32).filled(np.nan)
        canopy_height_x10 = src.read(6, out_shape=(ny, nx), resampling=Resampling.bilinear, masked=True).astype(np.float32).filled(np.nan)
        canopy_height_m = canopy_height_x10 / 10.0
        tree_mask = np.isfinite(canopy_height_m) & ((canopy_height_m >= 2.0) | (canopy_cover >= 30.0))

    roughness = np.full((ny, nx), grass_z0_m, dtype=np.float32)
    if canopy_cover is not None and canopy_height_m is not None:
        cover_fraction = np.clip(np.nan_to_num(canopy_cover, nan=0.0) / 100.0, 0.0, 1.0)
        height_fraction = np.clip(np.nan_to_num(canopy_height_m, nan=0.0) / 15.0, 0.0, 1.0)
        canopy_weight = np.maximum(cover_fraction, height_fraction)
        roughness = grass_z0_m + (tree_z0_m - grass_z0_m) * canopy_weight.astype(np.float32)
    roughness[tree_mask] = np.maximum(roughness[tree_mask], tree_z0_m * 0.65)
    roughness[water_mask] = water_z0_m
    finite = roughness[np.isfinite(roughness)]
    z0_m = float(np.clip(np.median(finite), min(water_z0_m, grass_z0_m, tree_z0_m), max(water_z0_m, grass_z0_m, tree_z0_m)))
    return roughness.astype(np.float32), {
        "z0_m": z0_m,
        "source": "spatial_landscape_fuel_model_and_canopy",
        "mode": "terrain_patch_nonuniform_z0",
        "water_fraction": float(np.count_nonzero(water_mask) / water_mask.size),
        "tree_fraction": float(np.count_nonzero(tree_mask & ~water_mask) / tree_mask.size),
        "min_z0_m": float(np.min(finite)),
        "max_z0_m": float(np.max(finite)),
        "mean_z0_m": float(np.mean(finite)),
        "median_z0_m": float(np.median(finite)),
        "p05_z0_m": float(np.percentile(finite, 5)),
        "p95_z0_m": float(np.percentile(finite, 95)),
    }


def _load_terrain(
    elevation_file: Path,
    mesh_resolution_m: float,
    max_horizontal_cells: int,
    *,
    water_z0_m: float,
    grass_z0_m: float,
    tree_z0_m: float,
) -> dict:
    with rasterio.open(elevation_file) as src:
        bounds = src.bounds
        width_m = float(bounds.right - bounds.left)
        height_m = float(bounds.top - bounds.bottom)
        nx = max(2, int(round(width_m / mesh_resolution_m)))
        ny = max(2, int(round(height_m / mesh_resolution_m)))
        while nx * ny > max_horizontal_cells:
            nx = max(2, int(math.floor(nx * 0.9)))
            ny = max(2, int(math.floor(ny * 0.9)))
        dem = (
            src.read(1, out_shape=(ny + 1, nx + 1), resampling=Resampling.bilinear, masked=True)
            .astype(np.float32)
            .filled(np.nan)
        )
        roughness_grid, roughness = _derive_roughness(
            src,
            ny=ny,
            nx=nx,
            water_z0_m=water_z0_m,
            grass_z0_m=grass_z0_m,
            tree_z0_m=tree_z0_m,
        )

    dem = np.flipud(dem)
    roughness_grid = np.flipud(roughness_grid).astype(np.float32)
    finite = dem[np.isfinite(dem)]
    if finite.size == 0:
        raise RuntimeError("DEM has no finite terrain values for OpenFOAM mesh generation.")
    dem = np.where(np.isfinite(dem), dem, float(np.nanmedian(finite))).astype(np.float32)
    terrain_z_offset_m = float(np.nanmin(dem))
    terrain_absolute_min_m = float(np.nanmin(dem))
    terrain_absolute_max_m = float(np.nanmax(dem))
    dem = (dem - terrain_z_offset_m).astype(np.float32)
    cellsize = max(width_m / nx, height_m / ny)
    dx = cellsize
    dy = cellsize
    return {
        "terrain": dem,
        "terrain_z_offset_m": terrain_z_offset_m,
        "terrain_absolute_min_m": terrain_absolute_min_m,
        "terrain_absolute_max_m": terrain_absolute_max_m,
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "cellsize": float(cellsize),
        "left": float(bounds.left),
        "bottom": float(bounds.bottom),
        "roughness": roughness,
        "roughness_z0_grid": roughness_grid,
    }


def _write_block_mesh_dict(case_dir: Path, terrain_info: dict, vertical_cells: int, domain_height_m: float) -> None:
    nx = terrain_info["nx"]
    ny = terrain_info["ny"]
    terrain = terrain_info["terrain"]
    left = terrain_info["left"]
    bottom = terrain_info["bottom"]
    dx = terrain_info["dx"]
    dy = terrain_info["dy"]
    top_z = float(np.nanmax(terrain) + max(domain_height_m, 4.0 * max(dx, dy)))
    system_dir = case_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    bottom_count = (nx + 1) * (ny + 1)

    def vid(i: int, j: int, top: bool = False) -> int:
        return (bottom_count if top else 0) + j * (nx + 1) + i

    vertices = []
    for layer in ("bottom", "top"):
        for j in range(ny + 1):
            for i in range(nx + 1):
                z = float(terrain[j, i]) if layer == "bottom" else top_z
                vertices.append(f"    ({left + i * dx:.6f} {bottom + j * dy:.6f} {z:.6f})")

    blocks: list[str] = []
    terrain_faces: list[str] = []
    top_faces: list[str] = []
    west_faces: list[str] = []
    east_faces: list[str] = []
    south_faces: list[str] = []
    north_faces: list[str] = []
    for j in range(ny):
        for i in range(nx):
            v0, v1, v2, v3 = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            v4, v5, v6, v7 = vid(i, j, True), vid(i + 1, j, True), vid(i + 1, j + 1, True), vid(i, j + 1, True)
            blocks.append(f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) (1 1 {vertical_cells}) simpleGrading (1 1 8)")
            terrain_faces.append(f"        ({v0} {v3} {v2} {v1})")
            top_faces.append(f"        ({v4} {v5} {v6} {v7})")
            if i == 0:
                west_faces.append(f"        ({v3} {v0} {v4} {v7})")
            if i == nx - 1:
                east_faces.append(f"        ({v1} {v2} {v6} {v5})")
            if j == 0:
                south_faces.append(f"        ({v0} {v1} {v5} {v4})")
            if j == ny - 1:
                north_faces.append(f"        ({v2} {v3} {v7} {v6})")

    def patch(name: str, patch_type: str, faces: list[str]) -> str:
        return "\n".join([f"    {name}", "    {", f"        type {patch_type};", "        faces", "        (", *faces, "        );", "    }"])

    text = "\n".join(
        [
            _foam_header("dictionary", "blockMeshDict"),
            "convertToMeters 1;",
            "vertices",
            "(",
            *vertices,
            ");",
            "blocks",
            "(",
            *blocks,
            ");",
            "edges ();",
            "boundary",
            "(",
            patch("terrain", "wall", terrain_faces),
            patch("top", "patch", top_faces),
            patch("west", "patch", west_faces),
            patch("east", "patch", east_faces),
            patch("south", "patch", south_faces),
            patch("north", "patch", north_faces),
            ");",
            "mergePatchPairs ();",
            "",
        ]
    )
    (system_dir / "blockMeshDict").write_text(text, encoding="utf-8")


def _side_roles(u: float, v: float) -> dict[str, str]:
    return {
        "west": "inlet" if u > 0 else "outlet",
        "east": "inlet" if u < 0 else "outlet",
        "south": "inlet" if v > 0 else "outlet",
        "north": "inlet" if v < 0 else "outlet",
    }


def _side_boundary(roles: dict[str, str], inlet: str, outlet: str) -> str:
    lines = []
    for name in ("west", "east", "south", "north"):
        lines.extend([f"    {name}", "    {", f"        {inlet if roles[name] == 'inlet' else outlet}", "    }"])
    return "\n".join(lines)


def _terrain_z0_field_text(terrain_info: dict) -> str:
    z0_grid = terrain_info.get("roughness_z0_grid")
    if z0_grid is None:
        z0_m = max(float(terrain_info["roughness"]["z0_m"]), 1.0e-6)
        return f"uniform {z0_m:.8f}"
    z0 = np.asarray(z0_grid, dtype=np.float32)
    expected_shape = (int(terrain_info["ny"]), int(terrain_info["nx"]))
    if z0.shape != expected_shape:
        raise RuntimeError(f"OpenFOAM roughness grid shape {z0.shape} did not match terrain cells {expected_shape}.")
    values = np.clip(np.where(np.isfinite(z0), z0, float(terrain_info["roughness"]["z0_m"])), 1.0e-6, 10.0).ravel()
    lines = ["nonuniform List<scalar>", str(values.size), "("]
    lines.extend(f"{float(value):.8f}" for value in values)
    lines.append(")")
    return "\n            ".join(lines)


def _write_case_files(case_dir: Path, terrain_info: dict, wind_u: float, wind_v: float, reference_height_m: float) -> None:
    zero_dir = case_dir / "0"
    constant_dir = case_dir / "constant"
    system_dir = case_dir / "system"
    zero_dir.mkdir(parents=True, exist_ok=True)
    constant_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)
    velocity = f"({wind_u:.8f} {wind_v:.8f} 0)"
    speed = max(math.hypot(wind_u, wind_v), 0.1)
    z0_m = max(float(terrain_info["roughness"]["z0_m"]), 1.0e-6)
    terrain_z0 = _terrain_z0_field_text(terrain_info)
    flow_mag = max(speed, 1.0e-6)
    flow_dir = f"({wind_u / flow_mag:.8f} {wind_v / flow_mag:.8f} 0)"
    kappa = 0.41
    cmu = 0.09
    u_star = speed * kappa / max(math.log((reference_height_m + z0_m) / z0_m), 1.0)
    k_value = max((u_star**2) / math.sqrt(cmu), 1.0e-5)
    epsilon_value = max((u_star**3) / (kappa * (reference_height_m + z0_m)), 1.0e-6)
    roles = _side_roles(wind_u, wind_v)

    def abl_boundary(field_type: str, fallback_value: str) -> str:
        return "\n".join(
            [
                f"type {field_type};",
                f"        flowDir {flow_dir};",
                "        zDir (0 0 1);",
                f"        Uref {speed:.8f};",
                f"        Zref {reference_height_m:.8f};",
                f"        z0 uniform {z0_m:.8f};",
                "        zGround uniform 0;",
                f"        value uniform {fallback_value};",
            ]
        )

    (zero_dir / "U").write_text(
        "\n".join(
            [
                _foam_header("volVectorField", "U"),
                "dimensions [0 1 -1 0 0 0 0];",
                f"internalField uniform {velocity};",
                "boundaryField",
                "{",
                _side_boundary(
                    roles,
                    abl_boundary("atmBoundaryLayerInletVelocity", velocity),
                    "type pressureInletOutletVelocity;\n        value uniform (0 0 0);",
                ),
                "    top { type slip; }",
                "    terrain { type noSlip; }",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (zero_dir / "p").write_text(
        "\n".join(
            [
                _foam_header("volScalarField", "p"),
                "dimensions [0 2 -2 0 0 0 0];",
                "internalField uniform 0;",
                "boundaryField",
                "{",
                _side_boundary(roles, "type zeroGradient;", "type fixedValue;\n        value uniform 0;"),
                "    top { type zeroGradient; }",
                "    terrain { type zeroGradient; }",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for name, dimensions, internal, wall_type in [
        ("k", "[0 2 -2 0 0 0 0]", f"{k_value:.8f}", "kqRWallFunction"),
        ("epsilon", "[0 2 -3 0 0 0 0]", f"{epsilon_value:.8f}", "epsilonWallFunction"),
    ]:
        (zero_dir / name).write_text(
            "\n".join(
                [
                    _foam_header("volScalarField", name),
                    f"dimensions {dimensions};",
                    f"internalField uniform {internal};",
                    "boundaryField",
                    "{",
                    _side_boundary(
                        roles,
                        abl_boundary(f"atmBoundaryLayerInlet{name[0].upper()}{name[1:]}", internal),
                        f"type inletOutlet;\n        inletValue uniform {internal};\n        value uniform {internal};",
                    ),
                    "    top { type zeroGradient; }",
                    f"    terrain {{ type {wall_type}; value uniform {internal}; }}",
                    "}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    (zero_dir / "nut").write_text(
        "\n".join(
            [
                _foam_header("volScalarField", "nut"),
                "dimensions [0 2 -1 0 0 0 0];",
                "internalField uniform 0;",
                "boundaryField",
                "{",
                _side_boundary(roles, "type calculated;\n        value uniform 0;", "type calculated;\n        value uniform 0;"),
                "    top { type calculated; value uniform 0; }",
                "    terrain",
                "    {",
                "        type nutkAtmRoughWallFunction;",
                f"        z0 {terrain_z0};",
                "        value uniform 0.1;",
                "    }",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for name, content in {
        "transportProperties": "transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] 1.5e-05;\n",
        "physicalProperties": "viscosityModel constant;\nnu [0 2 -1 0 0 0 0] 1.5e-05;\n",
        "turbulenceProperties": "simulationType RAS;\nRAS { RASModel kEpsilon; turbulence on; printCoeffs on; }\n",
        "momentumTransport": "simulationType RAS;\nRAS { model kEpsilon; turbulence on; printCoeffs on; }\n",
    }.items():
        (constant_dir / name).write_text(_foam_header("dictionary", name) + content, encoding="utf-8")

    (system_dir / "controlDict").write_text(
        _foam_header("dictionary", "controlDict")
        + "\n".join(
            [
                "application foamRun;",
                'libs ("libatmosphericModels.so");',
                "startFrom startTime;",
                "startTime 0;",
                "stopAt endTime;",
                "endTime 250;",
                "deltaT 1;",
                "writeControl timeStep;",
                "writeInterval 250;",
                "purgeWrite 0;",
                "writeFormat ascii;",
                "writePrecision 8;",
                "timeFormat general;",
                "runTimeModifiable true;",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (system_dir / "fvSchemes").write_text(
        _foam_header("dictionary", "fvSchemes")
        + "\n".join(
            [
                "ddtSchemes { default steadyState; }",
                "gradSchemes { default Gauss linear; }",
                "divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); div(phi,k) bounded Gauss upwind; div(phi,epsilon) bounded Gauss upwind; div((nuEff*dev2(T(grad(U))))) Gauss linear; }",
                "laplacianSchemes { default Gauss linear corrected; }",
                "interpolationSchemes { default linear; }",
                "snGradSchemes { default corrected; }",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (system_dir / "fvSolution").write_text(
        _foam_header("dictionary", "fvSolution")
        + "\n".join(
            [
                "solvers { p { solver PCG; preconditioner DIC; tolerance 1e-7; relTol 0.01; } Phi { solver PCG; preconditioner DIC; tolerance 1e-7; relTol 0.01; } U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.05; } k { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.05; } epsilon { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.05; } }",
                "SIMPLE { nNonOrthogonalCorrectors 1; residualControl { p 1e-3; U 1e-4; k 1e-4; epsilon 1e-3; } }",
                "relaxationFactors { fields { p 0.3; } equations { U 0.5; k 0.5; epsilon 0.5; } }",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (system_dir / "setAtmBoundaryLayerDict").write_text(
        _foam_header("dictionary", "setAtmBoundaryLayerDict")
        + "\n".join(
            [
                f"flowDir {flow_dir};",
                "zDir (0 0 1);",
                f"Uref {speed:.8f};",
                f"Zref {reference_height_m:.8f};",
                f"z0 uniform {z0_m:.8f};",
                "zGround uniform 0;",
                "",
            ]
        ),
        encoding="utf-8",
    )

    points = []
    terrain = terrain_info["terrain"]
    for row in range(terrain_info["ny"]):
        j = terrain_info["ny"] - 1 - row
        for i in range(terrain_info["nx"]):
            x = terrain_info["left"] + (i + 0.5) * terrain_info["dx"]
            y = terrain_info["bottom"] + (j + 0.5) * terrain_info["dy"]
            z = float(np.max(terrain[j : j + 2, i : i + 2])) + reference_height_m
            points.append(f"        ({x:.6f} {y:.6f} {z:.6f})")
    (system_dir / "sampleDict").write_text(
        "\n".join(
            [
                _foam_header("dictionary", "sampleDict"),
                "windGrid",
                "{",
                "    type sets;",
                '    libs ("libsampling.so");',
                "    executeControl writeTime;",
                "    writeControl writeTime;",
                "    interpolationScheme cellPoint;",
                "    setFormat raw;",
                "    fields (U k);",
                "    sets",
                "    {",
                "        windGrid",
                "        {",
                "            type points;",
                "            ordered yes;",
                "            points",
                "            (",
                *points,
                "            );",
                "        }",
                "    }",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _find_sample_output(case_dir: Path) -> Path:
    post_processing = case_dir / "postProcessing"
    candidates = [
        *post_processing.glob("**/windGrid.xy"),
        *post_processing.glob("**/windGrid_U*"),
        *post_processing.glob("**/points_U*"),
    ]
    candidates = sorted(candidate for candidate in candidates if candidate.is_file())
    if not candidates:
        raise RuntimeError("OpenFOAM sampling did not create windGrid output under postProcessing.")
    return candidates[-1]


def _parse_sample_vectors(path: Path, expected_count: int) -> np.ndarray:
    vectors, _ = _parse_sample_fields(path, expected_count)
    return vectors


def _parse_sample_fields(path: Path, expected_count: int) -> tuple[np.ndarray, np.ndarray | None]:
    number_pattern = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
    vectors = []
    k_values = []
    saw_k = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("#"):
            continue
        values = [float(item) for item in number_pattern.findall(line)]
        if len(values) >= 7:
            vectors.append(tuple(values[4:7]))
            if len(values) >= 8:
                k_values.append(values[7])
                saw_k = True
    if len(vectors) != expected_count:
        raise RuntimeError(f"Expected {expected_count} sampled vectors in {path}, found {len(vectors)}.")
    if saw_k and len(k_values) != expected_count:
        raise RuntimeError(f"Expected {expected_count} sampled k values in {path}, found {len(k_values)}.")
    return np.array(vectors, dtype=np.float32), (np.array(k_values, dtype=np.float32) if saw_k else None)


def _clear_solver_outputs(case_dir: Path) -> None:
    post_processing = case_dir / "postProcessing"
    if post_processing.exists():
        shutil.rmtree(post_processing)
    for child in case_dir.iterdir():
        if not child.is_dir() or child.name == "0":
            continue
        try:
            float(child.name)
        except ValueError:
            continue
        shutil.rmtree(child)


def _write_outputs(args: argparse.Namespace, terrain_info: dict, sample_path: Path) -> None:
    nx = terrain_info["nx"]
    ny = terrain_info["ny"]
    vectors, k_values = _parse_sample_fields(sample_path, nx * ny)
    u = vectors[:, 0].reshape(ny, nx).astype(np.float32)
    v = vectors[:, 1].reshape(ny, nx).astype(np.float32)
    speed, direction = _speed_direction_from_uv(u, v)
    cloud = np.zeros_like(speed, dtype=np.float32)
    for path, data in [(args.speed_output, speed), (args.direction_output, direction), (args.u_output, u), (args.v_output, v)]:
        _write_aaigrid(Path(path), data, xllcorner=terrain_info["left"], yllcorner=terrain_info["bottom"], cellsize=terrain_info["cellsize"])
    cloud_output = getattr(args, "cloud_output", "")
    if cloud_output:
        _write_aaigrid(Path(cloud_output), cloud, xllcorner=terrain_info["left"], yllcorner=terrain_info["bottom"], cellsize=terrain_info["cellsize"])
    if k_values is not None:
        k_grid = np.maximum(k_values.reshape(ny, nx).astype(np.float32), 0.0)
        turbulence_intensity_pct = (np.sqrt((2.0 / 3.0) * k_grid) / np.maximum(speed, 0.1) * 100.0).astype(np.float32)
        turbulence_intensity_pct[~np.isfinite(turbulence_intensity_pct)] = np.nan
        turbulence_output = getattr(args, "turbulence_intensity_output", "")
        if turbulence_output:
            _write_aaigrid(
                Path(turbulence_output),
                turbulence_intensity_pct,
                xllcorner=terrain_info["left"],
                yllcorner=terrain_info["bottom"],
                cellsize=terrain_info["cellsize"],
            )
        tke_output = getattr(args, "tke_output", "")
        if tke_output:
            _write_aaigrid(Path(tke_output), k_grid, xllcorner=terrain_info["left"], yllcorner=terrain_info["bottom"], cellsize=terrain_info["cellsize"])


def _sample_outputs(args: argparse.Namespace, terrain_info: dict) -> Path:
    post_processing = Path(args.case_dir) / "postProcessing"
    if post_processing.exists():
        shutil.rmtree(post_processing)
    _run_sampler(Path(args.case_dir))
    sample_path = _find_sample_output(Path(args.case_dir))
    _write_outputs(args, terrain_info, sample_path)
    if not _aaigrid_has_finite(Path(args.speed_output)) or not _aaigrid_has_finite(Path(args.direction_output)):
        raise RuntimeError("OpenFOAM sample output contained no finite wind values.")
    _validate_speed_output(args)
    return sample_path


def _parse_residual_summary(solver_log: str) -> dict:
    pattern = re.compile(
        r"Solving for\s+([^,]+),\s+Initial residual\s*=\s*([-+0-9.eE]+),\s+Final residual\s*=\s*([-+0-9.eE]+),\s+No Iterations\s+([0-9]+)"
    )
    by_field: dict[str, dict[str, float | int]] = {}
    for match in pattern.finditer(solver_log):
        field = match.group(1).strip()
        by_field[field] = {
            "initial": float(match.group(2)),
            "final": float(match.group(3)),
            "iterations": int(match.group(4)),
        }
    if not by_field:
        return {"fields": {}, "converged": False, "reason": "No residual lines were found in the solver log."}

    thresholds = {"p": 1.0e-3, "Ux": 1.0e-4, "Uy": 1.0e-4, "Uz": 1.0e-4, "k": 1.0e-4, "epsilon": 1.0e-3}
    required_fields = {"Ux", "Uy", "k", "epsilon"}
    failures = []
    missing_fields = sorted(field for field in required_fields if field not in by_field)
    for field, threshold in thresholds.items():
        if field in by_field and float(by_field[field]["final"]) > threshold:
            failures.append({"field": field, "final": float(by_field[field]["final"]), "threshold": threshold})
    return {
        "fields": by_field,
        "converged": not failures and not missing_fields,
        "failures": failures,
        "missing_fields": missing_fields,
    }


def _run_set_atm_boundary_layer(case_dir: Path) -> str:
    if not _foam_command_exists("setAtmBoundaryLayer"):
        raise RuntimeError("OpenFOAM setAtmBoundaryLayer utility was not found. OpenFOAM 13 is required for the ABL runner.")
    case_arg = _windows_to_wsl_path(case_dir)
    return _run_wsl(f'setAtmBoundaryLayer -case "{case_arg}"')


def _write_runner_artifacts(output_dir: Path, logs: list[str], summary: dict | None) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "openfoam_wsl_runner.log").write_text("\n\n".join(logs), encoding="utf-8")
        if summary is not None:
            (output_dir / "openfoam_wsl_terrain_runner_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    request_path = Path(args.request_json)
    if request_path.exists():
        json.loads(request_path.read_text(encoding="utf-8"))

    output_dir = Path(args.output_dir)
    case_dir = Path(args.case_dir)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    summary: dict | None = None

    try:
        wind_u, wind_v = _uv_from_speed_direction(float(args.speed_mps), float(args.direction_deg))
        terrain_info = _load_terrain(
            Path(args.elevation_file),
            float(args.mesh_resolution_m),
            int(args.max_horizontal_cells),
            water_z0_m=float(args.water_z0_m),
            grass_z0_m=float(args.grass_z0_m),
            tree_z0_m=float(args.tree_z0_m),
        )
        _write_block_mesh_dict(case_dir, terrain_info, int(args.vertical_cells), float(args.domain_height_m))
        _write_case_files(case_dir, terrain_info, wind_u, wind_v, float(args.reference_height_m))

        logs.append("[stage] environment")
        logs.append(_run_wsl('foamRun -help >/dev/null', timeout_seconds=60))
        logs.append("[stage] blockMesh")
        logs.append(_run_wsl(f'blockMesh -case "{_windows_to_wsl_path(case_dir)}"'))
        logs.append("[stage] checkMesh")
        check_mesh = _run_check_mesh(case_dir)
        logs.append(check_mesh["summary"])
        if not check_mesh["passed"]:
            raise RuntimeError(f"OpenFOAM checkMesh failed. Summary:\n{check_mesh['summary']}")
        logs.append("[stage] renumberMesh")
        logs.append(_renumber_mesh(case_dir))
        logs.append("[stage] setAtmBoundaryLayer")
        logs.append(_run_set_atm_boundary_layer(case_dir))
        potential_init = False
        should_run_potential = args.enable_potential_init and not args.skip_potential_init and _foam_command_exists("potentialFoam")
        if should_run_potential:
            logs.append("[stage] potentialFoam")
            try:
                logs.append(_run_potential_solver(case_dir))
                potential_init = True
            except Exception as exc:
                logs.append(f"potentialFoam skipped after failure: {exc}")
        logs.append("[stage] foamRun")
        solver_log = _run_solver(case_dir)
        logs.append(solver_log)
        residual_summary = _parse_residual_summary(solver_log)
        if not residual_summary["converged"]:
            raise RuntimeError(f"OpenFOAM RANS solve did not meet residual checks: {residual_summary}")
        logs.append("[stage] sample")
        sample_path = _sample_outputs(args, terrain_info)

        summary = {
            "runner": "openfoam_wsl_terrain_runner",
            "note": "Experimental WSL/OpenFOAM neutral ABL terrain-following RANS case. Compare against WindNinja before scientific use.",
            "case_dir": str(case_dir),
            "sample_path": str(sample_path),
            "solver_mode": "steady_incompressible_rans_abl",
            "status": "completed",
            "potential_initialization": potential_init,
            "wind_speed_mps": float(args.speed_mps),
            "wind_direction_deg": float(args.direction_deg),
            "speed_output_stats": _aaigrid_stats(Path(args.speed_output)),
            "turbulence_intensity_output_stats_pct": _aaigrid_stats(Path(args.turbulence_intensity_output))
            if getattr(args, "turbulence_intensity_output", "") and Path(args.turbulence_intensity_output).exists()
            else None,
            "tke_output_stats": _aaigrid_stats(Path(args.tke_output)) if getattr(args, "tke_output", "") and Path(args.tke_output).exists() else None,
            "max_output_speed_mps": float(args.max_output_speed_mps),
            "reference_height_m": float(args.reference_height_m),
            "mesh_resolution_m": float(args.mesh_resolution_m),
            "horizontal_cells": int(terrain_info["nx"] * terrain_info["ny"]),
            "vertical_cells": int(args.vertical_cells),
            "total_cells": int(terrain_info["nx"] * terrain_info["ny"] * int(args.vertical_cells)),
            "domain_height_m": float(args.domain_height_m),
            "terrain_z_offset_m": float(terrain_info["terrain_z_offset_m"]),
            "terrain_absolute_min_m": float(terrain_info["terrain_absolute_min_m"]),
            "terrain_absolute_max_m": float(terrain_info["terrain_absolute_max_m"]),
            "roughness": terrain_info["roughness"],
            "check_mesh": check_mesh,
            "residual_summary": residual_summary,
        }
    except Exception as exc:
        summary = {
            "runner": "openfoam_wsl_terrain_runner",
            "status": "failed",
            "error": str(exc),
            "case_dir": str(case_dir),
        }
        logs.append("[stage] failure")
        logs.append(traceback.format_exc())
        _write_runner_artifacts(output_dir, logs, summary)
        raise
    else:
        _write_runner_artifacts(output_dir, logs, summary)


if __name__ == "__main__":
    main()
