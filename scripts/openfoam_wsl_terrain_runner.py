from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
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
    parser.add_argument("--vertical-cells", type=int, default=int(os.environ.get("PONDWIND_OPENFOAM_VERTICAL_CELLS", "8")))
    parser.add_argument("--domain-height-m", type=float, default=float(os.environ.get("PONDWIND_OPENFOAM_DOMAIN_HEIGHT_M", "250")))
    parser.add_argument("--max-horizontal-cells", type=int, default=int(os.environ.get("PONDWIND_OPENFOAM_MAX_HORIZONTAL_CELLS", "12000")))
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


def _run_wsl(command: str, timeout_seconds: int = 1800) -> str:
    bashrc_override = os.environ.get("PONDWIND_OPENFOAM_BASHRC", "").strip()
    source_lines = ["set +e", 'foam_bashrc=""']
    if bashrc_override:
        source_lines.append(f'if [ -f "{bashrc_override}" ]; then foam_bashrc="{bashrc_override}"; fi')
    source_lines.extend(
        [
            'if [ -z "\\$foam_bashrc" ] && [ -f "/opt/openfoam13/etc/bashrc" ]; then foam_bashrc="/opt/openfoam13/etc/bashrc"; fi',
            'if [ -z "\\$foam_bashrc" ]; then for f in /opt/openfoam*/etc/bashrc /usr/lib/openfoam/openfoam*/etc/bashrc; do if [ -f "\\$f" ]; then foam_bashrc="\\$f"; break; fi; done; fi',
            'if [ -n "\\$foam_bashrc" ]; then . "\\$foam_bashrc"; fi',
            "set -e",
        ]
    )
    shell_command = "\n".join([*source_lines, command])
    completed = subprocess.run(
        ["wsl", "bash", "-lc", shell_command],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return (completed.stdout or "") + (completed.stderr or "")


def _foam_command_exists(command: str) -> bool:
    try:
        _run_wsl(f"command -v {command} >/dev/null 2>&1", timeout_seconds=30)
        return True
    except Exception:
        return False


def _run_solver(case_dir: Path) -> str:
    case_arg = _windows_to_wsl_path(case_dir)
    if _foam_command_exists("foamRun"):
        return _run_wsl(f'foamRun -solver incompressibleFluid -case "{case_arg}"')
    return _run_wsl(f'simpleFoam -case "{case_arg}"')


def _run_sampler(case_dir: Path) -> str:
    case_arg = _windows_to_wsl_path(case_dir)
    try:
        return _run_wsl(f'postProcess -case "{case_arg}" -latestTime -func sample')
    except Exception:
        return _run_wsl(f'sample -case "{case_arg}" -latestTime')


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


def _load_terrain(elevation_file: Path, mesh_resolution_m: float, max_horizontal_cells: int) -> dict:
    with rasterio.open(elevation_file) as src:
        bounds = src.bounds
        width_m = float(bounds.right - bounds.left)
        height_m = float(bounds.top - bounds.bottom)
        nx = max(2, int(round(width_m / mesh_resolution_m)))
        ny = max(2, int(round(height_m / mesh_resolution_m)))
        while nx * ny > max_horizontal_cells:
            nx = max(2, int(math.floor(nx * 0.9)))
            ny = max(2, int(math.floor(ny * 0.9)))
        dem = src.read(1, out_shape=(ny + 1, nx + 1), resampling=Resampling.bilinear, masked=True).filled(np.nan).astype(np.float32)

    dem = np.flipud(dem)
    finite = dem[np.isfinite(dem)]
    if finite.size == 0:
        raise RuntimeError("DEM has no finite terrain values for OpenFOAM mesh generation.")
    dem = np.where(np.isfinite(dem), dem, float(np.nanmedian(finite))).astype(np.float32)
    dx = width_m / nx
    dy = height_m / ny
    return {
        "terrain": dem,
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "cellsize": float((dx + dy) * 0.5),
        "left": float(bounds.left),
        "bottom": float(bounds.bottom),
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
            blocks.append(f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) (1 1 {vertical_cells}) simpleGrading (1 1 1)")
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


def _write_case_files(case_dir: Path, terrain_info: dict, wind_u: float, wind_v: float) -> None:
    zero_dir = case_dir / "0"
    constant_dir = case_dir / "constant"
    system_dir = case_dir / "system"
    zero_dir.mkdir(parents=True, exist_ok=True)
    constant_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)
    velocity = f"({wind_u:.8f} {wind_v:.8f} 0)"
    speed = max(math.hypot(wind_u, wind_v), 0.1)
    k_value = 1.5 * (speed * 0.08) ** 2
    epsilon_value = max((0.09 ** 0.75) * (k_value ** 1.5) / 10.0, 1.0e-5)
    roles = _side_roles(wind_u, wind_v)

    (zero_dir / "U").write_text(
        "\n".join(
            [
                _foam_header("volVectorField", "U"),
                "dimensions [0 1 -1 0 0 0 0];",
                f"internalField uniform {velocity};",
                "boundaryField",
                "{",
                _side_boundary(roles, f"type fixedValue;\n        value uniform {velocity};", "type zeroGradient;"),
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
                    _side_boundary(roles, f"type fixedValue;\n        value uniform {internal};", "type zeroGradient;"),
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
                "    terrain { type nutkWallFunction; value uniform 0; }",
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
                "application simpleFoam;",
                "startFrom startTime;",
                "startTime 0;",
                "stopAt endTime;",
                "endTime 150;",
                "deltaT 1;",
                "writeControl timeStep;",
                "writeInterval 150;",
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
                "divSchemes { default none; div(phi,U) bounded Gauss upwind; div(phi,k) bounded Gauss upwind; div(phi,epsilon) bounded Gauss upwind; div((nuEff*dev2(T(grad(U))))) Gauss linear; }",
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
                "solvers { p { solver PCG; preconditioner DIC; tolerance 1e-7; relTol 0.01; } U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; } k { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; } epsilon { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; } }",
                "SIMPLE { nNonOrthogonalCorrectors 1; residualControl { p 1e-3; U 1e-4; k 1e-4; epsilon 1e-4; } }",
                "relaxationFactors { equations { U 0.7; k 0.7; epsilon 0.7; } }",
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
            z = float(np.mean(terrain[j : j + 2, i : i + 2])) + 10.0
            points.append(f"        ({x:.6f} {y:.6f} {z:.6f})")
    (system_dir / "sampleDict").write_text(
        "\n".join(
            [
                _foam_header("dictionary", "sampleDict"),
                "interpolationScheme cellPoint;",
                "setFormat raw;",
                "sets",
                "(",
                "    windGrid",
                "    {",
                "        type cloud;",
                "        axis xyz;",
                "        points",
                "        (",
                *points,
                "        );",
                "    }",
                ");",
                "fields (U);",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _find_sample_output(case_dir: Path) -> Path:
    candidates = sorted((case_dir / "postProcessing").glob("**/windGrid_U*"))
    if not candidates:
        raise RuntimeError("OpenFOAM sampling did not create windGrid_U output under postProcessing.")
    return candidates[-1]


def _parse_sample_vectors(path: Path, expected_count: int) -> np.ndarray:
    pattern = re.compile(r"^\s*[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+(?:\(\s*)?([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)?")
    vectors = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            vectors.append((float(match.group(1)), float(match.group(2)), float(match.group(3))))
    if len(vectors) != expected_count:
        raise RuntimeError(f"Expected {expected_count} sampled vectors in {path}, found {len(vectors)}.")
    return np.array(vectors, dtype=np.float32)


def _latest_time_dir(case_dir: Path) -> Path:
    candidates = []
    for child in case_dir.iterdir():
        if child.is_dir():
            try:
                candidates.append((float(child.name), child))
            except ValueError:
                continue
    if not candidates:
        raise RuntimeError("OpenFOAM case has no numeric time output directory.")
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _parse_internal_u_vectors(u_path: Path, expected_count: int) -> np.ndarray:
    vector_pattern = re.compile(r"^\s*\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)")
    vectors = []
    in_internal_field = False
    for line in u_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("internalField"):
            in_internal_field = True
            continue
        if in_internal_field and stripped == ");":
            break
        if not in_internal_field:
            continue
        match = vector_pattern.match(stripped)
        if match:
            vectors.append((float(match.group(1)), float(match.group(2)), float(match.group(3))))
    if len(vectors) != expected_count:
        raise RuntimeError(f"Expected {expected_count} internal U vectors in {u_path}, found {len(vectors)}.")
    return np.array(vectors, dtype=np.float32)


def _write_outputs_from_volume_field(args: argparse.Namespace, terrain_info: dict) -> Path:
    nx = terrain_info["nx"]
    ny = terrain_info["ny"]
    vertical_cells = int(args.vertical_cells)
    latest_dir = _latest_time_dir(Path(args.case_dir))
    vectors = _parse_internal_u_vectors(latest_dir / "U", nx * ny * vertical_cells)
    columns = vectors.reshape(ny * nx, vertical_cells, 3)
    lowest_layer = columns[:, 0, :]
    u_south_to_north = lowest_layer[:, 0].reshape(ny, nx).astype(np.float32)
    v_south_to_north = lowest_layer[:, 1].reshape(ny, nx).astype(np.float32)
    u = np.flipud(u_south_to_north)
    v = np.flipud(v_south_to_north)
    speed, direction = _speed_direction_from_uv(u, v)
    for path, data in [(args.speed_output, speed), (args.direction_output, direction), (args.u_output, u), (args.v_output, v)]:
        _write_aaigrid(Path(path), data, xllcorner=terrain_info["left"], yllcorner=terrain_info["bottom"], cellsize=terrain_info["cellsize"])
    return latest_dir / "U"


def _write_outputs(args: argparse.Namespace, terrain_info: dict, sample_path: Path) -> None:
    nx = terrain_info["nx"]
    ny = terrain_info["ny"]
    vectors = _parse_sample_vectors(sample_path, nx * ny)
    u = vectors[:, 0].reshape(ny, nx).astype(np.float32)
    v = vectors[:, 1].reshape(ny, nx).astype(np.float32)
    speed, direction = _speed_direction_from_uv(u, v)
    for path, data in [(args.speed_output, speed), (args.direction_output, direction), (args.u_output, u), (args.v_output, v)]:
        _write_aaigrid(Path(path), data, xllcorner=terrain_info["left"], yllcorner=terrain_info["bottom"], cellsize=terrain_info["cellsize"])


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

    wind_u, wind_v = _uv_from_speed_direction(float(args.speed_mps), float(args.direction_deg))
    terrain_info = _load_terrain(Path(args.elevation_file), float(args.mesh_resolution_m), int(args.max_horizontal_cells))
    _write_block_mesh_dict(case_dir, terrain_info, int(args.vertical_cells), float(args.domain_height_m))
    _write_case_files(case_dir, terrain_info, wind_u, wind_v)

    logs = []
    logs.append(_run_wsl('foamRun -help >/dev/null || simpleFoam -help >/dev/null', timeout_seconds=60))
    logs.append(_run_wsl(f'blockMesh -case "{_windows_to_wsl_path(case_dir)}"'))
    logs.append(_run_solver(case_dir))
    sampling_mode = "function_object"
    try:
        logs.append(_run_sampler(case_dir))
        sample_path = _find_sample_output(case_dir)
        _write_outputs(args, terrain_info, sample_path)
    except Exception as exc:
        sampling_mode = "lowest_volume_layer"
        logs.append(f"Function-object sampling failed; using lowest volume layer fallback: {exc!r}")
        sample_path = _write_outputs_from_volume_field(args, terrain_info)

    summary = {
        "runner": "openfoam_wsl_terrain_runner",
        "note": "Experimental WSL/OpenFOAM terrain-following case. Validate before scientific use.",
        "case_dir": str(case_dir),
        "sample_path": str(sample_path),
        "sampling_mode": sampling_mode,
        "wind_speed_mps": float(args.speed_mps),
        "wind_direction_deg": float(args.direction_deg),
        "mesh_resolution_m": float(args.mesh_resolution_m),
        "horizontal_cells": int(terrain_info["nx"] * terrain_info["ny"]),
        "vertical_cells": int(args.vertical_cells),
    }
    (output_dir / "openfoam_wsl_terrain_runner_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "openfoam_wsl_runner.log").write_text("\n\n".join(logs), encoding="utf-8")


if __name__ == "__main__":
    main()
