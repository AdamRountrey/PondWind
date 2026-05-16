from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference PondWind OpenFOAM runner contract implementation.")
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
    parser.add_argument("--vertical-cells", type=int, default=20)
    parser.add_argument("--domain-height-m", type=float, default=400.0)
    parser.add_argument("--max-horizontal-cells", type=int, default=12000)
    parser.add_argument("--water-z0-m", type=float, default=0.0002)
    parser.add_argument("--grass-z0-m", type=float, default=0.03)
    parser.add_argument("--tree-z0-m", type=float, default=0.3)
    return parser.parse_args()


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


def _uv_from_speed_direction(speed_mps: float, direction_from_deg: float) -> tuple[float, float]:
    direction_to_rad = math.radians((direction_from_deg + 180.0) % 360.0)
    u = speed_mps * math.sin(direction_to_rad)
    v = speed_mps * math.cos(direction_to_rad)
    return u, v


def main() -> None:
    args = parse_args()
    request_path = Path(args.request_json)
    if request_path.exists():
        json.loads(request_path.read_text(encoding="utf-8"))

    output_dir = Path(args.output_dir)
    case_dir = Path(args.case_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.elevation_file) as src:
        transform = src.transform
        mask = src.read(1, masked=True).mask
        height = src.height
        width = src.width
        xllcorner = float(transform.c)
        yllcorner = float(transform.f + transform.e * height)
        cellsize = float(abs(transform.a))

    speed = np.full((height, width), float(args.speed_mps), dtype=np.float32)
    direction = np.full((height, width), float(args.direction_deg), dtype=np.float32)
    u_value, v_value = _uv_from_speed_direction(float(args.speed_mps), float(args.direction_deg))
    u = np.full((height, width), u_value, dtype=np.float32)
    v = np.full((height, width), v_value, dtype=np.float32)

    if np.any(mask):
        speed[mask] = np.nan
        direction[mask] = np.nan
        u[mask] = np.nan
        v[mask] = np.nan

    _write_aaigrid(Path(args.speed_output), speed, xllcorner=xllcorner, yllcorner=yllcorner, cellsize=cellsize)
    _write_aaigrid(Path(args.direction_output), direction, xllcorner=xllcorner, yllcorner=yllcorner, cellsize=cellsize)
    _write_aaigrid(Path(args.u_output), u, xllcorner=xllcorner, yllcorner=yllcorner, cellsize=cellsize)
    _write_aaigrid(Path(args.v_output), v, xllcorner=xllcorner, yllcorner=yllcorner, cellsize=cellsize)

    summary = {
        "runner": "openfoam_uniform_runner",
        "note": "Reference contract runner only; this is not a CFD/OpenFOAM solve.",
        "speed_mps": float(args.speed_mps),
        "direction_deg": float(args.direction_deg),
        "mesh_resolution_m": float(args.mesh_resolution_m),
        "grid_shape": [height, width],
    }
    (output_dir / "openfoam_uniform_runner_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
