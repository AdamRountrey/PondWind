from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictweather.runtime import configure_geospatial_runtime

configure_geospatial_runtime()

from predictweather.boundary import load_latest_hrdps_boundary_summary, sample_boundary_wind_at_site
from predictweather.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, OUTPUTS_DIR
from predictweather.hrdps import download_hrdps_for_valid_time
from predictweather.windninja import (
    _read_aaigrid,
    expected_windninja_ascii_paths,
    run_windninja_domain_average,
    windninja_cli_path,
    write_array_to_geotiff_from_header,
    write_scalar_diagnostic_preview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a WindNinja ensemble and compute per-cell variability.")
    parser.add_argument("--mesh-resolution", type=float, default=30.0, help="WindNinja mesh resolution in meters.")
    parser.add_argument("--solver", choices=("mass", "momentum"), default="momentum", help="WindNinja solver mode.")
    parser.add_argument("--iterations", type=int, default=300, help="Momentum-solver iteration count.")
    parser.add_argument("--threads", type=int, default=1, help="Number of WindNinja threads.")
    parser.add_argument(
        "--ensemble-shape",
        choices=("full", "compact"),
        default="compact",
        help="Use a full Cartesian perturbation grid or a smaller paired perturbation set.",
    )
    parser.add_argument(
        "--speed-deltas",
        type=float,
        nargs="+",
        default=[-1.4, -0.6, 0.0, 0.8, 1.8],
        help="Boundary wind speed perturbations in m/s.",
    )
    parser.add_argument(
        "--direction-deltas",
        type=float,
        nargs="+",
        default=[-16.0, -7.0, 0.0, 6.0, 14.0],
        help="Boundary wind direction perturbations in degrees.",
    )
    parser.add_argument(
        "--time-offset-hours",
        type=float,
        nargs="+",
        default=[-1.0, 0.0, 1.0, 2.0],
        help="Forecast valid-time offsets in hours relative to the base boundary time.",
    )
    parser.add_argument(
        "--allow-insecure-ssl",
        action="store_true",
        help="Allow insecure SSL when downloading extra HRDPS times.",
    )
    return parser.parse_args()


def compact_perturbation_pairs() -> list[tuple[float, float]]:
    return [
        (0.0, 0.0),
        (-1.4, -16.0),
        (-0.6, 6.0),
        (0.8, -7.0),
        (1.8, 14.0),
    ]


def main() -> None:
    args = parse_args()
    if args.allow_insecure_ssl:
        import os

        os.environ["PREDICTWEATHER_ALLOW_INSECURE_SSL"] = "1"

    boundary_summary, boundary_summary_path = load_latest_hrdps_boundary_summary(DATA_RAW_DIR / "hrdps")
    cli_path = windninja_cli_path(PROJECT_ROOT)
    elevation_tif = DATA_PROCESSED_DIR / "site_dem.tif"

    solver_tag = "momentum" if args.solver == "momentum" else "mass"
    base_tag = f"windninja_{int(round(args.mesh_resolution))}m_{solver_tag}_ensemble"
    ensemble_root = OUTPUTS_DIR / base_tag
    if ensemble_root.exists():
        shutil.rmtree(ensemble_root)
    ensemble_root.mkdir(parents=True, exist_ok=True)

    speed_members: list[np.ndarray] = []
    direction_members_deg: list[np.ndarray] = []
    member_records: list[dict] = []
    speed_header: dict[str, float] | None = None

    base_valid_time = datetime.fromisoformat(boundary_summary["valid_time_utc"]).replace(tzinfo=timezone.utc)
    time_members: list[dict] = []
    for time_offset_hours in args.time_offset_hours:
        target_valid_time = base_valid_time + timedelta(hours=float(time_offset_hours))
        selection = download_hrdps_for_valid_time(DATA_RAW_DIR, target_valid_time)
        sampled = sample_boundary_wind_at_site(
            ugrd_path=Path(selection.files["UGRD"]),
            vgrd_path=Path(selection.files["VGRD"]),
            lat=float(boundary_summary["site_lat"]),
            lon=float(boundary_summary["site_lon"]),
        )
        sampled["time_offset_hours"] = float(time_offset_hours)
        time_members.append(sampled)

    if args.ensemble_shape == "compact":
        perturbation_pairs = compact_perturbation_pairs()
    else:
        perturbation_pairs = [
            (float(speed_delta), float(direction_delta))
            for speed_delta in args.speed_deltas
            for direction_delta in args.direction_deltas
        ]

    member_index = 0
    for time_member in time_members:
        base_speed = float(time_member["wind_speed_mps"])
        base_direction = float(time_member["wind_from_direction_deg"])
        for speed_delta, direction_delta in perturbation_pairs:
            member_speed = max(0.1, base_speed + float(speed_delta))
            member_direction = (base_direction + float(direction_delta)) % 360.0
            member_dir = ensemble_root / f"member_{member_index:02d}"
            run_windninja_domain_average(
                cli_path=cli_path,
                elevation_tif=elevation_tif,
                output_dir=member_dir,
                wind_speed_mps=member_speed,
                wind_direction_deg=member_direction,
                mesh_resolution_m=args.mesh_resolution,
                momentum=args.solver == "momentum",
                iterations=args.iterations,
                turbulence_output=False,
                num_threads=max(1, args.threads),
            )
            ascii_paths = expected_windninja_ascii_paths(
                elevation_tif=elevation_tif,
                wind_speed_mps=member_speed,
                wind_direction_deg=member_direction,
                mesh_resolution_m=args.mesh_resolution,
                output_dir=member_dir,
            )
            speed_grid, member_header = _read_aaigrid(ascii_paths["speed"])
            direction_grid_deg, _ = _read_aaigrid(ascii_paths["direction"])
            if speed_header is None:
                speed_header = member_header
            speed_members.append(speed_grid)
            direction_members_deg.append(direction_grid_deg)
            member_records.append(
                {
                    "member": member_index,
                    "time_offset_hours": float(time_member["time_offset_hours"]),
                    "valid_time_utc": time_member["valid_time_utc"],
                    "base_wind_speed_mps": base_speed,
                    "base_wind_from_direction_deg": base_direction,
                    "speed_delta_mps": float(speed_delta),
                    "direction_delta_deg": float(direction_delta),
                    "wind_speed_mps": member_speed,
                    "wind_speed_kts": member_speed * 1.94384449,
                    "wind_from_direction_deg": member_direction,
                    "speed_ascii": str(ascii_paths["speed"]),
                    "direction_ascii": str(ascii_paths["direction"]),
                }
            )
            member_index += 1

    if speed_header is None:
        raise RuntimeError("No ensemble members were produced.")

    speed_stack = np.stack(speed_members, axis=0).astype(np.float32)
    direction_stack_deg = np.stack(direction_members_deg, axis=0).astype(np.float32)

    speed_mean = np.nanmean(speed_stack, axis=0).astype(np.float32)
    speed_std = np.nanstd(speed_stack, axis=0).astype(np.float32)
    speed_mean_kts = (speed_mean * 1.94384449).astype(np.float32)
    speed_std_kts = (speed_std * 1.94384449).astype(np.float32)

    direction_rad = np.deg2rad(direction_stack_deg)
    sin_mean = np.nanmean(np.sin(direction_rad), axis=0)
    cos_mean = np.nanmean(np.cos(direction_rad), axis=0)
    mean_direction_rad = np.arctan2(sin_mean, cos_mean)
    mean_direction_deg = (np.rad2deg(mean_direction_rad) + 360.0) % 360.0
    resultant_length = np.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
    resultant_length = np.clip(resultant_length, 1.0e-6, 1.0)
    direction_std_deg = np.rad2deg(np.sqrt(-2.0 * np.log(resultant_length))).astype(np.float32)

    speed_mean_tif = OUTPUTS_DIR / f"{base_tag}_speed_mean.tif"
    speed_std_tif = OUTPUTS_DIR / f"{base_tag}_speed_std.tif"
    speed_mean_kts_tif = OUTPUTS_DIR / f"{base_tag}_speed_mean_kts.tif"
    speed_std_kts_tif = OUTPUTS_DIR / f"{base_tag}_speed_std_kts.tif"
    direction_mean_tif = OUTPUTS_DIR / f"{base_tag}_direction_mean.tif"
    direction_std_tif = OUTPUTS_DIR / f"{base_tag}_direction_std.tif"
    write_array_to_geotiff_from_header(speed_header, speed_mean, speed_mean_tif)
    write_array_to_geotiff_from_header(speed_header, speed_std, speed_std_tif)
    write_array_to_geotiff_from_header(speed_header, speed_mean_kts, speed_mean_kts_tif)
    write_array_to_geotiff_from_header(speed_header, speed_std_kts, speed_std_kts_tif)
    write_array_to_geotiff_from_header(speed_header, mean_direction_deg.astype(np.float32), direction_mean_tif)
    write_array_to_geotiff_from_header(speed_header, direction_std_deg, direction_std_tif)

    speed_std_png = OUTPUTS_DIR / f"{base_tag}_speed_std_kts.png"
    direction_std_png = OUTPUTS_DIR / f"{base_tag}_direction_std.png"
    spread_colormap = [
        (0.00, (237, 248, 251)),
        (0.25, (178, 226, 226)),
        (0.50, (102, 194, 164)),
        (0.75, (44, 162, 95)),
        (1.00, (0, 109, 44)),
    ]
    angle_colormap = [
        (0.00, (247, 251, 255)),
        (0.25, (198, 219, 239)),
        (0.50, (107, 174, 214)),
        (0.75, (33, 113, 181)),
        (1.00, (8, 48, 107)),
    ]
    write_scalar_diagnostic_preview(
        field=speed_std_kts,
        dem_basemap_tif=OUTPUTS_DIR / "site_dem_preview.tif",
        output_png=speed_std_png,
        title="spd",
        units="kts",
        colormap=spread_colormap,
        alpha=0.58,
        signed=False,
    )
    write_scalar_diagnostic_preview(
        field=direction_std_deg,
        dem_basemap_tif=OUTPUTS_DIR / "site_dem_preview.tif",
        output_png=direction_std_png,
        title="dir",
        units="deg",
        colormap=angle_colormap,
        alpha=0.58,
        signed=False,
    )

    summary = {
        "model": f"WindNinja_{solver_tag}_ensemble",
        "boundary_summary": boundary_summary,
        "boundary_summary_path": str(boundary_summary_path),
        "mesh_resolution_m": args.mesh_resolution,
        "member_count": len(member_records),
        "ensemble_shape": args.ensemble_shape,
        "paired_perturbations": [
            {"speed_delta_mps": speed_delta, "direction_delta_deg": direction_delta}
            for speed_delta, direction_delta in perturbation_pairs
        ],
        "time_offset_hours": [float(value) for value in args.time_offset_hours],
        "speed_deltas_mps": [float(value) for value in args.speed_deltas],
        "direction_deltas_deg": [float(value) for value in args.direction_deltas],
        "time_members": time_members,
        "members": member_records,
        "speed_mean_mps": {
            "min": float(np.nanmin(speed_mean)),
            "max": float(np.nanmax(speed_mean)),
            "mean": float(np.nanmean(speed_mean)),
        },
        "speed_mean_kts": {
            "min": float(np.nanmin(speed_mean_kts)),
            "max": float(np.nanmax(speed_mean_kts)),
            "mean": float(np.nanmean(speed_mean_kts)),
        },
        "speed_std_mps": {
            "min": float(np.nanmin(speed_std)),
            "max": float(np.nanmax(speed_std)),
            "mean": float(np.nanmean(speed_std)),
        },
        "speed_std_kts": {
            "min": float(np.nanmin(speed_std_kts)),
            "max": float(np.nanmax(speed_std_kts)),
            "mean": float(np.nanmean(speed_std_kts)),
        },
        "direction_std_deg": {
            "min": float(np.nanmin(direction_std_deg)),
            "max": float(np.nanmax(direction_std_deg)),
            "mean": float(np.nanmean(direction_std_deg)),
        },
        "outputs": {
            "speed_mean_tif": str(speed_mean_tif),
            "speed_std_tif": str(speed_std_tif),
            "speed_mean_kts_tif": str(speed_mean_kts_tif),
            "speed_std_kts_tif": str(speed_std_kts_tif),
            "direction_mean_tif": str(direction_mean_tif),
            "direction_std_tif": str(direction_std_tif),
            "speed_std_png": str(speed_std_png),
            "direction_std_png": str(direction_std_png),
        },
        "note": "This is an ensemble sensitivity spread from perturbed boundary wind speed, direction, and nearby forecast valid times, not a calibrated probabilistic forecast.",
    }
    summary_path = OUTPUTS_DIR / f"{base_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Ensemble summary: {summary_path}")
    print(f"Speed spread map: {speed_std_png}")
    print(f"Direction spread map: {direction_std_png}")


if __name__ == "__main__":
    main()
