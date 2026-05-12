from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictweather.runtime import configure_geospatial_runtime

configure_geospatial_runtime()

from predictweather.boundary import load_latest_hrdps_boundary_summary
from predictweather.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, OUTPUTS_DIR
from predictweather.wind import write_wind_color_preview, write_wind_preview
from predictweather.windninja import (
    aaigrid_to_geotiff,
    expected_windninja_ascii_paths,
    run_windninja_domain_average,
    windninja_cli_path,
    write_windninja_knots_vector_preview,
    write_windninja_knots_vector_preview_from_speed_angle,
    write_windninja_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WindNinja using the latest sampled HRDPS boundary wind.")
    parser.add_argument("--mesh-resolution", type=float, default=100.0, help="WindNinja mesh resolution in meters.")
    parser.add_argument(
        "--solver",
        choices=("mass", "momentum"),
        default="mass",
        help="WindNinja solver mode.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Momentum-solver iteration count.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of WindNinja threads to use.",
    )
    parser.add_argument(
        "--vector-resolution",
        type=float,
        default=100.0,
        help="Approximate spacing in meters between plotted vectors on the PNG output.",
    )
    parser.add_argument(
        "--vector-scale",
        type=float,
        default=1.8,
        help="Multiplier for arrow length on the PNG output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    boundary_summary, boundary_summary_path = load_latest_hrdps_boundary_summary(DATA_RAW_DIR / "hrdps")
    cli_path = windninja_cli_path(PROJECT_ROOT)
    if not cli_path.exists():
        raise FileNotFoundError(f"WindNinja CLI not found: {cli_path}")

    elevation_tif = DATA_PROCESSED_DIR / "site_dem.tif"
    solver_tag = "momentum" if args.solver == "momentum" else "mass"
    resolution_tag = f"{int(round(args.mesh_resolution))}m_{solver_tag}"
    windninja_output_dir = OUTPUTS_DIR / f"windninja_{resolution_tag}"
    run_metadata = run_windninja_domain_average(
        cli_path=cli_path,
        elevation_tif=elevation_tif,
        output_dir=windninja_output_dir,
        wind_speed_mps=float(boundary_summary["wind_speed_mps"]),
        wind_direction_deg=float(boundary_summary["wind_from_direction_deg"]),
        mesh_resolution_m=args.mesh_resolution,
        momentum=args.solver == "momentum",
        iterations=args.iterations,
        turbulence_output=False,
        num_threads=max(1, args.threads),
    )

    ascii_paths = expected_windninja_ascii_paths(
        elevation_tif=elevation_tif,
        wind_speed_mps=float(boundary_summary["wind_speed_mps"]),
        wind_direction_deg=float(boundary_summary["wind_from_direction_deg"]),
        mesh_resolution_m=args.mesh_resolution,
        output_dir=windninja_output_dir,
    )
    speed_tif = OUTPUTS_DIR / f"windninja_{resolution_tag}_speed.tif"
    aaigrid_to_geotiff(ascii_paths["speed"], speed_tif)
    write_wind_preview(speed_tif, OUTPUTS_DIR / f"windninja_{resolution_tag}_speed_preview.tif")
    write_wind_color_preview(speed_tif, OUTPUTS_DIR / f"windninja_{resolution_tag}_speed_colormap.png")
    preview_png = OUTPUTS_DIR / f"windninja_{resolution_tag}_speed_knots_vectors.png"
    if ascii_paths["u"].exists() and ascii_paths["v"].exists():
        write_windninja_knots_vector_preview(
            ascii_paths["speed"],
            ascii_paths["u"],
            ascii_paths["v"],
            preview_png,
            dem_basemap_tif=OUTPUTS_DIR / "site_dem_preview.tif",
            vector_stride=max(1, int(round(args.vector_resolution / args.mesh_resolution))),
            vector_scale=args.vector_scale,
        )
    else:
        write_windninja_knots_vector_preview_from_speed_angle(
            ascii_paths["speed"],
            ascii_paths["direction"],
            preview_png,
            dem_basemap_tif=OUTPUTS_DIR / "site_dem_preview.tif",
            vector_stride=max(1, int(round(args.vector_resolution / args.mesh_resolution))),
            vector_scale=args.vector_scale,
        )
    summary_path = write_windninja_summary(
        summary_path=OUTPUTS_DIR / f"windninja_{resolution_tag}_summary.json",
        speed_tif=speed_tif,
        direction_asc=ascii_paths["direction"],
        boundary_summary=boundary_summary,
        run_metadata=run_metadata,
        mesh_resolution_m=args.mesh_resolution,
        model_name="WindNinja_momentum" if args.solver == "momentum" else "WindNinja_COM",
    )

    print(f"Boundary summary: {boundary_summary_path}")
    print(f"WindNinja speed ASCII: {ascii_paths['speed']}")
    print(f"WindNinja speed GeoTIFF: {speed_tif}")
    print(f"WindNinja color preview: {OUTPUTS_DIR / f'windninja_{resolution_tag}_speed_colormap.png'}")
    print(f"WindNinja knots/vector preview: {OUTPUTS_DIR / f'windninja_{resolution_tag}_speed_knots_vectors.png'}")
    print(f"WindNinja summary: {summary_path}")


if __name__ == "__main__":
    main()
