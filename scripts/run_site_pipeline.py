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

from predictweather.config import DATA_RAW_DIR, OUTPUTS_DIR, SiteConfig
from predictweather.boundary import load_latest_hrdps_manifest, sample_boundary_wind_at_site
from predictweather.geo import square_bbox_from_center
from predictweather.site import prepare_site_domain
from predictweather.wind import build_coarse_diagnostic_wind, build_speedup_raster, write_summary
from predictweather.wind import write_wind_color_preview, write_wind_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terrain-only local wind pipeline.")
    parser.add_argument("--center-lat", type=float, default=SiteConfig.center_lat)
    parser.add_argument("--center-lon", type=float, default=SiteConfig.center_lon)
    parser.add_argument("--side-meters", type=float, default=SiteConfig.side_meters)
    parser.add_argument("--wind-direction", type=float, default=None)
    parser.add_argument("--wind-speed", type=float, default=None)
    parser.add_argument(
        "--no-boundary",
        action="store_true",
        help="Ignore the latest sampled HRDPS boundary summary and use manual/default wind arguments instead.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bbox = square_bbox_from_center(args.center_lat, args.center_lon, args.side_meters)
    print(f"Bounding box: {bbox.as_tnm_bbox()}")

    boundary_summary = None
    boundary_summary_path = None
    if not args.no_boundary:
        try:
            latest_manifest, boundary_summary_path = load_latest_hrdps_manifest(DATA_RAW_DIR / "hrdps")
            boundary_summary = sample_boundary_wind_at_site(
                ugrd_path=Path(latest_manifest["files"]["UGRD"]),
                vgrd_path=Path(latest_manifest["files"]["VGRD"]),
                lat=args.center_lat,
                lon=args.center_lon,
            )
        except FileNotFoundError:
            boundary_summary = None
            boundary_summary_path = None

    if args.wind_direction is not None:
        wind_direction_deg = args.wind_direction
        wind_speed_mps = args.wind_speed if args.wind_speed is not None else SiteConfig.wind_speed_mps
        wind_source = "manual"
    elif boundary_summary is not None:
        wind_direction_deg = float(boundary_summary["wind_from_direction_deg"])
        wind_speed_mps = float(boundary_summary["wind_speed_mps"])
        wind_source = "hrdps_boundary"
    else:
        wind_direction_deg = SiteConfig.wind_direction_deg
        wind_speed_mps = SiteConfig.wind_speed_mps if args.wind_speed is None else args.wind_speed
        wind_source = "default"

    print(f"Wind source: {wind_source}")
    print(f"Wind direction from deg: {wind_direction_deg:.1f}")
    print(f"Wind speed m/s: {wind_speed_mps:.3f}")
    if boundary_summary_path is not None:
        print(f"Boundary summary: {boundary_summary_path}")

    site = SiteConfig(center_lat=args.center_lat, center_lon=args.center_lon, side_meters=args.side_meters)
    domain = prepare_site_domain(site=site, working_dir=OUTPUTS_DIR / "site_pipeline")
    clipped_tif = domain.clipped_dem_tif
    print(f"Dataset: {domain.dataset}")
    print(f"Source DEM: {domain.source_dem_tif}")
    print(f"Clipped DEM: {clipped_tif}")

    dem_preview_tif = domain.dem_preview_tif
    print(f"DEM preview: {dem_preview_tif}")

    speedup_tif = OUTPUTS_DIR / "terrain_wind_speed.tif"
    summary = build_speedup_raster(
        dem_tif=clipped_tif,
        output_tif=speedup_tif,
        wind_direction_deg=wind_direction_deg,
        base_wind_speed_mps=wind_speed_mps,
    )
    summary["wind_source"] = wind_source
    if boundary_summary_path is not None:
        summary["boundary_summary_path"] = str(boundary_summary_path)
        summary["boundary_valid_time_utc"] = boundary_summary["valid_time_utc"]
    wind_preview_tif = OUTPUTS_DIR / "terrain_wind_speed_preview.tif"
    write_wind_preview(speedup_tif, wind_preview_tif)
    wind_color_preview_png = OUTPUTS_DIR / "terrain_wind_speed_colormap.png"
    write_wind_color_preview(speedup_tif, wind_color_preview_png)

    coarse_wind_tif = OUTPUTS_DIR / "terrain_wind_speed_100m.tif"
    coarse_summary = build_coarse_diagnostic_wind(
        dem_tif=clipped_tif,
        output_tif=coarse_wind_tif,
        wind_direction_deg=wind_direction_deg,
        base_wind_speed_mps=wind_speed_mps,
        target_resolution_m=100.0,
    )
    coarse_wind_color_preview_png = OUTPUTS_DIR / "terrain_wind_speed_100m_colormap.png"
    write_wind_color_preview(coarse_wind_tif, coarse_wind_color_preview_png)
    coarse_wind_preview_tif = OUTPUTS_DIR / "terrain_wind_speed_100m_preview.tif"
    write_wind_preview(coarse_wind_tif, coarse_wind_preview_tif)

    summary_path = OUTPUTS_DIR / "summary.json"
    summary["coarse_100m"] = coarse_summary
    write_summary(summary, summary_path)

    print(f"Wind raster: {speedup_tif}")
    print(f"Wind preview: {wind_preview_tif}")
    print(f"Wind color preview: {wind_color_preview_png}")
    print(f"Coarse wind raster: {coarse_wind_tif}")
    print(f"Coarse wind preview: {coarse_wind_preview_tif}")
    print(f"Coarse wind color preview: {coarse_wind_color_preview_png}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
