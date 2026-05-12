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

from predictweather.boundary import sample_latest_hrdps_boundary
from predictweather.config import DATA_RAW_DIR, SiteConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample the latest HRDPS boundary winds at the project site.")
    parser.add_argument("--site-lat", type=float, default=SiteConfig.center_lat)
    parser.add_argument("--site-lon", type=float, default=SiteConfig.center_lon)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, summary_path = sample_latest_hrdps_boundary(
        root=DATA_RAW_DIR / "hrdps",
        lat=args.site_lat,
        lon=args.site_lon,
    )
    print(f"Valid UTC: {summary['valid_time_utc']}")
    print(f"Grid point: {summary['grid_lat']:.6f}, {summary['grid_lon']:.6f}")
    print(f"Grid distance km: {summary['grid_distance_km']:.3f}")
    print(f"U10 m/s: {summary['u10_mps']:.3f}")
    print(f"V10 m/s: {summary['v10_mps']:.3f}")
    print(f"Wind speed m/s: {summary['wind_speed_mps']:.3f}")
    print(f"Wind from deg: {summary['wind_from_direction_deg']:.1f}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
