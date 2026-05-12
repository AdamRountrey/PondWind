from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictweather.landscape import LandscapeBuildOptions, build_landscape_geotiff
from predictweather.runtime import configure_geospatial_runtime

configure_geospatial_runtime()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an 8-band landscape GeoTIFF for WindNinja/FARSITE-style vegetation input.")
    parser.add_argument("--dem-tif", required=True)
    parser.add_argument("--output-tif", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--canopy-cover-tif", default=None)
    parser.add_argument("--canopy-height-tif", default=None)
    parser.add_argument("--canopy-base-height-tif", default=None)
    parser.add_argument("--canopy-bulk-density-tif", default=None)
    parser.add_argument("--water-mask-tif", default=None)
    parser.add_argument("--fuel-model-land", type=int, default=181)
    parser.add_argument("--fuel-model-water", type=int, default=98)
    parser.add_argument("--canopy-cover-units", choices=["percent", "fraction"], default="percent")
    parser.add_argument("--canopy-height-units", choices=["meters", "meters_x10"], default="meters")
    parser.add_argument("--canopy-base-height-units", choices=["meters", "meters_x10"], default="meters")
    parser.add_argument("--canopy-bulk-density-units", choices=["kg_m3", "kg_m3_x100"], default="kg_m3")
    parser.add_argument("--derived-cbh-fraction-of-ch", type=float, default=0.4)
    parser.add_argument("--default-cbd-kg-m3", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_landscape_geotiff(
        dem_tif=Path(args.dem_tif),
        output_tif=Path(args.output_tif),
        canopy_cover_tif=Path(args.canopy_cover_tif) if args.canopy_cover_tif else None,
        canopy_height_tif=Path(args.canopy_height_tif) if args.canopy_height_tif else None,
        canopy_base_height_tif=Path(args.canopy_base_height_tif) if args.canopy_base_height_tif else None,
        canopy_bulk_density_tif=Path(args.canopy_bulk_density_tif) if args.canopy_bulk_density_tif else None,
        water_mask_tif=Path(args.water_mask_tif) if args.water_mask_tif else None,
        options=LandscapeBuildOptions(
            fuel_model_land=args.fuel_model_land,
            fuel_model_water=args.fuel_model_water,
            canopy_cover_units=args.canopy_cover_units,
            canopy_height_units=args.canopy_height_units,
            canopy_base_height_units=args.canopy_base_height_units,
            canopy_bulk_density_units=args.canopy_bulk_density_units,
            derived_cbh_fraction_of_ch=args.derived_cbh_fraction_of_ch,
            default_cbd_kg_m3=args.default_cbd_kg_m3,
        ),
        summary_path=Path(args.summary_json) if args.summary_json else None,
    )
    print(json.dumps(summary.__dict__, indent=2))


if __name__ == "__main__":
    main()
