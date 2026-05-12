from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictweather.landfire import ExportSpec, export_image
from predictweather.landscape import LandscapeBuildOptions, build_landscape_geotiff
from predictweather.runtime import configure_geospatial_runtime

configure_geospatial_runtime()


def _resample_scl_water_mask(scl_tif: Path, reference_tif: Path, destination_tif: Path) -> Path:
    with rasterio.open(reference_tif) as ref:
        dest = np.zeros((ref.height, ref.width), dtype=np.uint8)
        with rasterio.open(scl_tif) as src:
            scl_resampled = np.full((ref.height, ref.width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=scl_resampled,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
        dest[np.rint(scl_resampled) == 6] = 1
        profile = ref.profile.copy()
        profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0, compress="deflate")
        destination_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(destination_tif, "w", **profile) as dst:
            dst.write(dest, 1)
    return destination_tif


def main() -> None:
    report_temp = PROJECT_ROOT / "outputs" / "reports" / "20260507_1400_barton_pond_regression" / "report_temp"
    domain_dem = report_temp / "domain" / "site_dem.tif"
    scl_tif = report_temp / "satellite" / "sentinel" / "sentinel-2-l2a_S2B_17TKG_20260423_0_L2A_scl.tif"
    out_dir = PROJECT_ROOT / "outputs" / "landscape_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(domain_dem) as src:
        bounds = src.bounds
        crs = src.crs
        if crs is None or crs.to_epsg() is None:
            raise ValueError("Projected DEM CRS is required to request LANDFIRE rasters.")
        spec = ExportSpec(
            xmin=float(bounds.left),
            ymin=float(bounds.bottom),
            xmax=float(bounds.right),
            ymax=float(bounds.top),
            bbox_sr=int(crs.to_epsg()),
            image_sr=int(crs.to_epsg()),
            width=int(src.width),
            height=int(src.height),
        )

    canopy_height_tif = out_dir / "landfire_canopy_height.tif"
    canopy_cover_tif = out_dir / "landfire_canopy_cover.tif"
    canopy_base_height_tif = out_dir / "landfire_canopy_base_height.tif"
    canopy_bulk_density_tif = out_dir / "landfire_canopy_bulk_density.tif"
    water_mask_tif = out_dir / "water_mask.tif"
    landscape_tif = out_dir / "barton_landscape.tif"
    summary_json = out_dir / "barton_landscape_summary.json"

    export_image(service_key="ch", spec=spec, destination=canopy_height_tif)
    export_image(service_key="cc", spec=spec, destination=canopy_cover_tif)
    export_image(service_key="cbh", spec=spec, destination=canopy_base_height_tif)
    export_image(service_key="cbd", spec=spec, destination=canopy_bulk_density_tif)
    _resample_scl_water_mask(scl_tif, domain_dem, water_mask_tif)

    summary = build_landscape_geotiff(
        dem_tif=domain_dem,
        output_tif=landscape_tif,
        canopy_cover_tif=canopy_cover_tif,
        canopy_height_tif=canopy_height_tif,
        canopy_base_height_tif=canopy_base_height_tif,
        canopy_bulk_density_tif=canopy_bulk_density_tif,
        water_mask_tif=water_mask_tif,
        options=LandscapeBuildOptions(
            fuel_model_land=181,
            fuel_model_water=98,
            canopy_cover_units="percent",
            canopy_height_units="meters_x10",
            canopy_base_height_units="meters_x10",
            canopy_bulk_density_units="kg_m3_x100",
            derived_cbh_fraction_of_ch=0.4,
            default_cbd_kg_m3=0.1,
        ),
        summary_path=summary_json,
    )
    print(json.dumps(summary.__dict__, indent=2))


if __name__ == "__main__":
    main()
