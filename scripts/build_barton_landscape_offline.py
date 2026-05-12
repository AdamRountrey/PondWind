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

from predictweather.landscape import LandscapeBuildOptions, build_landscape_geotiff
from predictweather.runtime import configure_geospatial_runtime

configure_geospatial_runtime()


def _resample_to_reference(source_tif: Path, reference_tif: Path, *, resampling: Resampling) -> np.ndarray:
    with rasterio.open(reference_tif) as ref:
        destination = np.full((ref.height, ref.width), np.nan, dtype=np.float32)
        with rasterio.open(source_tif) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
                resampling=resampling,
            )
    return destination


def _write_like(reference_tif: Path, destination_tif: Path, data: np.ndarray, dtype: str) -> Path:
    destination_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()
        profile.update(driver="GTiff", dtype=dtype, count=1, nodata=np.nan if "float" in dtype else 0, compress="deflate")
        with rasterio.open(destination_tif, "w", **profile) as dst:
            dst.write(data.astype(dtype), 1)
    return destination_tif


def main() -> None:
    report_temp = PROJECT_ROOT / "outputs" / "reports" / "20260507_1400_barton_pond_regression" / "report_temp"
    domain_dem = report_temp / "domain" / "site_dem.tif"
    sentinel_dir = report_temp / "satellite" / "sentinel"
    red_tif = sentinel_dir / "sentinel-2-l2a_S2B_17TKG_20260423_0_L2A_red.tif"
    nir_tif = sentinel_dir / "sentinel-2-l2a_S2B_17TKG_20260423_0_L2A_nir.tif"
    scl_tif = sentinel_dir / "sentinel-2-l2a_S2B_17TKG_20260423_0_L2A_scl.tif"

    out_dir = PROJECT_ROOT / "outputs" / "landscape_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    canopy_cover_tif = out_dir / "sentinel_canopy_cover_proxy.tif"
    canopy_height_tif = out_dir / "sentinel_canopy_height_proxy.tif"
    water_mask_tif = out_dir / "sentinel_water_mask.tif"
    landscape_tif = out_dir / "barton_landscape_offline.tif"
    summary_json = out_dir / "barton_landscape_offline_summary.json"

    red = _resample_to_reference(red_tif, domain_dem, resampling=Resampling.bilinear)
    nir = _resample_to_reference(nir_tif, domain_dem, resampling=Resampling.bilinear)
    scl = _resample_to_reference(scl_tif, domain_dem, resampling=Resampling.nearest)

    denominator = nir + red
    ndvi = np.divide(
        nir - red,
        denominator,
        out=np.full_like(red, np.nan, dtype=np.float32),
        where=np.isfinite(denominator) & (np.abs(denominator) > 1.0e-6),
    )
    scl_classes = np.rint(scl).astype(np.int16)
    vegetation_mask = scl_classes == 4
    water_mask = scl_classes == 6

    canopy_cover_pct = np.where(
        vegetation_mask,
        np.clip((ndvi - 0.30) / 0.40, 0.0, 1.0) * 100.0,
        0.0,
    ).astype(np.float32)
    canopy_height_m = np.where(
        canopy_cover_pct >= 15.0,
        np.clip(2.0 + 0.12 * canopy_cover_pct, 0.0, 18.0),
        0.0,
    ).astype(np.float32)
    water_mask_u8 = water_mask.astype(np.uint8)

    _write_like(domain_dem, canopy_cover_tif, canopy_cover_pct, "float32")
    _write_like(domain_dem, canopy_height_tif, canopy_height_m, "float32")
    _write_like(domain_dem, water_mask_tif, water_mask_u8, "uint8")

    summary = build_landscape_geotiff(
        dem_tif=domain_dem,
        output_tif=landscape_tif,
        canopy_cover_tif=canopy_cover_tif,
        canopy_height_tif=canopy_height_tif,
        water_mask_tif=water_mask_tif,
        options=LandscapeBuildOptions(
            fuel_model_land=181,
            fuel_model_water=98,
            canopy_cover_units="percent",
            canopy_height_units="meters",
            derived_cbh_fraction_of_ch=0.4,
            default_cbd_kg_m3=0.1,
        ),
        summary_path=summary_json,
    )

    extras = {
        "proxy_inputs": {
            "red_tif": str(red_tif),
            "nir_tif": str(nir_tif),
            "scl_tif": str(scl_tif),
        },
        "proxy_method": {
            "canopy_cover": "Sentinel-2 NDVI over SCL vegetation pixels, scaled to 0-100 percent.",
            "canopy_height": "Provisional height proxy derived from canopy cover: 2.0 + 0.12 * cover_pct meters, clipped to 18 m.",
            "water_mask": "Sentinel-2 SCL class 6.",
        },
    }
    payload = {**summary.__dict__, **extras}
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
