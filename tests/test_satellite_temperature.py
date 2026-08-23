from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from predictweather.geo import BoundingBox
from predictweather.satellite import derive_landsat_sst


def _write_raster(path: Path, data: np.ndarray, *, nodata: float | int | None, dtype: str) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=from_origin(-84.0, 43.0, 0.001, 0.001),
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype), 1)


class SatelliteTemperatureTests(unittest.TestCase):
    def test_landsat_sparse_clear_water_renders_limited_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            thermal = np.full((10, 10), 43000, dtype=np.uint16)
            qa = np.zeros((10, 10), dtype=np.uint16)
            qa.flat[:19] = 1 << 7
            dem = np.full((10, 10), 128, dtype=np.uint8)
            thermal_path = root / "thermal.tif"
            qa_path = root / "qa.tif"
            dem_path = root / "dem.tif"
            _write_raster(thermal_path, thermal, nodata=0, dtype="uint16")
            _write_raster(qa_path, qa, nodata=None, dtype="uint16")
            _write_raster(dem_path, dem, nodata=None, dtype="uint8")

            with patch.multiple(
                "predictweather.satellite",
                TARGET_MAP_WIDTH=120,
                TARGET_MAP_HEIGHT=100,
                TARGET_LEGEND_WIDTH=80,
            ):
                summary = derive_landsat_sst(
                    lwir11_tif=thermal_path,
                    qa_pixel_tif=qa_path,
                    bbox=BoundingBox(-84.0, 42.99, -83.99, 43.0),
                    lwir_scale=0.00341802,
                    lwir_offset=149.0,
                    lwir_nodata=0,
                    output_tif=root / "sst.tif",
                    output_png=root / "sst.png",
                    dem_basemap_tif=dem_path,
                    footer_text="surface temp collected 2026-08-23",
                )

            self.assertEqual(summary["status"], "limited")
            self.assertEqual(summary["usable_pixel_count"], 19)
            self.assertEqual(summary["min_recommended_pixel_count"], 25)
            self.assertTrue((root / "sst.tif").exists())
            self.assertTrue((root / "sst.png").exists())

    def test_landsat_zero_clear_water_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            thermal = np.full((10, 10), 43000, dtype=np.uint16)
            qa = np.zeros((10, 10), dtype=np.uint16)
            dem = np.full((10, 10), 128, dtype=np.uint8)
            thermal_path = root / "thermal.tif"
            qa_path = root / "qa.tif"
            dem_path = root / "dem.tif"
            _write_raster(thermal_path, thermal, nodata=0, dtype="uint16")
            _write_raster(qa_path, qa, nodata=None, dtype="uint16")
            _write_raster(dem_path, dem, nodata=None, dtype="uint8")

            with self.assertRaisesRegex(ValueError, "no usable pixels"):
                derive_landsat_sst(
                    lwir11_tif=thermal_path,
                    qa_pixel_tif=qa_path,
                    bbox=BoundingBox(-84.0, 42.99, -83.99, 43.0),
                    lwir_scale=0.00341802,
                    lwir_offset=149.0,
                    lwir_nodata=0,
                    output_tif=root / "sst.tif",
                    output_png=root / "sst.png",
                    dem_basemap_tif=dem_path,
                )


if __name__ == "__main__":
    unittest.main()
