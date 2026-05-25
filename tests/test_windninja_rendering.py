from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from predictweather.windninja import (
    diverging_blue_green_red_colormap,
    write_windninja_knots_vector_preview_from_arrays,
)


class WindNinjaRenderingTests(unittest.TestCase):
    def test_zero_wind_preview_renders_without_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            basemap = tmp / "dem_preview.tif"
            preview = tmp / "calm.png"
            map_size = 120
            dem = np.full((map_size, map_size), 160, dtype=np.uint8)
            with rasterio.open(
                basemap,
                "w",
                driver="GTiff",
                width=map_size,
                height=map_size,
                count=1,
                dtype="uint8",
                crs="EPSG:26917",
                transform=from_origin(0.0, float(map_size), 1.0, 1.0),
            ) as dst:
                dst.write(dem, 1)

            source_rows = source_cols = 12
            zeros = np.zeros((source_rows, source_cols), dtype=np.float32)
            source_header = {
                "ncols": float(source_cols),
                "nrows": float(source_rows),
                "xllcorner": 0.0,
                "yllcorner": 0.0,
                "cellsize": 10.0,
                "nodata_value": -9999.0,
            }

            write_windninja_knots_vector_preview_from_arrays(
                speed_mps=zeros,
                u_mps=zeros,
                v_mps=zeros,
                preview_png=preview,
                dem_basemap_tif=basemap,
                source_header=source_header,
                vector_stride=3,
                vector_scale=2.2,
                colormap=diverging_blue_green_red_colormap(),
                center_value=0.0,
                footer_text="calm diagnostic",
            )

            self.assertTrue(preview.exists())
            image = np.array(Image.open(preview).convert("RGB"))
            self.assertGreater(image.size, 0)

    def test_near_cardinal_vectors_do_not_make_full_height_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            basemap = tmp / "dem_preview.tif"
            preview = tmp / "preview.png"
            map_size = 800
            dem = np.full((map_size, map_size), 180, dtype=np.uint8)
            with rasterio.open(
                basemap,
                "w",
                driver="GTiff",
                width=map_size,
                height=map_size,
                count=1,
                dtype="uint8",
                crs="EPSG:26917",
                transform=from_origin(0.0, float(map_size), 1.0, 1.0),
            ) as dst:
                dst.write(dem, 1)

            source_rows = source_cols = 40
            speed_mps = np.tile(np.linspace(2.0, 8.0, source_cols, dtype=np.float32), (source_rows, 1))
            u_mps = np.zeros_like(speed_mps)
            v_mps = speed_mps.copy()
            source_header = {
                "ncols": float(source_cols),
                "nrows": float(source_rows),
                "xllcorner": 0.0,
                "yllcorner": 0.0,
                "cellsize": 20.0,
                "nodata_value": -9999.0,
            }

            write_windninja_knots_vector_preview_from_arrays(
                speed_mps=speed_mps,
                u_mps=u_mps,
                v_mps=v_mps,
                preview_png=preview,
                dem_basemap_tif=basemap,
                source_header=source_header,
                vector_stride=4,
                vector_scale=2.2,
                colormap=diverging_blue_green_red_colormap(),
                center_value=float(np.mean(speed_mps * 1.94384449)),
                footer_text="wind forecast diagnostic",
                bottom_table_rows=[
                    {"model": "ecmwf", "wind": "9.8", "gust": "21.2"},
                    {"model": "gfs", "wind": "10.5", "gust": "13.9"},
                    {"model": "icon", "wind": "6.5", "gust": "16.0"},
                    {"model": "nam", "wind": "11.6", "gust": "15.9"},
                    {"model": "hrdps", "wind": "8.3", "gust": "12.7"},
                    {"model": "hrrr", "wind": "8.1", "gust": "13.4"},
                ],
            )

            image = np.array(Image.open(preview).convert("RGB"))
            self.assertEqual(image.shape[0], map_size + 50)

            map_and_footer = image[:, :map_size]
            black = (map_and_footer[:, :, 0] < 25) & (map_and_footer[:, :, 1] < 25) & (map_and_footer[:, :, 2] < 25)
            column_black_counts = black.sum(axis=0)

            self.assertLess(int(column_black_counts.max()), int(image.shape[0] * 0.75))


if __name__ == "__main__":
    unittest.main()
