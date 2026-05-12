from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictweather.runtime import configure_geospatial_runtime

configure_geospatial_runtime()

import numpy as np
import rasterio

from predictweather.config import OUTPUTS_DIR
from predictweather.windninja import (
    _read_aaigrid,
    diverging_blue_black_red_colormap,
    write_scalar_diagnostic_preview,
    write_windninja_knots_vector_preview_from_speed_angle,
)


def main() -> None:
    colormap = diverging_blue_black_red_colormap()

    prediction_speed_asc = OUTPUTS_DIR / "windninja_30m_momentum" / "site_dem_215_7_30m_vel.asc"
    prediction_angle_asc = OUTPUTS_DIR / "windninja_30m_momentum" / "site_dem_215_7_30m_ang.asc"
    product1_png = OUTPUTS_DIR / "product_1_wind_speed_prediction_knots.png"
    speed_mps, _ = _read_aaigrid(prediction_speed_asc)
    speed_kts = speed_mps * 1.94384449
    prediction_center = float(np.nanmean(speed_kts))
    write_windninja_knots_vector_preview_from_speed_angle(
        prediction_speed_asc,
        prediction_angle_asc,
        product1_png,
        dem_basemap_tif=OUTPUTS_DIR / "site_dem_preview.tif",
        vector_stride=4,
        vector_scale=2.2,
        colormap=colormap,
        center_value=prediction_center,
        title="wind",
        units="knots",
    )

    speed_variance_tif = OUTPUTS_DIR / "windninja_30m_momentum_gefs_real_speed_std_kts.tif"
    direction_variance_tif = OUTPUTS_DIR / "windninja_30m_momentum_gefs_real_direction_std.tif"
    product2_png = OUTPUTS_DIR / "product_2_wind_speed_variance_knots.png"
    product3_png = OUTPUTS_DIR / "product_3_wind_direction_variance_degrees.png"
    with rasterio.open(speed_variance_tif) as src:
        speed_variance = src.read(1, masked=True).filled(np.nan).astype(np.float32)
    with rasterio.open(direction_variance_tif) as src:
        direction_variance = src.read(1, masked=True).filled(np.nan).astype(np.float32)

    write_scalar_diagnostic_preview(
        field=speed_variance,
        dem_basemap_tif=OUTPUTS_DIR / "site_dem_preview.tif",
        output_png=product2_png,
        title="sd wind",
        units="knots",
        colormap=colormap,
        alpha=0.58,
        signed=False,
        center_value=float(np.nanmean(speed_variance)),
    )
    write_scalar_diagnostic_preview(
        field=direction_variance,
        dem_basemap_tif=OUTPUTS_DIR / "site_dem_preview.tif",
        output_png=product3_png,
        title="sd az",
        units="deg",
        colormap=colormap,
        alpha=0.58,
        signed=False,
        center_value=float(np.nanmean(direction_variance)),
    )

    manifest = {
        "product_1": str(product1_png),
        "product_2": str(product2_png),
        "product_3": str(product3_png),
        "color_scale": "blue_low_black_mean_red_high",
        "units": {
            "prediction_speed": "knots",
            "speed_variance": "knots",
            "direction_variance": "degrees",
        },
        "centers": {
            "prediction_speed_knots_mean": prediction_center,
            "speed_variance_knots_mean": float(np.nanmean(speed_variance)),
            "direction_variance_degrees_mean": float(np.nanmean(direction_variance)),
        },
    }
    (OUTPUTS_DIR / "final_products_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(product1_png)
    print(product2_png)
    print(product3_png)


if __name__ == "__main__":
    main()
