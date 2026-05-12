from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request

import rasterio

from predictweather.http import _open_url_with_retry, env_allows_insecure_ssl


LANDFIRE_SERVICES = {
    "ch": "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2024/LF2024_CH_CONUS/ImageServer/exportImage",
    "cc": "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2024/LF2024_CC_CONUS/ImageServer/exportImage",
    "cbh": "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2024/LF2024_CBH_CONUS/ImageServer/exportImage",
    "cbd": "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2024/LF2024_CBD_CONUS/ImageServer/exportImage",
}


@dataclass(frozen=True)
class ExportSpec:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    bbox_sr: int
    image_sr: int
    width: int
    height: int


def export_image(
    *,
    service_key: str,
    spec: ExportSpec,
    destination: Path,
    interpolation: str = "RSP_BilinearInterpolation",
) -> Path:
    if service_key not in LANDFIRE_SERVICES:
        raise ValueError(f"Unsupported LANDFIRE service key: {service_key}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "f": "image",
        "format": "tiff",
        "bbox": f"{spec.xmin},{spec.ymin},{spec.xmax},{spec.ymax}",
        "bboxSR": str(spec.bbox_sr),
        "imageSR": str(spec.image_sr),
        "size": f"{spec.width},{spec.height}",
        "pixelType": "S16",
        "interpolation": interpolation,
        "compressionQuality": "100",
    }
    request = Request(f"{LANDFIRE_SERVICES[service_key]}?{urlencode(params)}")
    with _open_url_with_retry(request, allow_insecure=env_allows_insecure_ssl(), timeout=180, retries=4, backoff_seconds=2.0) as response:
        data = response.read()
    destination.write_bytes(data)
    with rasterio.open(destination) as src:
        if src.width <= 0 or src.height <= 0:
            raise ValueError(f"Downloaded LANDFIRE raster is invalid: {destination}")
    return destination
