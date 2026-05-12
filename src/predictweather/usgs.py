from __future__ import annotations

import json
import zipfile
from pathlib import Path
from urllib.parse import urlencode

from predictweather.http import download_url_to_file, env_allows_insecure_ssl, fetch_json

TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
THREEDEP_IMAGE_SERVER_EXPORT_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
DEFAULT_DATASETS = (
    "Digital Elevation Model (DEM) 1 meter",
)

def query_dem_products(bbox: str, dataset: str, max_items: int = 25) -> list[dict]:
    query = urlencode(
        {
            "datasets": dataset,
            "bbox": bbox,
            "prodFormats": "GeoTIFF",
            "outputFormat": "JSON",
            "max": max_items,
        }
    )
    payload = fetch_json(f"{TNM_PRODUCTS_URL}?{query}", allow_insecure=env_allows_insecure_ssl())
    return payload.get("items", [])


def query_best_available_dem_products(bbox: str, max_items: int = 100) -> tuple[str, list[dict]]:
    for dataset in DEFAULT_DATASETS:
        items = query_dem_products(bbox=bbox, dataset=dataset, max_items=max_items)
        if items:
            return dataset, items
    return DEFAULT_DATASETS[0], []


def best_download_url(item: dict) -> str:
    candidates = [
        item.get("downloadURL"),
        item.get("downloadUrl"),
        item.get("url"),
    ]
    urls = item.get("urls")
    if isinstance(urls, dict):
        candidates.extend(urls.values())

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.startswith("http"):
            return candidate

    raise ValueError(f"No download URL found in item: {item}")


def download_file(url: str, destination: Path) -> Path:
    return download_url_to_file(url, destination, allow_insecure=env_allows_insecure_ssl())


def download_seamless_dem(
    bbox: str,
    destination: Path,
    size_pixels: int | tuple[int, int],
    *,
    bbox_sr: int = 4326,
    image_sr: int = 4326,
) -> Path:
    if isinstance(size_pixels, tuple):
        width, height = size_pixels
    else:
        width = height = size_pixels
    query = urlencode(
        {
            "bbox": bbox,
            "bboxSR": bbox_sr,
            "imageSR": image_sr,
            "format": "tiff",
            "pixelType": "F32",
            "interpolation": "RSP_BilinearInterpolation",
            "size": f"{width},{height}",
            "f": "image",
        }
    )
    return download_url_to_file(
        f"{THREEDEP_IMAGE_SERVER_EXPORT_URL}?{query}",
        destination,
        allow_insecure=env_allows_insecure_ssl(),
    )


def extract_tifs_if_zip(path: Path, extract_dir: Path) -> list[Path]:
    if path.suffix.lower() != ".zip":
        return [path]

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(extract_dir)

    tif_files = sorted(extract_dir.rglob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No GeoTIFF found in extracted archive: {path}")
    return tif_files


def extract_if_zip(path: Path, extract_dir: Path) -> Path:
    tif_files = extract_tifs_if_zip(path, extract_dir)
    if len(tif_files) == 1:
        return tif_files[0]
    tif_files.sort(key=lambda tif_path: tif_path.stat().st_size, reverse=True)
    return tif_files[0]
