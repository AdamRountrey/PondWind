from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds, transform_geom
from rasterio.windows import Window, from_bounds

from predictweather.geo import BoundingBox
from predictweather.http import _open_url_with_retry, download_url_to_file, env_allows_insecure_ssl, parse_json_text
from predictweather.wind import _draw_text, _interpolate_colormap, _write_png

EARTH_SEARCH_STAC = "https://earth-search.aws.element84.com/v1/search"
PLANETARY_COMPUTER_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PLANETARY_COMPUTER_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="
CMR_GRANULES_JSON = "https://cmr.earthdata.nasa.gov/search/granules.json"
ECOSTRESS_COLLECTION_CONCEPT_ID = "C2076090826-LPCLOUD"
TARGET_MAP_WIDTH = 1664
TARGET_MAP_HEIGHT = 1659
TARGET_LEGEND_WIDTH = 190
SATELLITE_RENDER_BUFFER_M = 60.0
MAX_PARALLEL_ASSET_DOWNLOADS = 4


@dataclass(frozen=True)
class StacAsset:
    key: str
    href: str
    scale: float = 1.0
    offset: float = 0.0
    nodata: float | None = None


@dataclass(frozen=True)
class StacSelection:
    collection: str
    item_id: str
    item_datetime_utc: str
    cloud_cover: float | None
    bbox_coverage_fraction: float | None
    assets: dict[str, StacAsset]
    raw_item: dict


@dataclass(frozen=True)
class CmrAssetSelection:
    collection: str
    item_id: str
    item_datetime_utc: str
    bbox_coverage_fraction: float | None
    assets: dict[str, str]
    raw_item: dict


def _fetch_stac_search(payload: dict, stac_url: str) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(stac_url, data=body, headers={"Content-Type": "application/json"})
    with _open_url_with_retry(request, allow_insecure=env_allows_insecure_ssl(), timeout=120, retries=4, backoff_seconds=2.0) as response:
        response_text = response.read().decode("utf-8", errors="replace")
    return parse_json_text(response_text, source=stac_url)


def _fetch_cmr_granules(params: dict[str, str]) -> dict:
    from urllib.parse import urlencode

    query = urlencode(params)
    request = Request(f"{CMR_GRANULES_JSON}?{query}", headers={"Accept": "application/json"})
    with _open_url_with_retry(request, allow_insecure=env_allows_insecure_ssl(), timeout=120, retries=4, backoff_seconds=2.0) as response:
        response_text = response.read().decode("utf-8", errors="replace")
    return parse_json_text(response_text, source=CMR_GRANULES_JSON)


def _asset_scale_and_offset(asset: dict) -> tuple[float, float, float | None]:
    bands = asset.get("raster:bands") or []
    if not bands:
        return 1.0, 0.0, None
    band = bands[0]
    nodata = band.get("nodata")
    if nodata is None:
        return float(band.get("scale", 1.0)), float(band.get("offset", 0.0)), None
    return float(band.get("scale", 1.0)), float(band.get("offset", 0.0)), float(nodata)


def _score_item(item: dict, end_time_utc: datetime) -> tuple[float, float]:
    item_time = datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00"))
    age_days = max(0.0, (end_time_utc - item_time).total_seconds() / 86400.0)
    cloud_cover = float(item["properties"].get("eo:cloud_cover", 100.0))
    return age_days, cloud_cover


def _bounds_intersection_fraction(bounds: list[float] | tuple[float, float, float, float] | None, bbox: BoundingBox) -> float | None:
    if not bounds or len(bounds) != 4:
        return None
    min_lon, min_lat, max_lon, max_lat = [float(value) for value in bounds]
    inter_min_lon = max(min_lon, bbox.min_lon)
    inter_min_lat = max(min_lat, bbox.min_lat)
    inter_max_lon = min(max_lon, bbox.max_lon)
    inter_max_lat = min(max_lat, bbox.max_lat)
    if inter_max_lon <= inter_min_lon or inter_max_lat <= inter_min_lat:
        return 0.0
    intersection_area = (inter_max_lon - inter_min_lon) * (inter_max_lat - inter_min_lat)
    request_area = max((bbox.max_lon - bbox.min_lon) * (bbox.max_lat - bbox.min_lat), 1.0e-12)
    return float(intersection_area / request_area)


def select_best_item(
    *,
    collection: str,
    bbox: BoundingBox,
    end_time_utc: datetime,
    lookback_days: int,
    required_assets: list[str],
    limit: int = 24,
    stac_url: str = EARTH_SEARCH_STAC,
    max_cloud_cover: float | None = None,
) -> StacSelection:
    candidates = list_candidate_items(
        collection=collection,
        bbox=bbox,
        end_time_utc=end_time_utc,
        lookback_days=lookback_days,
        required_assets=required_assets,
        limit=limit,
        stac_url=stac_url,
        max_cloud_cover=max_cloud_cover,
    )
    return candidates[0]


def list_candidate_items(
    *,
    collection: str,
    bbox: BoundingBox,
    end_time_utc: datetime,
    lookback_days: int,
    required_assets: list[str],
    limit: int = 24,
    stac_url: str = EARTH_SEARCH_STAC,
    max_cloud_cover: float | None = None,
) -> list[StacSelection]:
    start_time_utc = end_time_utc - timedelta(days=lookback_days)
    payload = {
        "collections": [collection],
        "bbox": [bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat],
        "datetime": f"{start_time_utc.isoformat().replace('+00:00', 'Z')}/{end_time_utc.isoformat().replace('+00:00', 'Z')}",
        "limit": limit,
    }
    response = _fetch_stac_search(payload, stac_url)
    features = response.get("features", [])
    if not features:
        raise FileNotFoundError(f"No {collection} imagery found for {payload['datetime']}")

    features = [item for item in features if all(asset_name in item.get("assets", {}) for asset_name in required_assets)]
    if not features:
        raise FileNotFoundError(f"No {collection} imagery found with required assets: {', '.join(required_assets)}")

    if max_cloud_cover is not None:
        filtered_features = []
        for item in features:
            cloud_cover = item.get("properties", {}).get("eo:cloud_cover")
            if cloud_cover is None:
                filtered_features.append(item)
                continue
            try:
                if float(cloud_cover) <= float(max_cloud_cover):
                    filtered_features.append(item)
            except (TypeError, ValueError):
                filtered_features.append(item)
        if filtered_features:
            features = filtered_features

    features_with_coverage: list[tuple[dict, float | None]] = []
    for item in features:
        coverage_fraction = _bounds_intersection_fraction(item.get("bbox"), bbox)
        if coverage_fraction == 0.0:
            continue
        features_with_coverage.append((item, coverage_fraction))
    if not features_with_coverage:
        raise FileNotFoundError(f"No {collection} imagery intersects the requested bbox")

    ranked = sorted(
        features_with_coverage,
        key=lambda candidate: (
            0.0 if candidate[1] is None else -candidate[1],
            _score_item(candidate[0], end_time_utc),
        ),
    )
    selections: list[StacSelection] = []
    for item, coverage_fraction in ranked:
        assets: dict[str, StacAsset] = {}
        for key in required_assets:
            asset = item["assets"][key]
            scale, offset, nodata = _asset_scale_and_offset(asset)
            assets[key] = StacAsset(key=key, href=_http_href(asset["href"]), scale=scale, offset=offset, nodata=nodata)
        selections.append(
            StacSelection(
                collection=collection,
                item_id=item["id"],
                item_datetime_utc=item["properties"]["datetime"],
                cloud_cover=item["properties"].get("eo:cloud_cover"),
                bbox_coverage_fraction=coverage_fraction,
                assets=assets,
                raw_item=item,
            )
        )
    return selections


def _http_href(href: str) -> str:
    if href.startswith("https://") or href.startswith("http://"):
        if "blob.core.windows.net" in href:
            return _sign_planetary_computer_href(href)
        return href
    if href.startswith("s3://"):
        bucket_and_key = href[5:]
        bucket, key = bucket_and_key.split("/", 1)
        if bucket == "usgs-landsat":
            return f"https://{bucket}.s3.us-west-2.amazonaws.com/{key}"
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return href


def list_ecostress_lste_candidates(
    *,
    bbox: BoundingBox,
    end_time_utc: datetime,
    lookback_days: int,
    limit: int = 24,
) -> list[CmrAssetSelection]:
    start_time_utc = end_time_utc - timedelta(days=lookback_days)
    params = {
        "concept_id": ECOSTRESS_COLLECTION_CONCEPT_ID,
        "provider": "LPCLOUD",
        "temporal": f"{start_time_utc.isoformat().replace('+00:00', 'Z')},{end_time_utc.isoformat().replace('+00:00', 'Z')}",
        "bounding_box": f"{bbox.min_lon},{bbox.min_lat},{bbox.max_lon},{bbox.max_lat}",
        "page_size": str(limit),
        "sort_key[]": "-start_date",
    }
    response = _fetch_cmr_granules(params)
    entries = response.get("feed", {}).get("entry", [])
    if not entries:
        raise FileNotFoundError("No ECOSTRESS LSTE granules found for the requested area and time range.")

    selections: list[CmrAssetSelection] = []
    for entry in entries:
        assets: dict[str, str] = {}
        for link in entry.get("links", []):
            href = link.get("href")
            if not href or not href.startswith("https://") or ".tif" not in href.lower():
                continue
            href_lower = href.lower()
            title_lower = str(link.get("title", "")).lower()
            if href_lower.endswith("_lst.tif") or "_lst" in title_lower:
                assets["lst"] = href
            elif href_lower.endswith("_cloud.tif") or "_cloud" in title_lower:
                assets["cloud"] = href
            elif href_lower.endswith("_water.tif") or "_water" in title_lower:
                assets["water"] = href
            elif href_lower.endswith("_qc.tif") or "_qc" in title_lower:
                assets["qc"] = href
        if not {"lst", "cloud", "water"}.issubset(assets):
            continue
        box = None
        boxes = entry.get("boxes") or []
        if boxes:
            try:
                south, west, north, east = [float(value) for value in str(boxes[0]).split()]
                box = [west, south, east, north]
            except (TypeError, ValueError):
                box = None
        selections.append(
            CmrAssetSelection(
                collection="ecostress-l2t-lste",
                item_id=entry.get("producer_granule_id") or entry.get("title") or entry.get("id", "ecostress"),
                item_datetime_utc=entry.get("time_start") or entry.get("updated"),
                bbox_coverage_fraction=_bounds_intersection_fraction(box, bbox) if box else None,
                assets=assets,
                raw_item=entry,
            )
        )

    if not selections:
        raise FileNotFoundError("No ECOSTRESS LSTE granules with required LST/cloud/water assets were found.")

    ranked = sorted(
        selections,
        key=lambda selection: (
            0.0 if selection.bbox_coverage_fraction is None else -selection.bbox_coverage_fraction,
            max(
                0.0,
                (
                    end_time_utc - datetime.fromisoformat(selection.item_datetime_utc.replace("Z", "+00:00"))
                ).total_seconds() / 86400.0
            ),
        ),
    )
    return ranked


def _bounded_downloads(downloads: list[tuple[str, str, Path]], *, max_workers: int) -> dict[str, Path]:
    if not downloads:
        return {}

    worker_count = max(1, min(max_workers, len(downloads)))
    results: dict[str, Path] = {}

    def _download(item: tuple[str, str, Path]) -> tuple[str, Path]:
        key, href, destination = item
        if not destination.exists():
            download_url_to_file(href, destination, allow_insecure=env_allows_insecure_ssl())
        return key, destination

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_download, item) for item in downloads]
        for future in as_completed(futures):
            key, destination = future.result()
            results[key] = destination
    return results


def download_cmr_assets(
    selection: CmrAssetSelection,
    destination_dir: Path,
    *,
    asset_keys: list[str] | None = None,
    max_workers: int = MAX_PARALLEL_ASSET_DOWNLOADS,
) -> dict[str, Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    keys = list(selection.assets) if asset_keys is None else list(asset_keys)
    downloads: list[tuple[str, str, Path]] = []
    for key in keys:
        href = selection.assets[key]
        suffix = Path(urlparse(href).path).suffix or ".tif"
        destination = destination_dir / f"{selection.collection}_{selection.item_id}_{key}{suffix}"
        downloads.append((key, href, destination))
    downloaded = _bounded_downloads(downloads, max_workers=max_workers)
    return {key: downloaded[key] for key in keys}


def _sign_planetary_computer_href(href: str) -> str:
    request = Request(f"{PLANETARY_COMPUTER_SIGN}{quote(href, safe='')}")
    with _open_url_with_retry(request, allow_insecure=env_allows_insecure_ssl(), timeout=120, retries=4, backoff_seconds=2.0) as response:
        response_text = response.read().decode("utf-8", errors="replace")
    payload = parse_json_text(response_text, source=PLANETARY_COMPUTER_SIGN)
    return payload["href"]


def _local_asset_path(selection: StacSelection, key: str, destination_dir: Path) -> Path:
    suffix = Path(urlparse(selection.assets[key].href).path).suffix or ".tif"
    return destination_dir / f"{selection.collection}_{selection.item_id}_{key}{suffix}"


def download_selection_assets(
    selection: StacSelection,
    destination_dir: Path,
    *,
    asset_keys: list[str] | None = None,
    max_workers: int = MAX_PARALLEL_ASSET_DOWNLOADS,
) -> dict[str, Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    keys = list(selection.assets) if asset_keys is None else list(asset_keys)
    downloads: list[tuple[str, str, Path]] = []
    for key in keys:
        asset = selection.assets[key]
        destination = _local_asset_path(selection, key, destination_dir)
        downloads.append((key, asset.href, destination))
    downloaded = _bounded_downloads(downloads, max_workers=max_workers)
    return {key: downloaded[key] for key in keys}


def _bbox_geometry(bbox: BoundingBox) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[
            [bbox.min_lon, bbox.min_lat],
            [bbox.max_lon, bbox.min_lat],
            [bbox.max_lon, bbox.max_lat],
            [bbox.min_lon, bbox.max_lat],
            [bbox.min_lon, bbox.min_lat],
        ]],
    }


def _bounded_window(src: rasterio.io.DatasetReader, bounds: tuple[float, float, float, float]) -> Window:
    requested = from_bounds(*bounds, transform=src.transform)
    requested = requested.round_offsets().round_lengths()
    full = Window(col_off=0, row_off=0, width=src.width, height=src.height)
    return requested.intersection(full)


def _read_masked_crop(path: Path, bbox: BoundingBox, *, resampling: Resampling = Resampling.nearest) -> tuple[np.ma.MaskedArray, Affine, rasterio.crs.CRS]:
    with rasterio.open(path) as src:
        source_bounds = transform_bounds("EPSG:4326", src.crs, bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat, densify_pts=21)
        window = _bounded_window(src, source_bounds)
        data = src.read(window=window, masked=True)
        transform = src.window_transform(window)
        return data, transform, src.crs


def _resample_mask_to_match(
    source: np.ndarray,
    source_transform: Affine,
    source_crs,
    target_shape: tuple[int, int],
    target_transform: Affine,
    target_crs,
) -> np.ndarray:
    destination = np.full(target_shape, np.nan, dtype=np.float32)
    reproject(
        source=source.astype(np.float32),
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return destination


def _band_to_float(data: np.ndarray | np.ma.MaskedArray) -> np.ndarray:
    band = np.ma.asarray(data, dtype=np.float32)
    if np.ma.isMaskedArray(band):
        return band.filled(np.nan)
    return np.asarray(band, dtype=np.float32)


def _valid_fraction(data: np.ndarray | np.ma.MaskedArray) -> float:
    band = _band_to_float(data)
    if band.size == 0:
        return 0.0
    return float(np.isfinite(band).sum() / band.size)


def _validate_min_coverage(label: str, data: np.ndarray | np.ma.MaskedArray, *, min_fraction: float = 0.9) -> None:
    coverage = _valid_fraction(data)
    if coverage < min_fraction:
        raise ValueError(f"{label} coverage too low for requested bbox ({coverage:.1%} < {min_fraction:.1%})")


def _validate_min_pixels(label: str, mask: np.ndarray, *, min_pixels: int = 25) -> None:
    pixel_count = int(np.count_nonzero(mask))
    if pixel_count < min_pixels:
        raise ValueError(f"{label} has too few usable pixels for requested bbox ({pixel_count} < {min_pixels})")


def _apply_scale_offset(data: np.ndarray | np.ma.MaskedArray, scale: float, offset: float, nodata: float | None) -> np.ndarray:
    band = _band_to_float(data)
    if nodata is not None and np.isfinite(nodata):
        band[band == nodata] = np.nan
    return band * scale + offset


def _resample_to_match(source: np.ndarray, source_transform: Affine, source_crs, target_shape: tuple[int, int], target_transform: Affine, target_crs) -> np.ndarray:
    destination = np.full(target_shape, np.nan, dtype=np.float32)
    reproject(
        source=source.astype(np.float32),
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return destination


def _percentile_rgb_stretch(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.uint8)
    for index in range(3):
        channel = rgb[index].astype(np.float32)
        valid = channel[np.isfinite(channel)]
        if valid.size == 0:
            continue
        lo = float(np.percentile(valid, 2))
        hi = float(np.percentile(valid, 98))
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
        out[index] = np.round(scaled * 255.0).astype(np.uint8)
    return np.moveaxis(out, 0, -1)


def _resize_rgb_nearest(rgb: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    src_height, src_width = rgb.shape[:2]
    if src_height == target_height and src_width == target_width:
        return rgb
    row_idx = np.linspace(0, src_height - 1, target_height).round().astype(int)
    col_idx = np.linspace(0, src_width - 1, target_width).round().astype(int)
    return rgb[np.ix_(row_idx, col_idx)]


def _load_dem_basemap_rgb(dem_basemap_tif: Path) -> np.ndarray:
    with rasterio.open(dem_basemap_tif) as src:
        dem_base = src.read(1, masked=True).filled(0).astype(np.uint8)
    rgb = np.repeat(dem_base[:, :, None], 3, axis=2)
    return _resize_rgb_nearest(rgb, TARGET_MAP_HEIGHT, TARGET_MAP_WIDTH).astype(np.float32)


def _legend_canvas(
    *,
    rgb: np.ndarray,
    title: str,
    units: str,
    vmin: float,
    vmax: float,
    colormap: list[tuple[float, tuple[int, int, int]]],
    output_png: Path,
    show_gradient: bool = True,
    footer_text: str | None = None,
) -> Path:
    rgb = _resize_rgb_nearest(rgb, TARGET_MAP_HEIGHT, TARGET_MAP_WIDTH)
    map_height, map_width = rgb.shape[:2]
    top_margin = 80
    footer_band_height = 48 if footer_text else 0
    bottom_margin = (24 + footer_band_height) if footer_text else 24
    legend_width = TARGET_LEGEND_WIDTH
    canvas_height = max(map_height + footer_band_height, 320)
    canvas = np.full((canvas_height, map_width + legend_width, 3), 255, dtype=np.uint8)
    canvas[:map_height, :map_width] = rgb
    if footer_text:
        footer_top = canvas_height - footer_band_height
        canvas[footer_top:canvas_height, : map_width + legend_width] = 255

    legend_left = map_width + 40
    available_legend_height = max(80, canvas_height - top_margin - bottom_margin)
    legend_height = min(available_legend_height, 900)
    legend_top = top_margin
    legend_bottom = legend_top + legend_height
    legend_right = legend_left + 28
    if show_gradient:
        gradient_values = np.linspace(1.0, 0.0, legend_height, dtype=np.float32).reshape(-1, 1)
        gradient_rgb = np.repeat(_interpolate_colormap(gradient_values, colormap), legend_right - legend_left, axis=1)
        canvas[legend_top:legend_bottom, legend_left:legend_right] = gradient_rgb

        tick_values = np.linspace(vmax, vmin, 5, dtype=np.float32)
        for tick_value in tick_values:
            fraction = 0.0 if vmax <= vmin else float((tick_value - vmin) / (vmax - vmin))
            y = int(round(legend_bottom - fraction * legend_height))
            y = max(legend_top, min(legend_bottom - 1, y))
            canvas[y : y + 2, legend_right : legend_right + 12] = (0, 0, 0)
            _draw_text(canvas, legend_right + 18, y - 6, f"{tick_value:.1f}", (0, 0, 0), scale=2)

    _draw_text(canvas, legend_left - 4, 32, title, (0, 0, 0), scale=3)
    _draw_text(canvas, legend_left - 4, 58, units, (0, 0, 0), scale=2)
    if footer_text:
        footer_y = canvas_height - 34
        _draw_text(canvas, 24, footer_y, footer_text, (0, 0, 0), scale=3)
    return _write_png(output_png, canvas)


def render_rgb_preview(
    visual_tif: Path,
    bbox: BoundingBox,
    output_png: Path,
    dem_basemap_tif: Path,
    title: str = "rgb",
    alpha: float = 0.72,
    footer_text: str | None = None,
) -> Path:
    data, crop_transform, src_crs = _read_masked_crop(visual_tif, bbox)
    if src_crs is None:
        raise ValueError("RGB source CRS is required.")
    rgb_data = np.ma.asarray(data[:3], dtype=np.float32)
    valid_source = (~np.ma.getmaskarray(rgb_data)).all(axis=0).astype(np.float32)
    with rasterio.open(dem_basemap_tif) as base_src:
        base_height = base_src.height
        base_width = base_src.width
        base_transform = base_src.transform
        base_crs = base_src.crs
    reprojected_channels: list[np.ndarray] = []
    for band_index in range(3):
        destination = np.zeros((base_height, base_width), dtype=np.float32)
        reproject(
            source=rgb_data[band_index].filled(np.nan).astype(np.float32),
            destination=destination,
            src_transform=crop_transform,
            src_crs=src_crs,
            dst_transform=base_transform,
            dst_crs=base_crs,
            src_nodata=np.nan,
            dst_nodata=0.0,
            resampling=Resampling.bilinear,
        )
        reprojected_channels.append(destination)
    valid_mask = np.zeros((base_height, base_width), dtype=np.float32)
    reproject(
        source=valid_source,
        destination=valid_mask,
        src_transform=crop_transform,
        src_crs=src_crs,
        dst_transform=base_transform,
        dst_crs=base_crs,
        src_nodata=0.0,
        dst_nodata=0.0,
        resampling=Resampling.nearest,
    )
    rgb_overlay = np.clip(np.moveaxis(np.stack(reprojected_channels, axis=0), 0, -1), 0, 255)
    base_rgb = _load_dem_basemap_rgb(dem_basemap_tif)
    rgb_overlay = _resize_rgb_nearest(np.nan_to_num(rgb_overlay, nan=0.0).astype(np.uint8), TARGET_MAP_HEIGHT, TARGET_MAP_WIDTH).astype(np.float32)
    valid_mask = _resize_rgb_nearest((valid_mask > 0.5).astype(np.uint8)[:, :, None] * 255, TARGET_MAP_HEIGHT, TARGET_MAP_WIDTH)[:, :, 0] > 0
    blended = base_rgb.copy()
    blended[valid_mask] = np.clip((1.0 - alpha) * base_rgb[valid_mask] + alpha * rgb_overlay[valid_mask], 0, 255)
    return _legend_canvas(
        rgb=blended.astype(np.uint8),
        title=title,
        units="scene",
        vmin=0.0,
        vmax=255.0,
        colormap=[(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
        output_png=output_png,
        show_gradient=False,
        footer_text=footer_text,
    )


def count_sentinel_water_pixels(*, scl_tif: Path, bbox: BoundingBox) -> int:
    scl_data, _, _ = _read_masked_crop(scl_tif, bbox)
    scl = _band_to_float(scl_data[0])
    valid_scl = np.isfinite(scl)
    scl_classes = np.where(valid_scl, np.rint(scl), -1).astype(np.int16)
    water_mask = scl_classes == 6
    return int(np.count_nonzero(water_mask))


def sentinel_clear_fraction(*, scl_tif: Path, bbox: BoundingBox) -> float:
    scl_data, _, _ = _read_masked_crop(scl_tif, bbox)
    scl = _band_to_float(scl_data[0])
    valid_scl = np.isfinite(scl)
    scl_classes = np.where(valid_scl, np.rint(scl), -1).astype(np.int16)
    valid_pixels = valid_scl & (scl_classes > 0)
    if not np.any(valid_pixels):
        return 0.0
    clear_classes = np.isin(scl_classes, [4, 5, 6, 7])
    return float(np.count_nonzero(clear_classes & valid_pixels) / np.count_nonzero(valid_pixels))


def count_landsat_water_pixels(*, qa_pixel_tif: Path, bbox: BoundingBox) -> int:
    qa_data, _, _ = _read_masked_crop(qa_pixel_tif, bbox)
    qa = _band_to_float(qa_data[0])
    valid_qa = np.isfinite(qa)
    qa_bits = np.where(valid_qa, np.rint(qa), 0).astype(np.uint16)
    water_mask = valid_qa & (((qa_bits >> 7) & 1) == 1)
    return int(np.count_nonzero(water_mask))


def count_ecostress_water_pixels(*, water_tif: Path, bbox: BoundingBox) -> int:
    water_data, _, _ = _read_masked_crop(water_tif, bbox)
    water = _band_to_float(water_data[0])
    valid = np.isfinite(water)
    if not np.any(valid):
        return 0
    water_mask = valid & (water > 0.5)
    return int(np.count_nonzero(water_mask))


def derive_sentinel_chla(
    *,
    red_tif: Path,
    rededge1_tif: Path,
    scl_tif: Path,
    bbox: BoundingBox,
    red_scale: float,
    red_offset: float,
    red_nodata: float,
    rededge_scale: float,
    rededge_offset: float,
    rededge_nodata: float,
    output_tif: Path,
    output_png: Path,
    dem_basemap_tif: Path,
    footer_text: str | None = None,
) -> dict:
    red_data, red_transform, red_crs = _read_masked_crop(red_tif, bbox)
    rededge_data, rededge_transform, rededge_crs = _read_masked_crop(rededge1_tif, bbox)
    scl_data, scl_transform, scl_crs = _read_masked_crop(scl_tif, bbox)
    _validate_min_coverage("Sentinel red band", red_data[0])
    _validate_min_coverage("Sentinel red-edge band", rededge_data[0])
    _validate_min_coverage("Sentinel scene classification", scl_data[0])

    red = _apply_scale_offset(red_data[0], red_scale, red_offset, red_nodata)
    rededge = _apply_scale_offset(rededge_data[0], rededge_scale, rededge_offset, rededge_nodata)
    if red.shape != rededge.shape or red_transform != rededge_transform:
        red = _resample_to_match(red, red_transform, red_crs, rededge.shape, rededge_transform, rededge_crs)
        red_transform = rededge_transform
        red_crs = rededge_crs
    scl = _band_to_float(scl_data[0])
    if scl.shape != rededge.shape or scl_transform != rededge_transform:
        scl = _resample_mask_to_match(scl, scl_transform, scl_crs, rededge.shape, rededge_transform, rededge_crs)
    valid_scl = np.isfinite(scl)
    scl_classes = np.where(valid_scl, np.rint(scl), -1).astype(np.int16)
    water_mask = scl_classes == 6
    _validate_min_pixels("Sentinel-2 water-classified pixels", water_mask)

    denominator = rededge + red
    ndci = np.divide(rededge - red, denominator, out=np.full_like(rededge, np.nan), where=np.abs(denominator) > 1.0e-6)
    chla = 14.039 + 86.115 * ndci + 194.325 * ndci * ndci
    chla = np.where(water_mask, chla, np.nan)
    chla = np.clip(chla, 0.0, 250.0).astype(np.float32)
    _validate_min_pixels("Sentinel-2 chlorophyll-a output", np.isfinite(chla))

    _write_single_band_geotiff(output_tif, chla, red_transform, red_crs)
    _render_scalar_product(
        field=chla,
        transform=red_transform,
        crs=red_crs,
        output_png=output_png,
        dem_basemap_tif=dem_basemap_tif,
        title="chla",
        units="mg/m3",
        colormap=[
            (0.00, (49, 88, 173)),
            (0.25, (91, 155, 213)),
            (0.50, (0, 0, 0)),
            (0.75, (217, 95, 2)),
            (1.00, (200, 45, 45)),
        ],
        footer_text=footer_text,
    )
    return {
        "product": "estimated_chla",
        "item_bbox": [bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat],
        "min_mg_m3": float(np.nanmin(chla)),
        "max_mg_m3": float(np.nanmax(chla)),
        "mean_mg_m3": float(np.nanmean(chla)),
        "output_tif": str(output_tif),
        "output_png": str(output_png),
        "note": "Estimated chlorophyll-a from Sentinel-2 red/red-edge NDCI with a literature-based empirical conversion.",
    }


def derive_sentinel_turbidity(
    *,
    red_tif: Path,
    nir_tif: Path,
    scl_tif: Path,
    bbox: BoundingBox,
    red_scale: float,
    red_offset: float,
    red_nodata: float,
    nir_scale: float,
    nir_offset: float,
    nir_nodata: float,
    output_tif: Path,
    output_png: Path,
    dem_basemap_tif: Path,
    footer_text: str | None = None,
) -> dict:
    red_data, red_transform, red_crs = _read_masked_crop(red_tif, bbox)
    nir_data, nir_transform, nir_crs = _read_masked_crop(nir_tif, bbox)
    scl_data, scl_transform, scl_crs = _read_masked_crop(scl_tif, bbox)
    _validate_min_coverage("Sentinel red band", red_data[0])
    _validate_min_coverage("Sentinel NIR band", nir_data[0])
    _validate_min_coverage("Sentinel scene classification", scl_data[0])

    red = _apply_scale_offset(red_data[0], red_scale, red_offset, red_nodata)
    nir = _apply_scale_offset(nir_data[0], nir_scale, nir_offset, nir_nodata)
    if red.shape != nir.shape or red_transform != nir_transform:
        red = _resample_to_match(red, red_transform, red_crs, nir.shape, nir_transform, nir_crs)
        red_transform = nir_transform
        red_crs = nir_crs

    scl = _band_to_float(scl_data[0])
    if scl.shape != nir.shape or scl_transform != nir_transform:
        scl = _resample_mask_to_match(scl, scl_transform, scl_crs, nir.shape, nir_transform, nir_crs)
    valid_scl = np.isfinite(scl)
    scl_classes = np.where(valid_scl, np.rint(scl), -1).astype(np.int16)
    water_mask = scl_classes == 6
    _validate_min_pixels("Sentinel-2 water-classified pixels", water_mask)

    def _dogliotti_band_turbidity(rho: np.ndarray, a_coeff: float, c_coeff: float) -> np.ndarray:
        numerator = a_coeff * rho
        denominator = 1.0 - (rho / c_coeff)
        return np.divide(numerator, denominator, out=np.full_like(rho, np.nan), where=np.abs(denominator) > 1.0e-6)

    turb_red = _dogliotti_band_turbidity(red, 288.95, 0.1395)
    turb_nir = _dogliotti_band_turbidity(nir, 1830.7, 0.2033)

    red_valid = np.isfinite(red)
    use_red = red < 0.05
    use_nir = red > 0.07
    blend = np.clip((red - 0.05) / 0.02, 0.0, 1.0)
    turbidity = np.where(
        use_red,
        turb_red,
        np.where(use_nir, turb_nir, (1.0 - blend) * turb_red + blend * turb_nir),
    )
    turbidity = np.where(water_mask & red_valid, turbidity, np.nan)
    turbidity = np.clip(turbidity, 0.0, 500.0).astype(np.float32)
    _validate_min_pixels("Sentinel-2 turbidity output", np.isfinite(turbidity))

    _write_single_band_geotiff(output_tif, turbidity, red_transform, red_crs)
    _render_scalar_product(
        field=turbidity,
        transform=red_transform,
        crs=red_crs,
        output_png=output_png,
        dem_basemap_tif=dem_basemap_tif,
        title="turb",
        units="fnu",
        colormap=[
            (0.00, (49, 88, 173)),
            (0.25, (91, 155, 213)),
            (0.50, (0, 0, 0)),
            (0.75, (217, 95, 2)),
            (1.00, (200, 45, 45)),
        ],
        footer_text=footer_text,
    )
    return {
        "product": "estimated_turbidity",
        "item_bbox": [bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat],
        "min_fnu": float(np.nanmin(turbidity)),
        "max_fnu": float(np.nanmax(turbidity)),
        "mean_fnu": float(np.nanmean(turbidity)),
        "output_tif": str(output_tif),
        "output_png": str(output_png),
        "note": "Estimated turbidity from Sentinel-2 red and NIR reflectance using a Dogliotti-style switching algorithm.",
    }


def derive_landsat_sst(
    *,
    lwir11_tif: Path,
    qa_pixel_tif: Path,
    bbox: BoundingBox,
    lwir_scale: float,
    lwir_offset: float,
    lwir_nodata: float,
    output_tif: Path,
    output_png: Path,
    dem_basemap_tif: Path,
    footer_text: str | None = None,
) -> dict:
    lwir_data, transform, crs = _read_masked_crop(lwir11_tif, bbox)
    qa_data, qa_transform, qa_crs = _read_masked_crop(qa_pixel_tif, bbox)
    _validate_min_coverage("Landsat thermal band", lwir_data[0])
    _validate_min_coverage("Landsat QA band", qa_data[0])
    if qa_data.shape[1:] != lwir_data.shape[1:] or qa_transform != transform:
        qa = _resample_mask_to_match(_band_to_float(qa_data[0]), qa_transform, qa_crs, lwir_data.shape[1:], transform, crs)
    else:
        qa = _band_to_float(qa_data[0])

    kelvin = _apply_scale_offset(lwir_data[0], lwir_scale, lwir_offset, lwir_nodata)
    valid_qa = np.isfinite(qa)
    _validate_min_coverage("Landsat QA band", qa)
    qa_bits = np.where(valid_qa, np.rint(qa), 0).astype(np.uint16)
    fahrenheit = (kelvin - 273.15) * 9.0 / 5.0 + 32.0
    cloud_bits = ((qa_bits >> 1) & 1) | ((qa_bits >> 2) & 1) | ((qa_bits >> 3) & 1) | ((qa_bits >> 4) & 1) | ((qa_bits >> 5) & 1)
    water_mask = valid_qa & (((qa_bits >> 7) & 1) == 1)
    _validate_min_pixels("Landsat water-classified pixels", water_mask)
    sst = np.where((cloud_bits == 0) & water_mask, fahrenheit, np.nan).astype(np.float32)
    _validate_min_pixels("Landsat SST output", np.isfinite(sst))

    _write_single_band_geotiff(output_tif, sst, transform, crs)
    _render_scalar_product(
        field=sst,
        transform=transform,
        crs=crs,
        output_png=output_png,
        dem_basemap_tif=dem_basemap_tif,
        title="sst",
        units="deg f",
        colormap=[
            (0.00, (49, 88, 173)),
            (0.35, (91, 155, 213)),
            (0.50, (0, 0, 0)),
            (0.75, (244, 165, 130)),
            (1.00, (200, 45, 45)),
        ],
        footer_text=footer_text,
    )
    return {
        "product": "sst",
        "source": "landsat",
        "item_bbox": [bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat],
        "min_deg_f": float(np.nanmin(sst)),
        "max_deg_f": float(np.nanmax(sst)),
        "mean_deg_f": float(np.nanmean(sst)),
        "output_tif": str(output_tif),
        "output_png": str(output_png),
    }


def derive_ecostress_sst(
    *,
    lst_tif: Path,
    cloud_tif: Path,
    water_tif: Path,
    bbox: BoundingBox,
    output_tif: Path,
    output_png: Path,
    dem_basemap_tif: Path,
    footer_text: str | None = None,
) -> dict:
    lst_data, lst_transform, lst_crs = _read_masked_crop(lst_tif, bbox)
    cloud_data, cloud_transform, cloud_crs = _read_masked_crop(cloud_tif, bbox)
    water_data, water_transform, water_crs = _read_masked_crop(water_tif, bbox)
    _validate_min_coverage("ECOSTRESS LST band", lst_data[0])
    _validate_min_coverage("ECOSTRESS cloud mask", cloud_data[0], min_fraction=0.75)
    _validate_min_coverage("ECOSTRESS water mask", water_data[0], min_fraction=0.75)

    lst = _band_to_float(lst_data[0])
    with rasterio.open(lst_tif) as src:
        scales = list(src.scales or [])
        offsets = list(src.offsets or [])
        nodata = src.nodata
    scale = float(scales[0]) if scales and scales[0] not in (None, 0.0) else 1.0
    offset = float(offsets[0]) if offsets and offsets[0] is not None else 0.0
    if nodata is not None and np.isfinite(nodata):
        lst[lst == nodata] = np.nan
    if scale == 1.0 and np.nanpercentile(lst, 95) > 1000.0:
        # ECOSTRESS LSTE products are commonly stored as scaled integers.
        scale = 0.02
    kelvin = lst * scale + offset

    cloud = _band_to_float(cloud_data[0])
    if cloud.shape != kelvin.shape or cloud_transform != lst_transform:
        cloud = _resample_mask_to_match(cloud, cloud_transform, cloud_crs, kelvin.shape, lst_transform, lst_crs)
    water = _band_to_float(water_data[0])
    if water.shape != kelvin.shape or water_transform != lst_transform:
        water = _resample_mask_to_match(water, water_transform, water_crs, kelvin.shape, lst_transform, lst_crs)

    water_mask = np.isfinite(water) & (water > 0.5)
    _validate_min_pixels("ECOSTRESS water-classified pixels", water_mask)
    cloud_mask = np.isfinite(cloud) & (cloud > 0.5)
    fahrenheit = (kelvin - 273.15) * 9.0 / 5.0 + 32.0
    sst = np.where(water_mask & ~cloud_mask, fahrenheit, np.nan).astype(np.float32)
    _validate_min_pixels("ECOSTRESS SST output", np.isfinite(sst))

    _write_single_band_geotiff(output_tif, sst, lst_transform, lst_crs)
    _render_scalar_product(
        field=sst,
        transform=lst_transform,
        crs=lst_crs,
        output_png=output_png,
        dem_basemap_tif=dem_basemap_tif,
        title="sst",
        units="deg f",
        colormap=[
            (0.00, (49, 88, 173)),
            (0.35, (91, 155, 213)),
            (0.50, (0, 0, 0)),
            (0.75, (244, 165, 130)),
            (1.00, (200, 45, 45)),
        ],
        footer_text=footer_text,
    )
    return {
        "product": "sst",
        "source": "ecostress",
        "item_bbox": [bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat],
        "min_deg_f": float(np.nanmin(sst)),
        "max_deg_f": float(np.nanmax(sst)),
        "mean_deg_f": float(np.nanmean(sst)),
        "output_tif": str(output_tif),
        "output_png": str(output_png),
        "scale_applied": scale,
        "offset_applied": offset,
    }


def _render_scalar_product(
    *,
    field: np.ndarray,
    transform: Affine,
    crs,
    output_png: Path,
    dem_basemap_tif: Path,
    title: str,
    units: str,
    colormap: list[tuple[float, tuple[int, int, int]]],
    alpha: float = 0.62,
    footer_text: str | None = None,
) -> Path:
    valid = field[np.isfinite(field)]
    if valid.size == 0:
        raise ValueError(f"No finite values available for {title}")
    vmin = float(np.percentile(valid, 2))
    vmax = float(np.percentile(valid, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0
    normalized = np.clip((field - vmin) / (vmax - vmin), 0.0, 1.0)
    with rasterio.open(dem_basemap_tif) as base_src:
        base_height = base_src.height
        base_width = base_src.width
        base_transform = base_src.transform
        base_crs = base_src.crs
    normalized_full = np.full((base_height, base_width), np.nan, dtype=np.float32)
    reproject(
        source=normalized.astype(np.float32),
        destination=normalized_full,
        src_transform=transform,
        src_crs=crs,
        dst_transform=base_transform,
        dst_crs=base_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    valid_mask_full = np.zeros((base_height, base_width), dtype=np.float32)
    reproject(
        source=np.isfinite(field).astype(np.float32),
        destination=valid_mask_full,
        src_transform=transform,
        src_crs=crs,
        dst_transform=base_transform,
        dst_crs=base_crs,
        src_nodata=0.0,
        dst_nodata=0.0,
        resampling=Resampling.nearest,
    )
    overlay_rgb = _interpolate_colormap(np.clip(normalized_full, 0.0, 1.0), colormap).astype(np.float32)
    overlay_rgb[~np.isfinite(normalized_full)] = 0.0
    overlay_rgb = _resize_rgb_nearest(overlay_rgb.astype(np.uint8), TARGET_MAP_HEIGHT, TARGET_MAP_WIDTH).astype(np.float32)
    mask_full = _resize_rgb_nearest((valid_mask_full > 0.5).astype(np.uint8)[:, :, None] * 255, TARGET_MAP_HEIGHT, TARGET_MAP_WIDTH)[:, :, 0] > 0
    base_rgb = _load_dem_basemap_rgb(dem_basemap_tif)
    blended = base_rgb.copy()
    blended[mask_full] = np.clip((1.0 - alpha) * base_rgb[mask_full] + alpha * overlay_rgb[mask_full], 0, 255)
    return _legend_canvas(
        rgb=blended.astype(np.uint8),
        title=title,
        units=units,
        vmin=vmin,
        vmax=vmax,
        colormap=colormap,
        output_png=output_png,
        footer_text=footer_text,
    )


def _write_single_band_geotiff(path: Path, data: np.ndarray, transform: Affine, crs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path
