from __future__ import annotations

import json
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.fill import fillnodata
from rasterio.mask import mask
from rasterio.merge import merge as merge_rasters
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, transform_bounds, transform_geom, Resampling

from predictweather.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, RESOURCE_DATA_RAW_DIR, SiteConfig
from predictweather.geo import BoundingBox, buffered_square_bbox_from_center, square_bbox_from_center
from predictweather.usgs import best_download_url, download_file, download_seamless_dem, extract_if_zip, extract_tifs_if_zip, query_best_available_dem_products
from predictweather.wind import clip_dem_to_bbox, write_dem_preview


@dataclass(frozen=True)
class PreparedSiteDomain:
    site: SiteConfig
    bbox: BoundingBox
    solve_bbox: BoundingBox
    dataset: str
    source_dem_tif: Path
    clipped_dem_tif: Path
    dem_preview_tif: Path
    solve_dem_tif: Path
    solve_dem_preview_tif: Path
    raw_download_path: Path
    manifest_path: Path


def _bounds_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    a_minx, a_miny, a_maxx, a_maxy = a
    b_minx, b_miny, b_maxx, b_maxy = b
    return not (a_maxx < b_minx or b_maxx < a_minx or a_maxy < b_miny or b_maxy < a_miny)


def _bounds_contain(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    outer_minx, outer_miny, outer_maxx, outer_maxy = outer
    inner_minx, inner_miny, inner_maxx, inner_maxy = inner
    return (
        outer_minx <= inner_minx
        and outer_miny <= inner_miny
        and outer_maxx >= inner_maxx
        and outer_maxy >= inner_maxy
    )


def _candidate_local_dem_dirs(raw_data_dir: Path) -> list[Path]:
    dirs: list[Path] = []
    for candidate in (raw_data_dir, RESOURCE_DATA_RAW_DIR):
        resolved = candidate.resolve()
        if resolved.exists() and resolved not in dirs:
            dirs.append(resolved)
    return dirs


def _local_dem_candidates(directory: Path) -> list[Path]:
    candidates: list[Path] = []
    for pattern in ("*.tif", "*.tiff", "*.zip"):
        candidates.extend(sorted(directory.rglob(pattern)))
    return candidates


def _find_local_dem_covering_bbox(bbox: BoundingBox, raw_data_dir: Path) -> Path | None:
    local_dem, _ = _find_local_dem_covering_bbox_with_diagnostics(bbox, raw_data_dir)
    return local_dem


def _find_local_dem_covering_bbox_with_diagnostics(bbox: BoundingBox, raw_data_dir: Path) -> tuple[Path | None, list[str]]:
    bbox_bounds_4326 = (bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
    diagnostics: list[str] = []
    for directory in _candidate_local_dem_dirs(raw_data_dir):
        for tif_path in _local_dem_candidates(directory):
            try:
                candidate_tif = extract_if_zip(tif_path, directory / tif_path.stem) if tif_path.suffix.lower() == ".zip" else tif_path
                with rasterio.open(candidate_tif) as src:
                    src_bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
                if _bounds_contain(src_bounds_4326, bbox_bounds_4326):
                    diagnostics.append(f"local_cache: matched {candidate_tif}")
                    return candidate_tif, diagnostics
                diagnostics.append(f"local_cache: skipped {candidate_tif.name} (coverage bounds do not contain site bbox)")
            except Exception as exc:
                diagnostics.append(f"local_cache: skipped {tif_path.name} ({exc})")
                continue
    return None, diagnostics


def _intersecting_sources_for_bbox(bbox: BoundingBox, source_paths: list[Path]) -> list[Path]:
    bbox_bounds_4326 = (bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
    selected: list[Path] = []
    for source_path in source_paths:
        with rasterio.open(source_path) as src:
            src_bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        if _bounds_intersect(src_bounds_4326, bbox_bounds_4326):
            selected.append(source_path)
    return selected


def _build_dem_mosaic(source_paths: list[Path], destination_tif: Path) -> Path:
    if not source_paths:
        raise ValueError("No source rasters provided for DEM mosaic.")
    destination_tif.parent.mkdir(parents=True, exist_ok=True)
    sources = [rasterio.open(source_path) for source_path in source_paths]
    try:
        mosaic_data, mosaic_transform = merge_rasters(sources)
        meta = sources[0].meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "height": mosaic_data.shape[1],
                "width": mosaic_data.shape[2],
                "transform": mosaic_transform,
            }
        )
        with rasterio.open(destination_tif, "w", **meta) as dst:
            dst.write(mosaic_data)
    finally:
        for src in sources:
            src.close()
    return destination_tif


def _coverage_fraction(clipped_tif: Path) -> float:
    with rasterio.open(clipped_tif) as src:
        data = src.read(1, masked=True)
        if data.size == 0:
            return 0.0
        valid_mask = np.isfinite(data.filled(np.nan)) & ~np.ma.getmaskarray(data)
        return float(valid_mask.sum() / valid_mask.size)


def _coverage_metrics(clipped_tif: Path, edge_cells: int = 3) -> tuple[float, float]:
    with rasterio.open(clipped_tif) as src:
        data = src.read(1, masked=True)
        if data.size == 0:
            return 0.0, 0.0
        valid_mask = np.isfinite(data.filled(np.nan)) & ~np.ma.getmaskarray(data)
        overall = float(valid_mask.sum() / valid_mask.size)
        edge = np.zeros(valid_mask.shape, dtype=bool)
        edge[:edge_cells, :] = True
        edge[-edge_cells:, :] = True
        edge[:, :edge_cells] = True
        edge[:, -edge_cells:] = True
        edge_valid = valid_mask[edge]
        edge_fraction = float(edge_valid.sum() / max(edge_valid.size, 1))
        return overall, edge_fraction


def _fill_nodata_holes(source_tif: Path, destination_tif: Path, max_search_distance: float = 250.0) -> Path:
    destination_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source_tif) as src:
        data = src.read(1, masked=True).filled(np.nan).astype("float32")
        valid_mask = np.isfinite(data)
        if valid_mask.all():
            profile = src.profile.copy()
            profile.update(dtype="float32", nodata=np.nan)
            with rasterio.open(destination_tif, "w", **profile) as dst:
                dst.write(data, 1)
            return destination_tif
        filled = fillnodata(
            data.copy(),
            mask=valid_mask.astype(np.uint8),
            max_search_distance=max_search_distance,
            smoothing_iterations=1,
        ).astype("float32")
        filled[~np.isfinite(filled)] = np.nan
        profile = src.profile.copy()
        profile.update(dtype="float32", nodata=np.nan)
        with rasterio.open(destination_tif, "w", **profile) as dst:
            dst.write(filled, 1)
    return destination_tif


def _bbox_tiles(bbox: BoundingBox, tiles_per_side: int) -> list[BoundingBox]:
    lon_edges = np.linspace(bbox.min_lon, bbox.max_lon, tiles_per_side + 1)
    lat_edges = np.linspace(bbox.min_lat, bbox.max_lat, tiles_per_side + 1)
    tiles: list[BoundingBox] = []
    for row in range(tiles_per_side):
        for col in range(tiles_per_side):
            tiles.append(
                BoundingBox(
                    min_lon=float(lon_edges[col]),
                    min_lat=float(lat_edges[row]),
                    max_lon=float(lon_edges[col + 1]),
                    max_lat=float(lat_edges[row + 1]),
                )
            )
    return tiles


def _query_tnm_candidates_for_bboxes(
    bboxes: list[BoundingBox],
    *,
    max_items: int,
) -> tuple[str, list[dict], list[str]]:
    diagnostics: list[str] = []
    dataset_name = "Digital Elevation Model (DEM) 1 meter"
    deduped: dict[str, dict] = {}
    for bbox_index, query_bbox in enumerate(bboxes):
        try:
            dataset_name, items = query_best_available_dem_products(query_bbox.as_tnm_bbox(), max_items=max_items)
        except Exception as exc:
            diagnostics.append(f"tnm_catalog bbox {bbox_index}: {exc}")
            continue
        for item in items:
            try:
                url = best_download_url(item)
            except Exception:
                continue
            deduped.setdefault(url, item)
    return dataset_name, list(deduped.values()), diagnostics


def _download_tiled_seamless_dem(
    *,
    bbox: BoundingBox,
    destination_tif: Path,
    raw_data_dir: Path,
    tiles_per_side: int,
    dst_crs: CRS,
    overlap_m: float = 30.0,
) -> Path:
    tile_paths: list[Path] = []
    min_x, min_y, max_x, max_y = transform_bounds("EPSG:4326", dst_crs, bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat, densify_pts=21)
    x_edges = np.linspace(min_x, max_x, tiles_per_side + 1)
    y_edges = np.linspace(min_y, max_y, tiles_per_side + 1)
    epsg = int(dst_crs.to_epsg() or 4326)
    for row in range(tiles_per_side):
        for col in range(tiles_per_side):
            tile_min_x = x_edges[col] - (overlap_m if col > 0 else 0.0)
            tile_max_x = x_edges[col + 1] + (overlap_m if col < tiles_per_side - 1 else 0.0)
            tile_min_y = y_edges[row] - (overlap_m if row > 0 else 0.0)
            tile_max_y = y_edges[row + 1] + (overlap_m if row < tiles_per_side - 1 else 0.0)
            tile_width = max(256, int(math.ceil(tile_max_x - tile_min_x)))
            tile_height = max(256, int(math.ceil(tile_max_y - tile_min_y)))
            tile_bbox_text = f"{tile_min_x},{tile_min_y},{tile_max_x},{tile_max_y}"
            tile_path = raw_data_dir / f"{destination_tif.stem}_tile_{tiles_per_side}x{tiles_per_side}_{row:02d}_{col:02d}.tif"
            downloaded = download_seamless_dem(
                tile_bbox_text,
                tile_path,
                (tile_width, tile_height),
                bbox_sr=epsg,
                image_sr=epsg,
            )
            tile_paths.append(downloaded)
    return _build_dem_mosaic(tile_paths, destination_tif)


def _projected_bbox_export_spec(bbox: BoundingBox, dst_crs: CRS) -> tuple[str, int, tuple[int, int]]:
    min_x, min_y, max_x, max_y = transform_bounds("EPSG:4326", dst_crs, bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat, densify_pts=21)
    width = max(256, int(math.ceil(max_x - min_x)))
    height = max(256, int(math.ceil(max_y - min_y)))
    bbox_text = f"{min_x},{min_y},{max_x},{max_y}"
    epsg = int(dst_crs.to_epsg() or 4326)
    return bbox_text, epsg, (width, height)


def _utm_crs_for_site(site: SiteConfig) -> CRS:
    zone = int((site.center_lon + 180.0) // 6.0) + 1
    epsg = 32600 + zone if site.center_lat >= 0.0 else 32700 + zone
    return CRS.from_epsg(epsg)


def _reproject_dem_to_projected(source_tif: Path, destination_tif: Path, dst_crs: CRS, bbox: BoundingBox) -> Path:
    destination_tif.parent.mkdir(parents=True, exist_ok=True)
    unmasked_tif = destination_tif.with_name(f"{destination_tif.stem}_unmasked{destination_tif.suffix}")
    with rasterio.open(source_tif) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        profile = src.profile.copy()
        src_nodata = src.nodata if src.nodata is not None else np.nan
        profile.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "dtype": "float32",
                "nodata": np.nan,
            }
        )

        with rasterio.open(unmasked_tif, "w", **profile) as dst:
            for band_index in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_index),
                    destination=rasterio.band(dst, band_index),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src_nodata,
                    dst_nodata=np.nan,
                )
    clip_dem_to_bbox(
        source_tif=unmasked_tif,
        clipped_tif=destination_tif,
        bbox=(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat),
        bbox_crs="EPSG:4326",
    )

    unmasked_tif.unlink(missing_ok=True)
    return destination_tif


def _clip_projected_raster_to_bbox(source_tif: Path, destination_tif: Path, bbox: BoundingBox) -> Path:
    destination_tif.parent.mkdir(parents=True, exist_ok=True)
    return clip_dem_to_bbox(
        source_tif=source_tif,
        clipped_tif=destination_tif,
        bbox=(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat),
        bbox_crs="EPSG:4326",
    )


def prepare_site_domain(
    *,
    site: SiteConfig,
    working_dir: Path,
    raw_data_dir: Path = DATA_RAW_DIR,
    processed_data_dir: Path = DATA_PROCESSED_DIR,
) -> PreparedSiteDomain:
    bbox = square_bbox_from_center(site.center_lat, site.center_lon, site.side_meters)
    solve_buffer_m = max(0.0, site.solve_buffer_m)
    solve_bbox = buffered_square_bbox_from_center(site.center_lat, site.center_lon, site.side_meters, solve_buffer_m)
    utm_crs = _utm_crs_for_site(site)

    domain_dir = working_dir / "domain"
    domain_dir.mkdir(parents=True, exist_ok=True)
    final_geographic_tif_raw = domain_dir / "site_dem_final_geographic_raw.tif"
    final_geographic_tif = domain_dir / "site_dem_final_geographic.tif"
    solve_geographic_tif_raw = domain_dir / "site_dem_solve_geographic_raw.tif"
    solve_geographic_tif = domain_dir / "site_dem_solve_geographic.tif"
    final_projected_tif = domain_dir / "site_dem.tif"
    solve_projected_tif = domain_dir / "site_dem_solve.tif"
    dem_preview_tif = domain_dir / "site_dem_preview.tif"
    solve_dem_preview_tif = domain_dir / "site_dem_solve_preview.tif"
    coverage_fraction = 0.0
    attempt_errors: list[str] = []

    def _try_candidate(candidate_dataset: str, candidate_source_tif: Path, candidate_downloaded: Path) -> tuple[str, Path, Path, float] | None:
        nonlocal coverage_fraction
        try:
            clip_dem_to_bbox(
                source_tif=candidate_source_tif,
                clipped_tif=final_geographic_tif_raw,
                bbox=(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat),
            )
            overall_final_coverage, final_edge_coverage = _coverage_metrics(final_geographic_tif_raw)
            if final_edge_coverage < 0.995:
                attempt_errors.append(f"{candidate_dataset}: incomplete final-domain edge coverage ({final_edge_coverage:.3f})")
                return None
            if overall_final_coverage < 0.995:
                _fill_nodata_holes(final_geographic_tif_raw, final_geographic_tif)
                coverage_fraction = _coverage_fraction(final_geographic_tif)
            else:
                coverage_fraction = overall_final_coverage
                final_geographic_tif.write_bytes(final_geographic_tif_raw.read_bytes())
            if coverage_fraction < 0.995:
                attempt_errors.append(f"{candidate_dataset}: final-domain coverage remained partial after fill ({coverage_fraction:.3f})")
                return None

            clip_dem_to_bbox(
                source_tif=candidate_source_tif,
                clipped_tif=solve_geographic_tif_raw,
                bbox=(solve_bbox.min_lon, solve_bbox.min_lat, solve_bbox.max_lon, solve_bbox.max_lat),
            )
            overall_solve_coverage, solve_edge_coverage = _coverage_metrics(solve_geographic_tif_raw)
            if solve_edge_coverage < 0.995:
                attempt_errors.append(f"{candidate_dataset}: incomplete solve-domain edge coverage ({solve_edge_coverage:.3f})")
                return None
            if overall_solve_coverage < 0.995:
                _fill_nodata_holes(solve_geographic_tif_raw, solve_geographic_tif)
            else:
                solve_geographic_tif.write_bytes(solve_geographic_tif_raw.read_bytes())
            solve_coverage_fraction = _coverage_fraction(solve_geographic_tif)
            if solve_coverage_fraction < 0.995:
                attempt_errors.append(f"{candidate_dataset}: solve-domain coverage remained partial after fill ({solve_coverage_fraction:.3f})")
                return None
            solve_projected = _reproject_dem_to_projected(solve_geographic_tif, solve_projected_tif, utm_crs, solve_bbox)
            projected_tif = _clip_projected_raster_to_bbox(solve_projected, final_projected_tif, bbox)
            write_dem_preview(projected_tif, dem_preview_tif)
            write_dem_preview(solve_projected, solve_dem_preview_tif)
            return candidate_dataset, candidate_source_tif, candidate_downloaded, coverage_fraction
        except Exception as exc:
            attempt_errors.append(f"{candidate_dataset}: {exc}")
            return None

    def _try_downloaded_item_sources(candidate_dataset: str, downloaded_path: Path) -> tuple[str, Path, Path, float] | None:
        source_paths = extract_tifs_if_zip(downloaded_path, raw_data_dir / downloaded_path.stem)
        for index, source_path in enumerate(source_paths):
            label = f"{candidate_dataset}[{index}]"
            selected_candidate = _try_candidate(label, source_path, downloaded_path)
            if selected_candidate is not None:
                return selected_candidate
        return None

    selected: tuple[str, Path, Path, float] | None = None

    seamless_name = f"{site.slug()}_{int(round(site.side_meters))}m_3dep_seamless.tif"
    seamless_path = raw_data_dir / seamless_name
    seamless_pixels = max(256, min(8192, int(round(site.side_meters + 2.0 * solve_buffer_m))))
    projected_bbox_text, projected_epsg, projected_size = _projected_bbox_export_spec(solve_bbox, utm_crs)
    try:
        seamless_download = download_seamless_dem(
            projected_bbox_text,
            seamless_path,
            projected_size,
            bbox_sr=projected_epsg,
            image_sr=projected_epsg,
        )
        selected = _try_candidate("3dep_seamless_imageserver_projected", seamless_download, seamless_download)
    except Exception as exc:
        attempt_errors.append(f"3dep_seamless_imageserver_projected: {exc}")

    if selected is None:
        for tiles_per_side in (2, 3):
            tiled_name = f"{site.slug()}_{int(round(site.side_meters))}m_3dep_seamless_{tiles_per_side}x{tiles_per_side}.tif"
            tiled_path = raw_data_dir / tiled_name
            try:
                tiled_download = _download_tiled_seamless_dem(
                    bbox=solve_bbox,
                    destination_tif=tiled_path,
                    raw_data_dir=raw_data_dir,
                    tiles_per_side=tiles_per_side,
                    dst_crs=utm_crs,
                )
                selected = _try_candidate(f"3dep_seamless_tiled_{tiles_per_side}x{tiles_per_side}", tiled_download, tiled_download)
                if selected is not None:
                    break
            except Exception as exc:
                attempt_errors.append(f"3dep_seamless_tiled_{tiles_per_side}x{tiles_per_side}: {exc}")

    if selected is None:
        try:
            query_bboxes = [solve_bbox, bbox, *_bbox_tiles(solve_bbox, 2)]
            dataset, items, diagnostics = _query_tnm_candidates_for_bboxes(query_bboxes, max_items=80)
            attempt_errors.extend(diagnostics[-12:])
            if items:
                downloaded_sources: list[Path] = []
                downloaded_paths: list[Path] = []
                for item_index, item in enumerate(items):
                    download_url = best_download_url(item)
                    raw_download_path = raw_data_dir / Path(download_url).name
                    downloaded = download_file(download_url, raw_download_path)
                    downloaded_paths.append(downloaded)
                    selected = _try_downloaded_item_sources(f"{dataset} item {item_index}", downloaded)
                    if selected is not None:
                        break
                    extracted_sources = extract_tifs_if_zip(downloaded, raw_data_dir / downloaded.stem)
                    downloaded_sources.extend(extracted_sources)

                if selected is None and downloaded_sources:
                    intersecting_sources = _intersecting_sources_for_bbox(solve_bbox, downloaded_sources)
                    if intersecting_sources:
                        mosaic_download_path = raw_data_dir / f"{site.slug()}_{int(round(site.side_meters))}m_tnm_mosaic.tif"
                        mosaic_source = _build_dem_mosaic(intersecting_sources, mosaic_download_path)
                        selected = _try_candidate(f"{dataset}_mosaic", mosaic_source, mosaic_download_path)
                    else:
                        attempt_errors.append(f"{dataset}: downloaded {len(downloaded_sources)} source rasters but none intersect the solve bbox")
            else:
                attempt_errors.append("tnm_catalog: no items returned")
        except Exception as exc:
            attempt_errors.append(f"tnm_catalog: {exc}")

    if selected is None:
        local_dem, local_diagnostics = _find_local_dem_covering_bbox_with_diagnostics(solve_bbox, raw_data_dir)
        attempt_errors.extend(local_diagnostics[-12:])
        if local_dem is not None:
            selected = _try_candidate("local_cached_dem", local_dem, local_dem)

    if selected is None:
        details = "; ".join(attempt_errors) if attempt_errors else "no successful DEM source"
        raise RuntimeError(f"Unable to obtain full DEM coverage for the requested bounding box. Attempts: {details}")

    dataset, source_tif, downloaded, coverage_fraction = selected

    manifest_path = domain_dir / "site_domain_manifest.json"
    manifest = {
        "site": {
            "label": site.label,
            "center_lat": site.center_lat,
            "center_lon": site.center_lon,
            "side_meters": site.side_meters,
            "slug": site.slug(),
        },
        "bbox": {
            "min_lon": bbox.min_lon,
            "min_lat": bbox.min_lat,
            "max_lon": bbox.max_lon,
            "max_lat": bbox.max_lat,
        },
        "solve_bbox": {
            "min_lon": solve_bbox.min_lon,
            "min_lat": solve_bbox.min_lat,
            "max_lon": solve_bbox.max_lon,
            "max_lat": solve_bbox.max_lat,
        },
        "solve_buffer_m": solve_buffer_m,
        "dataset": dataset,
        "coverage_fraction": coverage_fraction,
        "source_dem_tif": str(source_tif),
        "raw_download_path": str(downloaded),
        "clipped_dem_tif": str(final_projected_tif),
        "dem_preview_tif": str(dem_preview_tif),
        "solve_dem_tif": str(solve_projected_tif),
        "solve_dem_preview_tif": str(solve_dem_preview_tif),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return PreparedSiteDomain(
        site=site,
        bbox=bbox,
        solve_bbox=solve_bbox,
        dataset=dataset,
        source_dem_tif=source_tif,
        clipped_dem_tif=final_projected_tif,
        dem_preview_tif=dem_preview_tif,
        solve_dem_tif=solve_projected_tif,
        solve_dem_preview_tif=solve_dem_preview_tif,
        raw_download_path=downloaded,
        manifest_path=manifest_path,
    )
