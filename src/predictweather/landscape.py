from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


@dataclass(frozen=True)
class LandscapeBuildOptions:
    fuel_model_land: int = 181
    fuel_model_water: int = 98
    canopy_cover_units: str = "percent"  # percent|fraction
    canopy_height_units: str = "meters"  # meters|meters_x10
    canopy_base_height_units: str = "meters"  # meters|meters_x10
    canopy_bulk_density_units: str = "kg_m3"  # kg_m3|kg_m3_x100
    derived_cbh_fraction_of_ch: float = 0.4
    default_cbd_kg_m3: float = 0.1


@dataclass(frozen=True)
class LandscapeBuildSummary:
    output_tif: str
    width: int
    height: int
    crs: str
    transform: tuple[float, float, float, float, float, float, float, float, float]
    bands: list[str]
    fuel_model_land: int
    fuel_model_water: int
    canopy_pixels: int
    water_pixels: int
    used_inputs: dict[str, str | None]


def _read_band(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        profile = src.profile.copy()
        profile["crs"] = src.crs
        profile["transform"] = src.transform
        return data, profile


def _resample_to_match(
    source_path: Path,
    target_profile: dict,
    *,
    resampling: Resampling,
) -> np.ndarray:
    with rasterio.open(source_path) as src:
        destination = np.full((target_profile["height"], target_profile["width"]), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_profile["transform"],
            dst_crs=target_profile["crs"],
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return destination


def _scale_cover_percent(data: np.ndarray, units: str) -> np.ndarray:
    if units == "fraction":
        scaled = data * 100.0
    elif units == "percent":
        scaled = data
    else:
        raise ValueError(f"Unsupported canopy_cover_units: {units}")
    return np.clip(np.round(scaled), 0, 100).astype(np.int16)


def _scale_height_x10(data: np.ndarray, units: str) -> np.ndarray:
    if units == "meters":
        scaled = data * 10.0
    elif units == "meters_x10":
        scaled = data
    else:
        raise ValueError(f"Unsupported height units: {units}")
    return np.clip(np.round(np.where(np.isfinite(data), scaled, 0.0)), 0, 32767).astype(np.int16)


def _scale_cbd_x100(data: np.ndarray, units: str) -> np.ndarray:
    if units == "kg_m3":
        scaled = data * 100.0
    elif units == "kg_m3_x100":
        scaled = data
    else:
        raise ValueError(f"Unsupported canopy_bulk_density_units: {units}")
    return np.clip(np.round(np.where(np.isfinite(data), scaled, 0.0)), 0, 32767).astype(np.int16)


def _compute_slope_aspect_from_dem(dem_m: np.ndarray, x_res_m: float, y_res_m: float) -> tuple[np.ndarray, np.ndarray]:
    grad_y, grad_x = np.gradient(dem_m.astype(np.float32), y_res_m, x_res_m)
    slope_rad = np.arctan(np.hypot(grad_x, grad_y))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Convert to fire-model style aspect degrees clockwise from north.
    aspect = (450.0 - np.degrees(np.arctan2(grad_y, -grad_x))) % 360.0
    flat_mask = (np.abs(grad_x) < 1.0e-6) & (np.abs(grad_y) < 1.0e-6)
    aspect[flat_mask] = 0.0
    slope_clean = np.where(np.isfinite(slope_deg), slope_deg, 0.0)
    aspect_clean = np.where(np.isfinite(aspect), aspect, 0.0)
    return np.round(slope_clean).astype(np.int16), np.round(aspect_clean).astype(np.int16)


def build_landscape_geotiff(
    *,
    dem_tif: Path,
    output_tif: Path,
    canopy_cover_tif: Path | None = None,
    canopy_height_tif: Path | None = None,
    canopy_base_height_tif: Path | None = None,
    canopy_bulk_density_tif: Path | None = None,
    water_mask_tif: Path | None = None,
    options: LandscapeBuildOptions | None = None,
    summary_path: Path | None = None,
) -> LandscapeBuildSummary:
    options = options or LandscapeBuildOptions()
    dem, dem_profile = _read_band(dem_tif)
    height = int(dem_profile["height"])
    width = int(dem_profile["width"])
    transform = dem_profile["transform"]
    crs = dem_profile["crs"]
    x_res_m = abs(float(transform.a))
    y_res_m = abs(float(transform.e))

    if not np.isfinite(dem).any():
        raise ValueError(f"DEM has no valid data: {dem_tif}")

    elevation = np.where(np.isfinite(dem), np.round(dem), -9999).astype(np.int16)
    slope, aspect = _compute_slope_aspect_from_dem(dem, x_res_m, y_res_m)

    if canopy_cover_tif is not None:
        canopy_cover_raw = _resample_to_match(canopy_cover_tif, dem_profile, resampling=Resampling.bilinear)
        canopy_cover = _scale_cover_percent(canopy_cover_raw, options.canopy_cover_units)
    else:
        canopy_cover = np.zeros((height, width), dtype=np.int16)

    if canopy_height_tif is not None:
        canopy_height_raw = _resample_to_match(canopy_height_tif, dem_profile, resampling=Resampling.bilinear)
        canopy_height = _scale_height_x10(canopy_height_raw, options.canopy_height_units)
    else:
        canopy_height_raw = np.zeros((height, width), dtype=np.float32)
        canopy_height = np.zeros((height, width), dtype=np.int16)

    if canopy_base_height_tif is not None:
        canopy_base_height_raw = _resample_to_match(canopy_base_height_tif, dem_profile, resampling=Resampling.bilinear)
        canopy_base_height = _scale_height_x10(canopy_base_height_raw, options.canopy_base_height_units)
    else:
        derived_cbh_m = np.where(
            canopy_height > 0,
            (canopy_height.astype(np.float32) / 10.0) * options.derived_cbh_fraction_of_ch,
            0.0,
        )
        canopy_base_height = _scale_height_x10(derived_cbh_m, "meters")

    if canopy_bulk_density_tif is not None:
        canopy_bulk_density_raw = _resample_to_match(canopy_bulk_density_tif, dem_profile, resampling=Resampling.bilinear)
        canopy_bulk_density = _scale_cbd_x100(canopy_bulk_density_raw, options.canopy_bulk_density_units)
    else:
        derived_cbd = np.where(
            canopy_height > 0,
            options.default_cbd_kg_m3 * np.clip(canopy_cover.astype(np.float32) / 40.0, 0.25, 1.5),
            0.0,
        )
        canopy_bulk_density = _scale_cbd_x100(derived_cbd, "kg_m3")

    if water_mask_tif is not None:
        water_mask_raw = _resample_to_match(water_mask_tif, dem_profile, resampling=Resampling.nearest)
        water_mask = np.isfinite(water_mask_raw) & (water_mask_raw > 0.5)
    else:
        water_mask = (canopy_cover == 0) & (canopy_height == 0)

    fuel_model = np.full((height, width), options.fuel_model_land, dtype=np.int16)
    fuel_model[water_mask] = options.fuel_model_water

    valid_dem = np.isfinite(dem)
    for band in (slope, aspect, canopy_cover, canopy_height, canopy_base_height, canopy_bulk_density, fuel_model):
        band[~valid_dem] = 0
    elevation[~valid_dem] = -9999

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    profile = dem_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="int16",
        count=8,
        nodata=-9999,
        compress="deflate",
        interleave="pixel",
    )
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(elevation, 1)
        dst.write(slope, 2)
        dst.write(aspect, 3)
        dst.write(fuel_model, 4)
        dst.write(canopy_cover, 5)
        dst.write(canopy_height, 6)
        dst.write(canopy_base_height, 7)
        dst.write(canopy_bulk_density, 8)
        dst.set_band_description(1, "elevation_m")
        dst.set_band_description(2, "slope_deg")
        dst.set_band_description(3, "aspect_deg")
        dst.set_band_description(4, "fuel_model")
        dst.set_band_description(5, "canopy_cover_pct")
        dst.set_band_description(6, "canopy_height_m_x10")
        dst.set_band_description(7, "canopy_base_height_m_x10")
        dst.set_band_description(8, "canopy_bulk_density_kg_m3_x100")

    summary = LandscapeBuildSummary(
        output_tif=str(output_tif),
        width=width,
        height=height,
        crs=str(crs),
        transform=tuple(float(value) for value in transform),
        bands=[
            "elevation_m",
            "slope_deg",
            "aspect_deg",
            "fuel_model",
            "canopy_cover_pct",
            "canopy_height_m_x10",
            "canopy_base_height_m_x10",
            "canopy_bulk_density_kg_m3_x100",
        ],
        fuel_model_land=options.fuel_model_land,
        fuel_model_water=options.fuel_model_water,
        canopy_pixels=int(np.count_nonzero(canopy_height > 0)),
        water_pixels=int(np.count_nonzero(water_mask)),
        used_inputs={
            "dem_tif": str(dem_tif),
            "canopy_cover_tif": str(canopy_cover_tif) if canopy_cover_tif else None,
            "canopy_height_tif": str(canopy_height_tif) if canopy_height_tif else None,
            "canopy_base_height_tif": str(canopy_base_height_tif) if canopy_base_height_tif else None,
            "canopy_bulk_density_tif": str(canopy_bulk_density_tif) if canopy_bulk_density_tif else None,
            "water_mask_tif": str(water_mask_tif) if water_mask_tif else None,
        },
    )
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary
