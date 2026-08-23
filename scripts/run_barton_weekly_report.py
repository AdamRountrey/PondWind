from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if not getattr(sys, "frozen", False) and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictweather.runtime import configure_geospatial_runtime

configure_geospatial_runtime()

from predictweather.boundary import build_model_point_forecast, load_cached_hrdps_manifest_for_valid_time, sample_boundary_wind_at_site, sample_scalar_field_at_site
from predictweather.config import DATA_RAW_DIR, OUTPUTS_DIR, RESOURCE_ROOT as RUNTIME_RESOURCE_ROOT, SiteConfig
from predictweather.ecmwf import download_ecmwf_for_valid_time, sample_ecmwf_point_forecast
from predictweather.forecast_models import choose_consensus_boundary, model_table_rows
from predictweather.gefs import (
    download_gefs_mean_spread_and_members,
    floor_to_3h,
    load_cached_gefs_manifest_for_valid_time,
    sample_gefs_mean_and_spread_at_site,
    sample_gefs_members_at_site,
)
from predictweather.geo import buffered_square_bbox_from_center
from predictweather.gfs import download_gfs_for_valid_time, sample_gfs_point_forecast
from predictweather.hrdps import GUST_VARIABLE_CANDIDATES, download_hrdps_for_valid_time, try_download_hrdps_variable
from predictweather.hrrr import download_hrrr_for_valid_time, sample_hrrr_point_forecast
from predictweather.icon import download_icon_for_valid_time, sample_icon_point_forecast
from predictweather.landfire import ExportSpec, export_image
from predictweather.landscape import LandscapeBuildOptions, build_landscape_geotiff
from predictweather.nam import download_nam_for_valid_time, sample_nam_point_forecast
from predictweather.nws import sample_nws_hourly_forecast
from predictweather.observations import nearest_station_candidates, recent_same_local_hour_observations
from predictweather.openfoam import OpenFoamRunError, _default_max_horizontal_cells, compare_wind_outputs, run_openfoam_domain_average
from predictweather.satellite import (
    PLANETARY_COMPUTER_STAC,
    count_ecostress_water_pixels,
    count_landsat_water_pixels,
    count_sentinel_water_pixels,
    derive_ecostress_sst,
    derive_landsat_sst,
    derive_sentinel_chla,
    derive_sentinel_turbidity,
    download_cmr_assets,
    download_selection_assets,
    list_ecostress_lste_candidates,
    list_candidate_items,
    render_rgb_preview,
    sentinel_clear_fraction,
    select_best_item,
)
from predictweather.site import PreparedSiteDomain, prepare_site_domain
from predictweather.wind import _draw_text, _write_png
from predictweather.windninja import (
    _read_aaigrid,
    expected_windninja_ascii_paths,
    run_windninja_domain_average,
    windninja_cli_path,
    write_array_to_geotiff_from_header,
    write_reprojected_array_like_reference,
    write_scalar_diagnostic_preview,
    write_windninja_knots_vector_preview_from_arrays,
    write_windninja_knots_vector_preview_from_speed_angle,
    diverging_blue_green_red_colormap,
)
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject


LOCAL_TZ = ZoneInfo("America/New_York")
DEFAULT_SKILL_SAMPLE_COUNT = 5
MAX_WATER_SCENE_ATTEMPTS = 8
MAX_RGB_SCENE_ATTEMPTS = 8
MAX_RGB_SCENE_CLOUD_COVER = 50.0
MAX_WATER_SCENE_CLOUD_COVER = 70.0
MAX_LANDSAT_SCENE_CLOUD_COVER = 80.0
LANDSAT_SST_MAX_AGE_DAYS = 10.0
MAX_PARALLEL_DOWNLOADS = 4
MAX_PARALLEL_MODEL_DOWNLOADS = 4
MAX_PARALLEL_WINDNINJA_MEMBER_SOLVES = 2
CALM_WIND_SOLVER_THRESHOLD_MPS = 0.2
VECTOR_REFERENCE_MESH_RESOLUTION_M = 30.0
VECTOR_REFERENCE_STRIDE = 4
VECTOR_REFERENCE_SCALE = 2.2
OPENFOAM_HIGH_RESOLUTION_WARNING_BASELINE_M = 30.0


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return max(minimum, int(default))


def _progress(progress_callback: Callable[[int, str], None] | None, percent: int, message: str) -> None:
    if progress_callback is not None:
        progress_callback(percent, message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a weekly race-day report for a user-selected 1 square mile site.")
    parser.add_argument(
        "--race-local-datetime",
        default=None,
        help="Race time in America/New_York, e.g. 2026-05-10T14:00:00. Defaults to next Sunday at 2pm local.",
    )
    parser.add_argument("--center-lat", type=float, default=SiteConfig.center_lat, help="Center latitude of the target area.")
    parser.add_argument("--center-lon", type=float, default=SiteConfig.center_lon, help="Center longitude of the target area.")
    parser.add_argument("--side-meters", type=float, default=SiteConfig.side_meters, help="Square side length in meters. Default is 1 square mile.")
    parser.add_argument("--site-label", default=SiteConfig.label, help="Short label used in report titles and output folder names.")
    parser.add_argument("--mesh-resolution", type=float, default=30.0)
    parser.add_argument(
        "--wind-solver",
        choices=("windninja", "openfoam"),
        default="windninja",
        help="WindNinja builds production products. Use openfoam to add an experimental CFD comparison product.",
    )
    parser.add_argument("--solve-buffer-m", type=float, default=SiteConfig.solve_buffer_m, help="Extra buffer around the final area for WindNinja solves.")
    parser.add_argument("--report-output-dir", default=None, help="Optional directory where the report folder should be created. Defaults to outputs/reports.")
    parser.add_argument("--allow-insecure-ssl", action="store_true")
    parser.add_argument("--force-ecostress-sst", action="store_true", help="Force ECOSTRESS surface-temperature discovery even when Landsat is recent enough.")
    parser.add_argument("--no-satellite-rgb", dest="satellite_rgb", action="store_false", help="Skip the latest RGB satellite product.")
    parser.add_argument("--no-satellite-sst", dest="satellite_sst", action="store_false", help="Skip the satellite surface-temperature-over-water product.")
    parser.add_argument("--no-satellite-chla", dest="satellite_chla", action="store_false", help="Skip the chlorophyll-a satellite product.")
    parser.add_argument("--no-satellite-turbidity", dest="satellite_turbidity", action="store_false", help="Skip the turbidity satellite product.")
    return parser.parse_args()


def _satellite_product_options(
    *,
    rgb: bool = True,
    sst: bool = True,
    chla: bool = True,
    turbidity: bool = True,
) -> dict[str, bool]:
    return {
        "rgb": bool(rgb),
        "sst": bool(sst),
        "chla": bool(chla),
        "turbidity": bool(turbidity),
    }


def _skipped_satellite_summary(product: str) -> dict:
    return {
        "product": product,
        "status": "skipped",
        "reason": "Product was not selected for this report.",
        "output_png": None,
    }


def _default_race_local_datetime(now_local: datetime) -> datetime:
    days_ahead = (6 - now_local.weekday()) % 7
    candidate = (now_local + timedelta(days=days_ahead)).replace(hour=14, minute=0, second=0, microsecond=0)
    if candidate <= now_local:
        candidate += timedelta(days=7)
    return candidate


def _parse_race_local_datetime(value: str | None) -> datetime:
    if value:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=LOCAL_TZ)
        return parsed.astimezone(LOCAL_TZ)
    return _default_race_local_datetime(datetime.now(LOCAL_TZ))


def _uv_to_speed_dir(u_mps: float, v_mps: float) -> tuple[float, float]:
    speed_mps = float(math.hypot(u_mps, v_mps))
    direction_from_deg = float((270.0 - math.degrees(math.atan2(v_mps, u_mps))) % 360.0)
    return speed_mps, direction_from_deg


def _sigma_scenarios() -> list[tuple[float, float, str]]:
    return [
        (0.0, 0.0, "mean"),
        (1.0, 0.0, "u_plus"),
        (-1.0, 0.0, "u_minus"),
        (0.0, 1.0, "v_plus"),
        (0.0, -1.0, "v_minus"),
    ]


def _gefs_member_download_limit() -> int:
    return _env_int("PONDWIND_GEFS_MEMBER_DOWNLOAD_LIMIT", 31, minimum=0)


def _gefs_member_solve_limit() -> int:
    return _env_int("PONDWIND_GEFS_MEMBER_SOLVE_LIMIT", 9, minimum=0)


def _windninja_member_workers() -> int:
    return _env_int("PONDWIND_WINDNINJA_MEMBER_WORKERS", MAX_PARALLEL_WINDNINJA_MEMBER_SOLVES, minimum=1)


def _model_download_workers() -> int:
    return _env_int("PONDWIND_MODEL_DOWNLOAD_WORKERS", MAX_PARALLEL_MODEL_DOWNLOADS, minimum=1)


def _vector_overlay_style(mesh_resolution_m: float) -> tuple[int, float]:
    requested_resolution = max(float(mesh_resolution_m), 1.0)
    scale_ratio = max(1.0, VECTOR_REFERENCE_MESH_RESOLUTION_M / requested_resolution)
    stride = max(1, int(round(VECTOR_REFERENCE_STRIDE * scale_ratio)))
    scale = VECTOR_REFERENCE_SCALE * scale_ratio
    return stride, scale


def _openfoam_vertical_cells() -> int:
    return _env_int("PONDWIND_OPENFOAM_VERTICAL_CELLS", 20, minimum=1)


def _estimate_openfoam_solve_cells_from_extent(
    width_m: float,
    height_m: float,
    mesh_resolution_m: float,
    vertical_cells: int,
    max_horizontal_cells: int,
) -> dict[str, int]:
    nx = max(2, int(round(float(width_m) / max(float(mesh_resolution_m), 1.0))))
    ny = max(2, int(round(float(height_m) / max(float(mesh_resolution_m), 1.0))))
    while nx * ny > max_horizontal_cells:
        nx = max(2, int(math.floor(nx * 0.9)))
        ny = max(2, int(math.floor(ny * 0.9)))
    horizontal_cells = int(nx * ny)
    return {
        "nx": int(nx),
        "ny": int(ny),
        "horizontal_cells": horizontal_cells,
        "vertical_cells": int(vertical_cells),
        "solve_cells": int(horizontal_cells * int(vertical_cells)),
    }


def _estimate_openfoam_solve_cells(elevation_tif: Path, mesh_resolution_m: float, vertical_cells: int) -> dict[str, int]:
    with rasterio.open(elevation_tif) as src:
        width_m = float(src.bounds.right - src.bounds.left)
        height_m = float(src.bounds.top - src.bounds.bottom)
    return _estimate_openfoam_solve_cells_from_extent(
        width_m=width_m,
        height_m=height_m,
        mesh_resolution_m=mesh_resolution_m,
        vertical_cells=vertical_cells,
        max_horizontal_cells=_default_max_horizontal_cells(mesh_resolution_m),
    )


def _select_representative_gefs_members(members: list[dict], max_members: int) -> list[dict]:
    finite_members = [
        member
        for member in members
        if math.isfinite(float(member.get("u10_mps", float("nan"))))
        and math.isfinite(float(member.get("v10_mps", float("nan"))))
        and math.isfinite(float(member.get("speed_mps", float("nan"))))
    ]
    if max_members <= 0 or not finite_members:
        return []
    if len(finite_members) <= max_members:
        total = max(1, len(finite_members))
        return [
            dict(
                member,
                selection_rank=index,
                selection_reason="all_available",
                member_weight=1.0,
                cluster_member_count=1,
                cluster_fraction=1.0 / total,
                represented_member_ids=[str(member.get("member_id", index))],
            )
            for index, member in enumerate(finite_members)
        ]

    vectors = np.array([[float(member["u10_mps"]), float(member["v10_mps"])] for member in finite_members], dtype=np.float64)
    center = np.nanmean(vectors, axis=0)
    scale = np.nanstd(vectors, axis=0)
    scale = np.where(scale < 0.25, 0.25, scale)
    normalized = (vectors - center) / scale

    selected: list[int] = []
    def add(index: int) -> None:
        if index not in selected and len(selected) < max_members:
            selected.append(index)

    add(int(np.argmin(np.linalg.norm(normalized, axis=1))))

    while len(selected) < max_members:
        remaining = [index for index in range(len(finite_members)) if index not in selected]
        if not remaining:
            break
        selected_points = normalized[selected]
        best_index = max(
            remaining,
            key=lambda index: float(np.min(np.linalg.norm(selected_points - normalized[index], axis=1))),
        )
        add(best_index)

    selected_array = np.array(selected, dtype=np.int64)
    assignments = np.zeros(len(finite_members), dtype=np.int64)
    for _ in range(6):
        selected_points = normalized[selected_array]
        distances = np.linalg.norm(normalized[:, None, :] - selected_points[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)
        next_selected = selected_array.copy()
        for cluster_index in range(len(selected_array)):
            members_in_cluster = np.where(assignments == cluster_index)[0]
            if members_in_cluster.size == 0:
                continue
            cluster_points = normalized[members_in_cluster]
            within_cluster_distance = np.sum(
                np.linalg.norm(cluster_points[:, None, :] - cluster_points[None, :, :], axis=2),
                axis=1,
            )
            next_selected[cluster_index] = int(members_in_cluster[int(np.argmin(within_cluster_distance))])
        if np.array_equal(next_selected, selected_array):
            break
        selected_array = next_selected

    selected_points = normalized[selected_array]
    distances = np.linalg.norm(normalized[:, None, :] - selected_points[None, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)

    selected_members: list[dict] = []
    total_members = len(finite_members)
    for rank, index in enumerate(selected_array.tolist()):
        members_in_cluster = np.where(assignments == rank)[0]
        represented_ids = [str(finite_members[int(member_index)].get("member_id", member_index)) for member_index in members_in_cluster]
        selected_members.append(
            {
                **finite_members[index],
                "selection_rank": rank,
                "selection_reason": "weighted_uv_cluster_medoid",
                "member_weight": float(len(members_in_cluster)),
                "cluster_member_count": int(len(members_in_cluster)),
                "cluster_fraction": float(len(members_in_cluster) / total_members),
                "represented_member_ids": represented_ids,
            }
        )
    return selected_members


def _weighted_nanstd(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = stack.astype(np.float64)
    finite = np.isfinite(values)
    safe_values = np.where(finite, values, 0.0)
    safe_weights = np.where(finite, weights[:, None, None], 0.0)
    weight_sum = np.sum(safe_weights, axis=0)
    mean = np.divide(
        np.sum(safe_values * safe_weights, axis=0),
        weight_sum,
        out=np.full(values.shape[1:], np.nan, dtype=np.float64),
        where=weight_sum > 0.0,
    )
    variance = np.divide(
        np.sum(((safe_values - mean[None, :, :]) ** 2) * safe_weights, axis=0),
        weight_sum,
        out=np.full(values.shape[1:], np.nan, dtype=np.float64),
        where=weight_sum > 0.0,
    )
    return np.sqrt(np.maximum(variance, 0.0))


def _weighted_circular_std_deg(direction_stack_deg: np.ndarray, weights: np.ndarray) -> np.ndarray:
    direction_rad = np.deg2rad(direction_stack_deg.astype(np.float64))
    finite = np.isfinite(direction_rad)
    safe_weights = np.where(finite, weights[:, None, None], 0.0)
    weight_sum = np.sum(safe_weights, axis=0)
    sin_mean = np.divide(
        np.sum(np.where(finite, np.sin(direction_rad), 0.0) * safe_weights, axis=0),
        weight_sum,
        out=np.full(direction_rad.shape[1:], np.nan, dtype=np.float64),
        where=weight_sum > 0.0,
    )
    cos_mean = np.divide(
        np.sum(np.where(finite, np.cos(direction_rad), 0.0) * safe_weights, axis=0),
        weight_sum,
        out=np.full(direction_rad.shape[1:], np.nan, dtype=np.float64),
        where=weight_sum > 0.0,
    )
    resultant_length = np.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
    resultant_length = np.clip(resultant_length, 1.0e-6, 1.0)
    return np.rad2deg(np.sqrt(-2.0 * np.log(resultant_length)))


def _aligned_boundary_valid_time(race_time_utc: datetime) -> datetime:
    return floor_to_3h(race_time_utc)


def _format_optional_number(value: float | None, suffix: str) -> str:
    if value is None:
        return "n/a"
    return f"{int(round(value))}{suffix}"


def _load_fresh_or_cached_hrdps_boundary(
    site: SiteConfig,
    target_valid_time_utc: datetime,
    raw_data_dir: Path = DATA_RAW_DIR,
) -> tuple[dict, str]:
    fresh_error: Exception | None = None
    try:
        hrdps_selection = download_hrdps_for_valid_time(raw_data_dir, target_valid_time_utc)
        boundary = sample_boundary_wind_at_site(
            ugrd_path=Path(hrdps_selection.files["UGRD"]),
            vgrd_path=Path(hrdps_selection.files["VGRD"]),
            lat=site.center_lat,
            lon=site.center_lon,
        )
        return boundary, "fresh_hrdps"
    except Exception as exc:
        fresh_error = exc

    try:
        cached_manifest, cached_manifest_path = load_cached_hrdps_manifest_for_valid_time(raw_data_dir / "hrdps", target_valid_time_utc)
        boundary = sample_boundary_wind_at_site(
            ugrd_path=Path(cached_manifest["files"]["UGRD"]),
            vgrd_path=Path(cached_manifest["files"]["VGRD"]),
            lat=site.center_lat,
            lon=site.center_lon,
        )
        boundary["manifest_path"] = str(cached_manifest_path)
        boundary["requested_valid_time_utc"] = target_valid_time_utc.isoformat()
        return boundary, "cached_hrdps"
    except Exception as cached_exc:
        message = f"Unable to acquire HRDPS boundary for {target_valid_time_utc.isoformat()}: fresh={fresh_error!r}; cached={cached_exc!r}"
        raise RuntimeError(message) from cached_exc


def _load_fresh_or_cached_gefs_bundle(
    site: SiteConfig,
    target_valid_time_utc: datetime,
    raw_data_dir: Path = DATA_RAW_DIR,
) -> tuple[Path, Path, dict[str, Path], str, dict]:
    fresh_error: Exception | None = None
    try:
        gefs_selection = download_gefs_mean_spread_and_members(
            raw_data_dir,
            target_valid_time_utc,
            lat=site.center_lat,
            lon=site.center_lon,
            max_members=_gefs_member_download_limit(),
        )
        member_paths = {
            member_id: Path(path)
            for member_id, path in (gefs_selection.member_files or {}).items()
            if Path(path).exists()
        }
        mode = "fresh_gefs_members" if member_paths else "fresh_gefs_spread_only"
        return Path(gefs_selection.mean_files["geavg"]), Path(gefs_selection.spread_files["gespr"]), member_paths, mode, gefs_selection.__dict__
    except Exception as exc:
        fresh_error = exc

    try:
        cached_manifest, cached_manifest_path = load_cached_gefs_manifest_for_valid_time(raw_data_dir / "gefs", target_valid_time_utc)
        member_paths = {
            member_id: Path(path)
            for member_id, path in cached_manifest.get("member_files", {}).items()
            if Path(path).exists()
        }
        mode = "cached_gefs_members" if member_paths else "cached_gefs_spread_only"
        cached_manifest = dict(cached_manifest)
        cached_manifest["manifest_path"] = str(cached_manifest_path)
        return Path(cached_manifest["mean_files"]["geavg"]), Path(cached_manifest["spread_files"]["gespr"]), member_paths, mode, cached_manifest
    except Exception as cached_exc:
        message = f"Unable to acquire GEFS boundary for {target_valid_time_utc.isoformat()}: fresh={fresh_error!r}; cached={cached_exc!r}"
        raise RuntimeError(message) from cached_exc


def _load_hrdps_point_forecast(
    lat: float,
    lon: float,
    target_valid_time_utc: datetime,
    raw_data_dir: Path = DATA_RAW_DIR,
) -> dict:
    site = SiteConfig(center_lat=lat, center_lon=lon)
    boundary, acquisition_mode = _load_fresh_or_cached_hrdps_boundary(site, target_valid_time_utc, raw_data_dir)
    gust_summary = None
    run_at_utc = target_valid_time_utc.isoformat()
    forecast_hour = 0
    files = {
        "UGRD": boundary["ugrd_path"],
        "VGRD": boundary["vgrd_path"],
    }

    try:
        if acquisition_mode == "cached_hrdps" and boundary.get("manifest_path"):
            cached_manifest = json.loads(Path(boundary["manifest_path"]).read_text(encoding="utf-8"))
            run_at_utc = cached_manifest["run_at_utc"]
            forecast_hour = int(cached_manifest["forecast_hour"])
            files = dict(cached_manifest["files"])
        else:
            hrdps_selection = download_hrdps_for_valid_time(raw_data_dir, target_valid_time_utc)
            run_at_utc = hrdps_selection.run_at_utc
            forecast_hour = hrdps_selection.forecast_hour
            files = dict(hrdps_selection.files)
        gust_result = try_download_hrdps_variable(
            raw_data_dir,
            run_at_utc=datetime.fromisoformat(run_at_utc.replace("Z", "+00:00")),
            forecast_hour=forecast_hour,
            variable_names=GUST_VARIABLE_CANDIDATES,
        )
        if gust_result is not None:
            _, gust_path = gust_result
            gust_summary = sample_scalar_field_at_site(gust_path, lat, lon)
            files["GUST"] = str(gust_path)
    except Exception:
        pass

    forecast = build_model_point_forecast(
        source="hrdps",
        display_name="hrdps",
        run_at_utc=run_at_utc,
        forecast_hour=forecast_hour,
        acquisition_mode=acquisition_mode,
        wind_summary=boundary,
        gust_summary=gust_summary,
        files=files,
    )
    return forecast.as_dict()


def _run_ordered_tasks(tasks: list[tuple[str, Callable[[], object]]], *, max_workers: int = MAX_PARALLEL_DOWNLOADS) -> list[tuple[str, object | None, Exception | None]]:
    if not tasks:
        return []
    results: list[tuple[str, object | None, Exception | None] | None] = [None] * len(tasks)
    worker_count = max(1, min(max_workers, len(tasks)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(loader): (index, name)
            for index, (name, loader) in enumerate(tasks)
        }
        for future in as_completed(futures):
            index, name = futures[future]
            try:
                results[index] = (name, future.result(), None)
            except Exception as exc:
                results[index] = (name, None, exc)
    return [result for result in results if result is not None]


def _load_model_point_forecasts(
    lat: float,
    lon: float,
    target_valid_time_utc: datetime,
    raw_data_dir: Path = DATA_RAW_DIR,
) -> tuple[list[dict], dict[str, str]]:
    errors: dict[str, str] = {}
    forecasts: list[dict] = []

    loaders: list[tuple[str, Callable[[], object]]] = [
        ("hrdps", lambda: _load_hrdps_point_forecast(lat, lon, target_valid_time_utc, raw_data_dir)),
        ("hrrr", lambda: sample_hrrr_point_forecast(download_hrrr_for_valid_time(raw_data_dir, target_valid_time_utc, lat, lon), lat, lon, "fresh_hrrr")),
        ("gfs", lambda: sample_gfs_point_forecast(download_gfs_for_valid_time(raw_data_dir, target_valid_time_utc, lat, lon), lat, lon, "fresh_gfs")),
        ("nam", lambda: sample_nam_point_forecast(download_nam_for_valid_time(raw_data_dir, target_valid_time_utc, lat, lon), lat, lon, "fresh_nam")),
        ("icon", lambda: sample_icon_point_forecast(download_icon_for_valid_time(raw_data_dir, target_valid_time_utc), lat, lon, "fresh_icon")),
        ("ecmwf", lambda: sample_ecmwf_point_forecast(download_ecmwf_for_valid_time(raw_data_dir, target_valid_time_utc), lat, lon, "fresh_ecmwf")),
    ]
    for loader_name, result, exc in _run_ordered_tasks(loaders, max_workers=_model_download_workers()):
        if exc is not None:
            errors[loader_name] = repr(exc)
        elif result is not None:
            forecasts.append(result)

    return forecasts, errors


def _compute_station_skill_adjustments(
    *,
    target_valid_time_utc: datetime,
    site_lat: float,
    site_lon: float,
    raw_data_dir: Path = DATA_RAW_DIR,
    sample_count: int = DEFAULT_SKILL_SAMPLE_COUNT,
) -> tuple[dict[str, float], dict]:
    station_candidates = nearest_station_candidates(raw_data_dir, site_lat, site_lon)
    selected_station = None
    selected_station_distance_km = None
    observations = []
    candidate_records: list[dict] = []
    selection_errors: dict[str, str] = {}

    for candidate in station_candidates:
        station = candidate["station"]
        distance_km = float(candidate["distance_km"])
        try:
            candidate_observations = recent_same_local_hour_observations(
                raw_data_dir,
                station.station_id,
                target_time_utc=target_valid_time_utc,
                timezone_name="America/New_York",
                sample_count=sample_count,
                max_lookback_days=15,
            )
        except Exception as exc:
            selection_errors[station.station_id] = repr(exc)
            candidate_records.append(
                {
                    "station_id": station.station_id,
                    "distance_km": distance_km,
                    "observations_found": 0,
                    "error": repr(exc),
                }
            )
            continue

        candidate_records.append(
            {
                "station_id": station.station_id,
                "distance_km": distance_km,
                "observations_found": len(candidate_observations),
            }
        )
        if candidate_observations:
            selected_station = station
            selected_station_distance_km = distance_km
            observations = candidate_observations
            break

    if not observations:
        return {}, {
            "station_id": None,
            "station_distance_km": None,
            "site_lat": site_lat,
            "site_lon": site_lon,
            "observations_used": 0,
            "samples": [],
            "per_model": {},
            "candidate_stations": candidate_records,
            "selection_errors": selection_errors,
        }

    per_model_errors: dict[str, dict[str, list[float]]] = {}
    sample_records: list[dict] = []
    for observation in observations:
        valid_time_utc = datetime.fromisoformat(observation.report_time_utc.replace("Z", "+00:00"))
        model_forecasts, model_errors = _load_model_point_forecasts(
            observation.lat,
            observation.lon,
            valid_time_utc,
            raw_data_dir,
        )
        sample_record = {
            "observation": observation.as_dict(),
            "model_errors": model_errors,
            "models": [],
        }
        for forecast in model_forecasts:
            wind_kts = float(forecast["wind_speed_mps"]) * 1.94384449
            gust_kts = None if forecast["gust_mps"] is None else float(forecast["gust_mps"]) * 1.94384449
            speed_abs_error = abs(wind_kts - float(observation.wind_speed_kts))
            gust_abs_error = None
            if observation.gust_kts is not None and gust_kts is not None:
                gust_abs_error = abs(gust_kts - float(observation.gust_kts))
            per_model_errors.setdefault(forecast["source"], {"speed": [], "gust": []})
            per_model_errors[forecast["source"]]["speed"].append(speed_abs_error)
            if gust_abs_error is not None:
                per_model_errors[forecast["source"]]["gust"].append(gust_abs_error)
            sample_record["models"].append(
                {
                    "source": forecast["source"],
                    "wind_kts": wind_kts,
                    "gust_kts": gust_kts,
                    "speed_abs_error_kts": speed_abs_error,
                    "gust_abs_error_kts": gust_abs_error,
                }
            )
        sample_records.append(sample_record)

    speed_mae_by_model = {
        source: float(np.mean(metrics["speed"]))
        for source, metrics in per_model_errors.items()
        if metrics["speed"]
    }
    if not speed_mae_by_model:
        return {}, {
            "station_id": None if selected_station is None else selected_station.station_id,
            "station_distance_km": selected_station_distance_km,
            "site_lat": site_lat,
            "site_lon": site_lon,
            "observations_used": len(observations),
            "samples": sample_records,
            "per_model": {},
            "candidate_stations": candidate_records,
            "selection_errors": selection_errors,
        }

    speed_mae_values = sorted(speed_mae_by_model.values())
    middle = len(speed_mae_values) // 2
    if len(speed_mae_values) % 2 == 1:
        median_speed_mae = speed_mae_values[middle]
    else:
        median_speed_mae = 0.5 * (speed_mae_values[middle - 1] + speed_mae_values[middle])

    skill_adjustments: dict[str, float] = {}
    per_model_summary: dict[str, dict] = {}
    for source, metrics in per_model_errors.items():
        if not metrics["speed"]:
            continue
        speed_mae = float(np.mean(metrics["speed"]))
        gust_mae = float(np.mean(metrics["gust"])) if metrics["gust"] else None
        relative_speed_skill = median_speed_mae / max(speed_mae, 0.25)
        blended_skill = relative_speed_skill
        if gust_mae is not None:
            gust_reference = max(gust_mae, 1.0)
            gust_skill = median_speed_mae / gust_reference
            blended_skill = math.sqrt(max(relative_speed_skill, 0.1) * max(gust_skill, 0.1))
        raw_skill_factor = min(1.15, max(0.85, blended_skill ** 0.35))
        coverage = min(1.0, len(metrics["speed"]) / 3.0)
        skill_factor = 1.0 + (raw_skill_factor - 1.0) * coverage
        skill_adjustments[source] = float(skill_factor)
        per_model_summary[source] = {
            "samples": len(metrics["speed"]),
            "speed_mae_kts": speed_mae,
            "gust_mae_kts": gust_mae,
            "coverage_factor": coverage,
            "raw_skill_factor": raw_skill_factor,
            "skill_factor": skill_factor,
        }

    return skill_adjustments, {
        "station_id": None if selected_station is None else selected_station.station_id,
        "station_distance_km": selected_station_distance_km,
        "station_name": None if selected_station is None else selected_station.name,
        "site_lat": site_lat,
        "site_lon": site_lon,
        "observations_used": len(observations),
        "median_speed_mae_kts": median_speed_mae,
        "samples": sample_records,
        "per_model": per_model_summary,
        "candidate_stations": candidate_records,
        "selection_errors": selection_errors,
    }


def _app_path(path: str | Path) -> str:
    text = str(Path(path).resolve()).replace("\\", "/")
    if not text.startswith("/"):
        text = "/" + text
    return text


def _write_unavailable_panel(
    output_png: Path,
    *,
    title: str,
    line1: str,
    line2: str | None = None,
    width: int = 1854,
    height: int = 1659,
) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    border = (70, 70, 70)
    text = (25, 25, 25)
    accent = (140, 140, 140)
    canvas[24 : 24 + 4, 24 : width - 24] = border
    canvas[height - 28 : height - 24, 24 : width - 24] = border
    canvas[24 : height - 24, 24 : 28] = border
    canvas[24 : height - 24, width - 28 : width - 24] = border
    canvas[24:180, 24 : width - 24] = (232, 232, 232)
    canvas[24 : 24 + 4, 24 : width - 24] = border
    canvas[176:180, 24 : width - 24] = border
    canvas[24:180, 24:28] = border
    canvas[24:180, width - 28 : width - 24] = border
    _draw_text(canvas, 60, 68, title, text, scale=5)
    _draw_text(canvas, 60, 280, line1, text, scale=4)
    if line2:
        _draw_text(canvas, 60, 360, line2, accent, scale=3)
    return _write_png(output_png, canvas)


def _footer_timestamp_text(timestamp: datetime, prefix: str) -> str:
    local_time = timestamp.astimezone(LOCAL_TZ)
    return f"{prefix} {local_time.strftime('%Y-%m-%d %H:%M %Z').lower()}"


def _report_dir_name(race_time_local: datetime, site: SiteConfig) -> str:
    return f"{race_time_local.strftime('%Y%m%d_%H%M')}_{site.slug()}"


def _clean_report_outputs(report_dir: Path) -> list[str]:
    removed: list[str] = []
    for pattern in ("product_*.png", "satellite_*.png", "weekly_report.md", "report_manifest.json"):
        for path in sorted(report_dir.glob(pattern)):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    return removed


def _resolve_report_root(report_output_dir: str | Path | None) -> Path:
    if report_output_dir is None:
        if getattr(sys, "frozen", False):
            documents = Path.home() / "Documents"
            visible_root = documents if documents.is_dir() else Path.home()
            return visible_root / "PondWind Reports"
        return OUTPUTS_DIR / "reports"
    return Path(report_output_dir).expanduser().resolve()


def _tidy_report_root(report_dir: Path, temp_dir: Path) -> list[str]:
    moved: list[str] = []
    temp_dir.mkdir(parents=True, exist_ok=True)

    for name in ("wind", "satellite"):
        source = report_dir / name
        if source.exists() and source != temp_dir / name:
            target = temp_dir / name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(source), str(target))
            moved.append(f"{source} -> {target}")

    for tif_path in sorted(report_dir.glob("*.tif")):
        target = temp_dir / tif_path.name
        if target.exists():
            target.unlink()
        shutil.move(str(tif_path), str(target))
        moved.append(f"{tif_path} -> {target}")

    return moved


def _download_satellite_inputs(
    temp_dir: Path,
    race_time_utc: datetime,
    domain: PreparedSiteDomain,
    force_ecostress_sst: bool = False,
    satellite_products: dict[str, bool] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict:
    _progress(progress_callback, 14, "Finding recent satellite imagery...")
    product_options = _satellite_product_options(**(satellite_products or {}))
    satellite_root = temp_dir / "satellite"
    satellite_root.mkdir(parents=True, exist_ok=True)
    selection_bbox = domain.bbox
    render_bbox = buffered_square_bbox_from_center(
        domain.site.center_lat,
        domain.site.center_lon,
        domain.site.side_meters,
        60.0,
    )
    diagnostics_errors: dict[str, str] = {}

    def safe_list_candidate_items(label: str, **kwargs) -> list:
        try:
            return list_candidate_items(**kwargs)
        except Exception as exc:
            diagnostics_errors[label] = repr(exc)
            return []

    sentinel_rgb_candidates = safe_list_candidate_items(
        "sentinel_rgb_search",
        collection="sentinel-2-l2a",
        bbox=render_bbox,
        end_time_utc=race_time_utc,
        lookback_days=45,
        required_assets=["visual", "scl"],
        limit=48,
        max_cloud_cover=MAX_RGB_SCENE_CLOUD_COVER,
    )
    sentinel_rgb = None
    sentinel_rgb_paths: dict[str, Path] | None = None
    sentinel_rgb_rank = None
    sentinel_rgb_rejections: list[dict] = []
    for index, candidate in enumerate(sentinel_rgb_candidates[:MAX_RGB_SCENE_ATTEMPTS]):
        _progress(
            progress_callback,
            14,
            f"Finding recent satellite imagery... RGB scene {index + 1}/{min(len(sentinel_rgb_candidates), MAX_RGB_SCENE_ATTEMPTS)}",
        )
        candidate_dir = satellite_root / f"sentinel_rgb_{index:02d}"
        screening_paths = download_selection_assets(candidate, candidate_dir, asset_keys=["scl"])
        clear_fraction = sentinel_clear_fraction(scl_tif=screening_paths["scl"], bbox=selection_bbox)
        if clear_fraction >= 0.35:
            sentinel_rgb = candidate
            sentinel_rgb_paths = download_selection_assets(candidate, candidate_dir, asset_keys=None if product_options["rgb"] else ["scl"])
            sentinel_rgb_rank = index
            break
        sentinel_rgb_rejections.append(
            {
                "item_id": candidate.item_id,
                "rank": index,
                "clear_fraction": clear_fraction,
            }
        )
    if (sentinel_rgb is None or sentinel_rgb_paths is None) and sentinel_rgb_candidates:
        sentinel_rgb = sentinel_rgb_candidates[0]
        sentinel_rgb_paths = download_selection_assets(
            sentinel_rgb,
            satellite_root / "sentinel_rgb_fallback",
            asset_keys=None if product_options["rgb"] else ["scl"],
        )
        sentinel_rgb_rank = 0

    sentinel_asset_keys = ["scl"]
    if product_options["chla"]:
        sentinel_asset_keys.extend(["red", "rededge1"])
    if product_options["turbidity"]:
        sentinel_asset_keys.extend(["red", "nir"])
    sentinel_asset_keys = sorted(set(sentinel_asset_keys))
    should_find_sentinel_analysis = product_options["chla"] or product_options["turbidity"]
    sentinel_candidates = (
        safe_list_candidate_items(
            "sentinel_analysis_search",
            collection="sentinel-2-l2a",
            bbox=render_bbox,
            end_time_utc=race_time_utc,
            lookback_days=45,
            required_assets=sentinel_asset_keys,
            limit=48,
            max_cloud_cover=MAX_WATER_SCENE_CLOUD_COVER,
        )
        if should_find_sentinel_analysis
        else []
    )
    sentinel = None
    sentinel_paths: dict[str, Path] | None = None
    sentinel_rank = None
    sentinel_rejections: list[dict] = []
    sentinel_unavailable_reason = None
    for index, candidate in enumerate(sentinel_candidates[:MAX_WATER_SCENE_ATTEMPTS]):
        _progress(
            progress_callback,
            15,
            f"Finding recent satellite imagery... Sentinel water scene {index + 1}/{min(len(sentinel_candidates), MAX_WATER_SCENE_ATTEMPTS)}",
        )
        candidate_dir = satellite_root / f"sentinel_analysis_{index:02d}"
        screening_paths = download_selection_assets(candidate, candidate_dir, asset_keys=["scl"])
        water_pixels = count_sentinel_water_pixels(scl_tif=screening_paths["scl"], bbox=selection_bbox)
        if water_pixels >= 25:
            sentinel = candidate
            sentinel_paths = download_selection_assets(candidate, candidate_dir, asset_keys=sentinel_asset_keys)
            sentinel_rank = index
            break
        sentinel_rejections.append({"item_id": candidate.item_id, "rank": index, "water_pixels": water_pixels})
    if should_find_sentinel_analysis and (sentinel is None or sentinel_paths is None):
        sentinel_unavailable_reason = diagnostics_errors.get(
            "sentinel_analysis_search",
            "No Sentinel-2 analytical scene had enough water pixels for the requested area.",
        )

    landsat_candidates = (
        safe_list_candidate_items(
            "landsat_search",
            collection="landsat-c2-l2",
            bbox=render_bbox,
            end_time_utc=race_time_utc,
            lookback_days=60,
            required_assets=["lwir11", "qa_pixel"],
            limit=48,
            stac_url=PLANETARY_COMPUTER_STAC,
            max_cloud_cover=MAX_LANDSAT_SCENE_CLOUD_COVER,
        )
        if product_options["sst"]
        else []
    )
    landsat = None
    landsat_paths: dict[str, Path] | None = None
    landsat_rank = None
    landsat_rejections: list[dict] = []
    landsat_unavailable_reason = None
    for index, candidate in enumerate(landsat_candidates[:MAX_WATER_SCENE_ATTEMPTS]):
        _progress(
            progress_callback,
            16,
            f"Finding recent satellite imagery... Landsat scene {index + 1}/{min(len(landsat_candidates), MAX_WATER_SCENE_ATTEMPTS)}",
        )
        candidate_dir = satellite_root / f"landsat_{index:02d}"
        screening_paths = download_selection_assets(candidate, candidate_dir, asset_keys=["qa_pixel"])
        water_pixels = count_landsat_water_pixels(qa_pixel_tif=screening_paths["qa_pixel"], bbox=selection_bbox)
        if water_pixels > 0:
            landsat = candidate
            landsat_paths = download_selection_assets(candidate, candidate_dir)
            landsat_rank = index
            break
        landsat_rejections.append({"item_id": candidate.item_id, "rank": index, "water_pixels": water_pixels})
    if product_options["sst"] and (landsat is None or landsat_paths is None):
        landsat_unavailable_reason = diagnostics_errors.get(
            "landsat_search",
            "No Landsat surface-temperature scene contained water pixels for the requested area.",
        )

    ecostress = None
    ecostress_paths: dict[str, Path] | None = None
    ecostress_rank = None
    ecostress_rejections: list[dict] = []
    ecostress_unavailable_reason = None
    landsat_age_days = None
    if landsat is not None:
        landsat_time = datetime.fromisoformat(landsat.item_datetime_utc.replace("Z", "+00:00"))
        landsat_age_days = max(0.0, (race_time_utc - landsat_time).total_seconds() / 86400.0)
    require_ecostress = product_options["sst"] and (
        force_ecostress_sst or landsat is None or landsat_age_days is None or landsat_age_days > LANDSAT_SST_MAX_AGE_DAYS
    )
    if require_ecostress:
        try:
            ecostress_candidates = list_ecostress_lste_candidates(
                bbox=render_bbox,
                end_time_utc=race_time_utc,
                lookback_days=90,
                limit=24,
            )
        except Exception as exc:
            ecostress_candidates = []
            ecostress_unavailable_reason = repr(exc)
        for index, candidate in enumerate(ecostress_candidates[:MAX_WATER_SCENE_ATTEMPTS]):
            _progress(
                progress_callback,
                17,
                f"Finding recent satellite imagery... ECOSTRESS scene {index + 1}/{min(len(ecostress_candidates), MAX_WATER_SCENE_ATTEMPTS)}",
            )
            candidate_dir = satellite_root / f"ecostress_{index:02d}"
            screening_paths = download_cmr_assets(candidate, candidate_dir, asset_keys=["water"])
            water_pixels = count_ecostress_water_pixels(water_tif=screening_paths["water"], bbox=selection_bbox)
            if water_pixels > 0:
                ecostress = candidate
                ecostress_paths = download_cmr_assets(candidate, candidate_dir)
                ecostress_rank = index
                break
            ecostress_rejections.append({"item_id": candidate.item_id, "rank": index, "water_pixels": water_pixels})
        if require_ecostress and ecostress is None and ecostress_unavailable_reason is None:
            ecostress_unavailable_reason = "No ECOSTRESS surface-temperature scene contained water pixels for the requested area."
    else:
        ecostress_candidates = []

    return {
        "sentinel_rgb": sentinel_rgb,
        "sentinel_rgb_paths": sentinel_rgb_paths,
        "sentinel": sentinel,
        "sentinel_paths": sentinel_paths,
        "landsat": landsat,
        "landsat_paths": landsat_paths,
        "ecostress": ecostress,
        "ecostress_paths": ecostress_paths,
        "render_bbox": render_bbox,
        "product_options": product_options,
        "selection_diagnostics": {
            "selected_products": product_options,
            "search_errors": diagnostics_errors,
            "sentinel_rgb_candidate_count": len(sentinel_rgb_candidates),
            "rgb_scene_attempt_cap": MAX_RGB_SCENE_ATTEMPTS,
            "sentinel_rgb_selected_rank": sentinel_rgb_rank,
            "sentinel_rgb_rejections": sentinel_rgb_rejections,
            "sentinel_analysis_candidate_count": len(sentinel_candidates),
            "sentinel_selected_rank": sentinel_rank,
            "sentinel_rejections": sentinel_rejections,
            "sentinel_unavailable_reason": sentinel_unavailable_reason,
            "landsat_candidate_count": len(landsat_candidates),
            "landsat_selected_rank": landsat_rank,
            "landsat_rejections": landsat_rejections,
            "landsat_unavailable_reason": landsat_unavailable_reason,
            "landsat_age_days": landsat_age_days,
            "landsat_sst_max_age_days": LANDSAT_SST_MAX_AGE_DAYS,
            "force_ecostress_sst": force_ecostress_sst,
            "ecostress_required": require_ecostress,
            "ecostress_lookback_days": 90,
            "ecostress_candidate_count": len(ecostress_candidates),
            "ecostress_selected_rank": ecostress_rank,
            "ecostress_rejections": ecostress_rejections,
            "ecostress_unavailable_reason": ecostress_unavailable_reason,
            "water_scene_attempt_cap": MAX_WATER_SCENE_ATTEMPTS,
            "rgb_cloud_cover_cap": MAX_RGB_SCENE_CLOUD_COVER,
            "sentinel_cloud_cover_cap": MAX_WATER_SCENE_CLOUD_COVER,
            "landsat_cloud_cover_cap": MAX_LANDSAT_SCENE_CLOUD_COVER,
        },
    }


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


def _prepare_landscape_input(
    temp_dir: Path,
    domain: PreparedSiteDomain,
    satellite_inputs: dict,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[Path, dict]:
    _progress(progress_callback, 18, "Building vegetation landscape...")
    landscape_dir = temp_dir / "landscape"
    landscape_dir.mkdir(parents=True, exist_ok=True)
    domain_dem = domain.solve_dem_tif
    if satellite_inputs.get("sentinel_paths") is not None:
        scl_tif = Path(satellite_inputs["sentinel_paths"]["scl"])
    elif satellite_inputs.get("sentinel_rgb_paths") is not None:
        scl_tif = Path(satellite_inputs["sentinel_rgb_paths"]["scl"])
    else:
        scl_tif = None

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

    canopy_height_tif = landscape_dir / "landfire_canopy_height.tif"
    canopy_cover_tif = landscape_dir / "landfire_canopy_cover.tif"
    canopy_base_height_tif = landscape_dir / "landfire_canopy_base_height.tif"
    canopy_bulk_density_tif = landscape_dir / "landfire_canopy_bulk_density.tif"
    water_mask_tif = landscape_dir / "water_mask.tif"
    landscape_tif = landscape_dir / "site_landscape.tif"
    summary_json = landscape_dir / "site_landscape_summary.json"

    landfire_tasks: list[tuple[str, Callable[[], object]]] = [
        ("ch", lambda: export_image(service_key="ch", spec=spec, destination=canopy_height_tif)),
        ("cc", lambda: export_image(service_key="cc", spec=spec, destination=canopy_cover_tif)),
        ("cbh", lambda: export_image(service_key="cbh", spec=spec, destination=canopy_base_height_tif)),
        ("cbd", lambda: export_image(service_key="cbd", spec=spec, destination=canopy_bulk_density_tif)),
    ]
    for service_key, _, exc in _run_ordered_tasks(landfire_tasks):
        if exc is not None:
            raise RuntimeError(f"Unable to export LANDFIRE {service_key} raster: {exc}") from exc
    if scl_tif is not None:
        _resample_scl_water_mask(scl_tif, domain_dem, water_mask_tif)
    else:
        with rasterio.open(domain_dem) as src:
            profile = src.profile.copy()
            profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0, compress="deflate")
            water_mask_tif.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(water_mask_tif, "w", **profile) as dst:
                dst.write(np.zeros((src.height, src.width), dtype=np.uint8), 1)

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
    summary_dict = summary.__dict__
    if scl_tif is None:
        summary_dict["water_mask_source"] = "none_satellite_unavailable"
    return landscape_tif, summary_dict


def _transform_from_aaigrid_header(header: dict[str, float]):
    return from_origin(
        header["xllcorner"],
        header["yllcorner"] + header["nrows"] * header["cellsize"],
        header["cellsize"],
        header["cellsize"],
    )


def _wind_grid_template_from_raster(reference_tif: Path) -> tuple[dict[str, float], tuple[int, int]]:
    with rasterio.open(reference_tif) as src:
        transform = src.transform
        cell_width = float(abs(transform.a))
        cell_height = float(abs(transform.e))
        if cell_width <= 0.0 or cell_height <= 0.0:
            raise ValueError(f"Invalid wind grid transform for {reference_tif}: {transform}")
        if abs(cell_width - cell_height) > max(cell_width, cell_height) * 0.01:
            raise ValueError(f"Wind grid cells must be square for AAIGrid-style rendering: {transform}")
        header = {
            "ncols": float(src.width),
            "nrows": float(src.height),
            "xllcorner": float(transform.c),
            "yllcorner": float(transform.f + src.height * transform.e),
            "cellsize": cell_width,
            "nodata_value": -9999.0,
        }
        return header, (int(src.height), int(src.width))


def _water_mask_for_wind_grid(landscape_tif: Path, source_header: dict[str, float]) -> np.ndarray | None:
    try:
        height = int(source_header["nrows"])
        width = int(source_header["ncols"])
        destination = np.full((height, width), -9999.0, dtype=np.float32)
        destination_transform = _transform_from_aaigrid_header(source_header)
        with rasterio.open(landscape_tif) as src:
            if src.count < 4:
                return None
            reproject(
                source=rasterio.band(src, 4),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=destination_transform,
                dst_crs=src.crs,
                src_nodata=src.nodata,
                dst_nodata=-9999.0,
                resampling=Resampling.nearest,
            )
        return np.isfinite(destination) & (np.round(destination).astype(np.int16) == 98)
    except Exception:
        return None


def _openfoam_error_payload(exc: Exception) -> dict:
    return {
        "type": exc.__class__.__name__,
        "message": getattr(exc, "message", str(exc)),
        "stage": getattr(exc, "stage", "openfoam_comparison"),
        "details": getattr(exc, "details", {}) or {},
        "stdout": getattr(exc, "stdout", ""),
        "stderr": getattr(exc, "stderr", ""),
    }


def _array_finite_summary(name: str, field: np.ndarray) -> dict:
    valid = field[np.isfinite(field)]
    if valid.size == 0:
        return {
            "name": name,
            "finite_count": 0,
            "shape": list(field.shape),
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "name": name,
        "finite_count": int(valid.size),
        "shape": list(field.shape),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
    }


def _model_spread_summary_from_forecasts(model_forecasts: list[dict]) -> tuple[float, float]:
    speed_values_mps: list[float] = []
    direction_values_deg: list[float] = []
    for forecast in model_forecasts:
        try:
            speed = float(forecast["wind_speed_mps"])
            direction = float(forecast["wind_from_direction_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(speed) or speed < 0.0:
            continue
        speed_values_mps.append(speed)
        if speed >= CALM_WIND_SOLVER_THRESHOLD_MPS and math.isfinite(direction):
            direction_values_deg.append(direction % 360.0)

    if len(speed_values_mps) >= 2:
        speed_std_kts = float(np.std(np.asarray(speed_values_mps, dtype=np.float32)) * 1.94384449)
    else:
        speed_std_kts = 0.0

    if len(direction_values_deg) >= 2:
        direction_rad = np.deg2rad(np.asarray(direction_values_deg, dtype=np.float32))
        resultant = float(math.hypot(float(np.mean(np.sin(direction_rad))), float(np.mean(np.cos(direction_rad)))))
        resultant = max(1.0e-6, min(1.0, resultant))
        direction_std_deg = float(math.degrees(math.sqrt(-2.0 * math.log(resultant))))
    else:
        direction_std_deg = 0.0

    return speed_std_kts, direction_std_deg


def _write_sailing_polar_overlay_product(
    *,
    speed_asc: Path,
    direction_asc: Path,
    dem_basemap_tif: Path,
    output_png: Path,
    wind_source: str,
    sailor_weight_lb: float = 175.0,
    overlay_radius_px: int = 280,
) -> dict:
    try:
        from experimental_laser_polar_point import _choose_point, _draw_polar_dem_overlay, _grid_xy, _polar_samples

        speed_mps, header = _read_aaigrid(speed_asc)
        direction_deg, _ = _read_aaigrid(direction_asc)
        row, col = _choose_point(speed_mps, header, argparse.Namespace(row=None, col=None, x=None, y=None))
        x, y = _grid_xy(header, row, col)
        tws_knots = float(speed_mps[row, col] * 1.94384449)
        wind_from_deg = float(direction_deg[row, col] % 360.0)
        samples = _polar_samples(tws_knots, wind_from_deg, sailor_weight_lb)
        _draw_polar_dem_overlay(
            output_png,
            dem_basemap_tif,
            samples,
            wind_from_deg,
            tws_knots,
            sailor_weight_lb,
            x,
            y,
            overlay_radius_px,
            wind_source,
        )
        return {
            "enabled": True,
            "status": "completed",
            "product_png": str(output_png),
            "wind_source": wind_source,
            "sailor_weight_lb": sailor_weight_lb,
            "row": int(row),
            "col": int(col),
            "x": float(x),
            "y": float(y),
            "local_wind_speed_knots": tws_knots,
            "local_wind_from_direction_deg": wind_from_deg,
            "note": (
                "Experimental ILCA 7/Laser relative point polar centered on the sampled wind cell and overlaid "
                "on the cropped DEM; heuristic sailing aid, not a calibrated VPP or routing model."
            ),
        }
    except Exception as exc:
        _write_unavailable_panel(
            output_png,
            title="sail",
            line1="Sailing polar unavailable",
            line2=str(exc),
        )
        return {
            "enabled": True,
            "status": "failed",
            "product_png": str(output_png),
            "wind_source": wind_source,
            "sailor_weight_lb": sailor_weight_lb,
            "error": {"type": exc.__class__.__name__, "message": str(exc)},
        }


def _build_wind_products(
    report_dir: Path,
    temp_dir: Path,
    race_time_utc: datetime,
    site: SiteConfig,
    domain: PreparedSiteDomain,
    wind_input_tif: Path,
    mesh_resolution_m: float,
    wind_solver: str = "windninja",
    progress_callback: Callable[[int, str], None] | None = None,
    raw_data_dir: Path = DATA_RAW_DIR,
) -> dict:
    if wind_solver not in {"windninja", "openfoam"}:
        raise ValueError(f"Unsupported wind solver: {wind_solver}")

    _progress(progress_callback, 20, "Building wind boundary conditions...")
    boundary_target_time_utc = _aligned_boundary_valid_time(race_time_utc)
    model_forecasts, model_errors = _load_model_point_forecasts(
        site.center_lat,
        site.center_lon,
        boundary_target_time_utc,
        raw_data_dir,
    )
    if not model_forecasts:
        raise RuntimeError(f"No deterministic model forecasts were available for {boundary_target_time_utc.isoformat()}: {model_errors}")
    _progress(progress_callback, 24, "Scoring recent nearby model skill...")
    skill_adjustments, skill_metadata = _compute_station_skill_adjustments(
        target_valid_time_utc=boundary_target_time_utc,
        site_lat=site.center_lat,
        site_lon=site.center_lon,
        raw_data_dir=raw_data_dir,
    )

    gefs_mean_path, gefs_spread_path, gefs_member_paths, gefs_mode, gefs_manifest = _load_fresh_or_cached_gefs_bundle(
        site,
        boundary_target_time_utc,
        raw_data_dir,
    )
    sampled_gefs = sample_gefs_mean_and_spread_at_site(
        mean_path=gefs_mean_path,
        spread_path=gefs_spread_path,
        lat=site.center_lat,
        lon=site.center_lon,
    )
    sampled_gefs["manifest"] = gefs_manifest
    sampled_gefs_members: list[dict] = []
    gefs_member_sample_error: str | None = None
    if gefs_member_paths:
        try:
            sampled_gefs_members = sample_gefs_members_at_site(gefs_member_paths, site.center_lat, site.center_lon)
        except Exception as exc:
            gefs_member_sample_error = repr(exc)
    available_sources = sorted({str(forecast.get("source", "")) for forecast in model_forecasts if forecast.get("source")})
    expected_sources = {"ecmwf", "gfs", "hrrr", "hrdps", "icon", "nam"}
    missing_sources = sorted(expected_sources - set(available_sources))
    wind_warnings: list[str] = []
    if model_errors:
        wind_warnings.append(
            "Some deterministic model downloads failed: "
            + ", ".join(f"{source} ({error})" for source, error in sorted(model_errors.items()))
        )
    if len(available_sources) < 4:
        wind_warnings.append(
            f"Deterministic boundary is based on only {len(available_sources)} model(s): {', '.join(available_sources) or 'none'}."
        )
    if missing_sources:
        wind_warnings.append("Missing deterministic sources: " + ", ".join(missing_sources) + ".")
    if any(str(forecast.get("acquisition_mode", "")).startswith("cached") for forecast in model_forecasts):
        wind_warnings.append("One or more deterministic model inputs came from a local cache for the exact requested valid time.")
    if not gefs_mode.startswith("fresh"):
        wind_warnings.append(f"GEFS inputs came from local cache mode `{gefs_mode}`.")
    if gefs_member_sample_error is not None:
        wind_warnings.append(f"GEFS member sampling failed; falling back to mean/spread sensitivity: {gefs_member_sample_error}")
    elif not sampled_gefs_members:
        wind_warnings.append("No usable GEFS member files were available; falling back to mean/spread sensitivity.")
    consensus = choose_consensus_boundary(
        [
            build_model_point_forecast(
                source=forecast["source"],
                display_name=forecast["display_name"],
                run_at_utc=forecast["run_at_utc"],
                forecast_hour=forecast["forecast_hour"],
                acquisition_mode=forecast["acquisition_mode"],
                wind_summary=forecast,
                files=forecast["files"],
                gust_summary=None if forecast["gust_mps"] is None else {"value": forecast["gust_mps"]},
                sustained_kind=forecast["sustained_kind"],
                gust_kind=forecast["gust_kind"],
            )
            for forecast in model_forecasts
        ],
        skill_adjustments=skill_adjustments,
        skill_metadata=skill_metadata,
    )
    boundary = consensus.boundary.as_dict()
    wind_note = f"{consensus.selected_source}_{gefs_mode}"

    wind_root = temp_dir / "wind"
    wind_root.mkdir(parents=True, exist_ok=True)
    deterministic_dir = wind_root / "deterministic"
    solver_metadata: list[dict] = []
    openfoam_comparison: dict | None = None
    enable_openfoam_comparison = wind_solver == "openfoam"
    vector_stride, vector_scale = _vector_overlay_style(mesh_resolution_m)
    openfoam_high_resolution_warning = False
    if enable_openfoam_comparison:
        try:
            vertical_cells = _openfoam_vertical_cells()
            requested_openfoam_cells = _estimate_openfoam_solve_cells(wind_input_tif, mesh_resolution_m, vertical_cells)
            baseline_openfoam_cells = _estimate_openfoam_solve_cells(
                wind_input_tif,
                OPENFOAM_HIGH_RESOLUTION_WARNING_BASELINE_M,
                vertical_cells,
            )
            if requested_openfoam_cells["solve_cells"] > baseline_openfoam_cells["solve_cells"]:
                openfoam_high_resolution_warning = True
                wind_warnings.append(
                    "OpenFOAM high-resolution solve is estimated at "
                    f"{requested_openfoam_cells['solve_cells']:,} cells "
                    f"({requested_openfoam_cells['nx']} x {requested_openfoam_cells['ny']} x "
                    f"{requested_openfoam_cells['vertical_cells']}) versus "
                    f"{baseline_openfoam_cells['solve_cells']:,} cells at "
                    f"{OPENFOAM_HIGH_RESOLUTION_WARNING_BASELINE_M:.0f} m; OpenFOAM may take a while."
                )
        except Exception as exc:
            wind_warnings.append(f"OpenFOAM high-resolution solve-size warning could not be calculated: {exc}")

    weather_inset: dict | None = None
    inset_lines: list[str] | None = None
    try:
        point_forecast = sample_nws_hourly_forecast(site.center_lat, site.center_lon, race_time_utc)
        weather_inset = {
            "temperature_f": point_forecast.temperature_f,
            "precipitation_probability_pct": point_forecast.precipitation_probability_pct,
            "sky_cover_pct": point_forecast.sky_cover_pct,
            "source_time_utc": point_forecast.source_time_utc,
            "target_time_utc": point_forecast.target_time_utc,
            "selection_mode": point_forecast.selection_mode,
            "forecast_grid_url": point_forecast.forecast_grid_url,
        }
        inset_lines = [
            f"temp {_format_optional_number(point_forecast.temperature_f, 'f')}",
            f"pop {_format_optional_number(point_forecast.precipitation_probability_pct, '%')}",
            f"sky {_format_optional_number(point_forecast.sky_cover_pct, '%')}",
        ]
    except Exception as exc:
        weather_inset = {
            "temperature_f": None,
            "precipitation_probability_pct": None,
            "sky_cover_pct": None,
            "error": str(exc),
            "target_time_utc": race_time_utc.isoformat(),
            "source": "nws",
        }
        inset_lines = [
            "temp n/a",
            "pop n/a",
            "sky n/a",
        ]

    bottom_table_rows = model_table_rows(
        [
            build_model_point_forecast(
                source=forecast["source"],
                display_name=forecast["display_name"],
                run_at_utc=forecast["run_at_utc"],
                forecast_hour=forecast["forecast_hour"],
                acquisition_mode=forecast["acquisition_mode"],
                wind_summary=forecast,
                files=forecast["files"],
                gust_summary=None if forecast["gust_mps"] is None else {"value": forecast["gust_mps"]},
                sustained_kind=forecast["sustained_kind"],
                gust_kind=forecast["gust_kind"],
            )
            for forecast in model_forecasts
        ]
    )

    def run_windninja_solver(output_dir: Path, wind_speed_mps: float, wind_direction_deg: float) -> dict:
        cli_path = windninja_cli_path(RUNTIME_RESOURCE_ROOT)
        run = run_windninja_domain_average(
            cli_path=cli_path,
            elevation_tif=wind_input_tif,
            output_dir=output_dir,
            wind_speed_mps=wind_speed_mps,
            wind_direction_deg=wind_direction_deg,
            mesh_resolution_m=mesh_resolution_m,
            momentum=True,
            iterations=300,
            turbulence_output=False,
            num_threads=1,
        )
        return {"solver": "windninja", "solver_mode": "momentum", **run}

    boundary_speed_mps = float(boundary["wind_speed_mps"])
    boundary_direction_deg = float(boundary["wind_from_direction_deg"])
    if not math.isfinite(boundary_direction_deg):
        boundary_direction_deg = 0.0
    if not math.isfinite(boundary_speed_mps) or boundary_speed_mps < CALM_WIND_SOLVER_THRESHOLD_MPS:
        _progress(progress_callback, 32, "Rendering calm wind products...")
        deterministic_dir.mkdir(parents=True, exist_ok=True)
        speed_header, grid_shape = _wind_grid_template_from_raster(wind_input_tif)
        speed_mps = np.zeros(grid_shape, dtype=np.float32)
        speed_kts = np.zeros(grid_shape, dtype=np.float32)
        direction_deg = np.full(grid_shape, boundary_direction_deg % 360.0, dtype=np.float32)
        u_mps = np.zeros(grid_shape, dtype=np.float32)
        v_mps = np.zeros(grid_shape, dtype=np.float32)
        product1_png = report_dir / "product_1_wind_speed_prediction_knots.png"
        write_windninja_knots_vector_preview_from_arrays(
            speed_mps=speed_mps,
            u_mps=u_mps,
            v_mps=v_mps,
            preview_png=product1_png,
            dem_basemap_tif=domain.dem_preview_tif,
            source_header=speed_header,
            vector_stride=vector_stride,
            vector_scale=vector_scale,
            colormap=diverging_blue_green_red_colormap(),
            center_value=0.0,
            title="calm",
            units="knots",
            footer_text=(
                f"{_footer_timestamp_text(boundary_target_time_utc, 'wind forecast')} | "
                f"terrain solvers skipped below {CALM_WIND_SOLVER_THRESHOLD_MPS:.1f} m/s"
            ),
            inset_lines=inset_lines,
            bottom_table_rows=bottom_table_rows,
        )

        sailing_polar_png = report_dir / "product_6_sailing_polar_dem_overlay.png"
        _write_unavailable_panel(
            sailing_polar_png,
            title="sail",
            line1="Sailing polar skipped",
            line2=f"Calm boundary wind below {CALM_WIND_SOLVER_THRESHOLD_MPS:.1f} m/s.",
        )
        sailing_polar = {
            "enabled": True,
            "status": "skipped",
            "product_png": str(sailing_polar_png),
            "wind_source": "calm_direct",
            "reason": f"Boundary wind below {CALM_WIND_SOLVER_THRESHOLD_MPS:.1f} m/s.",
        }

        model_speed_std_kts, model_direction_std_deg = _model_spread_summary_from_forecasts(model_forecasts)
        speed_std_kts = np.full(grid_shape, model_speed_std_kts, dtype=np.float32)
        direction_std_deg = np.full(grid_shape, model_direction_std_deg, dtype=np.float32)
        solve_speed_std_tif = temp_dir / "wind_speed_variance_knots_solve.tif"
        solve_direction_std_tif = temp_dir / "wind_direction_variance_degrees_solve.tif"
        speed_std_tif = temp_dir / "wind_speed_variance_knots.tif"
        direction_std_tif = temp_dir / "wind_direction_variance_degrees.tif"
        write_array_to_geotiff_from_header(speed_header, speed_std_kts, solve_speed_std_tif)
        write_array_to_geotiff_from_header(speed_header, direction_std_deg, solve_direction_std_tif)
        write_reprojected_array_like_reference(
            speed_std_kts,
            source_header=speed_header,
            reference_tif=domain.clipped_dem_tif,
            destination_tif=speed_std_tif,
            resampling=Resampling.bilinear,
        )
        write_reprojected_array_like_reference(
            direction_std_deg,
            source_header=speed_header,
            reference_tif=domain.clipped_dem_tif,
            destination_tif=direction_std_tif,
            resampling=Resampling.bilinear,
        )

        product2_png = report_dir / "product_2_wind_speed_variance_knots.png"
        product3_png = report_dir / "product_3_wind_direction_variance_degrees.png"
        _progress(progress_callback, 58, "Rendering calm wind maps...")
        colormap = diverging_blue_green_red_colormap()
        write_scalar_diagnostic_preview(
            field=speed_std_kts,
            dem_basemap_tif=domain.dem_preview_tif,
            output_png=product2_png,
            title="sd wind",
            units="knots",
            colormap=colormap,
            source_header=speed_header,
            alpha=0.58,
            signed=False,
            center_value=model_speed_std_kts,
            footer_text=_footer_timestamp_text(boundary_target_time_utc, "model wind speed SD"),
        )
        write_scalar_diagnostic_preview(
            field=direction_std_deg,
            dem_basemap_tif=domain.dem_preview_tif,
            output_png=product3_png,
            title="sd az",
            units="deg",
            colormap=colormap,
            source_header=speed_header,
            alpha=0.58,
            signed=False,
            center_value=model_direction_std_deg,
            footer_text=_footer_timestamp_text(boundary_target_time_utc, "model wind dir SD"),
        )

        if enable_openfoam_comparison:
            openfoam_png = report_dir / "product_4_openfoam_experimental_cfd_knots.png"
            _write_unavailable_panel(
                openfoam_png,
                title="cfd exp",
                line1="OpenFOAM CFD skipped",
                line2=f"Calm boundary wind below {CALM_WIND_SOLVER_THRESHOLD_MPS:.1f} m/s.",
            )
            openfoam_comparison = {
                "enabled": True,
                "status": "skipped",
                "product_png": str(openfoam_png),
                "reason": f"Boundary wind below {CALM_WIND_SOLVER_THRESHOLD_MPS:.1f} m/s.",
            }

        wind_warnings.append(
            f"Consensus boundary wind was {boundary_speed_mps:.2f} m/s; terrain solvers were skipped below "
            f"{CALM_WIND_SOLVER_THRESHOLD_MPS:.1f} m/s and calm products were rendered directly."
        )
        solver_metadata.append(
            {
                "role": "deterministic",
                "solver": "calm_direct",
                "solver_mode": "calm_guard",
                "status": "skipped",
                "threshold_mps": CALM_WIND_SOLVER_THRESHOLD_MPS,
                "boundary_wind_speed_mps": boundary_speed_mps,
                "boundary_wind_from_direction_deg": boundary_direction_deg,
                "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
            }
        )
        return {
            "boundary": boundary,
            "boundary_consensus": consensus.as_dict(),
            "model_forecasts": model_forecasts,
            "model_errors": model_errors,
            "skill_adjustments": skill_adjustments,
            "skill_metadata": skill_metadata,
            "gefs_boundary": sampled_gefs,
            "wind_data_mode": wind_note,
            "requested_wind_solver": wind_solver,
            "wind_solver": "calm_direct",
            "wind_solver_display": "Calm direct render",
            "openfoam_comparison": openfoam_comparison
            if openfoam_comparison is not None
            else {
                "enabled": False,
                "status": "not_requested",
            },
            "solver_runs": solver_metadata,
            "wind_input_tif": str(wind_input_tif),
            "final_dem_tif": str(domain.clipped_dem_tif),
            "solve_dem_tif": str(domain.solve_dem_tif),
            "boundary_target_time_utc": boundary_target_time_utc.isoformat(),
            "mesh_resolution_m": mesh_resolution_m,
            "available_model_sources": available_sources,
            "missing_model_sources": missing_sources,
            "degraded": True,
            "warnings": wind_warnings,
            "member_count": 0,
            "members": [],
            "product_1": str(product1_png),
            "product_2": str(product2_png),
            "product_3": str(product3_png),
            "sailing_polar_overlay": sailing_polar,
            "speed_std_knots_mean": float(np.nanmean(speed_std_kts)),
            "direction_std_deg_mean": float(np.nanmean(direction_std_deg)),
            "weather_inset": weather_inset,
            "spread_product_label": "deterministic-model disagreement standard deviation",
            "spread_product_note": (
                "Products 2 and 3 are constant calm-wind standard-deviation summaries from deterministic model "
                "disagreement; terrain solvers were skipped and these are not probabilistic variance maps."
            ),
        }

    solver_display = "WindNinja momentum"
    _progress(progress_callback, 32, "Running production terrain wind prediction with WindNinja...")
    deterministic_run = run_windninja_solver(
        output_dir=deterministic_dir,
        wind_speed_mps=boundary_speed_mps,
        wind_direction_deg=boundary_direction_deg,
    )
    solver_metadata.append({"role": "deterministic", **deterministic_run})
    ascii_paths = expected_windninja_ascii_paths(
        elevation_tif=wind_input_tif,
        wind_speed_mps=boundary_speed_mps,
        wind_direction_deg=boundary_direction_deg,
        mesh_resolution_m=mesh_resolution_m,
        output_dir=deterministic_dir,
    )
    speed_mps, speed_header = _read_aaigrid(ascii_paths["speed"])
    speed_kts = speed_mps * 1.94384449

    product1_png = report_dir / "product_1_wind_speed_prediction_knots.png"
    write_windninja_knots_vector_preview_from_speed_angle(
        ascii_paths["speed"],
        ascii_paths["direction"],
        product1_png,
        dem_basemap_tif=domain.dem_preview_tif,
        source_header=speed_header,
        vector_stride=vector_stride,
        vector_scale=vector_scale,
        colormap=diverging_blue_green_red_colormap(),
        center_value=float(np.nanmean(speed_kts)),
        title="wind",
        units="knots",
        footer_text=_footer_timestamp_text(boundary_target_time_utc, "wind forecast"),
        inset_lines=inset_lines,
        bottom_table_rows=bottom_table_rows,
    )
    sailing_polar_png = report_dir / "product_6_sailing_polar_dem_overlay.png"
    sailing_polar = _write_sailing_polar_overlay_product(
        speed_asc=ascii_paths["speed"],
        direction_asc=ascii_paths["direction"],
        dem_basemap_tif=domain.dem_preview_tif,
        output_png=sailing_polar_png,
        wind_source="windninja",
    )

    if enable_openfoam_comparison:
        openfoam_png = report_dir / "product_4_openfoam_experimental_cfd_knots.png"
        openfoam_turbulence_png = report_dir / "product_5_openfoam_turbulence_intensity_percent.png"
        openfoam_sailing_polar_png = report_dir / "product_7_openfoam_sailing_polar_dem_overlay.png"
        openfoam_dir = wind_root / "openfoam_comparison"
        _progress(
            progress_callback,
            40,
            "Running experimental OpenFOAM CFD comparison; high cell count may take a while..."
            if openfoam_high_resolution_warning
            else "Running experimental OpenFOAM CFD comparison...",
        )
        try:
            openfoam_run = run_openfoam_domain_average(
                elevation_tif=wind_input_tif,
                output_dir=openfoam_dir,
                wind_speed_mps=float(boundary["wind_speed_mps"]),
                wind_direction_deg=float(boundary["wind_from_direction_deg"]),
                mesh_resolution_m=mesh_resolution_m,
            )
            is_scientific_cfd_candidate = bool(openfoam_run.get("is_scientific_cfd_candidate"))
            openfoam_product_label = (
                "Experimental CFD comparison" if is_scientific_cfd_candidate else "Custom wind-grid adapter comparison"
            )
            openfoam_product_title = "cfd exp" if is_scientific_cfd_candidate else "adapter"
            openfoam_footer_label = "experimental cfd" if is_scientific_cfd_candidate else "adapter comparison"
            openfoam_scientific_note = (
                "Experimental WSL/OpenFOAM neutral ABL RANS comparison; not validated for production decisions."
                if is_scientific_cfd_candidate
                else "Custom runner output matched the PondWind wind-grid contract, but it is not a validated OpenFOAM/CFD solve."
            )
            openfoam_paths = {key: Path(value) for key, value in openfoam_run["expected_outputs"].items()}
            openfoam_speed, openfoam_header = _read_aaigrid(openfoam_paths["speed"])
            openfoam_speed_kts = openfoam_speed * 1.94384449
            water_mask = _water_mask_for_wind_grid(wind_input_tif, openfoam_header)
            comparison_metrics = compare_wind_outputs(
                windninja_paths=ascii_paths,
                openfoam_paths=openfoam_paths,
                water_mask=water_mask,
            )
            write_windninja_knots_vector_preview_from_speed_angle(
                openfoam_paths["speed"],
                openfoam_paths["direction"],
                openfoam_png,
                dem_basemap_tif=domain.dem_preview_tif,
                source_header=openfoam_header,
                vector_stride=vector_stride,
                vector_scale=vector_scale,
                colormap=diverging_blue_green_red_colormap(),
                center_value=float(np.nanmean(openfoam_speed_kts)),
                title=openfoam_product_title,
                units="knots",
                footer_text=(
                    f"{_footer_timestamp_text(boundary_target_time_utc, openfoam_footer_label)} | "
                    f"bias {comparison_metrics['full_domain'].get('speed_bias_mps', float('nan')):.1f} m/s | "
                    f"rmse {comparison_metrics['full_domain'].get('speed_rmse_mps', float('nan')):.1f} m/s | not production"
                ),
                inset_lines=inset_lines,
                bottom_table_rows=bottom_table_rows,
            )
            turbulence_stats: dict | None = None
            if openfoam_paths.get("turbulence_intensity") is not None and openfoam_paths["turbulence_intensity"].exists():
                turbulence_intensity_pct, _ = _read_aaigrid(openfoam_paths["turbulence_intensity"])
                turbulence_stats = _array_finite_summary("turbulence_intensity_pct", turbulence_intensity_pct)
                write_scalar_diagnostic_preview(
                    field=turbulence_intensity_pct,
                    dem_basemap_tif=domain.dem_preview_tif,
                    output_png=openfoam_turbulence_png,
                    title="cfd ti",
                    units="%",
                    colormap=diverging_blue_green_red_colormap(),
                    source_header=openfoam_header,
                    alpha=0.58,
                    signed=False,
                    center_value=float(np.nanmean(turbulence_intensity_pct)),
                    footer_text=_footer_timestamp_text(boundary_target_time_utc, "experimental cfd turbulence"),
                )
            openfoam_sailing_polar = _write_sailing_polar_overlay_product(
                speed_asc=openfoam_paths["speed"],
                direction_asc=openfoam_paths["direction"],
                dem_basemap_tif=domain.dem_preview_tif,
                output_png=openfoam_sailing_polar_png,
                wind_source="openfoam_comparison",
            )
            openfoam_comparison = {
                "enabled": True,
                "status": "completed" if is_scientific_cfd_candidate else "adapter_completed",
                "product_label": openfoam_product_label,
                "scientific_note": openfoam_scientific_note,
                "product_png": str(openfoam_png),
                "turbulence_png": str(openfoam_turbulence_png) if openfoam_turbulence_png.exists() else None,
                "sailing_polar_overlay_png": str(openfoam_sailing_polar_png),
                "sailing_polar_overlay": openfoam_sailing_polar,
                "turbulence_intensity_stats_pct": turbulence_stats,
                "run": openfoam_run,
                "windninja_comparison": comparison_metrics,
                "metar_proxy_validation": {
                    "label": "off_pond_nearby_station_proxy",
                    "station_id": skill_metadata.get("station_id"),
                    "station_distance_km": skill_metadata.get("station_distance_km"),
                    "observations_used": skill_metadata.get("observations_used", 0),
                    "note": "Nearby METAR/AWOS observations are used only as off-pond proxy context; they are not direct pond validation.",
                },
            }
            solver_metadata.append({"role": "openfoam_comparison", **openfoam_run})
        except OpenFoamRunError as exc:
            skipped = exc.stage == "openfoam_unavailable" or bool((exc.details or {}).get("skipped"))
            status = "skipped" if skipped else "failed"
            line1 = "OpenFOAM CFD skipped" if skipped else "OpenFOAM comparison failed"
            line2 = exc.message if skipped else f"{exc.stage}: {exc.message}"
            _write_unavailable_panel(
                openfoam_png,
                title="cfd exp",
                line1=line1,
                line2=line2,
            )
            openfoam_comparison = {
                "enabled": True,
                "status": status,
                "product_png": str(openfoam_png),
                "error": _openfoam_error_payload(exc),
            }
        except Exception as exc:
            _write_unavailable_panel(
                openfoam_png,
                title="cfd exp",
                line1="OpenFOAM comparison failed",
                line2=str(exc),
            )
            openfoam_comparison = {
                "enabled": True,
                "status": "failed",
                "product_png": str(openfoam_png),
                "error": _openfoam_error_payload(exc),
            }

    deterministic_speed_header = speed_header
    deterministic_grid_shape = speed_mps.shape
    selected_gefs_members = _select_representative_gefs_members(sampled_gefs_members, _gefs_member_solve_limit())
    use_real_gefs_members = len(selected_gefs_members) >= 3
    if sampled_gefs_members and not use_real_gefs_members:
        wind_warnings.append(
            f"Only {len(selected_gefs_members)} usable GEFS member(s) were selected; falling back to mean/spread sensitivity."
        )
    ensemble_dir = wind_root / ("gefs_members" if use_real_gefs_members else "gefs_sigma")
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    _progress(
        progress_callback,
        45,
        "Estimating WindNinja wind variability from GEFS members..."
        if use_real_gefs_members
        else "Estimating WindNinja wind variability from GEFS spread...",
    )
    speed_members: list[np.ndarray] = []
    direction_members_deg: list[np.ndarray] = []
    speed_header: dict[str, float] | None = None
    member_records: list[dict] = []
    if use_real_gefs_members:
        variability_members = [
            {
                "label": str(member["member_id"]),
                "member_id": str(member["member_id"]),
                "member_source": "gefs_member",
                "member_u": float(member["u10_mps"]),
                "member_v": float(member["v10_mps"]),
                "selection_reason": member.get("selection_reason"),
                "selection_rank": member.get("selection_rank"),
                "path": member.get("path"),
                "member_weight": float(member.get("member_weight", 1.0)),
                "cluster_member_count": int(member.get("cluster_member_count", 1)),
                "cluster_fraction": float(member.get("cluster_fraction", 0.0)),
                "represented_member_ids": member.get("represented_member_ids", [str(member["member_id"])]),
            }
            for member in selected_gefs_members
        ]
    else:
        variability_members = [
            {
                "label": label,
                "member_id": f"sigma_{label}",
                "member_source": "gefs_mean_spread_sigma",
                "member_u": float(sampled_gefs["mean_u10_mps"]) + sigma_u * float(sampled_gefs["spread_u10_mps"]),
                "member_v": float(sampled_gefs["mean_v10_mps"]) + sigma_v * float(sampled_gefs["spread_v10_mps"]),
                "selection_reason": "fallback_sigma_point",
                "selection_rank": index,
                "path": None,
                "member_weight": 1.0,
                "cluster_member_count": 1,
                "cluster_fraction": 1.0 / max(1, len(_sigma_scenarios())),
                "represented_member_ids": [f"sigma_{label}"],
            }
            for index, (sigma_u, sigma_v, label) in enumerate(_sigma_scenarios())
        ]

    def solve_variability_member(index: int, member: dict) -> dict:
        label = str(member["label"])
        member_u = float(member["member_u"])
        member_v = float(member["member_v"])
        member_speed_mps, member_direction_deg = _uv_to_speed_dir(member_u, member_v)
        member_direction_safe_deg = member_direction_deg if math.isfinite(member_direction_deg) else 0.0
        safe_label = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in label)
        member_dir = ensemble_dir / f"member_{index:02d}_{safe_label}"
        member_ascii: dict[str, Path] | None = None
        if not math.isfinite(member_speed_mps) or member_speed_mps < CALM_WIND_SOLVER_THRESHOLD_MPS:
            speed_grid = np.zeros(deterministic_grid_shape, dtype=np.float32)
            direction_grid_deg = np.full(deterministic_grid_shape, member_direction_safe_deg % 360.0, dtype=np.float32)
            member_header = deterministic_speed_header
            solver_run_metadata = {
                "role": f"member_{index:02d}",
                "solver": "calm_direct",
                "solver_mode": "calm_guard",
                "status": "skipped",
                "threshold_mps": CALM_WIND_SOLVER_THRESHOLD_MPS,
                "wind_speed_mps": member_speed_mps,
                "wind_from_direction_deg": member_direction_safe_deg,
            }
        else:
            member_run = run_windninja_solver(
                output_dir=member_dir,
                wind_speed_mps=member_speed_mps,
                wind_direction_deg=member_direction_deg,
            )
            solver_run_metadata = {"role": f"member_{index:02d}", **member_run}
            member_ascii = expected_windninja_ascii_paths(
                elevation_tif=wind_input_tif,
                wind_speed_mps=member_speed_mps,
                wind_direction_deg=member_direction_deg,
                mesh_resolution_m=mesh_resolution_m,
                output_dir=member_dir,
            )
            speed_grid, member_header = _read_aaigrid(member_ascii["speed"])
            direction_grid_deg, _ = _read_aaigrid(member_ascii["direction"])
        speed_finite_cells = int(np.isfinite(speed_grid).sum())
        direction_finite_cells = int(np.isfinite(direction_grid_deg).sum())
        record = {
            "member": index,
            "scenario": label,
            "member_id": member["member_id"],
            "member_source": member["member_source"],
            "selection_rank": member.get("selection_rank"),
            "selection_reason": member.get("selection_reason"),
            "member_weight": float(member.get("member_weight", 1.0)),
            "cluster_member_count": int(member.get("cluster_member_count", 1)),
            "cluster_fraction": float(member.get("cluster_fraction", 0.0)),
            "represented_member_ids": member.get("represented_member_ids", [str(member["member_id"])]),
            "member_path": member.get("path"),
            "wind_speed_mps": member_speed_mps,
            "wind_speed_kts": member_speed_mps * 1.94384449,
            "wind_from_direction_deg": member_direction_safe_deg,
            "speed_finite_cells": speed_finite_cells,
            "direction_finite_cells": direction_finite_cells,
            "speed_output": None if member_ascii is None else str(member_ascii["speed"]),
            "direction_output": None if member_ascii is None else str(member_ascii["direction"]),
            "status": "skipped_calm" if member_ascii is None else "completed",
        }
        return {
            "solver_metadata": solver_run_metadata,
            "speed_grid": speed_grid,
            "direction_grid_deg": direction_grid_deg,
            "member_header": member_header,
            "record": record,
        }

    member_tasks = [
        (
            str(member["label"]),
            lambda index=index, member=member: solve_variability_member(index, member),
        )
        for index, member in enumerate(variability_members)
    ]
    member_worker_count = min(_windninja_member_workers(), len(member_tasks))
    for member_name, result, exc in _run_ordered_tasks(member_tasks, max_workers=member_worker_count):
        if exc is not None or result is None:
            raise RuntimeError(f"WindNinja GEFS variability member `{member_name}` failed: {exc!r}") from exc
        solver_metadata.append(result["solver_metadata"])
        speed_grid = result["speed_grid"]
        direction_grid_deg = result["direction_grid_deg"]
        member_header = result["member_header"]
        if speed_header is None:
            speed_header = member_header
        speed_members.append(speed_grid)
        direction_members_deg.append(direction_grid_deg)
        member_records.append(result["record"])

    speed_stack = np.stack(speed_members, axis=0).astype(np.float32)
    direction_stack_deg = np.stack(direction_members_deg, axis=0).astype(np.float32)
    if not np.isfinite(speed_stack).any():
        raise RuntimeError(f"{solver_display} GEFS variability ensemble produced no finite speed cells: {member_records}")
    if not np.isfinite(direction_stack_deg).any():
        raise RuntimeError(f"{solver_display} GEFS variability ensemble produced no finite direction cells: {member_records}")
    member_weights = np.array([max(0.0, float(record.get("member_weight", 1.0))) for record in member_records], dtype=np.float64)
    if not np.any(member_weights > 0.0):
        member_weights = np.ones(len(member_records), dtype=np.float64)
    speed_std_kts = (_weighted_nanstd(speed_stack, member_weights) * 1.94384449).astype(np.float32)
    direction_std_deg = _weighted_circular_std_deg(direction_stack_deg, member_weights).astype(np.float32)
    if not np.isfinite(speed_std_kts).any():
        raise RuntimeError(f"Wind speed spread grid has no finite cells after ensemble reduction: {member_records}")
    if not np.isfinite(direction_std_deg).any():
        raise RuntimeError(f"Wind direction spread grid has no finite cells after ensemble reduction: {member_records}")

    if speed_header is None:
        raise RuntimeError("GEFS variability ensemble did not produce any members.")

    product1_whisker_label = (
        "faint whiskers show real GEFS member dir SD"
        if use_real_gefs_members
        else "faint whiskers show GEFS dir-spread sensitivity"
    )
    write_windninja_knots_vector_preview_from_speed_angle(
        ascii_paths["speed"],
        ascii_paths["direction"],
        product1_png,
        dem_basemap_tif=domain.dem_preview_tif,
        source_header=deterministic_speed_header,
        vector_stride=vector_stride,
        vector_scale=vector_scale,
        colormap=diverging_blue_green_red_colormap(),
        center_value=float(np.nanmean(speed_kts)),
        title="wind",
        units="knots",
        footer_text=f"{_footer_timestamp_text(boundary_target_time_utc, 'wind forecast')} | {product1_whisker_label}",
        inset_lines=inset_lines,
        direction_uncertainty_deg=direction_std_deg,
        direction_uncertainty_stride_multiplier=1,
        bottom_table_rows=bottom_table_rows,
    )

    solve_speed_std_tif = temp_dir / "wind_speed_variance_knots_solve.tif"
    solve_direction_std_tif = temp_dir / "wind_direction_variance_degrees_solve.tif"
    speed_std_tif = temp_dir / "wind_speed_variance_knots.tif"
    direction_std_tif = temp_dir / "wind_direction_variance_degrees.tif"
    write_array_to_geotiff_from_header(speed_header, speed_std_kts, solve_speed_std_tif)
    write_array_to_geotiff_from_header(speed_header, direction_std_deg, solve_direction_std_tif)
    write_reprojected_array_like_reference(
        speed_std_kts,
        source_header=speed_header,
        reference_tif=domain.clipped_dem_tif,
        destination_tif=speed_std_tif,
        resampling=Resampling.bilinear,
    )
    write_reprojected_array_like_reference(
        direction_std_deg,
        source_header=speed_header,
        reference_tif=domain.clipped_dem_tif,
        destination_tif=direction_std_tif,
        resampling=Resampling.bilinear,
    )

    product2_png = report_dir / "product_2_wind_speed_variance_knots.png"
    product3_png = report_dir / "product_3_wind_direction_variance_degrees.png"
    _progress(progress_callback, 58, "Rendering wind maps...")
    colormap = diverging_blue_green_red_colormap()
    speed_spread_footer_label = "wind speed ensemble SD" if use_real_gefs_members else "wind speed SD sensitivity"
    direction_spread_footer_label = "wind dir ensemble SD" if use_real_gefs_members else "wind dir SD sensitivity"
    try:
        write_scalar_diagnostic_preview(
            field=speed_std_kts,
            dem_basemap_tif=domain.dem_preview_tif,
            output_png=product2_png,
            title="sd wind",
            units="knots",
            colormap=colormap,
            source_header=speed_header,
            alpha=0.58,
            signed=False,
            center_value=float(np.nanmean(speed_std_kts)),
            footer_text=_footer_timestamp_text(boundary_target_time_utc, speed_spread_footer_label),
        )
        write_scalar_diagnostic_preview(
            field=direction_std_deg,
            dem_basemap_tif=domain.dem_preview_tif,
            output_png=product3_png,
            title="sd az",
            units="deg",
            colormap=colormap,
            source_header=speed_header,
            alpha=0.58,
            signed=False,
            center_value=float(np.nanmean(direction_std_deg)),
            footer_text=_footer_timestamp_text(boundary_target_time_utc, direction_spread_footer_label),
        )
    except ValueError as exc:
        diagnostics = {
            "error": str(exc),
            "speed_std_knots": _array_finite_summary("speed_std_knots", speed_std_kts),
            "direction_std_deg": _array_finite_summary("direction_std_deg", direction_std_deg),
            "source_header": speed_header,
            "dem_basemap_tif": str(domain.dem_preview_tif),
            "member_records": member_records,
        }
        raise RuntimeError(f"Unable to render WindNinja spread products: {json.dumps(diagnostics, indent=2)}") from exc
    return {
        "boundary": boundary,
        "boundary_consensus": consensus.as_dict(),
        "model_forecasts": model_forecasts,
        "model_errors": model_errors,
        "skill_adjustments": skill_adjustments,
        "skill_metadata": skill_metadata,
        "gefs_boundary": sampled_gefs,
        "wind_data_mode": wind_note,
        "requested_wind_solver": wind_solver,
        "wind_solver": "windninja",
        "wind_solver_display": solver_display,
        "openfoam_comparison": openfoam_comparison
        if openfoam_comparison is not None
        else {
            "enabled": False,
            "status": "not_requested",
        },
        "solver_runs": solver_metadata,
        "wind_input_tif": str(wind_input_tif),
        "final_dem_tif": str(domain.clipped_dem_tif),
        "solve_dem_tif": str(domain.solve_dem_tif),
        "boundary_target_time_utc": boundary_target_time_utc.isoformat(),
        "mesh_resolution_m": mesh_resolution_m,
        "available_model_sources": available_sources,
        "missing_model_sources": missing_sources,
        "degraded": bool(wind_warnings),
        "warnings": wind_warnings,
        "member_count": len(member_records),
        "members": member_records,
        "product_1": str(product1_png),
        "product_2": str(product2_png),
        "product_3": str(product3_png),
        "sailing_polar_overlay": sailing_polar,
        "speed_std_knots_mean": float(np.nanmean(speed_std_kts)),
        "direction_std_deg_mean": float(np.nanmean(direction_std_deg)),
        "weather_inset": weather_inset,
        "gefs_members_available_count": len(sampled_gefs_members),
        "gefs_members_selected_count": len(selected_gefs_members) if use_real_gefs_members else 0,
        "gefs_member_solve_limit": _gefs_member_solve_limit(),
        "gefs_member_download_limit": _gefs_member_download_limit(),
        "gefs_member_download_workers": _env_int("PONDWIND_GEFS_MEMBER_DOWNLOAD_WORKERS", 3, minimum=1),
        "windninja_member_workers": member_worker_count,
        "model_download_workers": _model_download_workers(),
        "gefs_member_selection_method": "weighted_uv_cluster_medoids" if use_real_gefs_members else "fallback_mean_spread_sigma_points",
        "gefs_member_weight_sum": float(np.sum(member_weights)),
        "spread_product_label": (
            "weighted real GEFS member standard deviation"
            if use_real_gefs_members
            else "GEFS component-spread standard-deviation sensitivity"
        ),
        "spread_product_note": (
            (
                "Products 2 and 3, plus the faint product-1 direction whiskers, are standard-deviation maps "
                f"from {len(selected_gefs_members)} weighted cluster-medoid GEFS member boundary winds representing "
                f"{len(sampled_gefs_members)} available members. They are downscaled with WindNinja and "
                "are still relative pond-scale ensemble guidance, not calibrated probabilities."
            )
            if use_real_gefs_members
            else (
                "Products 2 and 3, plus the faint product-1 direction whiskers, are standard-deviation sensitivity "
                "maps from five synthetic GEFS u/v component-spread perturbations because real GEFS members were "
                "not available. They are not calibrated probabilistic variance maps or raw GEFS member probabilities."
            )
        ),
    }


def _build_satellite_products(
    report_dir: Path,
    temp_dir: Path,
    domain: PreparedSiteDomain,
    satellite_inputs: dict,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict:
    bbox = satellite_inputs["render_bbox"]
    product_options = _satellite_product_options(**satellite_inputs.get("product_options", {}))
    dem_basemap_tif = domain.dem_preview_tif
    sentinel_rgb = satellite_inputs["sentinel_rgb"]
    sentinel_rgb_paths = satellite_inputs["sentinel_rgb_paths"]
    sentinel = satellite_inputs["sentinel"]
    sentinel_paths = satellite_inputs["sentinel_paths"]
    landsat = satellite_inputs["landsat"]
    landsat_paths = satellite_inputs["landsat_paths"]
    ecostress = satellite_inputs["ecostress"]
    ecostress_paths = satellite_inputs["ecostress_paths"]
    selection_diagnostics = satellite_inputs["selection_diagnostics"]
    _progress(progress_callback, 76, "Rendering satellite products...")

    rgb_png = report_dir / "satellite_rgb_latest.png"
    sentinel_rgb_selection = None
    if product_options["rgb"] and sentinel_rgb is not None and sentinel_rgb_paths is not None:
        sentinel_rgb_time = datetime.fromisoformat(sentinel_rgb.item_datetime_utc.replace("Z", "+00:00"))
        render_rgb_preview(
            sentinel_rgb_paths["visual"],
            bbox,
            rgb_png,
            dem_basemap_tif=dem_basemap_tif,
            title="rgb",
            footer_text=_footer_timestamp_text(sentinel_rgb_time, "rgb collected"),
        )
        sentinel_rgb_selection = {
            "item_id": sentinel_rgb.item_id,
            "datetime_utc": sentinel_rgb.item_datetime_utc,
            "cloud_cover": sentinel_rgb.cloud_cover,
        }
    elif product_options["rgb"]:
        reason = selection_diagnostics.get("search_errors", {}).get("sentinel_rgb_search") or "No usable Sentinel-2 RGB scene was available."
        _write_unavailable_panel(
            rgb_png,
            title="rgb",
            line1="N/A - satellite unavailable",
            line2=reason,
        )

    chla_png = report_dir / "satellite_chla_estimated.png"
    turbidity_png = report_dir / "satellite_turbidity_estimated.png"
    sst_png = report_dir / "satellite_sst_latest.png"

    sentinel_selection = None
    if product_options["chla"] and sentinel is not None and sentinel_paths is not None:
        sentinel_time = datetime.fromisoformat(sentinel.item_datetime_utc.replace("Z", "+00:00"))
        chla_tif = temp_dir / "satellite_chla_estimated.tif"
        chla_summary = derive_sentinel_chla(
            red_tif=sentinel_paths["red"],
            rededge1_tif=sentinel_paths["rededge1"],
            scl_tif=sentinel_paths["scl"],
            bbox=bbox,
            red_scale=sentinel.assets["red"].scale,
            red_offset=sentinel.assets["red"].offset,
            red_nodata=sentinel.assets["red"].nodata,
            rededge_scale=sentinel.assets["rededge1"].scale,
            rededge_offset=sentinel.assets["rededge1"].offset,
            rededge_nodata=sentinel.assets["rededge1"].nodata,
            output_tif=chla_tif,
            output_png=chla_png,
            dem_basemap_tif=dem_basemap_tif,
            footer_text=_footer_timestamp_text(sentinel_time, "chl a collected"),
        )
    elif product_options["chla"]:
        reason = selection_diagnostics.get("sentinel_unavailable_reason") or "Insufficient water coverage in the selected area."
        _write_unavailable_panel(
            chla_png,
            title="chl a",
            line1="N/A - insufficient water coverage",
            line2=reason,
        )
        chla_summary = {
            "product": "estimated_chlorophyll_a",
            "status": "unavailable",
            "reason": reason,
            "output_png": str(chla_png),
        }
    else:
        chla_summary = _skipped_satellite_summary("estimated_chlorophyll_a")

    if product_options["turbidity"] and sentinel is not None and sentinel_paths is not None:
        sentinel_time = datetime.fromisoformat(sentinel.item_datetime_utc.replace("Z", "+00:00"))
        turbidity_tif = temp_dir / "satellite_turbidity_estimated.tif"
        turbidity_summary = derive_sentinel_turbidity(
            red_tif=sentinel_paths["red"],
            nir_tif=sentinel_paths["nir"],
            scl_tif=sentinel_paths["scl"],
            bbox=bbox,
            red_scale=sentinel.assets["red"].scale,
            red_offset=sentinel.assets["red"].offset,
            red_nodata=sentinel.assets["red"].nodata,
            nir_scale=sentinel.assets["nir"].scale,
            nir_offset=sentinel.assets["nir"].offset,
            nir_nodata=sentinel.assets["nir"].nodata,
            output_tif=turbidity_tif,
            output_png=turbidity_png,
            dem_basemap_tif=dem_basemap_tif,
            footer_text=_footer_timestamp_text(sentinel_time, "turbidity collected"),
        )
    elif product_options["turbidity"]:
        reason = selection_diagnostics.get("sentinel_unavailable_reason") or "Insufficient water coverage in the selected area."
        _write_unavailable_panel(
            turbidity_png,
            title="turbidity",
            line1="N/A - insufficient water coverage",
            line2=reason,
        )
        turbidity_summary = {
            "product": "estimated_turbidity",
            "status": "unavailable",
            "reason": reason,
            "output_png": str(turbidity_png),
        }
    else:
        turbidity_summary = _skipped_satellite_summary("estimated_turbidity")

    if sentinel is not None and sentinel_paths is not None and (product_options["chla"] or product_options["turbidity"]):
        sentinel_selection = {
            "item_id": sentinel.item_id,
            "datetime_utc": sentinel.item_datetime_utc,
            "cloud_cover": sentinel.cloud_cover,
        }

    sst_selection = None
    sst_summary = None
    sst_derivation_errors: list[dict[str, str]] = []
    if not product_options["sst"]:
        sst_summary = _skipped_satellite_summary("surface_temperature_over_water")
    elif ecostress is not None and ecostress_paths is not None:
        ecostress_time = datetime.fromisoformat(ecostress.item_datetime_utc.replace("Z", "+00:00"))
        sst_tif = temp_dir / "satellite_sst_latest.tif"
        try:
            sst_summary = derive_ecostress_sst(
                lst_tif=ecostress_paths["lst"],
                cloud_tif=ecostress_paths["cloud"],
                water_tif=ecostress_paths["water"],
                bbox=bbox,
                output_tif=sst_tif,
                output_png=sst_png,
                dem_basemap_tif=dem_basemap_tif,
                footer_text=_footer_timestamp_text(ecostress_time, "surface temp collected"),
            )
            sst_selection = {
                "source": "ecostress",
                "item_id": ecostress.item_id,
                "datetime_utc": ecostress.item_datetime_utc,
                "cloud_cover": None,
            }
        except Exception as exc:
            sst_derivation_errors.append({"source": "ecostress", "error": repr(exc)})

    if product_options["sst"] and sst_summary is None and landsat is not None and landsat_paths is not None:
        landsat_time = datetime.fromisoformat(landsat.item_datetime_utc.replace("Z", "+00:00"))
        sst_tif = temp_dir / "satellite_sst_latest.tif"
        try:
            sst_summary = derive_landsat_sst(
                lwir11_tif=landsat_paths["lwir11"],
                qa_pixel_tif=landsat_paths["qa_pixel"],
                bbox=bbox,
                lwir_scale=landsat.assets["lwir11"].scale,
                lwir_offset=landsat.assets["lwir11"].offset,
                lwir_nodata=landsat.assets["lwir11"].nodata,
                output_tif=sst_tif,
                output_png=sst_png,
                dem_basemap_tif=dem_basemap_tif,
                footer_text=_footer_timestamp_text(landsat_time, "surface temp collected"),
            )
            sst_selection = {
                "source": "landsat",
                "item_id": landsat.item_id,
                "datetime_utc": landsat.item_datetime_utc,
                "cloud_cover": landsat.cloud_cover,
            }
        except Exception as exc:
            sst_derivation_errors.append({"source": "landsat", "error": repr(exc)})

    if product_options["sst"] and sst_summary is None:
        reason = (
            (sst_derivation_errors[-1]["error"] if sst_derivation_errors else None)
            or selection_diagnostics.get("ecostress_unavailable_reason")
            or selection_diagnostics.get("landsat_unavailable_reason")
            or "Insufficient water coverage in the selected area."
        )
        _write_unavailable_panel(
            sst_png,
            title="surf t",
            line1="N/A - insufficient water coverage",
            line2=reason,
        )
        sst_summary = {
            "product": "surface_temperature_over_water",
            "legacy_product": "sst",
            "status": "unavailable",
            "reason": reason,
            "output_png": str(sst_png),
        }
        sst_selection = None

    selection_diagnostics["sst_derivation_errors"] = sst_derivation_errors

    return {
        "selected_products": product_options,
        "sentinel_rgb_selection": sentinel_rgb_selection,
        "sentinel_selection": sentinel_selection,
        "sst_selection": sst_selection,
        "selection_diagnostics": selection_diagnostics,
        "rgb_png": str(rgb_png) if product_options["rgb"] else None,
        "chl_a": chla_summary,
        "turbidity": turbidity_summary,
        "sst": sst_summary,
    }


def _write_markdown_report(report_dir: Path, race_time_local: datetime, site: SiteConfig, wind: dict, satellite: dict) -> Path:
    sentinel_selection = satellite.get("sentinel_selection")
    sst_selection = satellite.get("sst_selection")
    chl_a = satellite["chl_a"]
    turbidity = satellite["turbidity"]
    sst = satellite["sst"]
    satellite_options = _satellite_product_options(**satellite.get("selected_products", {}))
    satellite_lines: list[str] = ["", "## Satellite products"]
    if satellite_options["rgb"]:
        rgb_selection = satellite.get("sentinel_rgb_selection")
        satellite_lines.extend(
            [
                (
                    f"Sentinel-2 RGB source: `{rgb_selection['item_id']}` at `{rgb_selection['datetime_utc']}`"
                    if rgb_selection is not None
                    else "Sentinel-2 RGB: unavailable"
                ),
                "",
                f"![Latest RGB]({_app_path(satellite['rgb_png'])})",
                "",
            ]
        )
    if satellite_options["chla"]:
        satellite_lines.extend(
            [
                (
                    f"Experimental chlorophyll-a index source: `{sentinel_selection['item_id']}`"
                    if sentinel_selection is not None
                    else f"Experimental chlorophyll-a index: unavailable ({chl_a.get('reason', 'insufficient water coverage')})"
                ),
                "",
                f"![Experimental chlorophyll-a index]({_app_path(chl_a['output_png'])})",
                "",
            ]
        )
    if satellite_options["turbidity"]:
        satellite_lines.extend(
            [
                (
                    f"Experimental turbidity index source: `{sentinel_selection['item_id']}`"
                    if sentinel_selection is not None
                    else f"Experimental turbidity index: unavailable ({turbidity.get('reason', 'insufficient water coverage')})"
                ),
                "",
                f"![Experimental turbidity index]({_app_path(turbidity['output_png'])})",
                "",
            ]
        )
    if satellite_options["sst"]:
        sst_coverage_note = ""
        if sst.get("status") == "limited":
            sst_coverage_note = f" (limited coverage: {sst.get('usable_pixel_count', 0)} usable pixels)"
        satellite_lines.extend(
            [
                (
                    f"{sst_selection['source'].upper()} surface-temperature-over-water source: `{sst_selection['item_id']}` at `{sst_selection['datetime_utc']}`{sst_coverage_note}"
                    if sst_selection is not None
                    else f"Surface temperature over water: unavailable ({sst.get('reason', 'insufficient water coverage')})"
                ),
                "",
                f"![Satellite surface temperature over water]({_app_path(sst['output_png'])})",
                "",
            ]
        )
    if not any(satellite_options.values()):
        satellite_lines.extend(["No satellite products were selected for this report.", ""])

    note_lines = [
        f"- {wind.get('spread_product_note', 'Wind ensemble products are standard-deviation maps, not calibrated probability maps.')}",
        f"- Wind data mode for this run: `{wind['wind_data_mode']}`.",
    ]
    if wind.get("warnings"):
        note_lines.extend(f"- Wind warning: {warning}" for warning in wind["warnings"])
    if satellite_options["chla"]:
        note_lines.append("- Chlorophyll-a is an experimental satellite index from Sentinel-2 L2A reflectance, not an in situ or locally validated aquatic retrieval.")
    if satellite_options["turbidity"]:
        note_lines.append("- Turbidity is an experimental Sentinel-2 red/NIR index, not an in situ or locally validated aquatic retrieval.")
    if satellite_options["sst"]:
        note_lines.append("- Surface temperature over water comes from Landsat/ECOSTRESS land-surface-temperature products masked to water pixels; it is not a direct in-water thermometer.")
    if any(satellite_options.values()):
        note_lines.append("- Cloudy scenes are allowed; this report uses the best recent scene available before race time.")
    if wind.get("sailing_polar_overlay", {}).get("product_png"):
        note_lines.append(
            "- Sailing polar overlays are experimental single-point ILCA relative estimates and should not be treated as calibrated boat-speed predictions."
        )

    openfoam_comparison = wind.get("openfoam_comparison", {"enabled": False})
    sailing_polar = wind.get("sailing_polar_overlay", {})
    openfoam_lines: list[str] = []
    if openfoam_comparison.get("enabled"):
        turbulence_png = openfoam_comparison.get("turbulence_png")
        openfoam_sailing_polar_png = openfoam_comparison.get("sailing_polar_overlay_png")
        openfoam_label = openfoam_comparison.get("product_label", "Experimental CFD comparison")
        openfoam_lines = [
            "",
            f"## {openfoam_label}",
            f"Status: `{openfoam_comparison.get('status', 'unknown')}`",
            "",
            f"![{openfoam_label}]({_app_path(openfoam_comparison['product_png'])})",
            "",
            *(
                [
                    "Experimental CFD sailing polar overlay:",
                    "",
                    f"![Experimental CFD sailing polar overlay]({_app_path(openfoam_sailing_polar_png)})",
                    "",
                ]
                if openfoam_sailing_polar_png
                else []
            ),
            *(
                [
                    "Experimental CFD turbulence intensity:",
                    "",
                    f"![Experimental CFD turbulence intensity]({_app_path(turbulence_png)})",
                    "",
                ]
                if turbulence_png
                else []
            ),
            openfoam_comparison.get(
                "scientific_note",
                "This product is experimental and is shown for comparison only; product 1 remains the WindNinja production wind prediction.",
            ),
            "Product 1 remains the WindNinja production wind prediction.",
            "",
        ]
    report_path = report_dir / "weekly_report.md"
    text = "\n".join(
        [
            f"# {site.display_name()} Weekly Report",
            "",
            f"Race time: {race_time_local.strftime('%Y-%m-%d %I:%M %p %Z')}",
            f"Center: `{site.center_lat:.6f}, {site.center_lon:.6f}`",
            f"Area: `{site.side_meters:.1f} m x {site.side_meters:.1f} m`",
            "",
            "## Wind products",
            f"Deterministic boundary source: `{wind['boundary_consensus']['selected_source']}`",
            f"Boundary selection note: {wind['boundary_consensus']['selected_reason']}",
            "",
            f"![Wind prediction]({_app_path(wind['product_1'])})",
            "",
            *(
                [
                    "Experimental single-point ILCA relative polar over the cropped DEM:",
                    "",
                    f"![Experimental single-point ILCA relative polar overlay]({_app_path(sailing_polar['product_png'])})",
                    "",
                ]
                if sailing_polar.get("product_png")
                else []
            ),
            f"![Wind speed ensemble SD]({_app_path(wind['product_2'])})",
            "",
            f"![Wind direction ensemble SD]({_app_path(wind['product_3'])})",
            *openfoam_lines,
            *satellite_lines,
            "",
            "## Notes",
            *note_lines,
        ]
    )
    report_path.write_text(text, encoding="utf-8")
    return report_path


def build_weekly_report(
    *,
    race_local_datetime: str | None = None,
    center_lat: float = SiteConfig.center_lat,
    center_lon: float = SiteConfig.center_lon,
    side_meters: float = SiteConfig.side_meters,
    site_label: str = SiteConfig.label,
    mesh_resolution: float = 30.0,
    wind_solver: str = "windninja",
    solve_buffer_m: float = SiteConfig.solve_buffer_m,
    report_output_dir: str | Path | None = None,
    allow_insecure_ssl: bool = False,
    force_ecostress_sst: bool = False,
    satellite_rgb: bool = True,
    satellite_sst: bool = True,
    satellite_chla: bool = True,
    satellite_turbidity: bool = True,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[Path, Path]:
    if allow_insecure_ssl:
        import os

        os.environ["PREDICTWEATHER_ALLOW_INSECURE_SSL"] = "1"

    site = SiteConfig(
        center_lat=center_lat,
        center_lon=center_lon,
        side_meters=side_meters,
        solve_buffer_m=solve_buffer_m,
        label=site_label,
    )
    race_time_local = _parse_race_local_datetime(race_local_datetime)
    race_time_utc = race_time_local.astimezone(timezone.utc)
    satellite_products = _satellite_product_options(
        rgb=satellite_rgb,
        sst=satellite_sst,
        chla=satellite_chla,
        turbidity=satellite_turbidity,
    )
    report_root = _resolve_report_root(report_output_dir)
    report_dir = report_root / _report_dir_name(race_time_local, site)
    report_dir.mkdir(parents=True, exist_ok=True)
    removed_previous_outputs = _clean_report_outputs(report_dir)
    temp_dir = report_dir / "report_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    working_data_dir = report_root / "PondWind Working Data"
    raw_data_dir = working_data_dir / "downloads"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    _progress(progress_callback, 8, "Preparing terrain domain...")
    domain = prepare_site_domain(
        site=site,
        working_dir=temp_dir,
        raw_data_dir=raw_data_dir,
        processed_data_dir=working_data_dir / "processed",
    )
    satellite_inputs = _download_satellite_inputs(
        temp_dir,
        race_time_utc,
        domain,
        force_ecostress_sst=force_ecostress_sst,
        satellite_products=satellite_products,
        progress_callback=progress_callback,
    )
    landscape_tif, landscape_summary = _prepare_landscape_input(temp_dir, domain, satellite_inputs, progress_callback)
    wind_summary = _build_wind_products(
        report_dir,
        temp_dir,
        race_time_utc,
        site,
        domain,
        landscape_tif,
        mesh_resolution,
        wind_solver,
        progress_callback,
        raw_data_dir,
    )
    satellite_summary = _build_satellite_products(report_dir, temp_dir, domain, satellite_inputs, progress_callback)
    _progress(progress_callback, 90, "Writing report outputs...")
    report_path = _write_markdown_report(report_dir, race_time_local, site, wind_summary, satellite_summary)
    moved_intermediates = _tidy_report_root(report_dir, temp_dir)

    manifest = {
        "site": {
            "label": site.label,
            "slug": site.slug(),
            "center_lat": site.center_lat,
            "center_lon": site.center_lon,
            "side_meters": site.side_meters,
            "solve_buffer_m": site.solve_buffer_m,
        },
        "race_time_local": race_time_local.isoformat(),
        "race_time_utc": race_time_utc.isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_dir": str(report_dir),
        "report_root": str(report_root),
        "report_markdown": str(report_path),
        "working_data": {
            "directory": str(working_data_dir),
            "downloads_dir": str(raw_data_dir),
            "note": "Shared downloads are kept beside the reports so later runs can reuse them.",
        },
        "domain": {
            "dataset": domain.dataset,
            "bbox": {
                "min_lon": domain.bbox.min_lon,
                "min_lat": domain.bbox.min_lat,
                "max_lon": domain.bbox.max_lon,
                "max_lat": domain.bbox.max_lat,
            },
            "solve_bbox": {
                "min_lon": domain.solve_bbox.min_lon,
                "min_lat": domain.solve_bbox.min_lat,
                "max_lon": domain.solve_bbox.max_lon,
                "max_lat": domain.solve_bbox.max_lat,
            },
            "source_dem_tif": str(domain.source_dem_tif),
            "raw_download_path": str(domain.raw_download_path),
            "final_dem_tif": str(domain.clipped_dem_tif),
            "final_dem_preview_tif": str(domain.dem_preview_tif),
            "solve_dem_tif": str(domain.solve_dem_tif),
            "solve_dem_preview_tif": str(domain.solve_dem_preview_tif),
            "domain_manifest_path": str(domain.manifest_path),
        },
        "landscape": landscape_summary,
        "cleanup": {
            "mode": "retain_in_report_temp",
            "report_temp_dir": str(temp_dir),
            "note": "Intermediate files are retained in report_temp for easier reruns and manual deletion.",
            "removed_previous_outputs": removed_previous_outputs,
            "moved_paths": moved_intermediates,
        },
        "wind": wind_summary,
        "satellite": satellite_summary,
    }
    manifest_path = report_dir / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _progress(progress_callback, 98, "Finalizing report...")
    return report_path, manifest_path


def main() -> None:
    args = parse_args()
    report_path, manifest_path = build_weekly_report(
        race_local_datetime=args.race_local_datetime,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        side_meters=args.side_meters,
        site_label=args.site_label,
        mesh_resolution=args.mesh_resolution,
        wind_solver=args.wind_solver,
        solve_buffer_m=args.solve_buffer_m,
        report_output_dir=args.report_output_dir,
        allow_insecure_ssl=args.allow_insecure_ssl,
        force_ecostress_sst=args.force_ecostress_sst,
        satellite_rgb=args.satellite_rgb,
        satellite_sst=args.satellite_sst,
        satellite_chla=args.satellite_chla,
        satellite_turbidity=args.satellite_turbidity,
    )
    print(report_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
