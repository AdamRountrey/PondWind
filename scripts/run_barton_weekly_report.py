from __future__ import annotations

import argparse
import json
import math
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
from predictweather.config import DATA_RAW_DIR, OUTPUTS_DIR, PROJECT_ROOT as RUNTIME_PROJECT_ROOT, RESOURCE_ROOT as RUNTIME_RESOURCE_ROOT, SiteConfig
from predictweather.ecmwf import download_ecmwf_for_valid_time, sample_ecmwf_point_forecast
from predictweather.forecast_models import choose_consensus_boundary, model_table_rows
from predictweather.gefs import (
    download_gefs_mean_and_spread,
    floor_to_3h,
    load_cached_gefs_manifest_for_valid_time,
    sample_gefs_mean_and_spread_at_site,
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
from predictweather.openfoam import run_openfoam_domain_average
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
    write_windninja_knots_vector_preview_from_speed_angle,
    diverging_blue_green_red_colormap,
)
import rasterio
from rasterio.enums import Resampling
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
    parser.add_argument("--wind-solver", choices=("windninja", "openfoam"), default="windninja", help="Terrain wind solver. OpenFOAM is experimental and requires PONDWIND_OPENFOAM_RUNNER.")
    parser.add_argument("--solve-buffer-m", type=float, default=SiteConfig.solve_buffer_m, help="Extra buffer around the final area for WindNinja solves.")
    parser.add_argument("--report-output-dir", default=None, help="Optional directory where the report folder should be created. Defaults to outputs/reports.")
    parser.add_argument("--allow-insecure-ssl", action="store_true")
    parser.add_argument("--force-ecostress-sst", action="store_true", help="Force ECOSTRESS SST discovery even when Landsat is recent enough.")
    return parser.parse_args()


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


def _aligned_boundary_valid_time(race_time_utc: datetime) -> datetime:
    return floor_to_3h(race_time_utc)


def _format_optional_number(value: float | None, suffix: str) -> str:
    if value is None:
        return "n/a"
    return f"{int(round(value))}{suffix}"


def _load_fresh_or_cached_hrdps_boundary(site: SiteConfig, target_valid_time_utc: datetime) -> tuple[dict, str]:
    fresh_error: Exception | None = None
    try:
        hrdps_selection = download_hrdps_for_valid_time(DATA_RAW_DIR, target_valid_time_utc)
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
        cached_manifest, cached_manifest_path = load_cached_hrdps_manifest_for_valid_time(DATA_RAW_DIR / "hrdps", target_valid_time_utc)
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


def _load_fresh_or_cached_gefs_paths(target_valid_time_utc: datetime) -> tuple[Path, Path, str]:
    fresh_error: Exception | None = None
    try:
        gefs_selection = download_gefs_mean_and_spread(DATA_RAW_DIR, target_valid_time_utc)
        return Path(gefs_selection.mean_files["geavg"]), Path(gefs_selection.spread_files["gespr"]), "fresh_gefs"
    except Exception as exc:
        fresh_error = exc

    try:
        cached_manifest, _ = load_cached_gefs_manifest_for_valid_time(DATA_RAW_DIR / "gefs", target_valid_time_utc)
        return Path(cached_manifest["mean_files"]["geavg"]), Path(cached_manifest["spread_files"]["gespr"]), "cached_gefs"
    except Exception as cached_exc:
        message = f"Unable to acquire GEFS boundary for {target_valid_time_utc.isoformat()}: fresh={fresh_error!r}; cached={cached_exc!r}"
        raise RuntimeError(message) from cached_exc


def _load_hrdps_point_forecast(lat: float, lon: float, target_valid_time_utc: datetime) -> dict:
    site = SiteConfig(center_lat=lat, center_lon=lon)
    boundary, acquisition_mode = _load_fresh_or_cached_hrdps_boundary(site, target_valid_time_utc)
    gust_summary = None
    run_at_utc = target_valid_time_utc.isoformat()
    forecast_hour = 0
    files = {
        "UGRD": boundary["ugrd_path"],
        "VGRD": boundary["vgrd_path"],
    }

    try:
        hrdps_selection = download_hrdps_for_valid_time(DATA_RAW_DIR, target_valid_time_utc)
        run_at_utc = hrdps_selection.run_at_utc
        forecast_hour = hrdps_selection.forecast_hour
        files = dict(hrdps_selection.files)
        gust_result = try_download_hrdps_variable(
            DATA_RAW_DIR,
            run_at_utc=datetime.fromisoformat(hrdps_selection.run_at_utc.replace("Z", "+00:00")),
            forecast_hour=hrdps_selection.forecast_hour,
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


def _load_model_point_forecasts(lat: float, lon: float, target_valid_time_utc: datetime) -> tuple[list[dict], dict[str, str]]:
    errors: dict[str, str] = {}
    forecasts: list[dict] = []

    loaders: list[tuple[str, Callable[[], object]]] = [
        ("hrdps", lambda: _load_hrdps_point_forecast(lat, lon, target_valid_time_utc)),
        ("hrrr", lambda: sample_hrrr_point_forecast(download_hrrr_for_valid_time(DATA_RAW_DIR, target_valid_time_utc, lat, lon), lat, lon, "fresh_hrrr")),
        ("gfs", lambda: sample_gfs_point_forecast(download_gfs_for_valid_time(DATA_RAW_DIR, target_valid_time_utc, lat, lon), lat, lon, "fresh_gfs")),
        ("nam", lambda: sample_nam_point_forecast(download_nam_for_valid_time(DATA_RAW_DIR, target_valid_time_utc, lat, lon), lat, lon, "fresh_nam")),
        ("icon", lambda: sample_icon_point_forecast(download_icon_for_valid_time(DATA_RAW_DIR, target_valid_time_utc), lat, lon, "fresh_icon")),
        ("ecmwf", lambda: sample_ecmwf_point_forecast(download_ecmwf_for_valid_time(DATA_RAW_DIR, target_valid_time_utc), lat, lon, "fresh_ecmwf")),
    ]
    for loader_name, result, exc in _run_ordered_tasks(loaders):
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
    sample_count: int = DEFAULT_SKILL_SAMPLE_COUNT,
) -> tuple[dict[str, float], dict]:
    station_candidates = nearest_station_candidates(DATA_RAW_DIR, site_lat, site_lon)
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
                DATA_RAW_DIR,
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
        model_forecasts, model_errors = _load_model_point_forecasts(observation.lat, observation.lon, valid_time_utc)
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


def _resolve_report_root(report_output_dir: str | Path | None) -> Path:
    if report_output_dir is None:
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
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict:
    _progress(progress_callback, 14, "Finding recent satellite imagery...")
    satellite_root = temp_dir / "satellite"
    satellite_root.mkdir(parents=True, exist_ok=True)
    selection_bbox = domain.bbox
    render_bbox = buffered_square_bbox_from_center(
        domain.site.center_lat,
        domain.site.center_lon,
        domain.site.side_meters,
        60.0,
    )

    sentinel_rgb_candidates = list_candidate_items(
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
            sentinel_rgb_paths = download_selection_assets(candidate, candidate_dir)
            sentinel_rgb_rank = index
            break
        sentinel_rgb_rejections.append(
            {
                "item_id": candidate.item_id,
                "rank": index,
                "clear_fraction": clear_fraction,
            }
        )
    if sentinel_rgb is None or sentinel_rgb_paths is None:
        sentinel_rgb = sentinel_rgb_candidates[0]
        sentinel_rgb_paths = download_selection_assets(sentinel_rgb, satellite_root / "sentinel_rgb_fallback")
        sentinel_rgb_rank = 0

    sentinel_candidates = list_candidate_items(
        collection="sentinel-2-l2a",
        bbox=render_bbox,
        end_time_utc=race_time_utc,
        lookback_days=45,
        required_assets=["red", "rededge1", "nir", "scl"],
        limit=48,
        max_cloud_cover=MAX_WATER_SCENE_CLOUD_COVER,
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
            sentinel_paths = download_selection_assets(candidate, candidate_dir)
            sentinel_rank = index
            break
        sentinel_rejections.append({"item_id": candidate.item_id, "rank": index, "water_pixels": water_pixels})
    if sentinel is None or sentinel_paths is None:
        sentinel_unavailable_reason = "No Sentinel-2 analytical scene had enough water pixels for the requested area."

    landsat_candidates = list_candidate_items(
        collection="landsat-c2-l2",
        bbox=render_bbox,
        end_time_utc=race_time_utc,
        lookback_days=60,
        required_assets=["lwir11", "qa_pixel"],
        limit=48,
        stac_url=PLANETARY_COMPUTER_STAC,
        max_cloud_cover=MAX_LANDSAT_SCENE_CLOUD_COVER,
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
        if water_pixels >= 25:
            landsat = candidate
            landsat_paths = download_selection_assets(candidate, candidate_dir)
            landsat_rank = index
            break
        landsat_rejections.append({"item_id": candidate.item_id, "rank": index, "water_pixels": water_pixels})
    if landsat is None or landsat_paths is None:
        landsat_unavailable_reason = "No Landsat SST scene had enough water pixels for the requested area."

    ecostress = None
    ecostress_paths: dict[str, Path] | None = None
    ecostress_rank = None
    ecostress_rejections: list[dict] = []
    ecostress_unavailable_reason = None
    landsat_age_days = None
    if landsat is not None:
        landsat_time = datetime.fromisoformat(landsat.item_datetime_utc.replace("Z", "+00:00"))
        landsat_age_days = max(0.0, (race_time_utc - landsat_time).total_seconds() / 86400.0)
    require_ecostress = force_ecostress_sst or landsat is None or landsat_age_days is None or landsat_age_days > LANDSAT_SST_MAX_AGE_DAYS
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
            if water_pixels >= 25:
                ecostress = candidate
                ecostress_paths = download_cmr_assets(candidate, candidate_dir)
                ecostress_rank = index
                break
            ecostress_rejections.append({"item_id": candidate.item_id, "rank": index, "water_pixels": water_pixels})
        if require_ecostress and ecostress is None and ecostress_unavailable_reason is None:
            ecostress_unavailable_reason = "No ECOSTRESS SST scene had enough water pixels for the requested area."
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
        "selection_diagnostics": {
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
    else:
        scl_tif = Path(satellite_inputs["sentinel_rgb_paths"]["scl"])

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
    _resample_scl_water_mask(scl_tif, domain_dem, water_mask_tif)

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
    return landscape_tif, summary.__dict__


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
) -> dict:
    if wind_solver not in {"windninja", "openfoam"}:
        raise ValueError(f"Unsupported wind solver: {wind_solver}")

    _progress(progress_callback, 20, "Building wind boundary conditions...")
    boundary_target_time_utc = _aligned_boundary_valid_time(race_time_utc)
    model_forecasts, model_errors = _load_model_point_forecasts(site.center_lat, site.center_lon, boundary_target_time_utc)
    if not model_forecasts:
        raise RuntimeError(f"No deterministic model forecasts were available for {boundary_target_time_utc.isoformat()}: {model_errors}")
    _progress(progress_callback, 24, "Scoring recent nearby model skill...")
    skill_adjustments, skill_metadata = _compute_station_skill_adjustments(
        target_valid_time_utc=boundary_target_time_utc,
        site_lat=site.center_lat,
        site_lon=site.center_lon,
    )

    gefs_mean_path, gefs_spread_path, gefs_mode = _load_fresh_or_cached_gefs_paths(boundary_target_time_utc)
    sampled_gefs = sample_gefs_mean_and_spread_at_site(
        mean_path=gefs_mean_path,
        spread_path=gefs_spread_path,
        lat=site.center_lat,
        lon=site.center_lon,
    )
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
    wind_note = f"{consensus.selected_source}_{gefs_mode}_spread"

    wind_root = temp_dir / "wind"
    wind_root.mkdir(parents=True, exist_ok=True)
    deterministic_dir = wind_root / "deterministic"
    solver_metadata: list[dict] = []

    def run_solver(output_dir: Path, wind_speed_mps: float, wind_direction_deg: float) -> dict:
        if wind_solver == "openfoam":
            return run_openfoam_domain_average(
                elevation_tif=wind_input_tif,
                output_dir=output_dir,
                wind_speed_mps=wind_speed_mps,
                wind_direction_deg=wind_direction_deg,
                mesh_resolution_m=mesh_resolution_m,
            )
        cli_path = windninja_cli_path(RUNTIME_RESOURCE_ROOT)
        return run_windninja_domain_average(
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

    solver_display = "OpenFOAM experimental" if wind_solver == "openfoam" else "WindNinja momentum"
    _progress(progress_callback, 32, f"Running terrain wind prediction with {solver_display}...")
    deterministic_run = run_solver(
        output_dir=deterministic_dir,
        wind_speed_mps=float(boundary["wind_speed_mps"]),
        wind_direction_deg=float(boundary["wind_from_direction_deg"]),
    )
    solver_metadata.append({"role": "deterministic", **deterministic_run})
    ascii_paths = expected_windninja_ascii_paths(
        elevation_tif=wind_input_tif,
        wind_speed_mps=float(boundary["wind_speed_mps"]),
        wind_direction_deg=float(boundary["wind_from_direction_deg"]),
        mesh_resolution_m=mesh_resolution_m,
        output_dir=deterministic_dir,
    )
    speed_mps, speed_header = _read_aaigrid(ascii_paths["speed"])
    speed_kts = speed_mps * 1.94384449
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

    product1_png = report_dir / "product_1_wind_speed_prediction_knots.png"
    write_windninja_knots_vector_preview_from_speed_angle(
        ascii_paths["speed"],
        ascii_paths["direction"],
        product1_png,
        dem_basemap_tif=domain.dem_preview_tif,
        source_header=speed_header,
        vector_stride=4,
        vector_scale=2.2,
        colormap=diverging_blue_green_red_colormap(),
        center_value=float(np.nanmean(speed_kts)),
        title="wind",
        units="knots",
        footer_text=_footer_timestamp_text(boundary_target_time_utc, "wind forecast"),
        inset_lines=inset_lines,
        bottom_table_rows=model_table_rows(
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
        ),
    )

    ensemble_dir = wind_root / "gefs_sigma"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    _progress(progress_callback, 45, f"Estimating wind variability with {solver_display}...")
    speed_members: list[np.ndarray] = []
    direction_members_deg: list[np.ndarray] = []
    speed_header: dict[str, float] | None = None
    member_records: list[dict] = []
    for index, (sigma_u, sigma_v, label) in enumerate(_sigma_scenarios()):
        member_u = float(sampled_gefs["mean_u10_mps"]) + sigma_u * float(sampled_gefs["spread_u10_mps"])
        member_v = float(sampled_gefs["mean_v10_mps"]) + sigma_v * float(sampled_gefs["spread_v10_mps"])
        member_speed_mps, member_direction_deg = _uv_to_speed_dir(member_u, member_v)
        member_dir = ensemble_dir / f"member_{index:02d}"
        member_run = run_solver(
            output_dir=member_dir,
            wind_speed_mps=member_speed_mps,
            wind_direction_deg=member_direction_deg,
        )
        solver_metadata.append({"role": f"member_{index:02d}", **member_run})
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
        if speed_header is None:
            speed_header = member_header
        speed_members.append(speed_grid)
        direction_members_deg.append(direction_grid_deg)
        member_records.append(
            {
                "member": index,
                "scenario": label,
                "wind_speed_mps": member_speed_mps,
                "wind_speed_kts": member_speed_mps * 1.94384449,
                "wind_from_direction_deg": member_direction_deg,
                "speed_finite_cells": speed_finite_cells,
                "direction_finite_cells": direction_finite_cells,
                "speed_output": str(member_ascii["speed"]),
                "direction_output": str(member_ascii["direction"]),
            }
        )

    speed_stack = np.stack(speed_members, axis=0).astype(np.float32)
    direction_stack_deg = np.stack(direction_members_deg, axis=0).astype(np.float32)
    if not np.isfinite(speed_stack).any():
        raise RuntimeError(f"{solver_display} GEFS sigma ensemble produced no finite speed cells: {member_records}")
    if not np.isfinite(direction_stack_deg).any():
        raise RuntimeError(f"{solver_display} GEFS sigma ensemble produced no finite direction cells: {member_records}")
    speed_std_kts = (np.nanstd(speed_stack, axis=0) * 1.94384449).astype(np.float32)

    direction_rad = np.deg2rad(direction_stack_deg)
    sin_mean = np.nanmean(np.sin(direction_rad), axis=0)
    cos_mean = np.nanmean(np.cos(direction_rad), axis=0)
    resultant_length = np.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
    resultant_length = np.clip(resultant_length, 1.0e-6, 1.0)
    direction_std_deg = np.rad2deg(np.sqrt(-2.0 * np.log(resultant_length))).astype(np.float32)
    if not np.isfinite(speed_std_kts).any():
        raise RuntimeError(f"Wind speed spread grid has no finite cells after ensemble reduction: {member_records}")
    if not np.isfinite(direction_std_deg).any():
        raise RuntimeError(f"Wind direction spread grid has no finite cells after ensemble reduction: {member_records}")

    if speed_header is None:
        raise RuntimeError("GEFS sigma ensemble did not produce any members.")

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
        footer_text=_footer_timestamp_text(boundary_target_time_utc, "wind speed spread"),
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
        footer_text=_footer_timestamp_text(boundary_target_time_utc, "wind dir spread"),
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
        "wind_solver": wind_solver,
        "wind_solver_display": solver_display,
        "solver_runs": solver_metadata,
        "wind_input_tif": str(wind_input_tif),
        "final_dem_tif": str(domain.clipped_dem_tif),
        "solve_dem_tif": str(domain.solve_dem_tif),
        "boundary_target_time_utc": boundary_target_time_utc.isoformat(),
        "mesh_resolution_m": mesh_resolution_m,
        "member_count": len(member_records),
        "members": member_records,
        "product_1": str(product1_png),
        "product_2": str(product2_png),
        "product_3": str(product3_png),
        "speed_std_knots_mean": float(np.nanmean(speed_std_kts)),
        "direction_std_deg_mean": float(np.nanmean(direction_std_deg)),
        "weather_inset": weather_inset,
    }


def _build_satellite_products(
    report_dir: Path,
    temp_dir: Path,
    domain: PreparedSiteDomain,
    satellite_inputs: dict,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict:
    bbox = satellite_inputs["render_bbox"]
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
    sentinel_rgb_time = datetime.fromisoformat(sentinel_rgb.item_datetime_utc.replace("Z", "+00:00"))
    render_rgb_preview(
        sentinel_rgb_paths["visual"],
        bbox,
        rgb_png,
        dem_basemap_tif=dem_basemap_tif,
        title="rgb",
        footer_text=_footer_timestamp_text(sentinel_rgb_time, "rgb collected"),
    )

    chla_png = report_dir / "satellite_chla_estimated.png"
    turbidity_png = report_dir / "satellite_turbidity_estimated.png"
    sst_png = report_dir / "satellite_sst_latest.png"

    if sentinel is not None and sentinel_paths is not None:
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
        sentinel_selection = {
            "item_id": sentinel.item_id,
            "datetime_utc": sentinel.item_datetime_utc,
            "cloud_cover": sentinel.cloud_cover,
        }
    else:
        reason = selection_diagnostics.get("sentinel_unavailable_reason") or "Insufficient water coverage in the selected area."
        _write_unavailable_panel(
            chla_png,
            title="chl a",
            line1="N/A - insufficient water coverage",
            line2=reason,
        )
        _write_unavailable_panel(
            turbidity_png,
            title="turbidity",
            line1="N/A - insufficient water coverage",
            line2=reason,
        )
        chla_summary = {
            "product": "estimated_chlorophyll_a",
            "status": "unavailable",
            "reason": reason,
            "output_png": str(chla_png),
        }
        turbidity_summary = {
            "product": "estimated_turbidity",
            "status": "unavailable",
            "reason": reason,
            "output_png": str(turbidity_png),
        }
        sentinel_selection = None

    sst_selection = None
    if ecostress is not None and ecostress_paths is not None:
        ecostress_time = datetime.fromisoformat(ecostress.item_datetime_utc.replace("Z", "+00:00"))
        sst_tif = temp_dir / "satellite_sst_latest.tif"
        sst_summary = derive_ecostress_sst(
            lst_tif=ecostress_paths["lst"],
            cloud_tif=ecostress_paths["cloud"],
            water_tif=ecostress_paths["water"],
            bbox=bbox,
            output_tif=sst_tif,
            output_png=sst_png,
            dem_basemap_tif=dem_basemap_tif,
            footer_text=_footer_timestamp_text(ecostress_time, "sst collected"),
        )
        sst_selection = {
            "source": "ecostress",
            "item_id": ecostress.item_id,
            "datetime_utc": ecostress.item_datetime_utc,
            "cloud_cover": None,
        }
    elif landsat is not None and landsat_paths is not None:
        landsat_time = datetime.fromisoformat(landsat.item_datetime_utc.replace("Z", "+00:00"))
        sst_tif = temp_dir / "satellite_sst_latest.tif"
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
            footer_text=_footer_timestamp_text(landsat_time, "sst collected"),
        )
        sst_selection = {
            "source": "landsat",
            "item_id": landsat.item_id,
            "datetime_utc": landsat.item_datetime_utc,
            "cloud_cover": landsat.cloud_cover,
        }
    else:
        reason = (
            selection_diagnostics.get("ecostress_unavailable_reason")
            or selection_diagnostics.get("landsat_unavailable_reason")
            or "Insufficient water coverage in the selected area."
        )
        _write_unavailable_panel(
            sst_png,
            title="sst",
            line1="N/A - insufficient water coverage",
            line2=reason,
        )
        sst_summary = {
            "product": "sst",
            "status": "unavailable",
            "reason": reason,
            "output_png": str(sst_png),
        }
        sst_selection = None

    return {
        "sentinel_rgb_selection": {
            "item_id": sentinel_rgb.item_id,
            "datetime_utc": sentinel_rgb.item_datetime_utc,
            "cloud_cover": sentinel_rgb.cloud_cover,
        },
        "sentinel_selection": sentinel_selection,
        "sst_selection": sst_selection,
        "selection_diagnostics": selection_diagnostics,
        "rgb_png": str(rgb_png),
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
            f"![Wind speed spread]({_app_path(wind['product_2'])})",
            "",
            f"![Wind direction spread]({_app_path(wind['product_3'])})",
            "",
            "## Satellite products",
            f"Sentinel-2 RGB source: `{satellite['sentinel_rgb_selection']['item_id']}` at `{satellite['sentinel_rgb_selection']['datetime_utc']}`",
            "",
            f"![Latest RGB]({_app_path(satellite['rgb_png'])})",
            "",
            (
                f"Estimated chlorophyll-a source: `{sentinel_selection['item_id']}`"
                if sentinel_selection is not None
                else f"Estimated chlorophyll-a: unavailable ({chl_a.get('reason', 'insufficient water coverage')})"
            ),
            "",
            f"![Estimated chlorophyll-a]({_app_path(chl_a['output_png'])})",
            "",
            (
                f"Estimated turbidity source: `{sentinel_selection['item_id']}`"
                if sentinel_selection is not None
                else f"Estimated turbidity: unavailable ({turbidity.get('reason', 'insufficient water coverage')})"
            ),
            "",
            f"![Estimated turbidity]({_app_path(turbidity['output_png'])})",
            "",
            (
                f"{sst_selection['source'].upper()} SST source: `{sst_selection['item_id']}` at `{sst_selection['datetime_utc']}`"
                if sst_selection is not None
                else f"SST: unavailable ({sst.get('reason', 'insufficient water coverage')})"
            ),
            "",
            f"![Latest SST]({_app_path(sst['output_png'])})",
            "",
            "## Notes",
            f"- Wind variance products are single-time GEFS sigma-spread maps downscaled with `{wind['wind_solver_display']}`.",
            f"- Wind data mode for this run: `{wind['wind_data_mode']}`.",
            "- Chlorophyll-a is an estimated satellite retrieval from Sentinel-2 red/red-edge reflectance, not an in situ measurement.",
            "- Turbidity is an estimated Sentinel-2 red/NIR remote-sensing retrieval, not an in situ measurement.",
            "- Cloudy scenes are allowed; this report uses the best recent scene available before race time.",
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
    report_root = _resolve_report_root(report_output_dir)
    report_dir = report_root / _report_dir_name(race_time_local, site)
    report_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = report_dir / "report_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    _progress(progress_callback, 8, "Preparing terrain domain...")
    domain = prepare_site_domain(site=site, working_dir=temp_dir)
    satellite_inputs = _download_satellite_inputs(
        temp_dir,
        race_time_utc,
        domain,
        force_ecostress_sst=force_ecostress_sst,
        progress_callback=progress_callback,
    )
    landscape_tif, landscape_summary = _prepare_landscape_input(temp_dir, domain, satellite_inputs, progress_callback)
    wind_summary = _build_wind_products(report_dir, temp_dir, race_time_utc, site, domain, landscape_tif, mesh_resolution, wind_solver, progress_callback)
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
        "report_dir": str(report_dir),
        "report_root": str(report_root),
        "report_markdown": str(report_path),
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
    )
    print(report_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
