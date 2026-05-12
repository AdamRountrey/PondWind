from __future__ import annotations

import bz2
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError

import eccodes
import numpy as np

from predictweather.boundary import build_model_point_forecast
from predictweather.http import download_url_to_file, env_allows_insecure_ssl, fetch_text, url_exists


RUN_HOURS_UTC = (0, 6, 12, 18)
MAX_HORIZON_HOURS = 180


@dataclass(frozen=True)
class IconSelection:
    requested_valid_time_utc: str
    selected_valid_time_utc: str
    run_at_utc: str
    forecast_hour: int
    files: dict[str, str]


def _step_hours(timestamp: datetime) -> int:
    timestamp = timestamp.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return timestamp.hour


def candidate_runs_for_valid_time(valid_at_utc: datetime) -> list[tuple[datetime, int]]:
    valid_at_utc = valid_at_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    candidates: list[tuple[datetime, int]] = []
    for offset_hours in range(0, MAX_HORIZON_HOURS + 25):
        run_at = valid_at_utc - timedelta(hours=offset_hours)
        if run_at.hour not in RUN_HOURS_UTC:
            continue
        forecast_hour = int((valid_at_utc - run_at).total_seconds() // 3600)
        max_horizon = 180 if run_at.hour in {0, 12} else 120
        if forecast_hour < 0 or forecast_hour > max_horizon:
            continue
        if forecast_hour > 78 and forecast_hour % 3 != 0:
            continue
        candidates.append((run_at, forecast_hour))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def build_icon_url(run_at_utc: datetime, forecast_hour: int, field_dir: str, field_name: str) -> str:
    return (
        f"https://opendata.dwd.de/weather/nwp/icon/grib/{run_at_utc:%H}/{field_dir}/"
        f"icon_global_icosahedral_single-level_{run_at_utc:%Y%m%d%H}_{forecast_hour:03d}_{field_name}.grib2.bz2"
    )


def _download_and_decompress(url: str, destination: Path) -> Path:
    compressed = destination.with_suffix(destination.suffix + ".bz2")
    download_url_to_file(url, compressed, allow_insecure=env_allows_insecure_ssl())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(compressed, "rb") as source, destination.open("wb") as target:
        target.write(source.read())
    compressed.unlink(missing_ok=True)
    return destination


def _latest_invariant_filename(field_dir: str, field_name: str) -> str:
    listing_url = f"https://opendata.dwd.de/weather/nwp/icon/grib/00/{field_dir}/"
    listing = fetch_text(listing_url, allow_insecure=env_allows_insecure_ssl())
    matches = re.findall(
        rf"(icon_global_icosahedral_time-invariant_[0-9]{{10}}_{field_name}\.grib2\.bz2)",
        listing,
    )
    if not matches:
        raise FileNotFoundError(f"No ICON invariant file found for {field_name}")
    return sorted(matches)[-1]


def _download_icon_invariant_field(destination_dir: Path, field_dir: str, field_name: str) -> Path:
    grid_dir = destination_dir / "icon" / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    file_name = _latest_invariant_filename(field_dir, field_name)
    destination = grid_dir / file_name.replace(".bz2", "")
    if destination.exists():
        return destination
    url = f"https://opendata.dwd.de/weather/nwp/icon/grib/00/{field_dir}/{file_name}"
    return _download_and_decompress(url, destination)


def _read_icon_values(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        gid = eccodes.codes_grib_new_from_file(handle)
        try:
            values = eccodes.codes_get_array(gid, "values").astype(np.float64)
        finally:
            eccodes.codes_release(gid)
    return values


def _icon_native_coordinates(destination_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    clat_path = _download_icon_invariant_field(destination_dir, "clat", "CLAT")
    clon_path = _download_icon_invariant_field(destination_dir, "clon", "CLON")
    return _read_icon_values(clat_path), _read_icon_values(clon_path)


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius_km = 6371.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(a))


def regrid_icon_unstructured_to_latlon(
    values: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    target_lats_deg: np.ndarray,
    target_lons_deg: np.ndarray,
    *,
    neighbors: int = 4,
) -> np.ndarray:
    output = np.full(target_lats_deg.shape, np.nan, dtype=np.float64)
    source_lon = np.where(longitudes_deg > 180.0, longitudes_deg - 360.0, longitudes_deg)
    for row in range(target_lats_deg.shape[0]):
        for col in range(target_lats_deg.shape[1]):
            lat = float(target_lats_deg[row, col])
            lon = float(target_lons_deg[row, col])
            lat_mask = np.abs(latitudes_deg - lat) <= 2.5
            lon_mask = np.abs(source_lon - lon) <= 2.5
            subset = np.where(lat_mask & lon_mask)[0]
            if subset.size == 0:
                subset = np.arange(values.size)
            distances = _haversine_km(lat, lon, latitudes_deg[subset], source_lon[subset])
            nearest_order = np.argsort(distances)[:neighbors]
            nearest_indices = subset[nearest_order]
            nearest_dist = distances[nearest_order]
            if nearest_dist[0] == 0.0:
                output[row, col] = float(values[nearest_indices[0]])
                continue
            weights = 1.0 / np.maximum(nearest_dist, 1.0e-6) ** 2
            output[row, col] = float(np.sum(values[nearest_indices] * weights) / np.sum(weights))
    return output


def _sample_icon_native_point(path: Path, lat: float, lon: float, native_lats: np.ndarray, native_lons: np.ndarray) -> dict:
    values = _read_icon_values(path)
    source_lon = np.where(native_lons > 180.0, native_lons - 360.0, native_lons)
    distances = _haversine_km(lat, lon, native_lats, source_lon)
    index = int(np.argmin(distances))
    return {
        "value": float(values[index]),
        "grid_lat": float(native_lats[index]),
        "grid_lon": float(source_lon[index]),
        "grid_distance_km": float(distances[index]),
        "grid_index": index,
    }


def select_best_run(valid_at_utc: datetime) -> tuple[datetime, int]:
    last_error: Exception | None = None
    for run_at_utc, forecast_hour in candidate_runs_for_valid_time(valid_at_utc):
        try:
            required_urls = (
                build_icon_url(run_at_utc, forecast_hour, "u_10m", "U_10M"),
                build_icon_url(run_at_utc, forecast_hour, "v_10m", "V_10M"),
            )
            if all(url_exists(url, allow_insecure=env_allows_insecure_ssl()) for url in required_urls):
                return run_at_utc, forecast_hour
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"No ICON run found for valid time {valid_at_utc.isoformat()}")


def download_icon_for_valid_time(destination_dir: Path, valid_at_utc: datetime) -> IconSelection:
    selected_valid_time_utc = valid_at_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    run_at_utc, forecast_hour = select_best_run(selected_valid_time_utc)
    output_dir = destination_dir / "icon" / run_at_utc.strftime("%Y%m%dT%HZ") / f"F{forecast_hour:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "UGRD": str(output_dir / f"icon_{run_at_utc:%Y%m%d%H}_{forecast_hour:03d}_u10.grib2"),
        "VGRD": str(output_dir / f"icon_{run_at_utc:%Y%m%d%H}_{forecast_hour:03d}_v10.grib2"),
        "GUST": str(output_dir / f"icon_{run_at_utc:%Y%m%d%H}_{forecast_hour:03d}_vmax10.grib2"),
    }
    specs = (
        ("UGRD", "u_10m", "U_10M"),
        ("VGRD", "v_10m", "V_10M"),
        ("GUST", "vmax_10m", "VMAX_10M"),
    )
    for file_key, field_dir, field_name in specs:
        destination = Path(files[file_key])
        if not destination.exists():
            url = build_icon_url(run_at_utc, forecast_hour, field_dir, field_name)
            if file_key == "GUST":
                try:
                    if not url_exists(url, allow_insecure=env_allows_insecure_ssl()):
                        files.pop("GUST", None)
                        continue
                except HTTPError:
                    files.pop("GUST", None)
                    continue
            _download_and_decompress(url, destination)

    selection = IconSelection(
        requested_valid_time_utc=valid_at_utc.astimezone(timezone.utc).isoformat(),
        selected_valid_time_utc=selected_valid_time_utc.isoformat(),
        run_at_utc=run_at_utc.isoformat(),
        forecast_hour=forecast_hour,
        files=files,
    )
    (output_dir / "manifest.json").write_text(json.dumps(selection.__dict__, indent=2), encoding="utf-8")
    return selection


def sample_icon_point_forecast(selection: IconSelection, lat: float, lon: float, acquisition_mode: str) -> dict:
    native_lats, native_lons = _icon_native_coordinates(Path(selection.files["UGRD"]).parents[3])
    u_summary = _sample_icon_native_point(Path(selection.files["UGRD"]), lat, lon, native_lats, native_lons)
    v_summary = _sample_icon_native_point(Path(selection.files["VGRD"]), lat, lon, native_lats, native_lons)
    wind_speed_mps = float(np.hypot(u_summary["value"], v_summary["value"]))
    wind_from_direction_deg = float((270.0 - np.degrees(np.arctan2(v_summary["value"], u_summary["value"]))) % 360.0)
    wind_summary = {
        "valid_time_utc": selection.selected_valid_time_utc,
        "site_lat": lat,
        "site_lon": lon,
        "grid_lat": u_summary["grid_lat"],
        "grid_lon": u_summary["grid_lon"],
        "grid_distance_km": u_summary["grid_distance_km"],
        "grid_indices": {"native_index": int(u_summary["grid_index"])},
        "u10_mps": float(u_summary["value"]),
        "v10_mps": float(v_summary["value"]),
        "wind_speed_mps": wind_speed_mps,
        "wind_from_direction_deg": wind_from_direction_deg,
    }
    gust_summary = None
    if "GUST" in selection.files:
        gust_point = _sample_icon_native_point(Path(selection.files["GUST"]), lat, lon, native_lats, native_lons)
        gust_summary = {"value": float(gust_point["value"])}
    forecast = build_model_point_forecast(
        source="icon",
        display_name="icon",
        run_at_utc=selection.run_at_utc,
        forecast_hour=selection.forecast_hour,
        acquisition_mode=acquisition_mode,
        wind_summary=wind_summary,
        gust_summary=gust_summary,
        files=selection.files,
        gust_kind="10m max wind",
    )
    return forecast.as_dict()
