from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

import cfgrib
import numpy as np

from predictweather.http import download_url_to_file, env_allows_insecure_ssl, url_exists

GEFS_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod"
RUN_HOURS_UTC = (0, 6, 12, 18)
FORECAST_STEP_HOURS = 3


@dataclass(frozen=True)
class GefsBoundarySelection:
    requested_valid_time_utc: str
    selected_valid_time_utc: str
    run_at_utc: str
    forecast_hour: int
    mean_files: dict[str, str]
    spread_files: dict[str, str]


def floor_to_3h(timestamp: datetime) -> datetime:
    timestamp = timestamp.astimezone(timezone.utc)
    floored_hour = timestamp.hour - (timestamp.hour % FORECAST_STEP_HOURS)
    return timestamp.replace(hour=floored_hour, minute=0, second=0, microsecond=0)


def build_run_datetime(valid_at_utc: datetime, run_hour_utc: int) -> datetime:
    run_at = valid_at_utc.replace(hour=run_hour_utc, minute=0, second=0, microsecond=0)
    if run_at > valid_at_utc:
        run_at -= timedelta(days=1)
    return run_at


def candidate_runs_for_valid_time(valid_at_utc: datetime) -> list[tuple[datetime, int]]:
    candidates: list[tuple[datetime, int]] = []
    latest_candidate = valid_at_utc.replace(minute=0, second=0, microsecond=0)
    for offset_hours in range(0, 385):
        run_at = latest_candidate - timedelta(hours=offset_hours)
        if run_at.hour not in RUN_HOURS_UTC:
            continue
        forecast_hour = int((valid_at_utc - run_at).total_seconds() // 3600)
        if forecast_hour >= 0 and forecast_hour % FORECAST_STEP_HOURS == 0 and forecast_hour <= 384:
            candidates.append((run_at, forecast_hour))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def build_gefs_filename(product: str, run_at_utc: datetime, forecast_hour: int) -> str:
    return f"{product}.t{run_at_utc:%H}z.pgrb2a.0p50.f{forecast_hour:03d}"


def build_gefs_url(product: str, run_at_utc: datetime, forecast_hour: int) -> str:
    filename = build_gefs_filename(product, run_at_utc, forecast_hour)
    return f"{GEFS_BASE_URL}/gefs.{run_at_utc:%Y%m%d}/{run_at_utc:%H}/atmos/pgrb2ap5/{filename}"


def _parse_iso_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _manifest_is_complete(manifest: dict) -> bool:
    mean_files = manifest.get("mean_files", {})
    spread_files = manifest.get("spread_files", {})
    for bucket, key in ((mean_files, "geavg"), (spread_files, "gespr")):
        path_text = bucket.get(key)
        if not path_text:
            return False
        if not Path(path_text).exists():
            return False
    return True


def _candidate_manifest_sort_key(manifest: dict, manifest_path: Path, target_valid_time_utc: datetime) -> tuple[float, float, str]:
    selected_valid = _parse_iso_utc(manifest["selected_valid_time_utc"])
    run_at = _parse_iso_utc(manifest["run_at_utc"])
    return (
        abs((selected_valid - target_valid_time_utc).total_seconds()),
        abs((run_at - target_valid_time_utc).total_seconds()),
        str(manifest_path),
    )


def load_cached_gefs_manifest_for_valid_time(root: Path, target_valid_time_utc: datetime) -> tuple[dict, Path]:
    target_valid_time_utc = target_valid_time_utc.astimezone(timezone.utc)
    candidates: list[tuple[dict, Path]] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "selected_valid_time_utc" not in manifest or "run_at_utc" not in manifest:
            continue
        if not _manifest_is_complete(manifest):
            continue
        candidates.append((manifest, manifest_path))

    if not candidates:
        raise FileNotFoundError(f"No usable GEFS manifest found under {root}")

    manifest, manifest_path = min(
        candidates,
        key=lambda item: _candidate_manifest_sort_key(item[0], item[1], target_valid_time_utc),
    )
    return manifest, manifest_path


def select_best_run(valid_at_utc: datetime) -> tuple[datetime, int]:
    last_error: Exception | None = None
    for run_at, forecast_hour in candidate_runs_for_valid_time(valid_at_utc):
        all_present = True
        for product in ("geavg", "gespr"):
            probe_url = build_gefs_url(product, run_at, forecast_hour)
            try:
                if not url_exists(probe_url, allow_insecure=env_allows_insecure_ssl()):
                    all_present = False
                    break
            except HTTPError:
                all_present = False
                break
            except URLError as exc:
                last_error = exc
                all_present = False
                break
        if all_present:
            return run_at, forecast_hour
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"No GEFS run found for valid time {valid_at_utc.isoformat()}")


def download_gefs_mean_and_spread(destination_dir: Path, valid_at_utc: datetime) -> GefsBoundarySelection:
    selected_valid_time_utc = floor_to_3h(valid_at_utc)
    run_at_utc, forecast_hour = select_best_run(selected_valid_time_utc)

    output_dir = destination_dir / "gefs" / run_at_utc.strftime("%Y%m%dT%HZ") / f"F{forecast_hour:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_files: dict[str, str] = {}
    spread_files: dict[str, str] = {}
    for product, bucket in (("geavg", mean_files), ("gespr", spread_files)):
        url = build_gefs_url(product, run_at_utc, forecast_hour)
        destination = output_dir / build_gefs_filename(product, run_at_utc, forecast_hour)
        if not destination.exists():
            download_url_to_file(url, destination, allow_insecure=env_allows_insecure_ssl())
        bucket[product] = str(destination)

    selection = GefsBoundarySelection(
        requested_valid_time_utc=valid_at_utc.astimezone(timezone.utc).isoformat(),
        selected_valid_time_utc=selected_valid_time_utc.isoformat(),
        run_at_utc=run_at_utc.isoformat(),
        forecast_hour=forecast_hour,
        mean_files=mean_files,
        spread_files=spread_files,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(selection.__dict__, indent=2), encoding="utf-8")
    return selection


def _open_component(path: Path, variable_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    dataset = cfgrib.open_dataset(
        path,
        indexpath="",
        filter_by_keys={
            "typeOfLevel": "heightAboveGround",
            "level": 10,
        },
    )
    values = dataset[variable_name].values.astype("float32")
    latitudes = dataset["latitude"].values.astype("float64")
    longitudes = dataset["longitude"].values.astype("float64")
    valid_time = np.datetime_as_string(dataset["valid_time"].values, unit="s")
    return values, latitudes, longitudes, valid_time


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def sample_gefs_mean_and_spread_at_site(mean_path: Path, spread_path: Path, lat: float, lon: float) -> dict:
    mean_u10, latitudes, longitudes, valid_time = _open_component(mean_path, "u10")
    mean_v10, _, _, _ = _open_component(mean_path, "v10")
    spread_u10, _, _, _ = _open_component(spread_path, "u10")
    spread_v10, _, _, _ = _open_component(spread_path, "v10")

    if latitudes.ndim == 1 and longitudes.ndim == 1:
        longitudes_2d, latitudes_2d = np.meshgrid(longitudes, latitudes)
    else:
        latitudes_2d = latitudes
        longitudes_2d = longitudes

    query_lon = lon
    if np.nanmax(longitudes_2d) > 180.0 and query_lon < 0.0:
        query_lon = lon + 360.0

    distance_metric = (latitudes_2d - lat) ** 2 + (longitudes_2d - query_lon) ** 2
    nearest_flat_index = int(np.argmin(distance_metric))
    y_index, x_index = np.unravel_index(nearest_flat_index, distance_metric.shape)

    point_lat = float(latitudes_2d[y_index, x_index])
    point_lon = float(longitudes_2d[y_index, x_index])
    if point_lon > 180.0:
        point_lon -= 360.0
    mu_u = float(mean_u10[y_index, x_index])
    mu_v = float(mean_v10[y_index, x_index])
    sigma_u = float(spread_u10[y_index, x_index])
    sigma_v = float(spread_v10[y_index, x_index])
    speed_mps = float(math.hypot(mu_u, mu_v))
    direction_from_deg = float((270.0 - math.degrees(math.atan2(mu_v, mu_u))) % 360.0)

    return {
        "valid_time_utc": valid_time,
        "site_lat": lat,
        "site_lon": lon,
        "grid_lat": point_lat,
        "grid_lon": point_lon,
        "grid_distance_km": _haversine_km(lat, lon, point_lat, point_lon),
        "grid_indices": {"y": int(y_index), "x": int(x_index)},
        "mean_u10_mps": mu_u,
        "mean_v10_mps": mu_v,
        "spread_u10_mps": sigma_u,
        "spread_v10_mps": sigma_v,
        "mean_speed_mps": speed_mps,
        "mean_wind_from_direction_deg": direction_from_deg,
        "mean_path": str(mean_path),
        "spread_path": str(spread_path),
    }
