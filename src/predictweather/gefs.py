from __future__ import annotations

import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

import cfgrib
import numpy as np

from predictweather.grib_lock import GRIB_DECODE_LOCK
from predictweather.http import download_url_to_file, env_allows_insecure_ssl, url_exists
from predictweather.nomads import site_subregion
from predictweather.noaa_pds import download_indexed_message, has_indexed_messages

GEFS_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod"
GEFS_PDS_BASE_URL = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_FILTER_SCRIPT = "filter_gefs_atmos_0p50a.pl"
RUN_HOURS_UTC = (0, 6, 12, 18)
FORECAST_STEP_HOURS = 3
GEFS_MEMBER_PRODUCTS = ("gec00",) + tuple(f"gep{index:02d}" for index in range(1, 31))


@dataclass(frozen=True)
class GefsBoundarySelection:
    requested_valid_time_utc: str
    selected_valid_time_utc: str
    run_at_utc: str
    forecast_hour: int
    mean_files: dict[str, str]
    spread_files: dict[str, str]
    member_files: dict[str, str] | None = None
    member_errors: dict[str, str] | None = None


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


def build_gefs_pds_url(product: str, run_at_utc: datetime, forecast_hour: int) -> str:
    filename = build_gefs_filename(product, run_at_utc, forecast_hour)
    return f"{GEFS_PDS_BASE_URL}/gefs.{run_at_utc:%Y%m%d}/{run_at_utc:%H}/atmos/pgrb2ap5/{filename}"


def build_gefs_subset_url(
    product: str,
    run_at_utc: datetime,
    forecast_hour: int,
    *,
    lat: float,
    lon: float,
    padding_deg: float = 0.45,
) -> str:
    """Build a small NOMADS subset URL containing only 10 m U/V wind near the site."""
    filename = build_gefs_filename(product, run_at_utc, forecast_hour)
    region = site_subregion(lat, lon, padding_deg=padding_deg, lon_360=True)
    query = {
        "file": filename,
        "lev_10_m_above_ground": "on",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "subregion": "",
        "leftlon": f"{region['leftlon']:.4f}",
        "rightlon": f"{region['rightlon']:.4f}",
        "toplat": f"{region['toplat']:.4f}",
        "bottomlat": f"{region['bottomlat']:.4f}",
        "dir": f"/gefs.{run_at_utc:%Y%m%d}/{run_at_utc:%H}/atmos/pgrb2ap5",
    }
    return f"https://nomads.ncep.noaa.gov/cgi-bin/{GEFS_FILTER_SCRIPT}?{urlencode(query)}"


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


def load_cached_gefs_manifest_for_valid_time(
    root: Path,
    target_valid_time_utc: datetime,
    *,
    max_valid_time_delta: timedelta = timedelta(minutes=0),
) -> tuple[dict, Path]:
    target_valid_time_utc = target_valid_time_utc.astimezone(timezone.utc)
    candidates: list[tuple[dict, Path]] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "selected_valid_time_utc" not in manifest or "run_at_utc" not in manifest:
            continue
        if not _manifest_is_complete(manifest):
            continue
        selected_valid = _parse_iso_utc(manifest["selected_valid_time_utc"])
        if abs(selected_valid - target_valid_time_utc) > max_valid_time_delta:
            continue
        candidates.append((manifest, manifest_path))

    if not candidates:
        raise FileNotFoundError(f"No usable GEFS manifest found under {root} for {target_valid_time_utc.isoformat()}")

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
            pds_url = build_gefs_pds_url(product, run_at, forecast_hour)
            nomads_url = build_gefs_url(product, run_at, forecast_hour)
            try:
                if not (
                    has_indexed_messages(pds_url, (("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")))
                    or url_exists(nomads_url, allow_insecure=env_allows_insecure_ssl())
                ):
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


def _download_gefs_product(path: Path, product: str, run_at_utc: datetime, forecast_hour: int) -> str:
    pds_url = build_gefs_pds_url(product, run_at_utc, forecast_hour)
    try:
        return _download_gefs_indexed_uv_product(path, pds_url)
    except Exception:
        nomads_url = build_gefs_url(product, run_at_utc, forecast_hour)
        download_url_to_file(nomads_url, path, allow_insecure=env_allows_insecure_ssl())
        return "nomads_full_file"


def _download_gefs_indexed_uv_product(destination: Path, grib_url: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_paths = [
        destination.with_name(f"{destination.name}.UGRD.partial"),
        destination.with_name(f"{destination.name}.VGRD.partial"),
    ]
    try:
        download_indexed_message(grib_url, part_paths[0], variable="UGRD", level_text="10 m above ground")
        download_indexed_message(grib_url, part_paths[1], variable="VGRD", level_text="10 m above ground")
        temp_path = destination.with_name(f"{destination.name}.partial")
        with temp_path.open("wb") as output:
            for part_path in part_paths:
                with part_path.open("rb") as part:
                    shutil.copyfileobj(part, output)
        temp_path.replace(destination)
        return "noaa_pds_indexed_range"
    finally:
        for part_path in part_paths:
            try:
                part_path.unlink()
            except FileNotFoundError:
                pass


def download_gefs_mean_and_spread(destination_dir: Path, valid_at_utc: datetime) -> GefsBoundarySelection:
    selected_valid_time_utc = floor_to_3h(valid_at_utc)
    run_at_utc, forecast_hour = select_best_run(selected_valid_time_utc)

    output_dir = destination_dir / "gefs" / run_at_utc.strftime("%Y%m%dT%HZ") / f"F{forecast_hour:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_files: dict[str, str] = {}
    spread_files: dict[str, str] = {}
    for product, bucket in (("geavg", mean_files), ("gespr", spread_files)):
        destination = output_dir / build_gefs_filename(product, run_at_utc, forecast_hour)
        if not destination.exists():
            _download_gefs_product(destination, product, run_at_utc, forecast_hour)
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


def download_gefs_mean_spread_and_members(
    destination_dir: Path,
    valid_at_utc: datetime,
    *,
    lat: float,
    lon: float,
    max_members: int | None = None,
) -> GefsBoundarySelection:
    selected_valid_time_utc = floor_to_3h(valid_at_utc)
    run_at_utc, forecast_hour = select_best_run(selected_valid_time_utc)

    output_dir = destination_dir / "gefs" / run_at_utc.strftime("%Y%m%dT%HZ") / f"F{forecast_hour:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_files: dict[str, str] = {}
    spread_files: dict[str, str] = {}
    source_modes: dict[str, str] = {}
    for product, bucket in (("geavg", mean_files), ("gespr", spread_files)):
        destination = output_dir / build_gefs_filename(product, run_at_utc, forecast_hour)
        if destination.exists():
            source_modes[product] = "local_existing"
        else:
            source_modes[product] = _download_gefs_product(destination, product, run_at_utc, forecast_hour)
        bucket[product] = str(destination)

    products = list(GEFS_MEMBER_PRODUCTS if max_members is None else GEFS_MEMBER_PRODUCTS[: max(0, int(max_members))])
    member_dir = output_dir / "members"
    member_dir.mkdir(parents=True, exist_ok=True)
    member_files: dict[str, str] = {}
    member_errors: dict[str, str] = {}
    max_workers = max(1, int(os.environ.get("PONDWIND_GEFS_MEMBER_DOWNLOAD_WORKERS", "3")))

    def download_member(product: str) -> tuple[str, Path | None, str | None]:
        destination = member_dir / build_gefs_filename(product, run_at_utc, forecast_hour)
        if destination.exists():
            return product, destination, None
        pds_url = build_gefs_pds_url(product, run_at_utc, forecast_hour)
        try:
            _download_gefs_indexed_uv_product(destination, pds_url)
            return product, destination, None
        except Exception as pds_exc:
            url = build_gefs_subset_url(product, run_at_utc, forecast_hour, lat=lat, lon=lon)
            try:
                download_url_to_file(url, destination, allow_insecure=env_allows_insecure_ssl())
            except Exception as exc:
                return product, None, f"pds={pds_exc!r}; nomads={exc!r}"
            return product, destination, None

    with ThreadPoolExecutor(max_workers=min(max_workers, len(products) or 1)) as executor:
        futures = [executor.submit(download_member, product) for product in products]
        for future in as_completed(futures):
            product, destination, error = future.result()
            if error is not None:
                member_errors[product] = error
            elif destination is not None:
                member_files[product] = str(destination)

    selection = GefsBoundarySelection(
        requested_valid_time_utc=valid_at_utc.astimezone(timezone.utc).isoformat(),
        selected_valid_time_utc=selected_valid_time_utc.isoformat(),
        run_at_utc=run_at_utc.isoformat(),
        forecast_hour=forecast_hour,
        mean_files=mean_files,
        spread_files=spread_files,
        member_files=member_files,
        member_errors=member_errors,
    )
    manifest = {**selection.__dict__, "source_modes": source_modes, "member_source_preference": "noaa_pds_indexed_range_then_nomads_filter"}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return selection


def _open_component(path: Path, variable_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with GRIB_DECODE_LOCK:
        dataset = cfgrib.open_dataset(
            path,
            indexpath="",
            filter_by_keys={
                "typeOfLevel": "heightAboveGround",
                "level": 10,
            },
        )
        try:
            values = dataset[variable_name].values.astype("float32")
            latitudes = dataset["latitude"].values.astype("float64")
            longitudes = dataset["longitude"].values.astype("float64")
            valid_time = np.datetime_as_string(dataset["valid_time"].values, unit="s")
        finally:
            dataset.close()
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

    latitudes_2d, longitudes_2d, y_index, x_index = _nearest_grid_point(latitudes, longitudes, lat, lon)

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


def _nearest_grid_point(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    lat: float,
    lon: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
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
    return latitudes_2d, longitudes_2d, int(y_index), int(x_index)


def sample_gefs_members_at_site(member_files: dict[str, str | Path], lat: float, lon: float) -> list[dict]:
    members: list[dict] = []
    for member_id, path_text in sorted(member_files.items()):
        path = Path(path_text)
        u10, latitudes, longitudes, valid_time = _open_component(path, "u10")
        v10, _, _, _ = _open_component(path, "v10")
        latitudes_2d, longitudes_2d, y_index, x_index = _nearest_grid_point(latitudes, longitudes, lat, lon)

        point_lat = float(latitudes_2d[y_index, x_index])
        point_lon = float(longitudes_2d[y_index, x_index])
        if point_lon > 180.0:
            point_lon -= 360.0
        u_value = float(u10[y_index, x_index])
        v_value = float(v10[y_index, x_index])
        speed_mps = float(math.hypot(u_value, v_value))
        direction_from_deg = float((270.0 - math.degrees(math.atan2(v_value, u_value))) % 360.0)
        members.append(
            {
                "member_id": member_id,
                "valid_time_utc": valid_time,
                "site_lat": lat,
                "site_lon": lon,
                "grid_lat": point_lat,
                "grid_lon": point_lon,
                "grid_distance_km": _haversine_km(lat, lon, point_lat, point_lon),
                "grid_indices": {"y": int(y_index), "x": int(x_index)},
                "u10_mps": u_value,
                "v10_mps": v_value,
                "speed_mps": speed_mps,
                "wind_from_direction_deg": direction_from_deg,
                "path": str(path),
            }
        )
    return members
