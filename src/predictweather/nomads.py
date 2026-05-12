from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

from predictweather.http import download_url_to_file, env_allows_insecure_ssl, url_exists


def site_subregion(lat: float, lon: float, padding_deg: float = 0.45) -> dict[str, float]:
    return {
        "leftlon": lon - padding_deg,
        "rightlon": lon + padding_deg,
        "toplat": lat + padding_deg,
        "bottomlat": lat - padding_deg,
    }


def run_candidates(
    valid_at_utc: datetime,
    *,
    run_hours_utc: tuple[int, ...],
    max_horizon_hours: int,
    step_hours: int = 1,
) -> list[tuple[datetime, int]]:
    valid_at_utc = valid_at_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    candidates: list[tuple[datetime, int]] = []
    for offset_hours in range(0, max_horizon_hours + max(run_hours_utc) + 25):
        run_at = valid_at_utc - timedelta(hours=offset_hours)
        if run_at.hour not in run_hours_utc:
            continue
        forecast_hour = int((valid_at_utc - run_at).total_seconds() // 3600)
        if forecast_hour < 0 or forecast_hour > max_horizon_hours or forecast_hour % step_hours != 0:
            continue
        candidates.append((run_at, forecast_hour))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def nomads_filter_url(
    *,
    filter_script: str,
    file_name: str,
    dir_path: str,
    variable_name: str,
    level_flag: str,
    lat: float,
    lon: float,
) -> str:
    region = site_subregion(lat, lon)
    query = {
        "file": file_name,
        level_flag: "on",
        f"var_{variable_name}": "on",
        "subregion": "",
        "leftlon": f"{region['leftlon']:.4f}",
        "rightlon": f"{region['rightlon']:.4f}",
        "toplat": f"{region['toplat']:.4f}",
        "bottomlat": f"{region['bottomlat']:.4f}",
        "dir": dir_path,
    }
    return f"https://nomads.ncep.noaa.gov/cgi-bin/{filter_script}?{urlencode(query)}"


def download_nomads_subset(
    *,
    filter_script: str,
    file_name: str,
    dir_path: str,
    variable_name: str,
    level_flag: str,
    lat: float,
    lon: float,
    destination: Path,
) -> Path:
    url = nomads_filter_url(
        filter_script=filter_script,
        file_name=file_name,
        dir_path=dir_path,
        variable_name=variable_name,
        level_flag=level_flag,
        lat=lat,
        lon=lon,
    )
    return download_url_to_file(url, destination, allow_insecure=env_allows_insecure_ssl())


def nomads_subset_exists(
    *,
    filter_script: str,
    file_name: str,
    dir_path: str,
    variable_name: str,
    level_flag: str,
    lat: float,
    lon: float,
) -> bool:
    url = nomads_filter_url(
        filter_script=filter_script,
        file_name=file_name,
        dir_path=dir_path,
        variable_name=variable_name,
        level_flag=level_flag,
        lat=lat,
        lon=lon,
    )
    return url_exists(url, allow_insecure=env_allows_insecure_ssl())
