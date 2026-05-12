from __future__ import annotations

import gzip
import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

from predictweather.http import download_url_to_file, env_allows_insecure_ssl, fetch_json

AVIATIONWEATHER_METAR_URL = "https://aviationweather.gov/api/data/metar"
AVIATIONWEATHER_STATIONS_CACHE_URL = "https://aviationweather.gov/data/cache/stations.cache.json.gz"


@dataclass(frozen=True)
class ObservationStation:
    station_id: str
    lat: float
    lon: float
    name: str | None = None
    country: str | None = None
    state: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SurfaceObservation:
    station_id: str
    report_time_utc: str
    lat: float
    lon: float
    wind_speed_kts: float | None
    gust_kts: float | None
    wind_direction_deg: float | None
    raw_observation: str

    def as_dict(self) -> dict:
        return asdict(self)


def _iso_z(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _cache_path(cache_dir: Path, station_id: str, start_utc: datetime, end_utc: datetime) -> Path:
    stamp = f"{station_id}_{start_utc:%Y%m%dT%H%MZ}_{end_utc:%Y%m%dT%H%MZ}_v2.json"
    return cache_dir / "observations" / "aviationweather" / station_id.upper() / stamp


def _stations_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "observations" / "aviationweather" / "stations.cache.json.gz"


def _fetch_metar_payload(station_id: str, start_utc: datetime, end_utc: datetime) -> list[dict]:
    hours = max(1, int((end_utc - start_utc).total_seconds() // 3600) + 1)
    params = urlencode(
        {
            "ids": station_id.upper(),
            "format": "json",
            "hours": hours,
        }
    )
    payload = fetch_json(f"{AVIATIONWEATHER_METAR_URL}?{params}", allow_insecure=env_allows_insecure_ssl())
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unexpected AviationWeather response shape for {station_id}: {type(payload)!r}")


def _cache_is_fresh(path: Path, *, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    age_seconds = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
    return age_seconds <= max_age_hours * 3600.0


def _extract_station_rows(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("features"), list):
            rows: list[dict] = []
            for feature in payload["features"]:
                if not isinstance(feature, dict):
                    continue
                properties = feature.get("properties")
                geometry = feature.get("geometry")
                row = dict(properties) if isinstance(properties, dict) else {}
                if isinstance(geometry, dict):
                    row["_geometry"] = geometry
                rows.append(row)
            return rows
        if isinstance(payload.get("data"), list):
            return [row for row in payload["data"] if isinstance(row, dict)]
    raise ValueError(f"Unexpected station cache payload shape: {type(payload)!r}")


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_station_coordinates(row: dict) -> tuple[float | None, float | None]:
    lat = _float_or_none(row.get("lat") or row.get("latitude"))
    lon = _float_or_none(row.get("lon") or row.get("longitude"))
    if lat is not None and lon is not None:
        return lat, lon
    geometry = row.get("_geometry")
    if isinstance(geometry, dict):
        coordinates = geometry.get("coordinates")
        if (
            isinstance(coordinates, (list, tuple))
            and len(coordinates) >= 2
            and coordinates[0] is not None
            and coordinates[1] is not None
        ):
            lon = _float_or_none(coordinates[0])
            lat = _float_or_none(coordinates[1])
            return lat, lon
    return None, None


def _extract_station_id(row: dict) -> str | None:
    for key in ("icaoId", "icao", "stationId", "ident", "id"):
        value = row.get(key)
        if isinstance(value, str):
            text = value.strip().upper()
            if len(text) >= 3:
                return text
    return None


def _great_circle_distance_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_km = 6371.0088
    lat_a_rad = math.radians(lat_a)
    lat_b_rad = math.radians(lat_b)
    delta_lat = math.radians(lat_b - lat_a)
    delta_lon = math.radians(lon_b - lon_a)
    sin_lat = math.sin(delta_lat / 2.0)
    sin_lon = math.sin(delta_lon / 2.0)
    hav = sin_lat * sin_lat + math.cos(lat_a_rad) * math.cos(lat_b_rad) * sin_lon * sin_lon
    return 2.0 * radius_km * math.asin(min(1.0, math.sqrt(max(hav, 0.0))))


def load_station_catalog(cache_dir: Path, *, max_age_hours: float = 36.0) -> list[ObservationStation]:
    cache_path = _stations_cache_path(cache_dir)
    if not _cache_is_fresh(cache_path, max_age_hours=max_age_hours):
        download_url_to_file(
            AVIATIONWEATHER_STATIONS_CACHE_URL,
            cache_path,
            allow_insecure=env_allows_insecure_ssl(),
        )

    with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    stations: list[ObservationStation] = []
    for row in _extract_station_rows(payload):
        station_id = _extract_station_id(row)
        lat, lon = _extract_station_coordinates(row)
        if station_id is None or lat is None or lon is None:
            continue
        stations.append(
            ObservationStation(
                station_id=station_id,
                lat=lat,
                lon=lon,
                name=(row.get("site") or row.get("name") or row.get("stationName")),
                country=(row.get("country") or row.get("cntry")),
                state=(row.get("state") or row.get("stateCode")),
            )
        )
    if not stations:
        raise ValueError("No observation stations found in AviationWeather station cache.")
    return stations


def nearest_station_candidates(
    cache_dir: Path,
    lat: float,
    lon: float,
    *,
    limit: int = 8,
    max_distance_km: float = 175.0,
) -> list[dict]:
    ranked: list[dict] = []
    for station in load_station_catalog(cache_dir):
        distance_km = _great_circle_distance_km(lat, lon, station.lat, station.lon)
        if distance_km > max_distance_km:
            continue
        ranked.append(
            {
                "station": station,
                "distance_km": distance_km,
            }
        )
    ranked.sort(key=lambda item: item["distance_km"])
    return ranked[:limit]


def fetch_metar_observations(cache_dir: Path, station_id: str, start_utc: datetime, end_utc: datetime) -> list[SurfaceObservation]:
    cache_path = _cache_path(cache_dir, station_id, start_utc, end_utc)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        payload = _fetch_metar_payload(station_id, start_utc, end_utc)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    observations: list[SurfaceObservation] = []
    for row in payload:
        report_time = row.get("reportTime")
        lat = row.get("lat")
        lon = row.get("lon")
        if report_time is None or lat is None or lon is None:
            continue
        direction = row.get("wdir")
        if isinstance(direction, str) and not direction.isdigit():
            direction_value = None
        else:
            direction_value = None if direction is None else float(direction)
        wind_speed = row.get("wspd")
        gust = row.get("wgst")
        observations.append(
            SurfaceObservation(
                station_id=str(row.get("icaoId", station_id)).upper(),
                report_time_utc=str(report_time),
                lat=float(lat),
                lon=float(lon),
                wind_speed_kts=None if wind_speed is None else float(wind_speed),
                gust_kts=None if gust is None else float(gust),
                wind_direction_deg=direction_value,
                raw_observation=str(row.get("rawOb", "")),
            )
        )
    observations.sort(key=lambda item: item.report_time_utc)
    return observations


def recent_same_local_hour_observations(
    cache_dir: Path,
    station_id: str,
    *,
    target_time_utc: datetime,
    timezone_name: str,
    sample_count: int = 5,
    max_lookback_days: int = 15,
) -> list[SurfaceObservation]:
    from zoneinfo import ZoneInfo

    target_time_utc = target_time_utc.astimezone(timezone.utc)
    anchor_time_utc = min(target_time_utc - timedelta(hours=1), datetime.now(timezone.utc) - timedelta(hours=1))
    start_utc = anchor_time_utc - timedelta(days=max_lookback_days)
    observations = fetch_metar_observations(cache_dir, station_id, start_utc, anchor_time_utc + timedelta(hours=1))

    local_tz = ZoneInfo(timezone_name)
    target_local = target_time_utc.astimezone(local_tz)
    selected: list[SurfaceObservation] = []
    seen_dates: set[str] = set()

    for observation in reversed(observations):
        report_time = datetime.fromisoformat(observation.report_time_utc.replace("Z", "+00:00")).astimezone(local_tz)
        if report_time >= target_local:
            continue
        if report_time.hour != target_local.hour:
            continue
        date_key = report_time.date().isoformat()
        if date_key in seen_dates:
            continue
        if observation.wind_speed_kts is None:
            continue
        seen_dates.add(date_key)
        selected.append(observation)
        if len(selected) >= sample_count:
            break

    selected.reverse()
    return selected
