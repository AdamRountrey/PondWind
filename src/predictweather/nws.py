from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.request import Request

from predictweather.http import _open_url_with_retry, env_allows_insecure_ssl, parse_json_text

NWS_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"
USER_AGENT = "PondWind/1.0 (local report builder)"


@dataclass(frozen=True)
class HourlyPointForecast:
    temperature_f: float | None
    precipitation_probability_pct: float | None
    sky_cover_pct: float | None
    source_time_utc: str
    target_time_utc: str
    selection_mode: str
    forecast_grid_url: str


class ExactForecastTimeUnavailable(ValueError):
    pass


def _fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/ld+json, application/json"})
    with _open_url_with_retry(request, allow_insecure=env_allows_insecure_ssl(), timeout=120, retries=4, backoff_seconds=2.0) as response:
        body = response.read().decode("utf-8", errors="replace")
    return parse_json_text(body, source=url)


def _parse_duration_hours(duration_text: str) -> float:
    text = duration_text.strip().upper()
    if not text.startswith("PT"):
        return 1.0
    text = text[2:]
    hours = 0.0
    number = ""
    for char in text:
        if char.isdigit() or char == ".":
            number += char
            continue
        if char == "H" and number:
            hours += float(number)
        elif char == "M" and number:
            hours += float(number) / 60.0
        number = ""
    return max(hours, 1.0)


def _parse_validity_interval(valid_time: str) -> tuple[datetime, datetime]:
    start_text, duration_text = valid_time.split("/", 1)
    start = datetime.fromisoformat(start_text.replace("Z", "+00:00")).astimezone(timezone.utc)
    end = start + timedelta(hours=_parse_duration_hours(duration_text))
    return start, end


def _extract_exact_hour_value(values: list[dict], target_time_utc: datetime) -> tuple[float, datetime]:
    for item in values:
        value = item.get("value")
        validity_time = item.get("validTime")
        if value is None or validity_time is None:
            continue
        start, end = _parse_validity_interval(validity_time)
        if start <= target_time_utc < end:
            return float(value), start
    raise ExactForecastTimeUnavailable("No exact hourly NWS forecast value available for target time.")


def _extract_optional_exact_hour_value(values: list[dict], target_time_utc: datetime) -> tuple[float | None, datetime | None]:
    try:
        return _extract_exact_hour_value(values, target_time_utc)
    except ExactForecastTimeUnavailable:
        return None, None


def _to_fahrenheit(value: float, unit_code: str | None) -> float:
    if not unit_code:
        return value
    normalized = unit_code.lower()
    if "degc" in normalized:
        return value * 9.0 / 5.0 + 32.0
    return value


def sample_nws_hourly_forecast(lat: float, lon: float, target_time_utc: datetime) -> HourlyPointForecast:
    target_time_utc = target_time_utc.astimezone(timezone.utc)
    points_url = NWS_POINTS_URL.format(lat=round(lat, 6), lon=round(lon, 6))
    points_payload = _fetch_json(points_url)
    forecast_grid_url = points_payload["properties"]["forecastGridData"]
    forecast_payload = _fetch_json(forecast_grid_url)
    properties = forecast_payload["properties"]

    temperature_value, temp_start = _extract_optional_exact_hour_value(properties["temperature"]["values"], target_time_utc)
    precip_pct, precip_start = _extract_optional_exact_hour_value(properties["probabilityOfPrecipitation"]["values"], target_time_utc)
    sky_pct, sky_start = _extract_optional_exact_hour_value(properties["skyCover"]["values"], target_time_utc)
    if temperature_value is None and precip_pct is None and sky_pct is None:
        raise ExactForecastTimeUnavailable("No exact hourly NWS forecast values available for target time.")

    temperature_f = None
    if temperature_value is not None:
        temperature_f = _to_fahrenheit(temperature_value, properties["temperature"].get("uom"))

    source_times = [timestamp for timestamp in (temp_start, precip_start, sky_start) if timestamp is not None]
    source_time = min(source_times) if source_times else target_time_utc
    return HourlyPointForecast(
        temperature_f=temperature_f,
        precipitation_probability_pct=precip_pct,
        sky_cover_pct=sky_pct,
        source_time_utc=source_time.isoformat().replace("+00:00", "Z"),
        target_time_utc=target_time_utc.isoformat().replace("+00:00", "Z"),
        selection_mode="exact",
        forecast_grid_url=forecast_grid_url,
    )
