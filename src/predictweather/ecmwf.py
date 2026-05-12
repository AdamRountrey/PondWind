from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from predictweather.boundary import build_model_point_forecast, sample_boundary_wind_at_site, sample_scalar_field_at_site
from predictweather.http import download_url_to_file, env_allows_insecure_ssl


RUN_HOURS_UTC = (0, 6, 12, 18)
STEP_HOURS = 3
MAX_HORIZON_HOURS = 144


@dataclass(frozen=True)
class EcmwfSelection:
    requested_valid_time_utc: str
    selected_valid_time_utc: str
    run_at_utc: str
    forecast_hour: int
    files: dict[str, str]


def floor_to_3h(timestamp: datetime) -> datetime:
    timestamp = timestamp.astimezone(timezone.utc)
    floored_hour = timestamp.hour - (timestamp.hour % STEP_HOURS)
    return timestamp.replace(hour=floored_hour, minute=0, second=0, microsecond=0)


def candidate_runs_for_valid_time(valid_at_utc: datetime) -> list[tuple[datetime, int]]:
    valid_at_utc = floor_to_3h(valid_at_utc)
    candidates: list[tuple[datetime, int]] = []
    for offset_hours in range(0, MAX_HORIZON_HOURS + 25):
        run_at = valid_at_utc - timedelta(hours=offset_hours)
        if run_at.hour not in RUN_HOURS_UTC:
            continue
        forecast_hour = int((valid_at_utc - run_at).total_seconds() // 3600)
        if forecast_hour < 0 or forecast_hour > MAX_HORIZON_HOURS or forecast_hour % STEP_HOURS != 0:
            continue
        candidates.append((run_at, forecast_hour))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _base_name(run_at_utc: datetime, forecast_hour: int) -> str:
    stream = stream_for_run(run_at_utc)
    return f"{run_at_utc:%Y%m%d%H}0000-{forecast_hour}h-{stream}-fc"


def stream_for_run(run_at_utc: datetime) -> str:
    return "oper" if run_at_utc.hour in {0, 12} else "scda"


def build_grib_url(run_at_utc: datetime, forecast_hour: int) -> str:
    stream = stream_for_run(run_at_utc)
    return f"https://data.ecmwf.int/forecasts/{run_at_utc:%Y%m%d}/{run_at_utc:%H}z/ifs/0p25/{stream}/{_base_name(run_at_utc, forecast_hour)}.grib2"


def build_index_url(run_at_utc: datetime, forecast_hour: int) -> str:
    return build_grib_url(run_at_utc, forecast_hour).replace(".grib2", ".index")


def _parse_index_lines(text: str) -> list[dict]:
    entries: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        entries.append(json.loads(stripped))
    return entries


def _fetch_text(url: str) -> str:
    with urlopen(url, timeout=120) as response:
        return response.read().decode("utf-8", errors="replace")


def _url_exists(url: str) -> bool:
    try:
        with urlopen(url, timeout=60) as response:
            return response.status == 200
    except HTTPError:
        return False


def _resolve_length(entries: list[dict], index: int) -> int:
    entry = entries[index]
    if "_length" in entry:
        return int(entry["_length"])
    start = int(entry["_offset"])
    if index + 1 < len(entries):
        return int(entries[index + 1]["_offset"]) - start
    raise ValueError("Unable to infer ECMWF byte-range length for final index entry.")


def _download_param_range(run_at_utc: datetime, forecast_hour: int, param_names: set[str], destination: Path) -> Path | None:
    index_url = build_index_url(run_at_utc, forecast_hour)
    index_entries = _parse_index_lines(_fetch_text(index_url))
    entry_index = next(
        (
            idx
            for idx, entry in enumerate(index_entries)
            if str(entry.get("levtype", "")).lower() == "sfc" and str(entry.get("param", "")).lower() in param_names
        ),
        None,
    )
    if entry_index is None:
        return None
    entry = index_entries[entry_index]
    offset = int(entry["_offset"])
    length = _resolve_length(index_entries, entry_index)
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    return download_url_to_file(
        build_grib_url(run_at_utc, forecast_hour),
        destination,
        allow_insecure=env_allows_insecure_ssl(),
        headers=headers,
    )


def select_best_run(valid_at_utc: datetime) -> tuple[datetime, int]:
    for run_at_utc, forecast_hour in candidate_runs_for_valid_time(valid_at_utc):
        if _url_exists(build_index_url(run_at_utc, forecast_hour)):
            return run_at_utc, forecast_hour
    raise FileNotFoundError(f"No ECMWF run found for valid time {valid_at_utc.isoformat()}")


def download_ecmwf_for_valid_time(destination_dir: Path, valid_at_utc: datetime) -> EcmwfSelection:
    selected_valid_time_utc = floor_to_3h(valid_at_utc)
    run_at_utc, forecast_hour = select_best_run(selected_valid_time_utc)
    output_dir = destination_dir / "ecmwf" / run_at_utc.strftime("%Y%m%dT%HZ") / f"F{forecast_hour:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = _base_name(run_at_utc, forecast_hour)
    files = {
        "UGRD": str(output_dir / f"{base_name}_10u.grib2"),
        "VGRD": str(output_dir / f"{base_name}_10v.grib2"),
        "GUST": str(output_dir / f"{base_name}_10fg.grib2"),
    }
    specs = (
        ("UGRD", {"10u"}),
        ("VGRD", {"10v"}),
        ("GUST", {"10fg", "10fg3"}),
    )
    for file_key, param_names in specs:
        destination = Path(files[file_key])
        if not destination.exists():
            downloaded = _download_param_range(run_at_utc, forecast_hour, param_names, destination)
            if downloaded is None:
                if file_key == "GUST":
                    files.pop("GUST", None)
                    continue
                raise FileNotFoundError(f"No ECMWF field found for {file_key} at {run_at_utc.isoformat()} +{forecast_hour}h")

    selection = EcmwfSelection(
        requested_valid_time_utc=valid_at_utc.astimezone(timezone.utc).isoformat(),
        selected_valid_time_utc=selected_valid_time_utc.isoformat(),
        run_at_utc=run_at_utc.isoformat(),
        forecast_hour=forecast_hour,
        files=files,
    )
    (output_dir / "manifest.json").write_text(json.dumps(selection.__dict__, indent=2), encoding="utf-8")
    return selection


def sample_ecmwf_point_forecast(selection: EcmwfSelection, lat: float, lon: float, acquisition_mode: str) -> dict:
    wind_summary = sample_boundary_wind_at_site(
        ugrd_path=Path(selection.files["UGRD"]),
        vgrd_path=Path(selection.files["VGRD"]),
        lat=lat,
        lon=lon,
        u_variable_name="u10",
        v_variable_name="v10",
    )
    gust_summary = None
    if "GUST" in selection.files and selection.forecast_hour > 0:
        gust_summary = sample_scalar_field_at_site(
            Path(selection.files["GUST"]),
            lat,
            lon,
        )
    forecast = build_model_point_forecast(
        source="ecmwf",
        display_name="ecmwf",
        run_at_utc=selection.run_at_utc,
        forecast_hour=selection.forecast_hour,
        acquisition_mode=acquisition_mode,
        wind_summary=wind_summary,
        gust_summary=gust_summary,
        files=selection.files,
        gust_kind="10m gust max since previous post-processing",
    )
    return forecast.as_dict()
