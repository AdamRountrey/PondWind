from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

from predictweather.boundary import build_model_point_forecast, sample_boundary_wind_at_site, sample_scalar_field_at_site
from predictweather.noaa_pds import download_indexed_message, has_indexed_message, has_indexed_messages
from predictweather.nomads import download_nomads_subset, nomads_subset_exists, run_candidates


RUN_HOURS_UTC = tuple(range(24))
MAX_HORIZON_HOURS = 48
HRRR_PDS_BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"


@dataclass(frozen=True)
class HrrrSelection:
    requested_valid_time_utc: str
    selected_valid_time_utc: str
    run_at_utc: str
    forecast_hour: int
    files: dict[str, str]


def build_hrrr_filename(run_at_utc: datetime, forecast_hour: int) -> str:
    return f"hrrr.t{run_at_utc:%H}z.wrfsfcf{forecast_hour:02d}.grib2"


def _dir_path(run_at_utc: datetime) -> str:
    return f"/hrrr.{run_at_utc:%Y%m%d}/conus"


def build_hrrr_pds_url(run_at_utc: datetime, forecast_hour: int) -> str:
    file_name = build_hrrr_filename(run_at_utc, forecast_hour)
    return f"{HRRR_PDS_BASE_URL}/hrrr.{run_at_utc:%Y%m%d}/conus/{file_name}"


def _pds_has_required_fields(run_at_utc: datetime, forecast_hour: int) -> bool:
    grib_url = build_hrrr_pds_url(run_at_utc, forecast_hour)
    return has_indexed_messages(grib_url, (("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")))


def _nomads_has_required_fields(run_at_utc: datetime, forecast_hour: int, lat: float, lon: float) -> bool:
    file_name = build_hrrr_filename(run_at_utc, forecast_hour)
    dir_path = _dir_path(run_at_utc)
    try:
        return all(
            nomads_subset_exists(
                filter_script="filter_hrrr_2d.pl",
                file_name=file_name,
                dir_path=dir_path,
                variable_name=variable_name,
                level_flag=level_flag,
                lat=lat,
                lon=lon,
            )
            for variable_name, level_flag in (("UGRD", "lev_10_m_above_ground"), ("VGRD", "lev_10_m_above_ground"))
        )
    except (HTTPError, URLError, TimeoutError):
        return False


def select_best_run(valid_at_utc: datetime, lat: float, lon: float) -> tuple[datetime, int]:
    for run_at_utc, forecast_hour in run_candidates(
        valid_at_utc,
        run_hours_utc=RUN_HOURS_UTC,
        max_horizon_hours=MAX_HORIZON_HOURS,
        step_hours=1,
    ):
        if _pds_has_required_fields(run_at_utc, forecast_hour) or _nomads_has_required_fields(run_at_utc, forecast_hour, lat, lon):
            return run_at_utc, forecast_hour
    raise FileNotFoundError(f"No HRRR run found for valid time {valid_at_utc.isoformat()}")


def download_hrrr_for_valid_time(destination_dir: Path, valid_at_utc: datetime, lat: float, lon: float) -> HrrrSelection:
    selected_valid_time_utc = valid_at_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    run_at_utc, forecast_hour = select_best_run(selected_valid_time_utc, lat, lon)
    output_dir = destination_dir / "hrrr" / run_at_utc.strftime("%Y%m%dT%HZ") / f"F{forecast_hour:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = build_hrrr_filename(run_at_utc, forecast_hour)
    dir_path = _dir_path(run_at_utc)

    files = {
        "UGRD": str(output_dir / f"{file_name}_ugrd.grib2"),
        "VGRD": str(output_dir / f"{file_name}_vgrd.grib2"),
    }
    for variable_name, level_flag in (
        ("UGRD", "lev_10_m_above_ground"),
        ("VGRD", "lev_10_m_above_ground"),
    ):
        destination = Path(files[variable_name])
        if not destination.exists():
            try:
                download_indexed_message(
                    build_hrrr_pds_url(run_at_utc, forecast_hour),
                    destination,
                    variable=variable_name,
                    level_text="10 m above ground",
                )
            except Exception:
                download_nomads_subset(
                    filter_script="filter_hrrr_2d.pl",
                    file_name=file_name,
                    dir_path=dir_path,
                    variable_name=variable_name,
                    level_flag=level_flag,
                    lat=lat,
                    lon=lon,
                    destination=destination,
                )

    grib_url = build_hrrr_pds_url(run_at_utc, forecast_hour)
    try:
        nomads_gust_exists = nomads_subset_exists(
            filter_script="filter_hrrr_2d.pl",
            file_name=file_name,
            dir_path=dir_path,
            variable_name="GUST",
            level_flag="lev_surface",
            lat=lat,
            lon=lon,
        )
    except (HTTPError, URLError, TimeoutError):
        nomads_gust_exists = False
    gust_exists = has_indexed_message(grib_url, variable="GUST", level_text="surface") or nomads_gust_exists
    if gust_exists:
        gust_destination = output_dir / f"{file_name}_gust.grib2"
        if not gust_destination.exists():
            try:
                download_indexed_message(grib_url, gust_destination, variable="GUST", level_text="surface")
            except Exception:
                download_nomads_subset(
                    filter_script="filter_hrrr_2d.pl",
                    file_name=file_name,
                    dir_path=dir_path,
                    variable_name="GUST",
                    level_flag="lev_surface",
                    lat=lat,
                    lon=lon,
                    destination=gust_destination,
                )
        files["GUST"] = str(gust_destination)

    selection = HrrrSelection(
        requested_valid_time_utc=valid_at_utc.astimezone(timezone.utc).isoformat(),
        selected_valid_time_utc=selected_valid_time_utc.isoformat(),
        run_at_utc=run_at_utc.isoformat(),
        forecast_hour=forecast_hour,
        files=files,
    )
    (output_dir / "manifest.json").write_text(json.dumps(selection.__dict__, indent=2), encoding="utf-8")
    return selection


def sample_hrrr_point_forecast(selection: HrrrSelection, lat: float, lon: float, acquisition_mode: str) -> dict:
    wind_summary = sample_boundary_wind_at_site(
        ugrd_path=Path(selection.files["UGRD"]),
        vgrd_path=Path(selection.files["VGRD"]),
        lat=lat,
        lon=lon,
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10},
    )
    gust_summary = None
    gust_path = selection.files.get("GUST")
    if gust_path:
        gust_summary = sample_scalar_field_at_site(
            Path(gust_path),
            lat,
            lon,
            filter_by_keys={"typeOfLevel": "surface"},
        )
    forecast = build_model_point_forecast(
        source="hrrr",
        display_name="hrrr",
        run_at_utc=selection.run_at_utc,
        forecast_hour=selection.forecast_hour,
        acquisition_mode=acquisition_mode,
        wind_summary=wind_summary,
        gust_summary=gust_summary,
        files=selection.files,
    )
    return forecast.as_dict()
