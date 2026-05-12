from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from predictweather.boundary import build_model_point_forecast, sample_boundary_wind_at_site, sample_scalar_field_at_site
from predictweather.nomads import download_nomads_subset, nomads_subset_exists, run_candidates


RUN_HOURS_UTC = (0, 6, 12, 18)
MAX_HORIZON_HOURS = 60


@dataclass(frozen=True)
class NamSelection:
    requested_valid_time_utc: str
    selected_valid_time_utc: str
    run_at_utc: str
    forecast_hour: int
    files: dict[str, str]


def build_nam_filename(run_at_utc: datetime, forecast_hour: int) -> str:
    return f"nam.t{run_at_utc:%H}z.conusnest.hiresf{forecast_hour:02d}.tm00.grib2"


def _dir_path(run_at_utc: datetime) -> str:
    return f"/nam.{run_at_utc:%Y%m%d}"


def select_best_run(valid_at_utc: datetime, lat: float, lon: float) -> tuple[datetime, int]:
    for run_at_utc, forecast_hour in run_candidates(
        valid_at_utc,
        run_hours_utc=RUN_HOURS_UTC,
        max_horizon_hours=MAX_HORIZON_HOURS,
        step_hours=1,
    ):
        file_name = build_nam_filename(run_at_utc, forecast_hour)
        dir_path = _dir_path(run_at_utc)
        if all(
            nomads_subset_exists(
                filter_script="filter_nam_conusnest.pl",
                file_name=file_name,
                dir_path=dir_path,
                variable_name=variable_name,
                level_flag=level_flag,
                lat=lat,
                lon=lon,
            )
            for variable_name, level_flag in (("UGRD", "lev_10_m_above_ground"), ("VGRD", "lev_10_m_above_ground"))
        ):
            return run_at_utc, forecast_hour
    raise FileNotFoundError(f"No NAM CONUS Nest run found for valid time {valid_at_utc.isoformat()}")


def download_nam_for_valid_time(destination_dir: Path, valid_at_utc: datetime, lat: float, lon: float) -> NamSelection:
    selected_valid_time_utc = valid_at_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    run_at_utc, forecast_hour = select_best_run(selected_valid_time_utc, lat, lon)
    output_dir = destination_dir / "nam" / run_at_utc.strftime("%Y%m%dT%HZ") / f"F{forecast_hour:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = build_nam_filename(run_at_utc, forecast_hour)
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
            download_nomads_subset(
                filter_script="filter_nam_conusnest.pl",
                file_name=file_name,
                dir_path=dir_path,
                variable_name=variable_name,
                level_flag=level_flag,
                lat=lat,
                lon=lon,
                destination=destination,
            )

    gust_exists = nomads_subset_exists(
        filter_script="filter_nam_conusnest.pl",
        file_name=file_name,
        dir_path=dir_path,
        variable_name="GUST",
        level_flag="lev_surface",
        lat=lat,
        lon=lon,
    )
    if gust_exists:
        gust_destination = output_dir / f"{file_name}_gust.grib2"
        if not gust_destination.exists():
            download_nomads_subset(
                filter_script="filter_nam_conusnest.pl",
                file_name=file_name,
                dir_path=dir_path,
                variable_name="GUST",
                level_flag="lev_surface",
                lat=lat,
                lon=lon,
                destination=gust_destination,
            )
        files["GUST"] = str(gust_destination)

    selection = NamSelection(
        requested_valid_time_utc=valid_at_utc.astimezone(timezone.utc).isoformat(),
        selected_valid_time_utc=selected_valid_time_utc.isoformat(),
        run_at_utc=run_at_utc.isoformat(),
        forecast_hour=forecast_hour,
        files=files,
    )
    (output_dir / "manifest.json").write_text(json.dumps(selection.__dict__, indent=2), encoding="utf-8")
    return selection


def sample_nam_point_forecast(selection: NamSelection, lat: float, lon: float, acquisition_mode: str) -> dict:
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
        source="nam",
        display_name="nam",
        run_at_utc=selection.run_at_utc,
        forecast_hour=selection.forecast_hour,
        acquisition_mode=acquisition_mode,
        wind_summary=wind_summary,
        gust_summary=gust_summary,
        files=selection.files,
    )
    return forecast.as_dict()
