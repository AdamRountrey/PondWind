from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cfgrib
import numpy as np

from predictweather.forecast_models import ModelPointForecast
from predictweather.forecast_models import vector_to_speed_direction
from predictweather.grib_lock import GRIB_DECODE_LOCK


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def _open_dataset_variable(
    path: Path,
    variable_name: str | None = None,
    *,
    filter_by_keys: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with GRIB_DECODE_LOCK:
        dataset = cfgrib.open_dataset(
            path,
            indexpath="",
            filter_by_keys=filter_by_keys
            or {
                "typeOfLevel": "heightAboveGround",
                "level": 10,
            },
        )
        try:
            data_vars = [name for name in dataset.data_vars.keys()]
            if variable_name is None:
                if not data_vars:
                    raise KeyError(f"No data variables found in {path}")
                variable_name = data_vars[0]
            elif variable_name not in data_vars and len(data_vars) == 1:
                variable_name = data_vars[0]
            values = dataset[variable_name].values.astype("float32")
            latitudes = dataset["latitude"].values.astype("float64")
            longitudes = dataset["longitude"].values.astype("float64")
            valid_time = np.datetime_as_string(dataset["valid_time"].values, unit="s")
        finally:
            dataset.close()
    return values, latitudes, longitudes, valid_time


def _open_wind_component(path: Path, variable_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    return _open_dataset_variable(path, variable_name)


def _grid_for_distance(latitudes: np.ndarray, longitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if latitudes.ndim == 1 and longitudes.ndim == 1:
        longitudes_2d, latitudes_2d = np.meshgrid(longitudes, latitudes)
        return latitudes_2d, longitudes_2d
    return latitudes, longitudes


def _sample_grid_point(latitudes: np.ndarray, longitudes: np.ndarray, lat: float, lon: float) -> tuple[int, int, float, float]:
    latitudes_2d, longitudes_2d = _grid_for_distance(latitudes, longitudes)
    query_lon = lon
    if np.nanmax(longitudes_2d) > 180.0 and query_lon < 0.0:
        query_lon = lon + 360.0

    distance_metric = (latitudes_2d - lat) ** 2 + (longitudes_2d - query_lon) ** 2
    nearest_flat_index = int(np.argmin(distance_metric))
    y_index, x_index = np.unravel_index(nearest_flat_index, latitudes_2d.shape)

    point_lat = float(latitudes_2d[y_index, x_index])
    point_lon = float(longitudes_2d[y_index, x_index])
    if point_lon > 180.0:
        point_lon -= 360.0
    return int(y_index), int(x_index), point_lat, point_lon


def sample_boundary_wind_at_site(
    ugrd_path: Path,
    vgrd_path: Path,
    lat: float,
    lon: float,
    *,
    u_variable_name: str = "u10",
    v_variable_name: str = "v10",
    filter_by_keys: dict | None = None,
) -> dict:
    u10, latitudes, longitudes, valid_time = _open_dataset_variable(
        ugrd_path,
        u_variable_name,
        filter_by_keys=filter_by_keys,
    )
    v10, _, _, _ = _open_dataset_variable(
        vgrd_path,
        v_variable_name,
        filter_by_keys=filter_by_keys,
    )

    y_index, x_index, point_lat, point_lon = _sample_grid_point(latitudes, longitudes, lat, lon)
    u_value = float(u10[y_index, x_index])
    v_value = float(v10[y_index, x_index])
    speed_mps, direction_from_deg = vector_to_speed_direction(u_value, v_value)

    return {
        "valid_time_utc": valid_time,
        "site_lat": lat,
        "site_lon": lon,
        "grid_lat": point_lat,
        "grid_lon": point_lon,
        "grid_distance_km": _haversine_km(lat, lon, point_lat, point_lon),
        "grid_indices": {"y": int(y_index), "x": int(x_index)},
        "u10_mps": u_value,
        "v10_mps": v_value,
        "wind_speed_mps": speed_mps,
        "wind_from_direction_deg": direction_from_deg,
        "ugrd_path": str(ugrd_path),
        "vgrd_path": str(vgrd_path),
    }


def sample_scalar_field_at_site(
    scalar_path: Path,
    lat: float,
    lon: float,
    *,
    variable_name: str | None = None,
    filter_by_keys: dict | None = None,
) -> dict:
    values, latitudes, longitudes, valid_time = _open_dataset_variable(
        scalar_path,
        variable_name,
        filter_by_keys=filter_by_keys,
    )
    y_index, x_index, point_lat, point_lon = _sample_grid_point(latitudes, longitudes, lat, lon)
    value = float(values[y_index, x_index])
    return {
        "valid_time_utc": valid_time,
        "site_lat": lat,
        "site_lon": lon,
        "grid_lat": point_lat,
        "grid_lon": point_lon,
        "grid_distance_km": _haversine_km(lat, lon, point_lat, point_lon),
        "grid_indices": {"y": int(y_index), "x": int(x_index)},
        "value": value,
        "path": str(scalar_path),
    }


def build_model_point_forecast(
    *,
    source: str,
    display_name: str,
    run_at_utc: str,
    forecast_hour: int,
    acquisition_mode: str,
    wind_summary: dict,
    files: dict[str, str],
    gust_summary: dict | None = None,
    sustained_kind: str = "10m sustained wind",
    gust_kind: str | None = "10m gust",
) -> ModelPointForecast:
    gust_mps = None if gust_summary is None else float(gust_summary["value"])
    return ModelPointForecast(
        source=source,
        display_name=display_name,
        run_at_utc=run_at_utc,
        valid_time_utc=str(wind_summary["valid_time_utc"]),
        forecast_hour=int(forecast_hour),
        site_lat=float(wind_summary["site_lat"]),
        site_lon=float(wind_summary["site_lon"]),
        grid_lat=float(wind_summary["grid_lat"]),
        grid_lon=float(wind_summary["grid_lon"]),
        grid_distance_km=float(wind_summary["grid_distance_km"]),
        u10_mps=float(wind_summary["u10_mps"]),
        v10_mps=float(wind_summary["v10_mps"]),
        wind_speed_mps=float(wind_summary["wind_speed_mps"]),
        wind_from_direction_deg=float(wind_summary["wind_from_direction_deg"]),
        gust_mps=gust_mps,
        sustained_kind=sustained_kind,
        gust_kind=None if gust_summary is None else gust_kind,
        acquisition_mode=acquisition_mode,
        files=files,
    )


def _parse_iso_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _manifest_is_complete(manifest: dict) -> bool:
    files = manifest.get("files", {})
    for variable in ("UGRD", "VGRD"):
        path_text = files.get(variable)
        if not path_text:
            return False
        if not Path(path_text).exists():
            return False
    return True


def _candidate_manifest_sort_key(manifest: dict, manifest_path: Path, target_valid_time_utc: datetime) -> tuple[float, float, str]:
    selected_valid = _parse_iso_utc(manifest["selected_valid_at_utc"])
    run_at = _parse_iso_utc(manifest["run_at_utc"])
    return (
        abs((selected_valid - target_valid_time_utc).total_seconds()),
        abs((run_at - target_valid_time_utc).total_seconds()),
        str(manifest_path),
    )


def latest_hrdps_manifest(root: Path) -> Path:
    manifests: list[tuple[dict, Path]] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "selected_valid_at_utc" not in manifest or "run_at_utc" not in manifest:
            continue
        if not _manifest_is_complete(manifest):
            continue
        manifests.append((manifest, manifest_path))
    if not manifests:
        raise FileNotFoundError(f"No HRDPS manifest found under {root}")
    manifests.sort(key=lambda item: (_parse_iso_utc(item[0]["selected_valid_at_utc"]), _parse_iso_utc(item[0]["run_at_utc"]), str(item[1])))
    return manifests[-1][1]


def load_cached_hrdps_manifest_for_valid_time(
    root: Path,
    target_valid_time_utc: datetime,
    *,
    max_valid_time_delta: timedelta = timedelta(minutes=0),
) -> tuple[dict, Path]:
    target_valid_time_utc = target_valid_time_utc.astimezone(timezone.utc)
    candidates: list[tuple[dict, Path]] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "selected_valid_at_utc" not in manifest or "run_at_utc" not in manifest:
            continue
        if not _manifest_is_complete(manifest):
            continue
        selected_valid = _parse_iso_utc(manifest["selected_valid_at_utc"])
        if abs(selected_valid - target_valid_time_utc) > max_valid_time_delta:
            continue
        candidates.append((manifest, manifest_path))

    if not candidates:
        raise FileNotFoundError(f"No usable HRDPS manifest found under {root} for {target_valid_time_utc.isoformat()}")

    manifest, manifest_path = min(
        candidates,
        key=lambda item: _candidate_manifest_sort_key(item[0], item[1], target_valid_time_utc),
    )
    return manifest, manifest_path


def load_latest_hrdps_manifest(root: Path) -> tuple[dict, Path]:
    manifest_path = latest_hrdps_manifest(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest, manifest_path


def sample_latest_hrdps_boundary(root: Path, lat: float, lon: float) -> tuple[dict, Path]:
    manifest_path = latest_hrdps_manifest(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = sample_boundary_wind_at_site(
        ugrd_path=Path(manifest["files"]["UGRD"]),
        vgrd_path=Path(manifest["files"]["VGRD"]),
        lat=lat,
        lon=lon,
    )
    summary["manifest_path"] = str(manifest_path)
    summary_path = manifest_path.with_name("site_boundary_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, summary_path


def load_latest_hrdps_boundary_summary(root: Path) -> tuple[dict, Path]:
    summary_paths = sorted(root.rglob("site_boundary_summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No HRDPS boundary summary found under {root}")
    summary_path = summary_paths[-1]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary, summary_path
