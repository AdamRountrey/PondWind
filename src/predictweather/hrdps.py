from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

from predictweather.http import download_url_to_file, env_allows_insecure_ssl, url_exists
from predictweather.usgs import download_file

HRDPS_BASE_URL = "https://dd.weather.gc.ca/today/model_hrdps/continental/2.5km"
RUN_HOURS_UTC = (0, 6, 12, 18)
BOUNDARY_VARIABLES = ("UGRD", "VGRD")
GUST_VARIABLE_CANDIDATES = ("WindGust", "GUST")


@dataclass(frozen=True)
class HrdpsSelection:
    requested_at_utc: str
    target_at_utc: str
    selected_valid_at_utc: str
    run_at_utc: str
    forecast_hour: int
    variables: tuple[str, ...]
    files: dict[str, str]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ceil_to_next_hour(timestamp: datetime) -> datetime:
    timestamp = timestamp.astimezone(timezone.utc)
    rounded = timestamp.replace(minute=0, second=0, microsecond=0)
    if rounded < timestamp:
        rounded += timedelta(hours=1)
    return rounded


def build_run_datetime(valid_at_utc: datetime, run_hour_utc: int) -> datetime:
    run_at = valid_at_utc.replace(hour=run_hour_utc, minute=0, second=0, microsecond=0)
    if run_at > valid_at_utc:
        run_at -= timedelta(days=1)
    return run_at


def candidate_runs_for_valid_time(valid_at_utc: datetime) -> list[tuple[datetime, int]]:
    candidates: list[tuple[datetime, int]] = []
    latest_candidate = valid_at_utc.replace(minute=0, second=0, microsecond=0)
    for offset_hours in range(0, 49):
        run_at = latest_candidate - timedelta(hours=offset_hours)
        if run_at.hour not in RUN_HOURS_UTC:
            continue
        forecast_hour = int((valid_at_utc - run_at).total_seconds() // 3600)
        if 0 <= forecast_hour <= 48:
            candidates.append((run_at, forecast_hour))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def build_hrdps_filename(run_at_utc: datetime, variable: str, forecast_hour: int) -> str:
    run_stamp = run_at_utc.strftime("%Y%m%dT%HZ")
    return f"{run_stamp}_MSC_HRDPS_{variable}_AGL-10m_RLatLon0.0225_PT{forecast_hour:03d}H.grib2"


def build_hrdps_url(run_at_utc: datetime, variable: str, forecast_hour: int) -> str:
    filename = build_hrdps_filename(run_at_utc, variable, forecast_hour)
    return f"{HRDPS_BASE_URL}/{run_at_utc:%H}/{forecast_hour:03d}/{filename}"


def try_download_hrdps_variable(
    destination_dir: Path,
    *,
    run_at_utc: datetime,
    forecast_hour: int,
    variable_names: tuple[str, ...],
) -> tuple[str, Path] | None:
    output_dir = destination_dir / "hrdps" / run_at_utc.strftime("%Y%m%dT%HZ") / f"PT{forecast_hour:03d}H"
    output_dir.mkdir(parents=True, exist_ok=True)
    for variable in variable_names:
        url = build_hrdps_url(run_at_utc, variable, forecast_hour)
        destination = output_dir / build_hrdps_filename(run_at_utc, variable, forecast_hour)
        try:
            if not destination.exists():
                if not _url_exists(url):
                    continue
                download_url_to_file(url, destination, allow_insecure=env_allows_insecure_ssl())
            return variable, destination
        except Exception:
            continue
    return None


def _url_exists(url: str) -> bool:
    try:
        return url_exists(url, allow_insecure=env_allows_insecure_ssl())
    except HTTPError:
        return False
    except URLError:
        raise


def _parse_iso_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _manifest_is_complete(manifest: dict) -> bool:
    files = manifest.get("files", {})
    for variable in BOUNDARY_VARIABLES:
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


def load_cached_hrdps_manifest_for_valid_time(root: Path, target_valid_time_utc: datetime) -> tuple[dict, Path]:
    target_valid_time_utc = target_valid_time_utc.astimezone(timezone.utc)
    candidates: list[tuple[dict, Path]] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "selected_valid_at_utc" not in manifest or "run_at_utc" not in manifest:
            continue
        if not _manifest_is_complete(manifest):
            continue
        candidates.append((manifest, manifest_path))

    if not candidates:
        raise FileNotFoundError(f"No usable HRDPS manifest found under {root}")

    manifest, manifest_path = min(
        candidates,
        key=lambda item: _candidate_manifest_sort_key(item[0], item[1], target_valid_time_utc),
    )
    return manifest, manifest_path


def select_best_run(target_at_utc: datetime) -> tuple[datetime, int]:
    last_error: Exception | None = None
    for run_at, forecast_hour in candidate_runs_for_valid_time(target_at_utc):
        all_present = True
        for variable in BOUNDARY_VARIABLES:
            probe_url = build_hrdps_url(run_at, variable, forecast_hour)
            try:
                if not _url_exists(probe_url):
                    all_present = False
                    break
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                all_present = False
                break
        if all_present:
            return run_at, forecast_hour

    if last_error is not None:
        raise last_error

    raise FileNotFoundError(f"No HRDPS run found for valid time {target_at_utc.isoformat()}")


def download_hrdps_boundary_conditions(
    destination_dir: Path,
    lead_hours: float = 2.0,
    requested_at_utc: datetime | None = None,
) -> HrdpsSelection:
    requested_at_utc = requested_at_utc or utc_now()
    target_at_utc = requested_at_utc + timedelta(hours=lead_hours)
    selected_valid_at_utc = ceil_to_next_hour(target_at_utc)

    run_at_utc, forecast_hour = select_best_run(selected_valid_at_utc)

    output_dir = destination_dir / "hrdps" / run_at_utc.strftime("%Y%m%dT%HZ") / f"PT{forecast_hour:03d}H"
    output_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    for variable in BOUNDARY_VARIABLES:
        url = build_hrdps_url(run_at_utc, variable, forecast_hour)
        destination = output_dir / build_hrdps_filename(run_at_utc, variable, forecast_hour)
        download_url_to_file(url, destination, allow_insecure=env_allows_insecure_ssl())
        files[variable] = str(destination)

    selection = HrdpsSelection(
        requested_at_utc=requested_at_utc.isoformat(),
        target_at_utc=target_at_utc.isoformat(),
        selected_valid_at_utc=selected_valid_at_utc.isoformat(),
        run_at_utc=run_at_utc.isoformat(),
        forecast_hour=forecast_hour,
        variables=BOUNDARY_VARIABLES,
        files=files,
    )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(selection.__dict__, indent=2), encoding="utf-8")
    return selection


def download_hrdps_for_valid_time(
    destination_dir: Path,
    valid_at_utc: datetime,
) -> HrdpsSelection:
    selected_valid_at_utc = ceil_to_next_hour(valid_at_utc)
    run_at_utc, forecast_hour = select_best_run(selected_valid_at_utc)

    output_dir = destination_dir / "hrdps" / run_at_utc.strftime("%Y%m%dT%HZ") / f"PT{forecast_hour:03d}H"
    output_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    for variable in BOUNDARY_VARIABLES:
        url = build_hrdps_url(run_at_utc, variable, forecast_hour)
        destination = output_dir / build_hrdps_filename(run_at_utc, variable, forecast_hour)
        if not destination.exists():
            download_url_to_file(url, destination, allow_insecure=env_allows_insecure_ssl())
        files[variable] = str(destination)

    selection = HrdpsSelection(
        requested_at_utc=utc_now().isoformat(),
        target_at_utc=selected_valid_at_utc.isoformat(),
        selected_valid_at_utc=selected_valid_at_utc.isoformat(),
        run_at_utc=run_at_utc.isoformat(),
        forecast_hour=forecast_hour,
        variables=BOUNDARY_VARIABLES,
        files=files,
    )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(selection.__dict__, indent=2), encoding="utf-8")
    return selection
