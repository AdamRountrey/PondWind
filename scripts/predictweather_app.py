from __future__ import annotations

import ctypes
import json
import os
import sys
import traceback
from pathlib import Path


def _candidate_log_paths() -> list[Path]:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "startup_error.txt")
    else:
        candidates.append(Path(__file__).resolve().parents[1] / "startup_error.txt")

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "PondWind" / "startup_error.txt")

    temp_dir = os.environ.get("TEMP")
    if temp_dir:
        candidates.append(Path(temp_dir) / "PondWind_startup_error.txt")
    return candidates


def _write_startup_log(text: str) -> Path | None:
    for path in _candidate_log_paths():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return path
        except Exception:
            continue
    return None


def _show_startup_error(message: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(0, message, "PondWind Startup Error", 0x10)
    except Exception:
        pass


def _runtime_project_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data).resolve() / "PondWind"
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "user_data"
    return Path(__file__).resolve().parents[1]


def _emit_worker_event(kind: str, **payload: object) -> None:
    message = {"kind": kind, **payload}
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _default_openfoam_runner_command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --openfoam-wsl-runner'
    runner_path = Path(__file__).resolve().parent / "openfoam_wsl_terrain_runner.py"
    return f'"{sys.executable}" "{runner_path}"'


def _run_report_worker(argv: list[str]) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if len(argv) not in {10, 14}:
        _emit_worker_event("error", message="Worker received invalid arguments.")
        return 2

    satellite_rgb = satellite_sst = satellite_chla = satellite_turbidity = "1"
    if len(argv) == 14:
        (
            race_time,
            center_lat,
            center_lon,
            side_meters,
            site_label,
            mesh_resolution,
            wind_solver,
            report_output_dir,
            allow_insecure_ssl,
            force_ecostress_sst,
            satellite_rgb,
            satellite_sst,
            satellite_chla,
            satellite_turbidity,
        ) = argv
    else:
        race_time, center_lat, center_lon, side_meters, site_label, mesh_resolution, wind_solver, report_output_dir, allow_insecure_ssl, force_ecostress_sst = argv
    if wind_solver == "openfoam" and not os.environ.get("PONDWIND_OPENFOAM_RUNNER"):
        os.environ["PONDWIND_OPENFOAM_RUNNER"] = _default_openfoam_runner_command()
    _emit_worker_event("progress", percent=1, message="Initializing report engine...")
    try:
        from run_barton_weekly_report import build_weekly_report

        def progress_callback(percent: int, message: str) -> None:
            _emit_worker_event("progress", percent=max(0, min(100, int(percent))), message=message)

        _emit_worker_event("log", text="Report worker started.\n")
        report_path, manifest_path = build_weekly_report(
            race_local_datetime=race_time,
            center_lat=float(center_lat),
            center_lon=float(center_lon),
            side_meters=float(side_meters),
            site_label=site_label,
            mesh_resolution=float(mesh_resolution),
            wind_solver=wind_solver,
            report_output_dir=(report_output_dir or None),
            allow_insecure_ssl=(allow_insecure_ssl == "1"),
            force_ecostress_sst=(force_ecostress_sst == "1"),
            satellite_rgb=(satellite_rgb == "1"),
            satellite_sst=(satellite_sst == "1"),
            satellite_chla=(satellite_chla == "1"),
            satellite_turbidity=(satellite_turbidity == "1"),
            progress_callback=progress_callback,
        )
        _emit_worker_event("log", text=f"Report written: {report_path}\n")
        _emit_worker_event("log", text=f"Manifest written: {manifest_path}\n")
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            wind = manifest.get("wind", {})
            for warning in wind.get("warnings", []) or []:
                _emit_worker_event("log", text=f"Wind acquisition warning: {warning}\n")
            done_message = (
                "Report build completed with wind acquisition warnings."
                if wind.get("degraded")
                else "Report build completed successfully."
            )
        except Exception:
            done_message = "Report build completed successfully."
        _emit_worker_event("report_path", path=str(report_path))
        _emit_worker_event("done", message=done_message)
        return 0
    except Exception as exc:
        error_root = Path(report_output_dir).expanduser().resolve() if report_output_dir else _runtime_project_root()
        error_root.mkdir(parents=True, exist_ok=True)
        error_log = error_root / "PondWind_report_build_error.txt"
        if hasattr(exc, "stage") or hasattr(exc, "details") or hasattr(exc, "message"):
            payload = {
                "stage": getattr(exc, "stage", "report_build"),
                "message": getattr(exc, "message", str(exc)),
                "details": getattr(exc, "details", {}) or {},
                "traceback": traceback.format_exc(),
            }
            error_log.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _emit_worker_event("progress", percent=0, message="Report build failed")
            _emit_worker_event("error", message=f"{payload['stage']}: {payload['message']}")
            _emit_worker_event("log", text=f"\nFailure stage: {payload['stage']}\n")
            _emit_worker_event("log", text=f"Detailed error log: {error_log}\n")
            if payload["details"]:
                _emit_worker_event("log", text=json.dumps(payload["details"], indent=2) + "\n")
            return 1

        error_log.write_text(traceback.format_exc(), encoding="utf-8")
        _emit_worker_event("progress", percent=0, message="Report build failed")
        _emit_worker_event("error", message=f"Unable to launch report build: {exc}")
        _emit_worker_event("log", text=f"\nDetailed traceback written to: {error_log}\n")
        return 1


def main() -> None:
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--worker-report-build":
            sys.exit(_run_report_worker(sys.argv[2:]))
        if len(sys.argv) > 1 and sys.argv[1] == "--openfoam-wsl-runner":
            from openfoam_wsl_terrain_runner import main as openfoam_runner_main

            sys.argv = [sys.argv[0], *sys.argv[2:]]
            openfoam_runner_main()
            return
        if len(sys.argv) > 1 and sys.argv[1] == "--openfoam-uniform-runner":
            from openfoam_uniform_runner import main as uniform_runner_main

            sys.argv = [sys.argv[0], *sys.argv[2:]]
            uniform_runner_main()
            return
        from predictweather_gui import main as gui_main

        gui_main()
    except Exception as exc:
        trace_text = traceback.format_exc()
        log_path = _write_startup_log(trace_text)
        log_hint = str(log_path) if log_path is not None else "unable to write startup log"
        _show_startup_error(f"{exc}\n\nTraceback saved to:\n{log_hint}")
        sys.exit(1)


if __name__ == "__main__":
    main()
