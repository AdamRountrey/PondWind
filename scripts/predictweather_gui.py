from __future__ import annotations

import queue
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, VERTICAL, W, Button, Checkbutton, Entry, Frame, IntVar, Label, StringVar, Text, Tk, Toplevel, filedialog
from tkinter import ttk
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if not getattr(sys, "frozen", False) and str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if not getattr(sys, "frozen", False) and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _runtime_project_root() -> Path:
    if getattr(sys, "frozen", False):
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data).resolve() / "PondWind"
        return Path(sys.executable).resolve().parent / "user_data"
    return PROJECT_ROOT


def _local_tz() -> ZoneInfo:
    try:
        return ZoneInfo("America/New_York")
    except Exception:
        return ZoneInfo("UTC")


def _default_race_local_datetime() -> str:
    now_local = datetime.now(_local_tz())
    days_ahead = (6 - now_local.weekday()) % 7
    candidate = (now_local + timedelta(days=days_ahead)).replace(hour=14, minute=0, second=0, microsecond=0)
    if candidate <= now_local:
        candidate += timedelta(days=7)
    return candidate.strftime("%Y-%m-%dT%H:%M:%S")


class PredictWeatherGui:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("PondWind Report Builder")
        self.root.geometry("860x680")

        self.race_time_var = StringVar(value=_default_race_local_datetime())
        self.center_lat_var = StringVar(value="42.31295753946108")
        self.center_lon_var = StringVar(value="-83.75641581917375")
        self.side_meters_var = StringVar(value="1609.344")
        self.site_label_var = StringVar(value="barton_pond")
        self.mesh_resolution_var = StringVar(value="30")
        self.wind_solver_var = StringVar(value="windninja")
        self.report_output_dir_var = StringVar(value=str(_runtime_project_root() / "outputs" / "reports"))
        self.allow_insecure_ssl_var = IntVar(value=0)
        self.force_ecostress_sst_var = IntVar(value=0)
        self.satellite_rgb_var = IntVar(value=1)
        self.satellite_sst_var = IntVar(value=1)
        self.satellite_chla_var = IntVar(value=1)
        self.satellite_turbidity_var = IntVar(value=1)
        self.status_var = StringVar(value="Ready.")
        self.last_report_path_var = StringVar(value="")
        self.progress_var = IntVar(value=0)

        self.output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.is_running = False
        self.last_report_file: Path | None = None
        self.worker_process: subprocess.Popen[str] | None = None

        self._build_layout()
        self.root.after(150, self._drain_queue)


    def _build_layout(self) -> None:
        container = Frame(self.root, padx=16, pady=16)
        container.pack(fill=BOTH, expand=True)

        form = ttk.LabelFrame(container, text="Report Inputs", padding=12)
        form.pack(fill="x")

        fields = [
            ("Race time", self.race_time_var, "YYYY-MM-DDTHH:MM:SS in America/New_York"),
            ("Center lat", self.center_lat_var, "Latitude"),
            ("Center lon", self.center_lon_var, "Longitude"),
            ("Side meters", self.side_meters_var, "Default is 1609.344"),
            ("Site label", self.site_label_var, "Used in folder names"),
            ("Wind grid size", self.mesh_resolution_var, "Meters per cell. 30 default; 10-15 is slower and more detailed"),
        ]

        for row, (label_text, variable, hint) in enumerate(fields):
            Label(form, text=label_text, anchor="w", width=16).grid(row=row, column=0, sticky=W, padx=(0, 8), pady=6)
            entry = Entry(form, textvariable=variable, width=36)
            entry.grid(row=row, column=1, sticky=W, pady=6)
            Label(form, text=hint, anchor="w").grid(row=row, column=2, sticky=W, padx=(10, 0), pady=6)

        solver_row = len(fields)
        Label(form, text="Wind products", anchor="w", width=16).grid(row=solver_row, column=0, sticky=W, padx=(0, 8), pady=6)
        solver_combo = ttk.Combobox(form, textvariable=self.wind_solver_var, values=("windninja", "openfoam"), state="readonly", width=33)
        solver_combo.grid(row=solver_row, column=1, sticky=W, pady=6)
        Label(form, text="openfoam is optional and requires WSL/OpenFOAM 13", anchor="w").grid(row=solver_row, column=2, sticky=W, padx=(10, 0), pady=6)

        output_row = solver_row + 1
        Label(form, text="Report folder", anchor="w", width=16).grid(row=output_row, column=0, sticky=W, padx=(0, 8), pady=6)
        output_entry = Entry(form, textvariable=self.report_output_dir_var, width=36)
        output_entry.grid(row=output_row, column=1, sticky=W, pady=6)
        Button(form, text="Browse...", command=self._choose_report_output_dir, padx=10, pady=2).grid(row=output_row, column=2, sticky=W, pady=6)

        satellite_row = output_row + 1
        Label(form, text="Satellite products", anchor="w", width=16).grid(row=satellite_row, column=0, sticky=W, padx=(0, 8), pady=(8, 0))
        satellite_checks = Frame(form)
        satellite_checks.grid(row=satellite_row, column=1, columnspan=2, sticky=W, pady=(8, 0))
        Checkbutton(satellite_checks, text="RGB", variable=self.satellite_rgb_var).pack(side=LEFT)
        Checkbutton(satellite_checks, text="SST", variable=self.satellite_sst_var).pack(side=LEFT, padx=(10, 0))
        Checkbutton(satellite_checks, text="chl-a", variable=self.satellite_chla_var).pack(side=LEFT, padx=(10, 0))
        Checkbutton(satellite_checks, text="turbidity", variable=self.satellite_turbidity_var).pack(side=LEFT, padx=(10, 0))

        Checkbutton(form, text="Allow insecure SSL", variable=self.allow_insecure_ssl_var).grid(row=output_row + 2, column=1, sticky=W, pady=(8, 0))
        Checkbutton(form, text="Force ECOSTRESS SST", variable=self.force_ecostress_sst_var).grid(row=output_row + 3, column=1, sticky=W, pady=(6, 0))

        actions = Frame(container, pady=12)
        actions.pack(fill="x")
        Button(actions, text="Build Weekly Report", command=self._start_report_build, padx=16, pady=6).pack(side=LEFT)
        Button(actions, text="Use Next Sunday 2 PM", command=self._reset_default_race_time, padx=16, pady=6).pack(side=LEFT, padx=(10, 0))
        Button(actions, text="Show Last Report Folder", command=self._show_last_report_folder, padx=16, pady=6).pack(side=RIGHT)

        status_frame = ttk.LabelFrame(container, text="Status", padding=12)
        status_frame.pack(fill="x")
        Label(status_frame, textvariable=self.status_var, anchor="w").pack(fill="x")
        self.progress_bar = ttk.Progressbar(status_frame, orient="horizontal", mode="determinate", maximum=100, variable=self.progress_var)
        self.progress_bar.pack(fill="x", pady=(10, 0))
        Label(status_frame, textvariable=self.last_report_path_var, anchor="w", justify="left", fg="#444444").pack(fill="x", pady=(8, 0))

        log_frame = ttk.LabelFrame(container, text="Run Log", padding=12)
        log_frame.pack(fill=BOTH, expand=True, pady=(12, 0))

        self.log_text = Text(log_frame, wrap="word", height=20)
        scrollbar = ttk.Scrollbar(log_frame, orient=VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill="y")


    def _reset_default_race_time(self) -> None:
        self.race_time_var.set(_default_race_local_datetime())
        self.status_var.set("Race time reset to next Sunday at 2 PM local.")


    def _append_log(self, text: str) -> None:
        self.log_text.insert(END, text)
        self.log_text.see(END)


    def _choose_report_output_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.report_output_dir_var.get() or str(_runtime_project_root()))
        if selected:
            self.report_output_dir_var.set(selected)


    def _show_error(self, message: str) -> None:
        dialog = Toplevel(self.root)
        dialog.title("Input Error")
        dialog.geometry("420x140")
        Label(dialog, text=message, justify="left", wraplength=380, padx=16, pady=20).pack(fill=BOTH, expand=True)
        Button(dialog, text="Close", command=dialog.destroy, padx=12, pady=4).pack(pady=(0, 16))


    def _validate_inputs(self) -> list[str] | None:
        try:
            datetime.fromisoformat(self.race_time_var.get().strip())
        except ValueError:
            self._show_error("Race time must look like 2026-05-10T14:00:00.")
            return None

        numeric_fields = [
            ("Center latitude", self.center_lat_var.get().strip()),
            ("Center longitude", self.center_lon_var.get().strip()),
            ("Side meters", self.side_meters_var.get().strip()),
            ("Wind grid size", self.mesh_resolution_var.get().strip()),
        ]
        parsed_numbers: dict[str, float] = {}
        for label, value in numeric_fields:
            try:
                parsed_numbers[label] = float(value)
            except ValueError:
                self._show_error(f"{label} must be numeric.")
                return None

        grid_size_m = parsed_numbers["Wind grid size"]
        if grid_size_m < 10.0 or grid_size_m > 250.0:
            self._show_error("Wind grid size must be between 10 and 250 meters.")
            return None

        if not self.site_label_var.get().strip():
            self._show_error("Site label cannot be empty.")
            return None
        if self.wind_solver_var.get().strip() not in {"windninja", "openfoam"}:
            self._show_error("Wind products must be windninja or openfoam.")
            return None

        return [
            self.race_time_var.get().strip(),
            self.center_lat_var.get().strip(),
            self.center_lon_var.get().strip(),
            self.side_meters_var.get().strip(),
            self.site_label_var.get().strip(),
            self.mesh_resolution_var.get().strip(),
            self.wind_solver_var.get().strip(),
            self.report_output_dir_var.get().strip(),
            str(self.allow_insecure_ssl_var.get()),
            str(self.force_ecostress_sst_var.get()),
            str(self.satellite_rgb_var.get()),
            str(self.satellite_sst_var.get()),
            str(self.satellite_chla_var.get()),
            str(self.satellite_turbidity_var.get()),
        ]


    def _start_report_build(self) -> None:
        if self.is_running:
            self.status_var.set("A report build is already running.")
            return

        command = self._validate_inputs()
        if command is None:
            return

        self.log_text.delete("1.0", END)
        self._append_log("Starting report build...\n\n")
        self._append_log(f"Race time: {command[0]}\n")
        self._append_log(f"Center: {command[1]}, {command[2]}\n")
        self._append_log(f"Side meters: {command[3]}\n")
        self._append_log(f"Site label: {command[4]}\n")
        self._append_log(f"Wind grid size: {command[5]} m\n\n")
        self._append_log(f"Wind products: {command[6]}\n")
        self._append_log(
            "Satellite products: "
            f"RGB={command[10]}, SST={command[11]}, chl-a={command[12]}, turbidity={command[13]}\n"
        )
        self._append_log(f"Report folder: {command[7]}\n\n")
        self.status_var.set("Running report build...")
        self.last_report_path_var.set("")
        self.progress_var.set(0)
        self.is_running = True

        thread = threading.Thread(target=self._run_subprocess, args=(command,), daemon=True)
        thread.start()


    def _worker_command(self, command: list[str]) -> list[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, "--worker-report-build", *command]
        return [sys.executable, str(SCRIPT_DIR / "predictweather_app.py"), "--worker-report-build", *command]


    def _emit_local_error(self, message: str, log_text: str | None = None) -> None:
        error_root = _runtime_project_root()
        error_root.mkdir(parents=True, exist_ok=True)
        error_log = error_root / "report_build_error.txt"
        if log_text:
            error_log.write_text(log_text, encoding="utf-8")
        self.output_queue.put(("progress", "0|Report build failed"))
        self.output_queue.put(("error", message))
        self.output_queue.put(("log", f"\nDetailed traceback written to: {error_log}\n"))


    def _handle_worker_message(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self.output_queue.put(("log", line + "\n"))
            return

        kind = payload.get("kind")
        if kind == "progress":
            percent = int(payload.get("percent", 0))
            message = str(payload.get("message", "Running report build..."))
            self.output_queue.put(("progress", f"{percent}|{message}"))
        elif kind == "log":
            self.output_queue.put(("log", str(payload.get("text", ""))))
        elif kind == "report_path":
            self.output_queue.put(("report_path", str(payload.get("path", ""))))
        elif kind == "done":
            self.output_queue.put(("done", str(payload.get("message", "Report build completed successfully."))))
        elif kind == "error":
            self.output_queue.put(("error", str(payload.get("message", "Report build failed."))))
        else:
            self.output_queue.put(("log", line + "\n"))


    def _run_subprocess(self, command: list[str]) -> None:
        try:
            worker_command = self._worker_command(command)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            saw_terminal_message = False
            self.output_queue.put(("progress", "1|Launching report worker..."))
            self.worker_process = subprocess.Popen(
                worker_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            if self.worker_process.stdout is not None:
                for raw_line in self.worker_process.stdout:
                    stripped = raw_line.strip()
                    if stripped:
                        try:
                            payload = json.loads(stripped)
                            if payload.get("kind") in {"done", "error"}:
                                saw_terminal_message = True
                        except json.JSONDecodeError:
                            pass
                    self._handle_worker_message(raw_line)
            return_code = self.worker_process.wait()
            if return_code != 0 and not saw_terminal_message:
                self.output_queue.put(("log", f"\nWorker exited with code {return_code}.\n"))
                self.output_queue.put(("error", f"Report worker exited with code {return_code}."))
        except Exception as exc:
            self._emit_local_error(f"Unable to launch report build: {exc}", traceback.format_exc())
        finally:
            self.worker_process = None
            self.is_running = False


    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self.output_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "progress":
                    progress_text, status_text = payload.split("|", 1)
                    self.progress_var.set(int(progress_text))
                    self.status_var.set(status_text)
                elif kind == "done":
                    self.progress_var.set(100)
                    self.status_var.set(payload)
                    if self.last_report_file is not None:
                        self.last_report_path_var.set(f"Last report: {self.last_report_file}")
                    else:
                        report_dir = Path(self.report_output_dir_var.get().strip() or str(_runtime_project_root() / "outputs" / "reports"))
                        self.last_report_path_var.set(f"Reports are under: {report_dir}")
                elif kind == "error":
                    self.status_var.set(payload)
                    if self.progress_var.get() > 0:
                        self.progress_var.set(0)
                    self.is_running = False
                elif kind == "report_path":
                    self.last_report_file = Path(payload)
        except queue.Empty:
            pass
        self.root.after(150, self._drain_queue)


    def _show_last_report_folder(self) -> None:
        reports_dir = Path(self.report_output_dir_var.get().strip() or str(_runtime_project_root() / "outputs" / "reports"))
        self.status_var.set(f"Reports folder: {reports_dir}")
        self.last_report_path_var.set(str(reports_dir))


def main() -> None:
    root = Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    PredictWeatherGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
