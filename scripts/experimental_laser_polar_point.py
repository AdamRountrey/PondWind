"""Experimental Laser/ILCA 7 point polar diagram from a PondWind wind grid."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

KNOTS_PER_MPS = 1.9438444924406048
DEFAULT_REPORT_NAME = "20260517_1400_barton_pond3"
TWS_BINS = np.array([2, 4, 6, 8, 10, 12, 14, 16, 20, 24], dtype=np.float64)
TWA_BINS = np.array([42, 45, 52, 60, 75, 90, 110, 135, 150, 165, 180], dtype=np.float64)
POLAR_SPEEDS = np.array(
    [
        [0.0, 1.2, 1.5, 1.7, 1.8, 1.9, 1.9, 1.7, 1.5, 1.3, 1.1],
        [0.0, 2.5, 2.8, 3.0, 3.2, 3.3, 3.4, 3.1, 2.8, 2.5, 2.2],
        [0.0, 3.6, 4.0, 4.2, 4.6, 4.8, 4.9, 4.4, 4.0, 3.5, 3.2],
        [0.0, 4.3, 4.8, 5.0, 5.4, 5.7, 5.8, 5.4, 4.9, 4.3, 3.9],
        [0.0, 4.8, 5.3, 5.6, 6.2, 6.6, 6.8, 6.2, 5.6, 4.9, 4.5],
        [0.0, 5.2, 5.6, 6.1, 7.0, 7.5, 8.0, 7.3, 6.2, 5.4, 5.0],
        [0.0, 5.4, 5.8, 6.5, 8.0, 9.0, 9.6, 8.7, 7.0, 6.0, 5.5],
        [0.0, 5.6, 6.0, 6.8, 9.3, 10.5, 11.2, 10.0, 8.0, 6.8, 6.1],
        [0.0, 5.8, 6.1, 7.0, 10.5, 12.0, 12.8, 11.6, 9.5, 8.2, 7.3],
        [0.0, 5.9, 6.2, 7.0, 11.5, 13.0, 13.6, 12.5, 10.5, 9.2, 8.4],
    ],
    dtype=np.float64,
)


def _read_aaigrid(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header: dict[str, float] = {}
    for line in lines[:6]:
        key, value = line.split()
        header[key.lower()] = float(value)
    rows = [[float(item) for item in line.split()] for line in lines[6:] if line.strip()]
    data = np.array(rows, dtype=np.float32)
    nodata = header.get("nodata_value")
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, header


def _default_report_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA is not set. Pass --report-dir explicitly.")
    return Path(local_app_data) / "PondWind" / "outputs" / "reports" / DEFAULT_REPORT_NAME


def _latest_report_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA is not set. Pass --report-dir explicitly.")
    root = Path(local_app_data) / "PondWind" / "outputs" / "reports"
    reports = [path for path in root.iterdir() if path.is_dir()]
    if not reports:
        raise RuntimeError(f"No reports found under {root}")
    return max(reports, key=lambda path: path.stat().st_mtime)


def _angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def _true_wind_angle_deg(heading_deg: float, wind_from_deg: float) -> float:
    return abs(_angle_diff_deg(heading_deg, wind_from_deg))


def _interp2(tws_knots: float, twa_deg: float) -> float:
    twa = float(np.clip(twa_deg, TWA_BINS[0], TWA_BINS[-1]))
    tws = float(np.clip(tws_knots, TWS_BINS[0], TWS_BINS[-1]))
    angle_values = np.array([np.interp(twa, TWA_BINS, row) for row in POLAR_SPEEDS], dtype=np.float64)
    return float(np.interp(tws, TWS_BINS, angle_values))


def _planing_threshold_knots(twa_deg: float, sailor_weight_lb: float) -> float:
    twa = abs(float(twa_deg))
    if twa < 65:
        return 99.0
    if twa <= 90:
        base = np.interp(twa, [65, 75, 90], [17.5, 14.5, 13.0])
    elif twa <= 135:
        base = np.interp(twa, [90, 110, 135], [13.0, 12.2, 12.8])
    elif twa <= 165:
        base = np.interp(twa, [135, 150, 165], [12.8, 14.5, 17.0])
    else:
        base = np.interp(twa, [165, 180], [17.0, 19.0])
    return float(base + (sailor_weight_lb - 175.0) * 0.04)


def _planing_probability(tws_knots: float, twa_deg: float, sailor_weight_lb: float) -> float:
    threshold = _planing_threshold_knots(twa_deg, sailor_weight_lb)
    if threshold > 50:
        return 0.0
    x = (tws_knots - threshold) / 1.65
    probability = 1.0 / (1.0 + math.exp(-x))
    angle_factor = np.interp(abs(twa_deg), [65, 80, 110, 140, 165, 180], [0.0, 0.65, 1.0, 0.85, 0.25, 0.08])
    return float(np.clip(probability * angle_factor, 0.0, 1.0))


def estimate_ilca7_speed(tws_knots: float, twa_deg: float, sailor_weight_lb: float) -> tuple[float, float, str]:
    if twa_deg < TWA_BINS[0]:
        return float("nan"), 0.0, "no-go"
    base = _interp2(tws_knots, twa_deg)
    planing = _planing_probability(tws_knots, twa_deg, sailor_weight_lb)

    light_factor = 1.0
    if tws_knots < 8:
        light_factor -= (sailor_weight_lb - 175.0) * 0.0018 * (8.0 - tws_knots) / 6.0
    control_factor = 1.0
    if tws_knots > 14 and twa_deg < 70:
        control_factor += (sailor_weight_lb - 175.0) * 0.0008 * min(tws_knots - 14.0, 10.0)
    if planing > 0:
        control_factor -= (sailor_weight_lb - 175.0) * 0.0009 * planing

    speed = base * float(np.clip(light_factor * control_factor, 0.86, 1.10))
    mode = "displacement"
    if planing >= 0.65:
        mode = "planing"
    elif planing >= 0.25:
        mode = "transition"
    return max(speed, 0.0), planing, mode


def _grid_xy(header: dict[str, float], row: int, col: int) -> tuple[float, float]:
    cellsize = float(header["cellsize"])
    x = float(header["xllcorner"]) + (col + 0.5) * cellsize
    nrows = int(header["nrows"])
    y = float(header["yllcorner"]) + (nrows - row - 0.5) * cellsize
    return x, y


def _row_col_from_xy(header: dict[str, float], x: float, y: float) -> tuple[int, int]:
    cellsize = float(header["cellsize"])
    col = int(round((x - float(header["xllcorner"])) / cellsize - 0.5))
    row = int(round(int(header["nrows"]) - (y - float(header["yllcorner"])) / cellsize - 0.5))
    return row, col


def _choose_point(speed: np.ndarray, header: dict[str, float], args: argparse.Namespace) -> tuple[int, int]:
    if args.row is not None and args.col is not None:
        row, col = int(args.row), int(args.col)
    elif args.x is not None and args.y is not None:
        row, col = _row_col_from_xy(header, float(args.x), float(args.y))
    else:
        finite = np.argwhere(np.isfinite(speed))
        if finite.size == 0:
            raise RuntimeError("No finite wind speed cells found.")
        center = np.array(speed.shape, dtype=np.float64) / 2.0
        row, col = min(finite, key=lambda item: float(np.sum((item - center) ** 2)))
        row, col = int(row), int(col)
    if row < 0 or col < 0 or row >= speed.shape[0] or col >= speed.shape[1]:
        raise RuntimeError(f"Selected row/col is outside the wind grid: row={row}, col={col}")
    if not np.isfinite(speed[row, col]):
        raise RuntimeError(f"Selected wind grid cell is not finite: row={row}, col={col}")
    return row, col


def _find_wind_paths(report_dir: Path, wind_source: str) -> tuple[Path, Path]:
    wind_root = report_dir / "report_temp" / "wind" / wind_source
    if not wind_root.exists():
        raise RuntimeError(f"Wind source folder not found: {wind_root}")
    speed_paths = sorted(wind_root.glob("*_vel.asc"))
    direction_paths = sorted(wind_root.glob("*_ang.asc"))
    if not speed_paths or not direction_paths:
        raise RuntimeError(f"Could not find *_vel.asc and *_ang.asc under {wind_root}")
    return speed_paths[0], direction_paths[0]


def _polar_samples(tws_knots: float, wind_from_deg: float, sailor_weight_lb: float) -> list[dict[str, float | str | None]]:
    samples = []
    for heading in range(360):
        twa = _true_wind_angle_deg(float(heading), wind_from_deg)
        speed, planing, mode = estimate_ilca7_speed(tws_knots, twa, sailor_weight_lb)
        upwind_vmg = speed * math.cos(math.radians(twa)) if math.isfinite(speed) else float("nan")
        downwind_vmg = -upwind_vmg if math.isfinite(upwind_vmg) else float("nan")
        samples.append(
            {
                "heading_deg": heading,
                "true_wind_angle_deg": round(twa, 3),
                "boat_speed_knots": None if not math.isfinite(speed) else round(speed, 3),
                "planing_probability": round(planing, 3),
                "mode": mode,
                "upwind_vmg_knots": None if not math.isfinite(upwind_vmg) else round(upwind_vmg, 3),
                "downwind_vmg_knots": None if not math.isfinite(downwind_vmg) else round(downwind_vmg, 3),
            }
        )
    return samples


def _point_for_heading(center: tuple[int, int], heading_deg: float, radius: float) -> tuple[float, float]:
    theta = math.radians(heading_deg)
    return center[0] + radius * math.sin(theta), center[1] - radius * math.cos(theta)


def _mode_color(mode: str, planing_probability: float) -> tuple[int, int, int]:
    if mode == "no-go":
        return (110, 118, 122)
    if mode == "planing":
        return (221, 89, 68)
    if mode == "transition":
        return (238, 184, 75)
    p = float(planing_probability)
    return (38, int(132 + 72 * p), int(202 - 80 * p))


def _draw_polar(
    output_path: Path,
    samples: list[dict[str, float | str | None]],
    wind_from_deg: float,
    tws_knots: float,
    sailor_weight_lb: float,
    row: int,
    col: int,
    x: float,
    y: float,
    wind_source: str,
) -> None:
    size = 1200
    image = Image.new("RGB", (size, size), (246, 248, 249))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    center = (size // 2, size // 2 + 28)
    plot_radius = 410
    finite_speeds = [float(s["boat_speed_knots"]) for s in samples if s["boat_speed_knots"] is not None]
    max_speed = max(finite_speeds) if finite_speeds else 1.0
    scale_max = max(2.0, math.ceil((max_speed + 0.5) * 2.0) / 2.0)

    draw.text((44, 32), "Experimental ILCA 7 / Laser Standard Speed Polar", fill=(22, 34, 42), font=font)
    draw.text((44, 58), f"{wind_source} point row {row}, col {col} | x {x:.1f}, y {y:.1f}", fill=(70, 86, 95), font=font)
    draw.text((44, 82), f"Local wind: {tws_knots:.1f} kt from {wind_from_deg:.0f} deg | sailor {sailor_weight_lb:.0f} lb", fill=(70, 86, 95), font=font)

    for speed in np.linspace(scale_max / 4.0, scale_max, 4):
        r = plot_radius * speed / scale_max
        draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), outline=(178, 189, 195, 105), width=1)
        draw.text((center[0] + 8, center[1] - r - 5), f"{speed:.1f} kt", fill=(104, 116, 122), font=font)
    for heading in range(0, 360, 30):
        end = _point_for_heading(center, heading, plot_radius)
        draw.line((center[0], center[1], end[0], end[1]), fill=(194, 202, 206, 70), width=1)
        label = _point_for_heading(center, heading, plot_radius + 28)
        draw.text((label[0] - 10, label[1] - 6), f"{heading}", fill=(91, 103, 110), font=font)

    no_go = 42.0
    wedge_points = [center]
    for angle in np.linspace(wind_from_deg - no_go, wind_from_deg + no_go, 42):
        wedge_points.append(_point_for_heading(center, angle, plot_radius))
    draw.polygon(wedge_points, fill=(112, 122, 128, 42))
    draw.text(_point_for_heading(center, wind_from_deg, plot_radius * 0.58), "NO-GO", fill=(93, 101, 106), font=font)

    points = []
    modes = []
    for sample in samples:
        speed = sample["boat_speed_knots"]
        if speed is None:
            points.append(None)
            modes.append("no-go")
            continue
        r = plot_radius * float(speed) / scale_max
        points.append(_point_for_heading(center, float(sample["heading_deg"]), r))
        modes.append(str(sample["mode"]))
    for idx in range(360):
        a = points[idx]
        b = points[(idx + 1) % 360]
        if a is None or b is None:
            continue
        sample = samples[idx]
        color = _mode_color(str(sample["mode"]), float(sample["planing_probability"]))
        draw.line((a[0], a[1], b[0], b[1]), fill=(*color, 236), width=5)

    wind_arrow_end = _point_for_heading(center, wind_from_deg, plot_radius * 0.92)
    draw.line((wind_arrow_end[0], wind_arrow_end[1], center[0], center[1]), fill=(20, 36, 44, 230), width=4)
    left = _point_for_heading(wind_arrow_end, wind_from_deg + 158, 22)
    right = _point_for_heading(wind_arrow_end, wind_from_deg - 158, 22)
    draw.polygon([wind_arrow_end, left, right], fill=(20, 36, 44, 230))
    draw.text((wind_arrow_end[0] + 10, wind_arrow_end[1] - 8), "wind from", fill=(20, 36, 44), font=font)

    finite_samples = [s for s in samples if s["boat_speed_knots"] is not None]
    fastest = max(finite_samples, key=lambda s: float(s["boat_speed_knots"]))
    upwind = max((s for s in finite_samples if float(s["true_wind_angle_deg"]) < 90), key=lambda s: float(s["upwind_vmg_knots"]), default=None)
    downwind = max((s for s in finite_samples if float(s["true_wind_angle_deg"]) > 90), key=lambda s: float(s["downwind_vmg_knots"]), default=None)
    notes = [
        ("Fastest", fastest, "boat_speed_knots"),
        ("Best upwind VMG", upwind, "upwind_vmg_knots"),
        ("Best downwind VMG", downwind, "downwind_vmg_knots"),
    ]
    x0, y0 = 815, 855
    draw.rounded_rectangle((790, 820, 1148, 1078), radius=8, fill=(255, 255, 255, 230), outline=(202, 212, 218, 255))
    draw.text((x0, y0 - 18), "Point summary", fill=(24, 35, 42), font=font)
    for i, (label, sample, metric) in enumerate(notes):
        if sample is None:
            continue
        line = f"{label}: {sample['heading_deg']:03.0f} deg, {sample[metric]:.1f} kt"
        draw.text((x0, y0 + i * 26), line, fill=(46, 61, 70), font=font)
    draw.text((x0, y0 + 94), "Blue: displacement", fill=(38, 132, 202), font=font)
    draw.text((x0, y0 + 120), "Gold: transition", fill=(196, 139, 42), font=font)
    draw.text((x0, y0 + 146), "Red: likely planing", fill=(190, 72, 58), font=font)
    draw.text((44, 1136), "Prototype polar: relative comparisons only; not calibrated to GPS tracks, waves, current, trim, or sailor skill.", fill=(89, 101, 108), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an experimental ILCA 7 point speed polar from a PondWind wind grid.")
    parser.add_argument("--report-dir", type=Path, default=None, help="Report folder. Defaults to latest PondWind report.")
    parser.add_argument("--wind-source", default="deterministic", help="Wind folder under report_temp/wind, e.g. deterministic or openfoam_comparison.")
    parser.add_argument("--sailor-weight-lb", type=float, default=175.0)
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--col", type=int, default=None)
    parser.add_argument("--x", type=float, default=None)
    parser.add_argument("--y", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    report_dir = args.report_dir or (_default_report_dir() if _default_report_dir().exists() else _latest_report_dir())
    speed_path, direction_path = _find_wind_paths(report_dir, args.wind_source)
    speed_mps, header = _read_aaigrid(speed_path)
    direction_deg, _ = _read_aaigrid(direction_path)
    row, col = _choose_point(speed_mps, header, args)
    x, y = _grid_xy(header, row, col)
    tws_knots = float(speed_mps[row, col] * KNOTS_PER_MPS)
    wind_from_deg = float(direction_deg[row, col] % 360.0)
    samples = _polar_samples(tws_knots, wind_from_deg, float(args.sailor_weight_lb))

    output_dir = args.output_dir or report_dir / "report_temp" / "wind" / "sailing_polar"
    stem = f"ilca7_polar_{args.wind_source}_r{row}_c{col}_{int(round(args.sailor_weight_lb))}lb"
    png_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"
    _draw_polar(png_path, samples, wind_from_deg, tws_knots, float(args.sailor_weight_lb), row, col, x, y, args.wind_source)

    summary = {
        "note": "Experimental ILCA 7 / Laser Standard point polar. Relative estimate only; not calibrated.",
        "report_dir": str(report_dir),
        "wind_source": args.wind_source,
        "speed_grid": str(speed_path),
        "direction_grid": str(direction_path),
        "row": row,
        "col": col,
        "x": x,
        "y": y,
        "local_wind_speed_knots": tws_knots,
        "local_wind_from_direction_deg": wind_from_deg,
        "sailor_weight_lb": float(args.sailor_weight_lb),
        "png": str(png_path),
        "samples": samples,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in summary if key != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
