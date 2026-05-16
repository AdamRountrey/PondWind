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
MPS_PER_KNOT = 1.0 / KNOTS_PER_MPS
ILCA7_LWL_M = 3.81
ILCA7_BASELINE_WEIGHT_LB = 176.37  # 80 kg baseline used in the Day 2017 Laser dinghy VPP.
NO_GO_TWA_DEG = 40.0
BEST_UPWIND_VMG_TWA_DEG = 43.0
DEFAULT_REPORT_NAME = "20260517_1400_barton_pond3"
MODEL_SOURCES = [
    "Day 2017, Performance Prediction for Sailing Dinghies, Ocean Engineering",
    "Binns et al. 2004, Development and Uses of the Virtual Sailing Dinghy",
    "Clark 2014, full-scale Laser simulator validation data",
    "Pennanen 2015, Olympic dinghy upwind CFD/VPP thesis",
]
TWS_BINS = np.array([2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 30], dtype=np.float64)
TWA_BINS = np.array([40, 43, 46, 52, 60, 75, 90, 105, 120, 135, 150, 165, 180], dtype=np.float64)
POLAR_SPEEDS = np.array(
    [
        [1.0, 1.12, 1.18, 1.36, 1.55, 1.76, 1.84, 1.88, 1.86, 1.72, 1.48, 1.26, 1.08],
        [2.08, 2.28, 2.42, 2.68, 2.92, 3.18, 3.32, 3.38, 3.32, 3.08, 2.76, 2.46, 2.18],
        [3.02, 3.28, 3.46, 3.78, 4.08, 4.48, 4.72, 4.82, 4.66, 4.32, 3.86, 3.46, 3.18],
        [3.58, 3.86, 4.04, 4.38, 4.75, 5.24, 5.55, 5.72, 5.56, 5.20, 4.70, 4.20, 3.84],
        [3.88, 4.16, 4.34, 4.72, 5.15, 5.82, 6.32, 6.58, 6.34, 5.88, 5.28, 4.70, 4.26],
        [4.02, 4.28, 4.48, 4.92, 5.48, 6.62, 7.22, 7.68, 7.26, 6.54, 5.72, 5.10, 4.72],
        [4.08, 4.34, 4.56, 5.06, 5.82, 7.58, 8.62, 9.25, 8.62, 7.56, 6.42, 5.66, 5.18],
        [4.14, 4.40, 4.62, 5.16, 6.12, 8.62, 10.0, 10.72, 9.92, 8.62, 7.30, 6.32, 5.72],
        [4.24, 4.48, 4.70, 5.28, 6.42, 10.0, 11.78, 12.58, 11.80, 10.35, 8.90, 7.70, 6.92],
        [4.30, 4.52, 4.74, 5.36, 6.62, 11.0, 12.85, 13.62, 12.95, 11.55, 10.05, 8.78, 7.95],
        [4.35, 4.56, 4.78, 5.44, 6.82, 12.15, 13.75, 14.25, 13.70, 12.55, 11.15, 9.95, 9.05],
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


def _logistic(value: float, midpoint: float, width: float) -> float:
    return 1.0 / (1.0 + math.exp(-(value - midpoint) / max(width, 1.0e-6)))


def _froude_number(boat_speed_knots: float) -> float:
    return (boat_speed_knots * MPS_PER_KNOT) / math.sqrt(9.80665 * ILCA7_LWL_M)


def _weight_factor(tws_knots: float, twa_deg: float, sailor_weight_lb: float) -> float:
    delta_kg = (sailor_weight_lb - ILCA7_BASELINE_WEIGHT_LB) / 2.20462262185
    if tws_knots < 7.0:
        light_air = (7.0 - tws_knots) / 5.0
        angle_factor = np.interp(twa_deg, [NO_GO_TWA_DEG, 90.0, 180.0], [0.7, 1.0, 0.85])
        return float(np.clip(1.0 - delta_kg * 0.0032 * light_air * angle_factor, 0.86, 1.10))
    if twa_deg < 70.0 and tws_knots > 11.0:
        heavy_hike = min((tws_knots - 11.0) / 10.0, 1.0)
        return float(np.clip(1.0 + delta_kg * 0.0020 * heavy_hike, 0.90, 1.12))
    if 75.0 <= twa_deg <= 150.0 and tws_knots > 12.0:
        planing_penalty = min((tws_knots - 12.0) / 12.0, 1.0)
        return float(np.clip(1.0 - delta_kg * 0.0012 * planing_penalty, 0.90, 1.08))
    return 1.0


def _planing_probability(tws_knots: float, twa_deg: float, boat_speed_knots: float, sailor_weight_lb: float) -> float:
    if twa_deg < 65.0:
        return 0.0
    angle_factor = float(np.interp(twa_deg, [65, 80, 105, 130, 155, 180], [0.0, 0.55, 1.0, 0.9, 0.35, 0.10]))
    wind_threshold = float(np.interp(twa_deg, [65, 80, 105, 130, 155, 180], [17.0, 14.2, 12.6, 13.0, 15.6, 18.5]))
    wind_threshold += (sailor_weight_lb - ILCA7_BASELINE_WEIGHT_LB) * 0.035
    wind_part = _logistic(tws_knots, wind_threshold, 1.8)
    froude_part = _logistic(_froude_number(boat_speed_knots), 0.78, 0.10)
    return float(np.clip(angle_factor * wind_part * froude_part, 0.0, 1.0))


def _apparent_wind(tws_knots: float, twa_deg: float, boat_speed_knots: float) -> tuple[float, float]:
    transverse = tws_knots * math.sin(math.radians(twa_deg))
    forward = tws_knots * math.cos(math.radians(twa_deg)) + max(boat_speed_knots, 0.0)
    apparent_speed = math.hypot(transverse, forward)
    apparent_angle = abs(math.degrees(math.atan2(transverse, forward)))
    return apparent_speed, apparent_angle


def estimate_ilca7_speed(tws_knots: float, twa_deg: float, sailor_weight_lb: float) -> tuple[float, float, str, float, float, float]:
    if twa_deg < NO_GO_TWA_DEG:
        return float("nan"), 0.0, "no-go", float("nan"), float("nan"), float("nan")
    base = _interp2(tws_knots, twa_deg)
    speed = base * _weight_factor(tws_knots, twa_deg, sailor_weight_lb)
    initial_planing = _planing_probability(tws_knots, twa_deg, speed, sailor_weight_lb)
    if initial_planing > 0.0:
        reach_factor = float(np.interp(twa_deg, [65, 90, 120, 155, 180], [0.0, 0.45, 0.55, 0.25, 0.05]))
        speed += initial_planing * reach_factor * max(tws_knots - 10.0, 0.0) * 0.18
    planing = _planing_probability(tws_knots, twa_deg, speed, sailor_weight_lb)
    apparent_speed, apparent_angle = _apparent_wind(tws_knots, twa_deg, speed)
    froude = _froude_number(speed)
    mode = "displacement"
    if planing >= 0.65:
        mode = "planing"
    elif planing >= 0.25:
        mode = "transition"
    return max(speed, 0.0), planing, mode, apparent_speed, apparent_angle, froude


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
        speed, planing, mode, apparent_speed, apparent_angle, froude = estimate_ilca7_speed(tws_knots, twa, sailor_weight_lb)
        upwind_vmg = speed * math.cos(math.radians(twa)) if math.isfinite(speed) else float("nan")
        downwind_vmg = -upwind_vmg if math.isfinite(upwind_vmg) else float("nan")
        samples.append(
            {
                "heading_deg": heading,
                "true_wind_angle_deg": round(twa, 3),
                "boat_speed_knots": None if not math.isfinite(speed) else round(speed, 3),
                "planing_probability": round(planing, 3),
                "mode": mode,
                "apparent_wind_speed_knots": None if not math.isfinite(apparent_speed) else round(apparent_speed, 3),
                "apparent_wind_angle_deg": None if not math.isfinite(apparent_angle) else round(apparent_angle, 3),
                "froude_lwl": None if not math.isfinite(froude) else round(froude, 3),
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


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _best_vmg_pairs(samples: list[dict[str, float | str | None]], wind_from_deg: float) -> tuple[list[dict[str, float | str | None]], list[dict[str, float | str | None]]]:
    finite = [sample for sample in samples if sample["boat_speed_knots"] is not None]
    upwind_candidates = [sample for sample in finite if float(sample["true_wind_angle_deg"]) < 90.0]
    downwind_candidates = [sample for sample in finite if float(sample["true_wind_angle_deg"]) > 90.0]

    def side(sample: dict[str, float | str | None]) -> int:
        return -1 if _angle_diff_deg(float(sample["heading_deg"]), wind_from_deg) < 0 else 1

    upwind = []
    for tack_side in (-1, 1):
        side_samples = [sample for sample in upwind_candidates if side(sample) == tack_side]
        if side_samples:
            upwind.append(max(side_samples, key=lambda sample: float(sample["upwind_vmg_knots"])))
    downwind = []
    for gybe_side in (-1, 1):
        side_samples = [sample for sample in downwind_candidates if side(sample) == gybe_side]
        if side_samples:
            downwind.append(max(side_samples, key=lambda sample: float(sample["downwind_vmg_knots"])))
    upwind.sort(key=lambda sample: float(sample["heading_deg"]))
    downwind.sort(key=lambda sample: float(sample["heading_deg"]))
    return upwind, downwind


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
    font = _font(18)
    small_font = _font(15)
    title_font = _font(22, bold=True)
    label_font = _font(16, bold=True)
    center = (size // 2, size // 2 + 28)
    plot_radius = 410
    finite_speeds = [float(s["boat_speed_knots"]) for s in samples if s["boat_speed_knots"] is not None]
    max_speed = max(finite_speeds) if finite_speeds else 1.0
    scale_max = max(2.0, math.ceil((max_speed + 0.5) * 2.0) / 2.0)

    draw.text((44, 30), "Experimental ILCA 7 / Laser Standard Speed Polar", fill=(22, 34, 42), font=title_font)
    draw.text((44, 62), f"{wind_source} point row {row}, col {col} | x {x:.1f}, y {y:.1f}", fill=(70, 86, 95), font=font)
    draw.text((44, 90), f"Local wind: {tws_knots:.1f} kt from {wind_from_deg:.0f} deg | sailor {sailor_weight_lb:.0f} lb", fill=(70, 86, 95), font=font)

    for speed in np.linspace(scale_max / 4.0, scale_max, 4):
        r = plot_radius * speed / scale_max
        draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), outline=(146, 160, 168, 150), width=2)
        label_pos = _point_for_heading(center, 8.0, r)
        draw.text((label_pos[0] + 8, label_pos[1] - 5), f"{speed:.1f} kt", fill=(70, 84, 92), font=small_font)
    for heading in range(0, 360, 30):
        end = _point_for_heading(center, heading, plot_radius)
        draw.line((center[0], center[1], end[0], end[1]), fill=(160, 172, 178, 105), width=2)
        label = _point_for_heading(center, heading, plot_radius + 28)
        draw.text((label[0] - 13, label[1] - 9), f"{heading}", fill=(64, 78, 86), font=small_font)

    no_go = NO_GO_TWA_DEG
    wedge_points = [center]
    for angle in np.linspace(wind_from_deg - no_go, wind_from_deg + no_go, 42):
        wedge_points.append(_point_for_heading(center, angle, plot_radius))
    draw.polygon(wedge_points, fill=(112, 122, 128, 42))
    draw.text(_point_for_heading(center, wind_from_deg, plot_radius * 0.58), "NO-GO", fill=(70, 78, 84), font=label_font)

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
    draw.line((wind_arrow_end[0], wind_arrow_end[1], center[0], center[1]), fill=(20, 36, 44, 235), width=6)
    left = _point_for_heading(wind_arrow_end, wind_from_deg + 158, 22)
    right = _point_for_heading(wind_arrow_end, wind_from_deg - 158, 22)
    draw.polygon([wind_arrow_end, left, right], fill=(20, 36, 44, 230))
    draw.text((wind_arrow_end[0] + 12, wind_arrow_end[1] - 10), "wind from", fill=(20, 36, 44), font=label_font)

    finite_samples = [s for s in samples if s["boat_speed_knots"] is not None]
    fastest = max(finite_samples, key=lambda s: float(s["boat_speed_knots"]))
    upwind_pair, downwind_pair = _best_vmg_pairs(samples, wind_from_deg)
    x0, y0 = 784, 822
    draw.rounded_rectangle((762, 790, 1164, 1114), radius=8, fill=(255, 255, 255, 236), outline=(186, 198, 206, 255), width=2)
    draw.text((x0, y0 - 20), "Point summary", fill=(24, 35, 42), font=label_font)
    draw.text((x0, y0 + 10), f"Fastest: {fastest['heading_deg']:03.0f} deg, {fastest['boat_speed_knots']:.1f} kt", fill=(38, 51, 60), font=font)
    y_cursor = y0 + 48
    draw.text((x0, y_cursor), "Best upwind VMG", fill=(38, 51, 60), font=label_font)
    y_cursor += 26
    for sample in upwind_pair:
        line = f"{sample['heading_deg']:03.0f} deg  {sample['upwind_vmg_knots']:.1f} kt VMG  boat {sample['boat_speed_knots']:.1f}"
        draw.text((x0, y_cursor), line, fill=(46, 61, 70), font=font)
        y_cursor += 24
    y_cursor += 8
    draw.text((x0, y_cursor), "Best downwind VMG", fill=(38, 51, 60), font=label_font)
    y_cursor += 26
    for sample in downwind_pair:
        line = f"{sample['heading_deg']:03.0f} deg  {sample['downwind_vmg_knots']:.1f} kt VMG  boat {sample['boat_speed_knots']:.1f}"
        draw.text((x0, y_cursor), line, fill=(46, 61, 70), font=font)
        y_cursor += 24
    y_cursor += 10
    draw.text((x0, y_cursor), "Blue: displacement", fill=(38, 132, 202), font=font)
    draw.text((x0, y_cursor + 24), "Gold: transition", fill=(196, 139, 42), font=font)
    draw.text((x0, y_cursor + 48), "Red: likely planing", fill=(190, 72, 58), font=font)
    draw.text((44, 1126), "Literature-informed prototype: Day 2017/Binns/Clark/Pennanen; relative estimate only.", fill=(70, 84, 92), font=font)
    draw.text((44, 1150), "Not calibrated to Barton GPS tracks, waves, current, trim, kinetics, or sailor skill.", fill=(70, 84, 92), font=font)

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
    upwind_pair, downwind_pair = _best_vmg_pairs(samples, wind_from_deg)
    finite_samples = [sample for sample in samples if sample["boat_speed_knots"] is not None]
    fastest = max(finite_samples, key=lambda sample: float(sample["boat_speed_knots"]))

    output_dir = args.output_dir or report_dir / "report_temp" / "wind" / "sailing_polar"
    stem = f"ilca7_polar_{args.wind_source}_r{row}_c{col}_{int(round(args.sailor_weight_lb))}lb"
    png_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"
    _draw_polar(png_path, samples, wind_from_deg, tws_knots, float(args.sailor_weight_lb), row, col, x, y, args.wind_source)

    summary = {
        "note": "Literature-informed experimental ILCA 7 / Laser Standard point polar. Relative estimate only; not calibrated.",
        "model": {
            "name": "ilca7_literature_informed_v1",
            "basis": MODEL_SOURCES,
            "baseline_weight_lb": ILCA7_BASELINE_WEIGHT_LB,
            "lwl_m": ILCA7_LWL_M,
            "no_go_twa_deg": NO_GO_TWA_DEG,
            "best_upwind_vmg_twa_deg_reference": BEST_UPWIND_VMG_TWA_DEG,
            "notes": [
                "Empirical polar table shaped from published Laser/ILCA VPP and measurement plots.",
                "Planing is a soft probability using reach angle, wind threshold, and Froude number from ILCA waterline length.",
                "Apparent wind and VMG are derived from the estimated boat speed for each heading.",
                "Use for relative comparison only until calibrated with GPS tracks and local water state.",
            ],
        },
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
        "fastest": fastest,
        "best_upwind_vmg_pair": upwind_pair,
        "best_downwind_vmg_pair": downwind_pair,
        "samples": samples,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in summary if key != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
