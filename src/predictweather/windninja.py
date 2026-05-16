from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

from predictweather.config import RESOURCE_ROOT
from predictweather.wind import _draw_text, _interpolate_colormap, _write_png


@dataclass
class WindNinjaRunError(RuntimeError):
    stage: str
    message: str
    command: list[str]
    output_dir: Path
    stdout: str = ""
    stderr: str = ""
    missing_outputs: list[str] | None = None

    def __post_init__(self) -> None:
        summary = f"{self.stage}: {self.message}"
        if self.missing_outputs:
            summary += f" Missing outputs: {', '.join(self.missing_outputs)}."
        super().__init__(summary)


def windninja_cli_path(project_root: Path) -> Path:
    candidates: list[Path] = []
    cli_override = os.environ.get("PONDWIND_WINDNINJA_CLI", "").strip()
    if cli_override:
        candidates.append(Path(cli_override))

    home_override = os.environ.get("PONDWIND_WINDNINJA_HOME", "").strip()
    if home_override:
        candidates.append(Path(home_override) / "bin" / "WindNinja_cli.exe")

    candidates.append(RESOURCE_ROOT / "tools" / "WindNinjaApp" / "bin" / "WindNinja_cli.exe")
    candidates.append(project_root / "tools" / "WindNinjaApp" / "bin" / "WindNinja_cli.exe")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _read_aaigrid(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header: dict[str, float] = {}
    for line in lines[:6]:
        key, value = line.split()
        header[key.lower()] = float(value)

    rows = []
    for line in lines[6:]:
        if line.strip():
            rows.append([float(item) for item in line.split()])
    data = np.array(rows, dtype="float32")
    nodata = header["nodata_value"]
    data[data == nodata] = np.nan
    return data, header


def aaigrid_to_geotiff(source_asc: Path, destination_tif: Path, crs: str = "EPSG:26917") -> Path:
    data, header = _read_aaigrid(source_asc)
    transform = from_origin(
        header["xllcorner"],
        header["yllcorner"] + header["nrows"] * header["cellsize"],
        header["cellsize"],
        header["cellsize"],
    )
    destination_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        destination_tif,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
    return destination_tif


def write_array_to_geotiff_from_header(
    header: dict[str, float],
    data: np.ndarray,
    destination_tif: Path,
    crs: str = "EPSG:26917",
) -> Path:
    transform = from_origin(
        header["xllcorner"],
        header["yllcorner"] + header["nrows"] * header["cellsize"],
        header["cellsize"],
        header["cellsize"],
    )
    destination_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        destination_tif,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return destination_tif


def write_reprojected_array_like_reference(
    source_array: np.ndarray,
    *,
    source_header: dict[str, float],
    reference_tif: Path,
    destination_tif: Path,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    reprojected, _, _ = _reproject_array_to_basemap(
        source_array.astype(np.float32),
        source_header=source_header,
        basemap_tif=reference_tif,
        resampling=resampling,
    )
    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()
        profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan, compress="deflate")
        destination_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(destination_tif, "w", **profile) as dst:
            dst.write(reprojected.astype(np.float32), 1)
    return destination_tif


def _transform_from_header(header: dict[str, float]):
    return from_origin(
        header["xllcorner"],
        header["yllcorner"] + header["nrows"] * header["cellsize"],
        header["cellsize"],
        header["cellsize"],
    )


def _reproject_array_to_basemap(
    source_array: np.ndarray,
    *,
    source_header: dict[str, float],
    basemap_tif: Path,
    resampling: Resampling,
) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    with rasterio.open(basemap_tif) as base_src:
        destination = np.full((base_src.height, base_src.width), np.nan, dtype=np.float32)
        base_transform = base_src.transform
        base_crs = base_src.crs
        source_transform = _transform_from_header(source_header)
        reproject(
            source=source_array.astype(np.float32),
            destination=destination,
            src_transform=source_transform,
            src_crs=base_crs,
            dst_transform=base_transform,
            dst_crs=base_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return destination, base_transform, base_crs


def run_windninja_domain_average(
    cli_path: Path,
    elevation_tif: Path,
    output_dir: Path,
    wind_speed_mps: float,
    wind_direction_deg: float,
    mesh_resolution_m: float,
    momentum: bool = False,
    iterations: int = 300,
    turbulence_output: bool = False,
    num_threads: int = 1,
    timeout_seconds: int = 900,
) -> dict:
    if not cli_path.exists():
        raise WindNinjaRunError(
            stage="windninja_setup",
            message=f"WindNinja CLI not found at {cli_path}",
            command=[str(cli_path)],
            output_dir=output_dir,
        )

    if not elevation_tif.exists():
        raise WindNinjaRunError(
            stage="windninja_setup",
            message=f"Elevation input not found at {elevation_tif}",
            command=[str(cli_path)],
            output_dir=output_dir,
        )

    if output_dir.exists():
        for existing in output_dir.iterdir():
            if existing.is_dir():
                import shutil

                shutil.rmtree(existing)
            else:
                existing.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(cli_path),
        "--num_threads",
        str(num_threads),
        "--elevation_file",
        str(elevation_tif),
        "--initialization_method",
        "domainAverageInitialization",
        "--input_speed",
        f"{wind_speed_mps:.6f}",
        "--input_speed_units",
        "mps",
        "--output_speed_units",
        "mps",
        "--input_direction",
        f"{wind_direction_deg:.6f}",
        "--input_wind_height",
        "10",
        "--units_input_wind_height",
        "m",
        "--output_wind_height",
        "10",
        "--units_output_wind_height",
        "m",
        "--vegetation",
        "grass",
        "--mesh_resolution",
        f"{mesh_resolution_m:.0f}",
        "--units_mesh_resolution",
        "m",
        "--write_ascii_output",
        "true",
        "--ascii_out_aaigrid",
        "true",
        "--ascii_out_json",
        "false",
        "--ascii_out_uv",
        "true",
        "--output_path",
        str(output_dir),
    ]
    if momentum:
        command.extend(
            [
                "--momentum_flag",
                "true",
                "--number_of_iterations",
                str(iterations),
            ]
        )
        if turbulence_output:
            command.extend(
                [
                    "--turbulence_output_flag",
                    "true",
                ]
            )
    expected_outputs = expected_windninja_ascii_paths(
        elevation_tif=elevation_tif,
        wind_speed_mps=wind_speed_mps,
        wind_direction_deg=wind_direction_deg,
        mesh_resolution_m=mesh_resolution_m,
        output_dir=output_dir,
    )

    try:
        run_kwargs = {
            "capture_output": True,
            "text": True,
            "check": True,
            "timeout": timeout_seconds,
        }
        if os.name == "nt":
            run_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        completed = subprocess.run(
            command,
            **run_kwargs,
        )
    except subprocess.TimeoutExpired as exc:
        raise WindNinjaRunError(
            stage="windninja_timeout",
            message=f"WindNinja exceeded timeout of {timeout_seconds} seconds",
            command=command,
            output_dir=output_dir,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise WindNinjaRunError(
            stage="windninja_subprocess",
            message=f"WindNinja exited with code {exc.returncode}",
            command=command,
            output_dir=output_dir,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        ) from exc

    required_keys = ["speed", "direction"]
    missing_outputs = [str(expected_outputs[key]) for key in required_keys if not expected_outputs[key].exists()]
    if missing_outputs:
        raise WindNinjaRunError(
            stage="windninja_outputs",
            message="WindNinja completed but did not produce all expected ASCII outputs",
            command=command,
            output_dir=output_dir,
            stdout=completed.stdout,
            stderr=completed.stderr,
            missing_outputs=missing_outputs,
        )

    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_dir": str(output_dir),
        "expected_outputs": {key: str(value) for key, value in expected_outputs.items()},
    }


def expected_windninja_ascii_paths(
    elevation_tif: Path,
    wind_speed_mps: float,
    wind_direction_deg: float,
    mesh_resolution_m: float,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    base_name = f"{elevation_tif.stem}_{int(round(wind_direction_deg))}_{int(round(wind_speed_mps))}_{int(round(mesh_resolution_m))}m"
    parent = output_dir if output_dir is not None else elevation_tif.parent
    return {
        "speed": parent / f"{base_name}_vel.asc",
        "direction": parent / f"{base_name}_ang.asc",
        "cloud": parent / f"{base_name}_cld.asc",
        "u": parent / f"{base_name}_u.asc",
        "v": parent / f"{base_name}_v.asc",
    }


def _draw_line(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], thickness: int = 1) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for step in range(steps + 1):
        t = step / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        y_start = max(0, y - thickness)
        y_end = min(image.shape[0], y + thickness + 1)
        x_start = max(0, x - thickness)
        x_end = min(image.shape[1], x + thickness + 1)
        image[y_start:y_end, x_start:x_end] = color


def _fill_rect(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    x0 = max(0, min(image.shape[1], x0))
    x1 = max(0, min(image.shape[1], x1))
    y0 = max(0, min(image.shape[0], y0))
    y1 = max(0, min(image.shape[0], y1))
    if x1 <= x0 or y1 <= y0:
        return
    image[y0:y1, x0:x1] = color


def _draw_arrow(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    flow_u: float,
    flow_v: float,
    length: int,
    color: tuple[int, int, int],
    thickness: int = 1,
    head_scale: float = 1.0,
) -> None:
    magnitude = float(np.hypot(flow_u, flow_v))
    if magnitude <= 0.0 or not np.isfinite(magnitude):
        return

    unit_x = flow_u / magnitude
    unit_y = -flow_v / magnitude
    end_x = int(round(center_x + unit_x * length))
    end_y = int(round(center_y + unit_y * length))
    _draw_line(image, center_x, center_y, end_x, end_y, color, thickness=thickness)

    head_length = max(4, int(round((length / 3.0) * head_scale)))
    angle = np.deg2rad(25.0)
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    back_x = -unit_x
    back_y = -unit_y
    left_x = back_x * cos_a - back_y * sin_a
    left_y = back_x * sin_a + back_y * cos_a
    right_x = back_x * cos_a + back_y * sin_a
    right_y = -back_x * sin_a + back_y * cos_a
    _draw_line(
        image,
        end_x,
        end_y,
        int(round(end_x + left_x * head_length)),
        int(round(end_y + left_y * head_length)),
        color,
        thickness=thickness,
    )
    _draw_line(
        image,
        end_x,
        end_y,
        int(round(end_x + right_x * head_length)),
        int(round(end_y + right_y * head_length)),
        color,
        thickness=thickness,
    )


def _arrow_head_length(length: int, head_scale: float) -> int:
    return max(4, int(round((length / 3.0) * head_scale)))


def _expand_single_band_to_rgb(data: np.ndarray, cell_px: int) -> np.ndarray:
    expanded = np.repeat(np.repeat(data, cell_px, axis=0), cell_px, axis=1)
    return np.repeat(expanded[:, :, None], 3, axis=2)


def _resample_to_shape(data: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    row_idx = np.linspace(0, data.shape[0] - 1, target_rows).round().astype(int)
    col_idx = np.linspace(0, data.shape[1] - 1, target_cols).round().astype(int)
    return data[np.ix_(row_idx, col_idx)]


def _bilinear_resize(data: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    src_rows, src_cols = data.shape
    if src_rows == target_rows and src_cols == target_cols:
        return data.astype(np.float32, copy=False)

    row_coords = np.linspace(0, src_rows - 1, target_rows, dtype=np.float32)
    col_coords = np.linspace(0, src_cols - 1, target_cols, dtype=np.float32)
    row0 = np.floor(row_coords).astype(int)
    col0 = np.floor(col_coords).astype(int)
    row1 = np.clip(row0 + 1, 0, src_rows - 1)
    col1 = np.clip(col0 + 1, 0, src_cols - 1)
    row_w = row_coords - row0
    col_w = col_coords - col0

    resized = np.empty((target_rows, target_cols), dtype=np.float32)
    for i in range(target_rows):
        top = data[row0[i], col0] * (1.0 - col_w) + data[row0[i], col1] * col_w
        bottom = data[row1[i], col0] * (1.0 - col_w) + data[row1[i], col1] * col_w
        resized[i] = top * (1.0 - row_w[i]) + bottom * row_w[i]
    return resized


def _mean_filter_nan(data: np.ndarray, radius: int) -> np.ndarray:
    height, width = data.shape
    output = np.full((height, width), np.nan, dtype=np.float32)
    for row in range(height):
        row0 = max(0, row - radius)
        row1 = min(height, row + radius + 1)
        for col in range(width):
            col0 = max(0, col - radius)
            col1 = min(width, col + radius + 1)
            window = data[row0:row1, col0:col1]
            valid = window[np.isfinite(window)]
            if valid.size:
                output[row, col] = float(valid.mean())
    return output


def _std_filter_nan(data: np.ndarray, radius: int) -> np.ndarray:
    height, width = data.shape
    output = np.full((height, width), np.nan, dtype=np.float32)
    for row in range(height):
        row0 = max(0, row - radius)
        row1 = min(height, row + radius + 1)
        for col in range(width):
            col0 = max(0, col - radius)
            col1 = min(width, col + radius + 1)
            window = data[row0:row1, col0:col1]
            valid = window[np.isfinite(window)]
            if valid.size >= 2:
                output[row, col] = float(valid.std())
            elif valid.size == 1:
                output[row, col] = 0.0
    return output


def diverging_blue_black_red_colormap() -> list[tuple[float, tuple[int, int, int]]]:
    return [
        (0.00, (49, 88, 173)),
        (0.50, (0, 0, 0)),
        (1.00, (200, 45, 45)),
    ]


def diverging_blue_green_red_colormap() -> list[tuple[float, tuple[int, int, int]]]:
    return [
        (0.00, (49, 88, 173)),
        (0.20, (35, 180, 255)),
        (0.40, (34, 139, 34)),
        (0.65, (235, 220, 60)),
        (0.82, (245, 140, 45)),
        (1.00, (200, 45, 45)),
    ]


def _normalize_with_center(data: np.ndarray, vmin: float, center: float, vmax: float) -> np.ndarray:
    normalized = np.full(data.shape, np.nan, dtype=np.float32)
    if center <= vmin:
        center = vmin + max((vmax - vmin) * 0.5, 1.0e-6)
    if center >= vmax:
        center = vmax - max((vmax - vmin) * 0.5, 1.0e-6)

    left_mask = np.isfinite(data) & (data <= center)
    right_mask = np.isfinite(data) & (data > center)

    left_span = max(center - vmin, 1.0e-6)
    right_span = max(vmax - center, 1.0e-6)
    normalized[left_mask] = 0.5 * np.clip((data[left_mask] - vmin) / left_span, 0.0, 1.0)
    normalized[right_mask] = 0.5 + 0.5 * np.clip((data[right_mask] - center) / right_span, 0.0, 1.0)
    return normalized


def _legend_canvas(
    field: np.ndarray,
    base_rgb: np.ndarray,
    colormap: list[tuple[float, tuple[int, int, int]]],
    title: str,
    units: str,
    output_png: Path,
    alpha: float = 0.45,
    signed: bool = False,
    center_value: float | None = None,
    footer_text: str | None = None,
) -> Path:
    valid = field[np.isfinite(field)]
    if valid.size == 0:
        raise ValueError(f"No finite data available for {title} preview.")

    if signed:
        max_abs = float(np.max(np.abs(valid)))
        if max_abs <= 0.0:
            max_abs = 1.0
        vmin = -max_abs
        vmax = max_abs
    else:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
        if vmax <= vmin:
            vmax = vmin + 1.0

    if center_value is None:
        center_value = float(np.mean(valid))
    normalized = _normalize_with_center(field, vmin=vmin, center=center_value, vmax=vmax)
    overlay_rgb = _interpolate_colormap(normalized, colormap).astype(np.float32)
    overlay_rgb[~np.isfinite(field)] = 255.0

    map_height, map_width = base_rgb.shape[:2]
    normalized_full = _bilinear_resize(normalized.astype(np.float32), map_height, map_width)
    overlay_full = _interpolate_colormap(np.clip(normalized_full, 0.0, 1.0), colormap).astype(np.float32)
    blended = np.clip((1.0 - alpha) * base_rgb + alpha * overlay_full, 0, 255).astype(np.uint8)

    top_margin = 80
    footer_band_height = 48 if footer_text else 0
    bottom_margin = (24 + footer_band_height) if footer_band_height else 24
    legend_width = 190
    canvas_height = max(map_height + footer_band_height, 320)
    canvas = np.full((canvas_height, map_width + legend_width, 3), 255, dtype=np.uint8)
    canvas[:map_height, :map_width] = blended
    if footer_text:
        footer_top = canvas_height - footer_band_height
        canvas[footer_top:canvas_height, : map_width + legend_width] = 255

    legend_left = map_width + 40
    available_legend_height = max(80, canvas_height - top_margin - bottom_margin)
    legend_height = min(available_legend_height, 900)
    legend_top = top_margin
    legend_bottom = legend_top + legend_height
    legend_right = legend_left + 28
    gradient_values = np.linspace(1.0, 0.0, legend_height, dtype=np.float32).reshape(-1, 1)
    gradient_rgb = np.repeat(_interpolate_colormap(gradient_values, colormap), legend_right - legend_left, axis=1)
    canvas[legend_top:legend_bottom, legend_left:legend_right] = gradient_rgb

    tick_values = np.array([vmax, 0.5 * (center_value + vmax), center_value, 0.5 * (vmin + center_value), vmin], dtype=np.float32)
    for tick_value in tick_values:
        fraction = float(_normalize_with_center(np.array([[tick_value]], dtype=np.float32), vmin, center_value, vmax)[0, 0])
        y = int(round(legend_bottom - fraction * legend_height))
        y = max(legend_top, min(legend_bottom - 1, y))
        canvas[y : y + 2, legend_right : legend_right + 12] = (0, 0, 0)
        _draw_text(canvas, legend_right + 18, y - 6, f"{tick_value:.2f}", (0, 0, 0), scale=2)

    _draw_text(canvas, legend_left - 4, 32, title, (0, 0, 0), scale=3)
    _draw_text(canvas, legend_left - 4, 58, units, (0, 0, 0), scale=2)
    if footer_text:
        footer_y = canvas_height - 34
        _draw_text(canvas, 24, footer_y, footer_text, (0, 0, 0), scale=3)
    return _write_png(output_png, canvas)


def write_windninja_knots_vector_preview(
    speed_asc: Path,
    u_asc: Path,
    v_asc: Path,
    preview_png: Path,
    dem_basemap_tif: Path | None = None,
    vector_stride: int = 1,
    vector_scale: float = 1.0,
) -> Path:
    speed_mps, _ = _read_aaigrid(speed_asc)
    u_mps, _ = _read_aaigrid(u_asc)
    v_mps, _ = _read_aaigrid(v_asc)
    return write_windninja_knots_vector_preview_from_arrays(
        speed_mps=speed_mps,
        u_mps=u_mps,
        v_mps=v_mps,
        preview_png=preview_png,
        dem_basemap_tif=dem_basemap_tif,
        vector_stride=vector_stride,
        vector_scale=vector_scale,
    )


def write_windninja_knots_vector_preview_from_speed_angle(
    speed_asc: Path,
    angle_asc: Path,
    preview_png: Path,
    dem_basemap_tif: Path | None = None,
    source_header: dict[str, float] | None = None,
    vector_stride: int = 1,
    vector_scale: float = 1.0,
    colormap: list[tuple[float, tuple[int, int, int]]] | None = None,
    center_value: float | None = None,
    title: str = "wind",
    units: str = "knots",
    footer_text: str | None = None,
    inset_lines: list[str] | None = None,
    bottom_table_rows: list[dict[str, str]] | None = None,
) -> Path:
    speed_mps, speed_header = _read_aaigrid(speed_asc)
    if source_header is None:
        source_header = speed_header
    direction_from_deg, _ = _read_aaigrid(angle_asc)
    theta = np.deg2rad(direction_from_deg)
    u_mps = -speed_mps * np.sin(theta)
    v_mps = -speed_mps * np.cos(theta)
    return write_windninja_knots_vector_preview_from_arrays(
        speed_mps=speed_mps,
        u_mps=u_mps,
        v_mps=v_mps,
        preview_png=preview_png,
        dem_basemap_tif=dem_basemap_tif,
        source_header=source_header,
        vector_stride=vector_stride,
        vector_scale=vector_scale,
        colormap=colormap,
        center_value=center_value,
        title=title,
        units=units,
        footer_text=footer_text,
        inset_lines=inset_lines,
        bottom_table_rows=bottom_table_rows,
    )


def write_windninja_knots_vector_preview_from_arrays(
    speed_mps: np.ndarray,
    u_mps: np.ndarray,
    v_mps: np.ndarray,
    preview_png: Path,
    dem_basemap_tif: Path | None = None,
    source_header: dict[str, float] | None = None,
    vector_stride: int = 1,
    vector_scale: float = 1.0,
    colormap: list[tuple[float, tuple[int, int, int]]] | None = None,
    center_value: float | None = None,
    title: str = "wind",
    units: str = "knots",
    footer_text: str | None = None,
    inset_lines: list[str] | None = None,
    edge_trim_cells: int = 1,
    bottom_table_rows: list[dict[str, str]] | None = None,
) -> Path:

    speed_kts = speed_mps * 1.94384449
    vector_speed_kts = speed_kts
    vector_u_mps = u_mps
    vector_v_mps = v_mps
    if edge_trim_cells > 0 and speed_kts.shape[0] > (2 * edge_trim_cells + 2) and speed_kts.shape[1] > (2 * edge_trim_cells + 2):
        vector_speed_kts = speed_kts.copy()
        vector_speed_kts[:edge_trim_cells, :] = np.nan
        vector_speed_kts[-edge_trim_cells:, :] = np.nan
        vector_speed_kts[:, :edge_trim_cells] = np.nan
        vector_speed_kts[:, -edge_trim_cells:] = np.nan
        vector_u_mps = u_mps.copy()
        vector_v_mps = v_mps.copy()
        vector_u_mps[:edge_trim_cells, :] = np.nan
        vector_u_mps[-edge_trim_cells:, :] = np.nan
        vector_u_mps[:, :edge_trim_cells] = np.nan
        vector_u_mps[:, -edge_trim_cells:] = np.nan
        vector_v_mps[:edge_trim_cells, :] = np.nan
        vector_v_mps[-edge_trim_cells:, :] = np.nan
        vector_v_mps[:, :edge_trim_cells] = np.nan
        vector_v_mps[:, -edge_trim_cells:] = np.nan

    valid = speed_kts[np.isfinite(speed_kts)]
    if valid.size == 0:
        raise ValueError("No finite source wind speeds available for wind preview.")
    vmin = float(np.percentile(valid, 2))
    vmax = float(np.percentile(valid, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0
    if center_value is None:
        center_value = float(np.mean(valid))

    normalized = _normalize_with_center(speed_kts, vmin=vmin, center=center_value, vmax=vmax)
    if colormap is None:
        colormap = diverging_blue_black_red_colormap()
    map_rgb = _interpolate_colormap(normalized, colormap)
    map_rgb[~np.isfinite(speed_kts)] = (255, 255, 255)

    rows, cols = speed_kts.shape
    top_margin = 80
    footer_lines = 1 if footer_text else 0
    table_lines = 0 if not bottom_table_rows else (2 + len(bottom_table_rows))
    footer_band_height = 0
    if footer_lines or table_lines:
        footer_band_height = 28 + footer_lines * 22 + table_lines * 20
    bottom_margin = (24 + footer_band_height) if footer_text else 24
    legend_width = 190
    if dem_basemap_tif is not None and dem_basemap_tif.exists():
        with rasterio.open(dem_basemap_tif) as src:
            dem_base = src.read(1, masked=True).filled(0).astype(np.uint8)
            base_transform = src.transform
        map_height, map_width = dem_base.shape
        base_rgb = np.repeat(dem_base[:, :, None], 3, axis=2).astype(np.float32)
    else:
        cell_px = max(24, min(40, 640 // max(rows, cols)))
        map_height = rows * cell_px
        map_width = cols * cell_px
        base_rgb = np.full((map_height, map_width, 3), 235.0, dtype=np.float32)
        base_transform = None

    canvas_height = max(map_height + footer_band_height, 320)
    canvas = np.full((canvas_height, map_width + legend_width, 3), 255, dtype=np.uint8)

    if dem_basemap_tif is not None and source_header is not None:
        normalized_full, _, _ = _reproject_array_to_basemap(
            normalized.astype(np.float32),
            source_header=source_header,
            basemap_tif=dem_basemap_tif,
            resampling=Resampling.bilinear,
        )
        if np.isfinite(normalized).any() and not np.isfinite(normalized_full).any():
            normalized_full, _, _ = _reproject_array_to_basemap(
                normalized.astype(np.float32),
                source_header=source_header,
                basemap_tif=dem_basemap_tif,
                resampling=Resampling.nearest,
            )
    else:
        normalized_full = _bilinear_resize(normalized.astype(np.float32), map_height, map_width)
    overlay_rgb = _interpolate_colormap(np.clip(normalized_full, 0.0, 1.0), colormap).astype(np.float32)
    overlay_rgb[~np.isfinite(normalized_full)] = 255.0
    blended = np.clip(0.55 * base_rgb + 0.45 * overlay_rgb, 0, 255).astype(np.uint8)
    canvas[:map_height, :map_width] = blended
    if footer_band_height:
        footer_top = canvas_height - footer_band_height
        canvas[footer_top:canvas_height, : map_width + legend_width] = 255

    speed_min = float(valid.min())
    speed_max = float(valid.max())
    speed_span = max(speed_max - speed_min, 0.1)
    if dem_basemap_tif is not None and source_header is not None and base_transform is not None:
        base_res = min(abs(base_transform.a), abs(base_transform.e))
        source_cell = float(source_header["cellsize"])
        base_cell_for_arrow = max(base_res, 1.0e-6)
        vector_speed_final, _, _ = _reproject_array_to_basemap(
            vector_speed_kts.astype(np.float32),
            source_header=source_header,
            basemap_tif=dem_basemap_tif,
            resampling=Resampling.bilinear,
        )
        if np.isfinite(vector_speed_kts).any() and not np.isfinite(vector_speed_final).any():
            vector_speed_final, _, _ = _reproject_array_to_basemap(
                vector_speed_kts.astype(np.float32),
                source_header=source_header,
                basemap_tif=dem_basemap_tif,
                resampling=Resampling.nearest,
            )
        vector_u_final, _, _ = _reproject_array_to_basemap(
            vector_u_mps.astype(np.float32),
            source_header=source_header,
            basemap_tif=dem_basemap_tif,
            resampling=Resampling.bilinear,
        )
        if np.isfinite(vector_u_mps).any() and not np.isfinite(vector_u_final).any():
            vector_u_final, _, _ = _reproject_array_to_basemap(
                vector_u_mps.astype(np.float32),
                source_header=source_header,
                basemap_tif=dem_basemap_tif,
                resampling=Resampling.nearest,
            )
        vector_v_final, _, _ = _reproject_array_to_basemap(
            vector_v_mps.astype(np.float32),
            source_header=source_header,
            basemap_tif=dem_basemap_tif,
            resampling=Resampling.bilinear,
        )
        if np.isfinite(vector_v_mps).any() and not np.isfinite(vector_v_final).any():
            vector_v_final, _, _ = _reproject_array_to_basemap(
                vector_v_mps.astype(np.float32),
                source_header=source_header,
                basemap_tif=dem_basemap_tif,
                resampling=Resampling.nearest,
            )
        pixel_stride = max(8, int(round((source_cell / base_cell_for_arrow) * max(1, int(vector_stride)))))
        row_start = pixel_stride // 2
        col_base = pixel_stride // 2
        for row in range(row_start, map_height, pixel_stride):
            col_offset = (pixel_stride // 2) if ((row - row_start) // pixel_stride) % 2 == 1 else 0
            for col in range(col_base + col_offset, map_width, pixel_stride):
                if not np.isfinite(vector_speed_final[row, col]):
                    continue
                speed_fraction = (float(vector_speed_final[row, col]) - speed_min) / speed_span
                arrow_length = int(round((0.10 + 0.22 * speed_fraction) * pixel_stride * vector_scale))
                arrow_length = max(6, arrow_length)
                head_length = _arrow_head_length(arrow_length, 1.4)
                edge_margin = max(4, head_length + 2)
                if (
                    col < edge_margin
                    or col >= (map_width - edge_margin)
                    or row < edge_margin
                    or row >= (map_height - edge_margin)
                ):
                    continue
                _draw_arrow(
                    canvas,
                    col,
                    row,
                    float(vector_u_final[row, col]),
                    float(vector_v_final[row, col]),
                    arrow_length,
                    (15, 15, 15),
                    thickness=1,
                    head_scale=1.4,
                )
    else:
        stride = max(1, int(vector_stride))
        row_start = stride // 2
        col_base = stride // 2
        cell_height = map_height / rows
        cell_width = map_width / cols
        for row in range(row_start, rows, stride):
            col_offset = (stride // 2) if ((row - row_start) // stride) % 2 == 1 else 0
            for col in range(col_base + col_offset, cols, stride):
                if not np.isfinite(vector_speed_kts[row, col]):
                    continue
                cx = int(round((col + 0.5) * cell_width))
                cy = int(round((row + 0.5) * cell_height))
                speed_fraction = (float(vector_speed_kts[row, col]) - speed_min) / speed_span
                arrow_length = int(round((0.20 + 0.50 * speed_fraction) * min(cell_width, cell_height) * vector_scale))
                arrow_length = max(6, arrow_length)
                _draw_arrow(
                    canvas,
                    cx,
                    cy,
                    float(vector_u_mps[row, col]),
                    float(vector_v_mps[row, col]),
                    arrow_length,
                    (15, 15, 15),
                    thickness=1,
                    head_scale=1.4,
                )

    if inset_lines:
        panel_width = 300
        panel_height = 28 + 24 * len(inset_lines)
        panel_right = map_width - 28
        panel_left = max(24, panel_right - panel_width)
        panel_bottom = map_height - 28
        panel_top = max(24, panel_bottom - panel_height)
        _fill_rect(canvas, panel_left, panel_top, panel_right, panel_bottom, (245, 245, 245))
        _draw_line(canvas, panel_left, panel_top, panel_right, panel_top, (20, 20, 20), thickness=1)
        _draw_line(canvas, panel_right - 1, panel_top, panel_right - 1, panel_bottom, (20, 20, 20), thickness=1)
        _draw_line(canvas, panel_left, panel_bottom - 1, panel_right, panel_bottom - 1, (20, 20, 20), thickness=1)
        _draw_line(canvas, panel_left, panel_top, panel_left, panel_bottom, (20, 20, 20), thickness=1)
        _draw_text(canvas, panel_left + 14, panel_top + 10, "race wx", (0, 0, 0), scale=2)
        text_y = panel_top + 38
        for line in inset_lines:
            _draw_text(canvas, panel_left + 14, text_y, line, (0, 0, 0), scale=2)
            text_y += 24

    legend_left = map_width + 40
    available_legend_height = max(80, canvas_height - top_margin - bottom_margin)
    legend_height = min(available_legend_height, 900)
    legend_top = top_margin
    legend_bottom = legend_top + legend_height
    legend_right = legend_left + 28
    gradient_values = np.linspace(1.0, 0.0, legend_height, dtype=np.float32).reshape(-1, 1)
    gradient_rgb = np.repeat(_interpolate_colormap(gradient_values, colormap), legend_right - legend_left, axis=1)
    canvas[legend_top:legend_bottom, legend_left:legend_right] = gradient_rgb

    tick_values = np.array([vmax, 0.5 * (center_value + vmax), center_value, 0.5 * (vmin + center_value), vmin], dtype=np.float32)
    for tick_value in tick_values:
        fraction = float(_normalize_with_center(np.array([[tick_value]], dtype=np.float32), vmin, center_value, vmax)[0, 0])
        y = int(round(legend_bottom - fraction * legend_height))
        y = max(legend_top, min(legend_bottom - 1, y))
        canvas[y : y + 2, legend_right : legend_right + 12] = (0, 0, 0)
        _draw_text(canvas, legend_right + 18, y - 6, f"{tick_value:.1f}", (0, 0, 0), scale=2)

    _draw_text(canvas, legend_left - 4, 32, title, (0, 0, 0), scale=3)
    _draw_text(canvas, legend_left - 4, 58, units, (0, 0, 0), scale=2)
    if footer_band_height:
        footer_y = canvas_height - footer_band_height + 10
        if footer_text:
            _draw_text(canvas, 24, footer_y, footer_text, (0, 0, 0), scale=2)
            footer_y += 24
        if bottom_table_rows:
            _draw_text(canvas, 24, footer_y, "model", (0, 0, 0), scale=2)
            _draw_text(canvas, 170, footer_y, "wind kt", (0, 0, 0), scale=2)
            _draw_text(canvas, 262, footer_y, "gust kt", (0, 0, 0), scale=2)
            footer_y += 22
            for row in bottom_table_rows:
                _draw_text(canvas, 24, footer_y, row["model"], (0, 0, 0), scale=2)
                _draw_text(canvas, 170, footer_y, row["wind"], (0, 0, 0), scale=2)
                _draw_text(canvas, 262, footer_y, row["gust"], (0, 0, 0), scale=2)
                footer_y += 20
    return _write_png(preview_png, canvas)


def write_windninja_summary(
    summary_path: Path,
    speed_tif: Path,
    direction_asc: Path,
    boundary_summary: dict,
    run_metadata: dict,
    mesh_resolution_m: float,
    model_name: str = "WindNinja_COM",
) -> Path:
    with rasterio.open(speed_tif) as src:
        wind_speed = src.read(1, masked=True).filled(np.nan)
    valid = wind_speed[np.isfinite(wind_speed)]
    summary = {
        "model": model_name,
        "boundary_valid_time_utc": boundary_summary["valid_time_utc"],
        "boundary_wind_speed_mps": boundary_summary["wind_speed_mps"],
        "boundary_wind_from_direction_deg": boundary_summary["wind_from_direction_deg"],
        "mesh_resolution_m": mesh_resolution_m,
        "grid_shape": [int(wind_speed.shape[0]), int(wind_speed.shape[1])],
        "min_local_speed_mps": float(valid.min()),
        "max_local_speed_mps": float(valid.max()),
        "mean_local_speed_mps": float(valid.mean()),
        "speed_raster": str(speed_tif),
        "direction_ascii": str(direction_asc),
        "stdout_tail": run_metadata["stdout"].splitlines()[-10:],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def write_flow_diagnostic_geotiff(source_asc: Path, destination_tif: Path, data: np.ndarray, crs: str = "EPSG:26917") -> Path:
    _, header = _read_aaigrid(source_asc)
    transform = from_origin(
        header["xllcorner"],
        header["yllcorner"] + header["nrows"] * header["cellsize"],
        header["cellsize"],
        header["cellsize"],
    )
    destination_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        destination_tif,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return destination_tif


def compute_flow_diagnostics(u_asc: Path, v_asc: Path, radius_cells: int = 2) -> dict[str, np.ndarray]:
    u_mps, u_header = _read_aaigrid(u_asc)
    v_mps, _ = _read_aaigrid(v_asc)
    cell_size_m = float(u_header["cellsize"])
    speed_mps = np.hypot(u_mps, v_mps).astype(np.float32)

    speed_mean = _mean_filter_nan(speed_mps, radius=radius_cells)
    speed_std = _std_filter_nan(speed_mps, radius=radius_cells)
    variability_index = np.divide(speed_std, np.maximum(speed_mean, 0.1), out=np.full_like(speed_std, np.nan), where=np.isfinite(speed_std))

    du_dy, du_dx = np.gradient(u_mps, cell_size_m, cell_size_m)
    dv_dy, dv_dx = np.gradient(v_mps, cell_size_m, cell_size_m)
    vorticity = (dv_dx - du_dy).astype(np.float32)
    divergence = (du_dx + dv_dy).astype(np.float32)
    shear = np.sqrt((du_dx - dv_dy) ** 2 + (dv_dx + du_dy) ** 2).astype(np.float32)

    mask = np.isfinite(speed_mps)
    for field in (speed_std, variability_index, vorticity, divergence, shear):
        field[~mask] = np.nan

    return {
        "speed_mps": speed_mps,
        "speed_std_mps": speed_std,
        "variability_index": variability_index,
        "vorticity_s-1": vorticity,
        "divergence_s-1": divergence,
        "shear_s-1": shear,
    }


def write_flow_diagnostic_previews(
    diagnostics: dict[str, np.ndarray],
    dem_basemap_tif: Path,
    variability_png: Path,
    vorticity_png: Path,
) -> tuple[Path, Path]:
    with rasterio.open(dem_basemap_tif) as src:
        dem_base = src.read(1, masked=True).filled(0).astype(np.uint8)
    base_rgb = np.repeat(dem_base[:, :, None], 3, axis=2).astype(np.float32)

    variability_colormap = [
        (0.00, (237, 248, 251)),
        (0.25, (178, 226, 226)),
        (0.50, (102, 194, 164)),
        (0.75, (44, 162, 95)),
        (1.00, (0, 109, 44)),
    ]
    vorticity_colormap = [
        (0.00, (33, 102, 172)),
        (0.25, (146, 197, 222)),
        (0.50, (247, 247, 247)),
        (0.75, (244, 165, 130)),
        (1.00, (178, 24, 43)),
    ]

    variability_path = _legend_canvas(
        diagnostics["variability_index"],
        base_rgb,
        variability_colormap,
        "var",
        "cv",
        variability_png,
        alpha=0.58,
        signed=False,
    )
    vorticity_path = _legend_canvas(
        diagnostics["vorticity_s-1"],
        base_rgb,
        vorticity_colormap,
        "rot",
        "s-1",
        vorticity_png,
        alpha=0.52,
        signed=True,
    )
    return variability_path, vorticity_path


def write_scalar_diagnostic_preview(
    field: np.ndarray,
    dem_basemap_tif: Path,
    output_png: Path,
    title: str,
    units: str,
    colormap: list[tuple[float, tuple[int, int, int]]],
    source_header: dict[str, float] | None = None,
    alpha: float = 0.55,
    signed: bool = False,
    center_value: float | None = None,
    footer_text: str | None = None,
) -> Path:
    with rasterio.open(dem_basemap_tif) as src:
        dem_base = src.read(1, masked=True).filled(0).astype(np.uint8)
    base_rgb = np.repeat(dem_base[:, :, None], 3, axis=2).astype(np.float32)
    if source_header is not None:
        source_field = field.astype(np.float32)
        source_finite_count = int(np.isfinite(source_field).sum())
        if source_finite_count == 0:
            raise ValueError(f"No finite source data available for {title} preview.")
        field, _, _ = _reproject_array_to_basemap(
            source_field,
            source_header=source_header,
            basemap_tif=dem_basemap_tif,
            resampling=Resampling.bilinear,
        )
        if np.isfinite(source_field).any() and not np.isfinite(field).any():
            field, _, _ = _reproject_array_to_basemap(
                source_field,
                source_header=source_header,
                basemap_tif=dem_basemap_tif,
                resampling=Resampling.nearest,
            )
        if not np.isfinite(field).any():
            raise ValueError(
                f"No finite {title} preview cells after mapping source grid to basemap. "
                f"source_finite_cells={source_finite_count}; source_header={source_header}; "
                f"basemap={dem_basemap_tif}"
            )
    return _legend_canvas(
        field=field,
        base_rgb=base_rgb,
        colormap=colormap,
        title=title,
        units=units,
        output_png=output_png,
        alpha=alpha,
        signed=signed,
        center_value=center_value,
        footer_text=footer_text,
    )
