from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds


def _bounded_window(src: rasterio.io.DatasetReader, bounds: tuple[float, float, float, float]) -> Window:
    requested = from_bounds(*bounds, transform=src.transform)
    requested = requested.round_offsets().round_lengths()
    full = Window(col_off=0, row_off=0, width=src.width, height=src.height)
    return requested.intersection(full)


def _as_float_nan(data: np.ma.MaskedArray | np.ndarray) -> np.ndarray:
    if np.ma.isMaskedArray(data):
        return data.astype("float32").filled(np.nan)
    return np.asarray(data, dtype="float32")


def clip_dem_to_bbox(
    source_tif: Path,
    clipped_tif: Path,
    bbox: tuple[float, float, float, float],
    *,
    bbox_crs: str = "EPSG:4326",
) -> Path:
    clipped_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source_tif) as src:
        source_bounds = transform_bounds(bbox_crs, src.crs, *bbox, densify_pts=21)
        window = _bounded_window(src, source_bounds)
        clipped_data = src.read(window=window, masked=True)
        clipped_transform = src.window_transform(window)
        clipped_array = _as_float_nan(clipped_data)
        clipped_meta = src.meta.copy()
        clipped_meta.update(
            {
                "height": clipped_array.shape[1],
                "width": clipped_array.shape[2],
                "transform": clipped_transform,
                "dtype": "float32",
                "nodata": np.nan,
            }
        )

        with rasterio.open(clipped_tif, "w", **clipped_meta) as dst:
            dst.write(clipped_array)

    return clipped_tif


def build_speedup_raster(dem_tif: Path, output_tif: Path, wind_direction_deg: float, base_wind_speed_mps: float) -> dict:
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_tif) as src:
        dem = _as_float_nan(src.read(1, masked=True))
        transform = src.transform
        res_x = abs(transform.a)
        res_y = abs(transform.e)

        grad_y, grad_x = np.gradient(dem, res_y, res_x)

        theta = math.radians((270.0 - wind_direction_deg) % 360.0)
        wind_x = math.cos(theta)
        wind_y = math.sin(theta)

        slope_along_wind = -(grad_x * wind_x + grad_y * wind_y)
        speedup = 1.0 + np.clip(slope_along_wind * 0.35, -0.4, 0.6)
        local_speed = np.clip(base_wind_speed_mps * speedup, 0.0, None).astype("float32")

        meta = src.meta.copy()
        meta.update(count=1, dtype="float32", nodata=np.nan)

        with rasterio.open(output_tif, "w", **meta) as dst:
            dst.write(local_speed, 1)

    summary = {
        "wind_direction_deg": wind_direction_deg,
        "base_wind_speed_mps": base_wind_speed_mps,
        "min_local_speed_mps": float(np.nanmin(local_speed)),
        "max_local_speed_mps": float(np.nanmax(local_speed)),
        "mean_local_speed_mps": float(np.nanmean(local_speed)),
    }
    return summary


def _nanmean_blocks(data: np.ndarray, block_size: int) -> np.ndarray:
    height, width = data.shape
    out_height = math.ceil(height / block_size)
    out_width = math.ceil(width / block_size)
    coarse = np.full((out_height, out_width), np.nan, dtype="float32")

    for row in range(out_height):
        for col in range(out_width):
            block = data[
                row * block_size : min((row + 1) * block_size, height),
                col * block_size : min((col + 1) * block_size, width),
            ]
            valid = block[np.isfinite(block)]
            if valid.size:
                coarse[row, col] = float(valid.mean())

    return coarse


def _box_mean(data: np.ndarray, radius: int) -> np.ndarray:
    height, width = data.shape
    out = np.full_like(data, np.nan, dtype="float32")
    for row in range(height):
        row0 = max(0, row - radius)
        row1 = min(height, row + radius + 1)
        for col in range(width):
            col0 = max(0, col - radius)
            col1 = min(width, col + radius + 1)
            window = data[row0:row1, col0:col1]
            valid = window[np.isfinite(window)]
            if valid.size:
                out[row, col] = float(valid.mean())
    return out


def build_coarse_diagnostic_wind(
    dem_tif: Path,
    output_tif: Path,
    wind_direction_deg: float,
    base_wind_speed_mps: float,
    target_resolution_m: float = 100.0,
) -> dict:
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_tif) as src:
        dem = _as_float_nan(src.read(1, masked=True))
        transform = src.transform
        src_res_x = abs(transform.a)
        src_res_y = abs(transform.e)
        block_size = max(1, int(round(target_resolution_m / max(src_res_x, src_res_y))))

        coarse_dem = _nanmean_blocks(dem, block_size)
        coarse_res_x = src_res_x * block_size
        coarse_res_y = src_res_y * block_size

        grad_y, grad_x = np.gradient(coarse_dem, coarse_res_y, coarse_res_x)
        theta = math.radians((270.0 - wind_direction_deg) % 360.0)
        wind_x = math.cos(theta)
        wind_y = math.sin(theta)

        slope_along_wind = -(grad_x * wind_x + grad_y * wind_y)
        terrain_background = _box_mean(coarse_dem, radius=1)
        terrain_position = coarse_dem - terrain_background

        # Diagnostic, terrain-aware speed adjustment at ~100 m scale.
        speedup = (
            1.0
            + np.clip(slope_along_wind * 1.8, -0.30, 0.45)
            + np.clip(terrain_position / 80.0, -0.12, 0.12)
        )
        local_speed = np.clip(base_wind_speed_mps * speedup, 0.0, None).astype("float32")
        local_speed = np.where(np.isfinite(coarse_dem), local_speed, np.nan)

        coarse_transform = rasterio.Affine(
            transform.a * block_size,
            transform.b,
            transform.c,
            transform.d,
            transform.e * block_size,
            transform.f,
        )
        meta = src.meta.copy()
        meta.update(
            count=1,
            dtype="float32",
            nodata=np.nan,
            width=coarse_dem.shape[1],
            height=coarse_dem.shape[0],
            transform=coarse_transform,
        )

        with rasterio.open(output_tif, "w", **meta) as dst:
            dst.write(local_speed, 1)

    return {
        "target_resolution_m": target_resolution_m,
        "grid_shape": [int(coarse_dem.shape[0]), int(coarse_dem.shape[1])],
        "min_local_speed_mps": float(np.nanmin(local_speed)),
        "max_local_speed_mps": float(np.nanmax(local_speed)),
        "mean_local_speed_mps": float(np.nanmean(local_speed)),
    }


def _percentile_scale(data: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return np.zeros_like(data, dtype="uint8")

    lo = float(np.percentile(valid, low))
    hi = float(np.percentile(valid, high))
    if hi <= lo:
        return np.where(np.isfinite(data), 128, 0).astype("uint8")

    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return np.where(np.isfinite(data), np.round(scaled * 255.0), 0).astype("uint8")


def _interpolate_colormap(values: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    rgb = np.zeros(values.shape + (3,), dtype=np.uint8)

    for index in range(len(stops) - 1):
        left_value, left_color = stops[index]
        right_value, right_color = stops[index + 1]
        mask = (values >= left_value) & (values <= right_value)
        if not np.any(mask):
            continue

        span = right_value - left_value
        weight = np.zeros_like(values, dtype=np.float32)
        if span > 0:
            weight[mask] = (values[mask] - left_value) / span

        left = np.array(left_color, dtype=np.float32)
        right = np.array(right_color, dtype=np.float32)
        blended = left + (right - left) * weight[mask, None]
        rgb[mask] = np.round(blended).astype(np.uint8)

    rgb[values < stops[0][0]] = stops[0][1]
    rgb[values > stops[-1][0]] = stops[-1][1]
    return rgb


_FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00100", "00100"],
    ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "%": ["11001", "11010", "00100", "01000", "10110", "00110", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "a": ["00000", "00000", "01110", "00001", "01111", "10001", "01111"],
    "b": ["10000", "10000", "11110", "10001", "10001", "10001", "11110"],
    "c": ["00000", "00000", "01110", "10001", "10000", "10001", "01110"],
    "d": ["00000", "00001", "00001", "01111", "10001", "10001", "01111"],
    "e": ["00000", "00000", "01110", "10001", "11111", "10000", "01110"],
    "f": ["00110", "01001", "01000", "11100", "01000", "01000", "01000"],
    "g": ["00000", "00000", "01111", "10001", "01111", "00001", "01110"],
    "h": ["10000", "10000", "11110", "10001", "10001", "10001", "10001"],
    "i": ["00100", "00000", "01100", "00100", "00100", "00100", "01110"],
    "j": ["00010", "00000", "00110", "00010", "00010", "10010", "01100"],
    "k": ["10000", "10010", "10100", "11000", "10100", "10010", "10001"],
    "l": ["01100", "00100", "00100", "00100", "00100", "00100", "01110"],
    "m": ["00000", "00000", "11010", "10101", "10101", "10101", "10101"],
    "n": ["00000", "00000", "11110", "10001", "10001", "10001", "10001"],
    "o": ["00000", "00000", "01110", "10001", "10001", "10001", "01110"],
    "p": ["00000", "00000", "11110", "10001", "11110", "10000", "10000"],
    "q": ["00000", "00000", "01111", "10001", "10001", "01111", "00001"],
    "r": ["00000", "00000", "10110", "11001", "10000", "10000", "10000"],
    "s": ["00000", "00000", "01111", "10000", "01110", "00001", "11110"],
    "t": ["00100", "00100", "11111", "00100", "00100", "00101", "00010"],
    "u": ["00000", "00000", "10001", "10001", "10001", "10011", "01101"],
    "v": ["00000", "00000", "10001", "10001", "10001", "01010", "00100"],
    "w": ["00000", "00000", "10001", "10001", "10101", "10101", "01010"],
    "x": ["00000", "00000", "10001", "01010", "00100", "01010", "10001"],
    "y": ["00000", "00000", "10001", "10001", "01111", "00001", "01110"],
    "z": ["00000", "00000", "11111", "00010", "00100", "01000", "11111"],
}


def _draw_text(image: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int], scale: int = 2) -> None:
    cursor_x = x
    for char in text:
        glyph = _FONT_5X7.get(char.lower(), _FONT_5X7[" "])
        for row_index, row in enumerate(glyph):
            for col_index, bit in enumerate(row):
                if bit != "1":
                    continue
                x0 = cursor_x + col_index * scale
                y0 = y + row_index * scale
                image[y0 : y0 + scale, x0 : x0 + scale] = color
        cursor_x += 6 * scale


def _write_png(path: Path, rgb: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = rgb.shape

    raw_rows = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    compressed = zlib.compress(raw_rows, level=9)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + chunk_type
            + data
            + struct.pack("!I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.write_bytes(png)
    return path


def write_dem_preview(dem_tif: Path, preview_tif: Path) -> Path:
    preview_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_tif) as src:
        dem = _as_float_nan(src.read(1, masked=True))
        transform = src.transform
        res_x = abs(transform.a)
        res_y = abs(transform.e)

        grad_y, grad_x = np.gradient(dem, res_y, res_x)
        slope = np.pi / 2.0 - np.arctan(np.sqrt(grad_x * grad_x + grad_y * grad_y))
        aspect = np.arctan2(-grad_x, grad_y)

        azimuth = math.radians(315.0)
        altitude = math.radians(45.0)
        hillshade = (
            np.sin(altitude) * np.sin(slope)
            + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
        )
        hillshade = np.where(np.isfinite(dem), hillshade, np.nan)
        preview = _percentile_scale(hillshade)

        meta = src.meta.copy()
        meta.update(count=1, dtype="uint8", nodata=0)

        with rasterio.open(preview_tif, "w", **meta) as dst:
            dst.write(preview, 1)

    return preview_tif


def write_wind_preview(wind_tif: Path, preview_tif: Path) -> Path:
    preview_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(wind_tif) as src:
        wind = _as_float_nan(src.read(1, masked=True))
        preview = _percentile_scale(wind)

        meta = src.meta.copy()
        meta.update(count=1, dtype="uint8", nodata=0)

        with rasterio.open(preview_tif, "w", **meta) as dst:
            dst.write(preview, 1)

    return preview_tif


def write_wind_color_preview(wind_tif: Path, preview_png: Path) -> Path:
    with rasterio.open(wind_tif) as src:
        wind = _as_float_nan(src.read(1, masked=True))

    valid = wind[np.isfinite(wind)]
    vmin = float(np.percentile(valid, 2))
    vmax = float(np.percentile(valid, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0

    normalized = np.clip((wind - vmin) / (vmax - vmin), 0.0, 1.0)
    colormap = [
        (0.00, (49, 54, 149)),
        (0.20, (69, 117, 180)),
        (0.40, (116, 173, 209)),
        (0.55, (171, 217, 233)),
        (0.70, (254, 224, 144)),
        (0.85, (253, 174, 97)),
        (1.00, (215, 48, 39)),
    ]
    map_rgb = _interpolate_colormap(normalized, colormap)
    map_rgb[~np.isfinite(wind)] = (255, 255, 255)

    height, width = map_rgb.shape[:2]
    top_margin = 80
    bottom_margin = 24
    min_canvas_height = 320
    canvas_height = max(height, min_canvas_height)
    legend_width = 180
    canvas = np.full((canvas_height, width + legend_width, 3), 255, dtype=np.uint8)
    canvas[:height, :width] = map_rgb

    legend_left = width + 40
    available_legend_height = max(80, canvas_height - top_margin - bottom_margin)
    legend_height = min(available_legend_height, 900)
    legend_top = top_margin
    legend_bottom = legend_top + legend_height
    legend_right = legend_left + 28

    gradient_values = np.linspace(1.0, 0.0, legend_height, dtype=np.float32).reshape(-1, 1)
    gradient_rgb = np.repeat(_interpolate_colormap(gradient_values, colormap), legend_right - legend_left, axis=1)
    canvas[legend_top:legend_bottom, legend_left:legend_right] = gradient_rgb

    tick_values = np.linspace(vmax, vmin, 5)
    for tick_value in tick_values:
        fraction = (tick_value - vmin) / (vmax - vmin)
        y = int(round(legend_bottom - fraction * legend_height))
        y = max(legend_top, min(legend_bottom - 1, y))
        canvas[y : y + 2, legend_right : legend_right + 12] = (0, 0, 0)
        _draw_text(canvas, legend_right + 18, y - 6, f"{tick_value:.1f}", (0, 0, 0), scale=2)

    _draw_text(canvas, legend_left - 4, 32, "wind", (0, 0, 0), scale=3)
    _draw_text(canvas, legend_left - 4, 58, "m/s", (0, 0, 0), scale=2)

    return _write_png(preview_png, canvas)


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
