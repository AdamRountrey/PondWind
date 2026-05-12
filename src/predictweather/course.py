from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy as transform_xy
from rasterio.warp import transform

from predictweather.wind import _draw_text, _write_png
from predictweather.windninja import _draw_arrow, _draw_line


@dataclass(frozen=True)
class CourseMark:
    name: str
    row: float
    col: float
    easting: float
    northing: float
    lat: float
    lon: float


@dataclass(frozen=True)
class SailCourseLayout:
    wind_from_direction_deg: float
    windward_leg_heading_deg: float
    leg_length_m: float
    start_line_length_m: float
    water_level_m: float
    water_level_tolerance_m: float
    start: CourseMark
    windward: CourseMark
    reach: CourseMark
    start_line_pin: CourseMark
    start_line_committee: CourseMark

    def to_dict(self) -> dict:
        return asdict(self)


def _dominant_water_level(dem: np.ndarray) -> float:
    valid = dem[np.isfinite(dem)]
    rounded = np.round(valid, 3)
    values, counts = np.unique(rounded, return_counts=True)
    return float(values[np.argmax(counts)])


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_component: list[tuple[int, int]] = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            queue = deque([(row, col)])
            visited[row, col] = True
            component: list[tuple[int, int]] = []
            while queue:
                y, x = queue.popleft()
                component.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(component) > len(best_component):
                best_component = component

    largest = np.zeros_like(mask, dtype=bool)
    for row, col in best_component:
        largest[row, col] = True
    return largest


def _shore_distance_cells(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    distances = np.full((height, width), -1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()

    for row in range(height):
        for col in range(width):
            if not mask[row, col]:
                continue
            on_shore = False
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny = row + dy
                nx = col + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width or not mask[ny, nx]:
                    on_shore = True
                    break
            if on_shore:
                distances[row, col] = 0
                queue.append((row, col))

    while queue:
        row, col = queue.popleft()
        next_distance = distances[row, col] + 1
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny = row + dy
            nx = col + dx
            if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and distances[ny, nx] < 0:
                distances[ny, nx] = next_distance
                queue.append((ny, nx))
    return distances.astype(np.float32)


def _sample_segment(start_row: float, start_col: float, end_row: float, end_col: float, sample_spacing_px: float) -> np.ndarray:
    distance = float(math.hypot(end_row - start_row, end_col - start_col))
    steps = max(2, int(math.ceil(distance / max(sample_spacing_px, 1.0))) + 1)
    points = np.empty((steps, 2), dtype=np.float32)
    for idx in range(steps):
        fraction = idx / (steps - 1)
        points[idx, 0] = start_row + (end_row - start_row) * fraction
        points[idx, 1] = start_col + (end_col - start_col) * fraction
    return points


def _all_points_water(points: np.ndarray, mask: np.ndarray) -> bool:
    rows = np.clip(np.round(points[:, 0]).astype(int), 0, mask.shape[0] - 1)
    cols = np.clip(np.round(points[:, 1]).astype(int), 0, mask.shape[1] - 1)
    return bool(np.all(mask[rows, cols]))


def _clearance_score(points: np.ndarray, shore_distance_m: np.ndarray) -> tuple[float, float]:
    rows = np.clip(np.round(points[:, 0]).astype(int), 0, shore_distance_m.shape[0] - 1)
    cols = np.clip(np.round(points[:, 1]).astype(int), 0, shore_distance_m.shape[1] - 1)
    sampled = shore_distance_m[rows, cols]
    return float(np.min(sampled)), float(np.mean(sampled))


def _to_mark(name: str, row: float, col: float, transform_obj, crs: str) -> CourseMark:
    easting, northing = transform_xy(transform_obj, row, col, offset="center")
    lon, lat = transform(crs, "EPSG:4326", [easting], [northing])
    return CourseMark(
        name=name,
        row=float(row),
        col=float(col),
        easting=float(easting),
        northing=float(northing),
        lat=float(lat[0]),
        lon=float(lon[0]),
    )


def _find_start_line_endpoints(
    start_row: float,
    start_col: float,
    crosswind_dx: float,
    crosswind_dy: float,
    preferred_half_length_m: float,
    mask: np.ndarray,
    shore_distance_m: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    half_length = preferred_half_length_m
    for _ in range(24):
        pin = (start_row + crosswind_dy * half_length, start_col + crosswind_dx * half_length)
        committee = (start_row - crosswind_dy * half_length, start_col - crosswind_dx * half_length)
        line_points = _sample_segment(pin[0], pin[1], committee[0], committee[1], sample_spacing_px=8.0)
        if _all_points_water(line_points, mask):
            min_clearance, _ = _clearance_score(line_points, shore_distance_m)
            if min_clearance >= 12.0:
                return pin, committee, half_length * 2.0
        half_length *= 0.88
    raise RuntimeError("Could not fit a start line inside the lake.")


def layout_conventional_triangle_course(
    dem_tif: Path,
    wind_from_direction_deg: float,
    water_level_tolerance_m: float = 0.06,
) -> SailCourseLayout:
    with rasterio.open(dem_tif) as src:
        dem = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        transform_obj = src.transform
        crs = str(src.crs)
        cell_size_m = float(max(abs(transform_obj.a), abs(transform_obj.e)))

    water_level = _dominant_water_level(dem)
    candidate_water = np.isfinite(dem) & (np.abs(dem - water_level) <= water_level_tolerance_m)
    water_mask = _largest_connected_component(candidate_water)
    shore_distance_m = _shore_distance_cells(water_mask) * cell_size_m

    max_clearance = float(np.max(shore_distance_m[water_mask]))
    candidate_points = np.argwhere(water_mask & (shore_distance_m >= max_clearance * 0.55))
    candidate_points = candidate_points[::6]
    if candidate_points.size == 0:
        raise RuntimeError("No suitable interior water points found for course layout.")

    direction_rad = math.radians(wind_from_direction_deg)
    upwind_dx = math.sin(direction_rad)
    upwind_dy = -math.cos(direction_rad)
    best_score = -1.0e9
    best_layout: tuple[float, float, float, float, float, float] | None = None

    for leg_length_m in range(420, 219, -20):
        for sign in (-1.0, 1.0):
            rotate_cos = 0.5
            rotate_sin = sign * math.sin(math.pi / 3.0)
            reach_dx = upwind_dx * rotate_cos - upwind_dy * rotate_sin
            reach_dy = upwind_dx * rotate_sin + upwind_dy * rotate_cos

            for start_row, start_col in candidate_points:
                start_row = float(start_row)
                start_col = float(start_col)
                windward_row = start_row + upwind_dy * leg_length_m
                windward_col = start_col + upwind_dx * leg_length_m
                reach_row = start_row + reach_dy * leg_length_m
                reach_col = start_col + reach_dx * leg_length_m

                mark_points = np.array(
                    [
                        [start_row, start_col],
                        [windward_row, windward_col],
                        [reach_row, reach_col],
                    ],
                    dtype=np.float32,
                )
                if not _all_points_water(mark_points, water_mask):
                    continue

                legs = (
                    _sample_segment(start_row, start_col, windward_row, windward_col, 8.0),
                    _sample_segment(windward_row, windward_col, reach_row, reach_col, 8.0),
                    _sample_segment(reach_row, reach_col, start_row, start_col, 8.0),
                )
                if not all(_all_points_water(points, water_mask) for points in legs):
                    continue

                min_clearances = []
                mean_clearances = []
                for points in legs:
                    min_clearance, mean_clearance = _clearance_score(points, shore_distance_m)
                    min_clearances.append(min_clearance)
                    mean_clearances.append(mean_clearance)

                course_min = min(min_clearances)
                if course_min < 18.0:
                    continue

                start_clearance = float(shore_distance_m[int(round(start_row)), int(round(start_col))])
                score = (
                    leg_length_m * 5.0
                    + course_min * 14.0
                    + float(np.mean(mean_clearances)) * 4.0
                    + start_clearance * 2.0
                )
                if score > best_score:
                    best_score = score
                    best_layout = (start_row, start_col, windward_row, windward_col, reach_row, reach_col)
        if best_layout is not None:
            break

    if best_layout is None:
        raise RuntimeError("Could not place a conventional triangle inside the lake.")

    start_row, start_col, windward_row, windward_col, reach_row, reach_col = best_layout
    crosswind_dx = math.cos(direction_rad)
    crosswind_dy = math.sin(direction_rad)
    preferred_half_length = min(80.0, float(shore_distance_m[int(round(start_row)), int(round(start_col))]) * 0.45)
    (pin_row, pin_col), (committee_row, committee_col), start_line_length_m = _find_start_line_endpoints(
        start_row,
        start_col,
        crosswind_dx,
        crosswind_dy,
        preferred_half_length,
        water_mask,
        shore_distance_m,
    )

    return SailCourseLayout(
        wind_from_direction_deg=float(wind_from_direction_deg),
        windward_leg_heading_deg=float(wind_from_direction_deg),
        leg_length_m=float(
            math.hypot((windward_row - start_row) * cell_size_m, (windward_col - start_col) * cell_size_m)
        ),
        start_line_length_m=float(start_line_length_m),
        water_level_m=float(water_level),
        water_level_tolerance_m=float(water_level_tolerance_m),
        start=_to_mark("start", start_row, start_col, transform_obj, crs),
        windward=_to_mark("windward", windward_row, windward_col, transform_obj, crs),
        reach=_to_mark("reach", reach_row, reach_col, transform_obj, crs),
        start_line_pin=_to_mark("pin", pin_row, pin_col, transform_obj, crs),
        start_line_committee=_to_mark("committee", committee_row, committee_col, transform_obj, crs),
    )


def _draw_filled_circle(image: np.ndarray, center_x: int, center_y: int, radius: int, color: tuple[int, int, int]) -> None:
    for y in range(max(0, center_y - radius), min(image.shape[0], center_y + radius + 1)):
        for x in range(max(0, center_x - radius), min(image.shape[1], center_x + radius + 1)):
            if (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y) <= radius * radius:
                image[y, x] = color


def _draw_mark(image: np.ndarray, row: float, col: float, fill: tuple[int, int, int]) -> None:
    center_x = int(round(col))
    center_y = int(round(row))
    _draw_filled_circle(image, center_x, center_y, 11, (0, 0, 0))
    _draw_filled_circle(image, center_x, center_y, 8, fill)


def _draw_outlined_text(image: np.ndarray, x: int, y: int, text: str, fill: tuple[int, int, int], scale: int = 3) -> None:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            _draw_text(image, x + dx, y + dy, text, (255, 255, 255), scale=scale)
    _draw_text(image, x, y, text, fill, scale=scale)


def render_course_overlay(base_png: Path, output_png: Path, course: SailCourseLayout, map_width_px: int) -> Path:
    with rasterio.open(base_png) as src:
        image = np.moveaxis(src.read([1, 2, 3]), 0, -1).copy()

    if image.shape[1] < map_width_px:
        raise ValueError("Base map is narrower than the expected map width.")

    map_width_px = min(map_width_px, image.shape[1])
    overlay = image[:, :map_width_px]

    def row_col(mark: CourseMark) -> tuple[int, int]:
        return int(round(mark.row)), int(round(mark.col))

    start_row, start_col = row_col(course.start)
    windward_row, windward_col = row_col(course.windward)
    reach_row, reach_col = row_col(course.reach)
    pin_row, pin_col = row_col(course.start_line_pin)
    committee_row, committee_col = row_col(course.start_line_committee)

    leg_color = (255, 198, 40)
    start_line_color = (34, 192, 195)
    for thickness, color in ((4, (0, 0, 0)), (2, leg_color)):
        _draw_line(overlay, start_col, start_row, windward_col, windward_row, color, thickness=thickness)
        _draw_line(overlay, windward_col, windward_row, reach_col, reach_row, color, thickness=thickness)
        _draw_line(overlay, reach_col, reach_row, start_col, start_row, color, thickness=thickness)
    for thickness, color in ((4, (0, 0, 0)), (2, start_line_color)):
        _draw_line(overlay, pin_col, pin_row, committee_col, committee_row, color, thickness=thickness)

    _draw_mark(overlay, start_row, start_col, (66, 165, 245))
    _draw_mark(overlay, windward_row, windward_col, (239, 83, 80))
    _draw_mark(overlay, reach_row, reach_col, (255, 202, 40))

    _draw_outlined_text(overlay, start_col + 16, start_row - 16, "s", (0, 0, 0), scale=3)
    _draw_outlined_text(overlay, windward_col + 14, windward_row - 16, "1", (0, 0, 0), scale=3)
    _draw_outlined_text(overlay, reach_col + 14, reach_row - 16, "2", (0, 0, 0), scale=3)

    wind_to_direction = (course.wind_from_direction_deg + 180.0) % 360.0
    theta = math.radians(wind_to_direction)
    arrow_u = math.sin(theta)
    arrow_v = math.cos(theta)
    arrow_center_x = 110
    arrow_center_y = 110
    _draw_arrow(overlay, arrow_center_x, arrow_center_y, arrow_u, arrow_v, 72, (0, 0, 0), thickness=2, head_scale=1.5)
    _draw_outlined_text(overlay, 24, 28, "wind", (0, 0, 0), scale=3)
    _draw_outlined_text(overlay, 24, 54, "to", (0, 0, 0), scale=2)

    return _write_png(output_png, image)


def write_course_layout_json(output_path: Path, course: SailCourseLayout) -> Path:
    output_path.write_text(json.dumps(course.to_dict(), indent=2), encoding="utf-8")
    return output_path
