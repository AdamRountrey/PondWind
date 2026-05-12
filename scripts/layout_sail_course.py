from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictweather.runtime import configure_geospatial_runtime

configure_geospatial_runtime()

from predictweather.boundary import load_latest_hrdps_boundary_summary
from predictweather.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, OUTPUTS_DIR
from predictweather.course import layout_conventional_triangle_course, render_course_overlay, write_course_layout_json


def main() -> None:
    boundary_summary, _ = load_latest_hrdps_boundary_summary(DATA_RAW_DIR / "hrdps")
    dem_tif = DATA_PROCESSED_DIR / "site_dem.tif"
    base_png = OUTPUTS_DIR / "windninja_10m_speed_knots_vectors.png"
    output_png = OUTPUTS_DIR / "windninja_10m_speed_knots_vectors_course.png"
    output_json = OUTPUTS_DIR / "sail_course_layout.json"

    course = layout_conventional_triangle_course(
        dem_tif=dem_tif,
        wind_from_direction_deg=float(boundary_summary["wind_from_direction_deg"]),
    )
    render_course_overlay(base_png, output_png, course, map_width_px=1664)
    write_course_layout_json(output_json, course)

    print(f"Course overlay: {output_png}")
    print(f"Course layout: {output_json}")


if __name__ == "__main__":
    main()
