from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def as_tnm_bbox(self) -> str:
        return f"{self.min_lon:.8f},{self.min_lat:.8f},{self.max_lon:.8f},{self.max_lat:.8f}"


def square_bbox_from_center(center_lat: float, center_lon: float, side_meters: float) -> BoundingBox:
    half_side = side_meters / 2.0
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(center_lat))

    delta_lat = half_side / meters_per_deg_lat
    delta_lon = half_side / meters_per_deg_lon

    return BoundingBox(
        min_lon=center_lon - delta_lon,
        min_lat=center_lat - delta_lat,
        max_lon=center_lon + delta_lon,
        max_lat=center_lat + delta_lat,
    )


def buffered_square_bbox_from_center(center_lat: float, center_lon: float, side_meters: float, buffer_meters: float) -> BoundingBox:
    return square_bbox_from_center(center_lat, center_lon, side_meters + 2.0 * buffer_meters)
