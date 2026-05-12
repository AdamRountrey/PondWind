from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys


@dataclass(frozen=True)
class SiteConfig:
    center_lat: float = 42.31295753946108
    center_lon: float = -83.75641581917375
    side_meters: float = 1609.344
    solve_buffer_m: float = 800.0
    wind_direction_deg: float = 270.0
    wind_speed_mps: float = 10.0
    label: str = "barton_pond"

    def slug(self) -> str:
        if self.label.strip():
            normalized = re.sub(r"[^a-z0-9]+", "_", self.label.strip().lower())
            normalized = normalized.strip("_")
            if normalized:
                return normalized
        lat_text = f"{self.center_lat:.5f}".replace("-", "m").replace(".", "p")
        lon_text = f"{self.center_lon:.5f}".replace("-", "m").replace(".", "p")
        return f"lat_{lat_text}_lon_{lon_text}"

    def display_name(self) -> str:
        return self.label.strip() or f"{self.center_lat:.5f}, {self.center_lon:.5f}"


def _frozen_project_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data).resolve() / "PondWind"
    return Path(sys.executable).resolve().parent / "user_data"


if getattr(sys, "frozen", False):
    PROJECT_ROOT = _frozen_project_root()
    RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    RESOURCE_ROOT = PROJECT_ROOT
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESOURCE_DATA_RAW_DIR = RESOURCE_ROOT / "data" / "raw"
