from __future__ import annotations

import os
import sys
from pathlib import Path

from predictweather.config import RESOURCE_ROOT


def _candidate_env_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in (RESOURCE_ROOT, Path(sys.prefix), Path(sys.base_prefix)):
        resolved = candidate.resolve()
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            roots.append(resolved)
    return roots


def configure_geospatial_runtime() -> None:
    """Set GDAL/PROJ/ecCodes paths when running from a Conda env without activation."""
    path_entries: list[str] = []
    gdal_data: Path | None = None
    proj_data: Path | None = None
    eccodes_root: Path | None = None

    for env_root in _candidate_env_roots():
        library_bin = env_root / "Library" / "bin"
        candidate_gdal_data = env_root / "Library" / "share" / "gdal"
        candidate_proj_data = env_root / "Library" / "share" / "proj"
        candidate_eccodes_root = env_root / "Library"

        if env_root.exists():
            path_entries.append(str(env_root))
        if library_bin.exists():
            path_entries.append(str(library_bin))
        if gdal_data is None and candidate_gdal_data.exists():
            gdal_data = candidate_gdal_data
        if proj_data is None and candidate_proj_data.exists():
            proj_data = candidate_proj_data
        if eccodes_root is None and candidate_eccodes_root.exists():
            eccodes_root = candidate_eccodes_root

    deduped_entries: list[str] = []
    seen_entries: set[str] = set()
    for entry in path_entries:
        key = entry.lower()
        if key not in seen_entries:
            seen_entries.add(key)
            deduped_entries.append(entry)
    path_entries = deduped_entries

    current_path = os.environ.get("PATH", "")
    for entry in reversed(path_entries):
        if not current_path.startswith(entry):
            current_path = entry + os.pathsep + current_path
    os.environ["PATH"] = current_path

    if hasattr(os, "add_dll_directory"):
        for entry in path_entries:
            try:
                os.add_dll_directory(entry)
            except (FileNotFoundError, OSError):
                pass

    if gdal_data is not None and gdal_data.exists() and "GDAL_DATA" not in os.environ:
        os.environ["GDAL_DATA"] = str(gdal_data)

    if proj_data is not None and proj_data.exists():
        os.environ.setdefault("PROJ_DATA", str(proj_data))
        os.environ.setdefault("PROJ_LIB", str(proj_data))

    if eccodes_root is not None and eccodes_root.exists():
        os.environ.setdefault("ECCODES_DIR", str(eccodes_root))
