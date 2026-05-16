# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project_root = Path.cwd()
src_dir = project_root / "src"
scripts_dir = project_root / "scripts"

conda_root = Path(os.environ.get("PONDWIND_CONDA_PREFIX", sys.prefix)).resolve()
fallback_conda_root = (project_root / ".conda-sitewind").resolve()
if not (conda_root / "Library").exists() and (fallback_conda_root / "Library").exists():
    conda_root = fallback_conda_root

windninja_home_env = os.environ.get("PONDWIND_WINDNINJA_HOME", "").strip()
windninja_home = Path(windninja_home_env).resolve() if windninja_home_env else (project_root / "tools" / "WindNinjaApp").resolve()

conda_library_dir = conda_root / "Library"
conda_tkinter_dir = conda_root / "Lib" / "tkinter"
conda_tcl_data_dir = conda_library_dir / "lib" / "tcl8.6"
conda_tk_data_dir = conda_library_dir / "lib" / "tk8.6"

missing_required = [
    str(path)
    for path in (
        conda_library_dir,
        conda_tkinter_dir,
        conda_tcl_data_dir,
        conda_tk_data_dir,
        windninja_home / "bin" / "WindNinja_cli.exe",
    )
    if not path.exists()
]
if missing_required:
    raise FileNotFoundError(
        "Missing required build inputs:\n- " + "\n- ".join(missing_required)
    )

datas = []
binaries = []
hiddenimports = []

for package_name in ("rasterio", "pyproj", "PIL", "cfgrib", "xarray", "certifi"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

datas += [
    (str(windninja_home), "tools/WindNinjaApp"),
    (str(src_dir), "src"),
    (str(conda_library_dir), "Library"),
    (str(conda_tkinter_dir), "tkinter"),
    (str(conda_tcl_data_dir), "_tcl_data"),
    (str(conda_tk_data_dir), "_tk_data"),
]


a = Analysis(
    [str(scripts_dir / "predictweather_app.py")],
    pathex=[str(project_root), str(src_dir), str(scripts_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        "predictweather_gui",
        "run_barton_weekly_report",
        "experimental_laser_polar_point",
        "openfoam_wsl_terrain_runner",
        "openfoam_uniform_runner",
        "tkinter",
        "_tkinter",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PondWind",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PondWind",
    contents_directory=".",
)
