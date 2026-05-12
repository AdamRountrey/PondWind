# PondWind

PondWind builds race-day wind and satellite reports for a user-selected square area. It combines terrain-aware WindNinja downscaling with multi-model forecast consensus and nearby observation skill weighting.

## Current capabilities

- User-selected site center and area size
- Buffered `1 m` USGS 3DEP terrain acquisition
- WindNinja solve on a larger buffered domain, cropped back to the report area
- Deterministic wind boundary from a weighted consensus of:
  - `HRDPS`
  - `HRRR`
  - `GFS`
  - `NAM`
  - `ICON`
  - `ECMWF`
- GEFS-based spread products
- Recent nearby observation skill adjustment for deterministic model weighting
- Satellite products on the same report footprint:
  - RGB
  - SST
  - estimated chlorophyll-a
  - estimated turbidity
- Graceful no-water handling:
  - wind and RGB still run
  - water products become `N/A` panels if the selected area does not have enough water coverage
- Packaged Windows GUI build

## Limitations

- This is not a full mesoscale weather model or CFD system.
- Deterministic winds are still only as good as the upstream forecast guidance.
- Observation weighting is based on nearby station observations, not on-pond observations.
- Satellite chlorophyll-a and turbidity are estimated remote-sensing products, not in situ measurements.
- Some model gust products are unavailable or not trustworthy at all forecast hours.

## Repository layout

- `src/predictweather`: core library
- `scripts/run_barton_weekly_report.py`: main report builder CLI
- `scripts/predictweather_gui.py`: desktop GUI
- `scripts/build_windows_exe.ps1`: Windows packaging script
- `PondWind.spec`: PyInstaller spec
- `tools/WindNinjaApp`: local untracked WindNinja runtime staging directory used for local builds and tests

## Environment

The project is currently built around a local Conda environment defined in [`environment.yml`](environment.yml).

Local builds and tests also expect a WindNinja runtime staged at:

- `tools/WindNinjaApp`

That directory is intentionally not committed to Git because vendor binaries can trigger repository size and secret-scanning issues on GitHub.

Create the environment:

```powershell
conda env create -f environment.yml
```

Or, if you use the existing local path-style environment:

```powershell
conda activate C:\Users\aroun\Documents\PredictWeather\.conda-sitewind
```

## Build a weekly report

Example:

```powershell
python scripts\run_barton_weekly_report.py `
  --race-local-datetime 2026-05-10T14:00:00 `
  --center-lat 42.31295753946108 `
  --center-lon -83.75641581917375 `
  --side-meters 1609.344 `
  --site-label barton_pond `
  --mesh-resolution 30 `
  --solve-buffer-m 800 `
  --report-output-dir C:\Reports\PondWind
```

Outputs are written under:

- `outputs\reports\<report_name>`

If `--report-output-dir` is provided, the report folder is created there instead.

Top-level report deliverables are:

- `weekly_report.md`
- `product_1_wind_speed_prediction_knots.png`
- `product_2_wind_speed_variance_knots.png`
- `product_3_wind_direction_variance_degrees.png`
- `satellite_rgb_latest.png`
- `satellite_sst_latest.png`
- `satellite_chla_estimated.png`
- `satellite_turbidity_estimated.png`
- `report_manifest.json`

Intermediates are retained under:

- `report_temp`

## How the wind forecast is built

PondWind uses a two-domain workflow:

1. A larger buffered solve domain is prepared around the user-selected area.
2. Terrain, vegetation, and the WindNinja solve run on that larger domain.
3. Final report products are cropped and rendered back to the requested report area.

This is done to reduce edge effects and give WindNinja a more realistic surrounding terrain context.

### Deterministic wind

The deterministic wind boundary for product 1 is built from a weighted multi-model consensus using:

- `HRDPS`
- `HRRR`
- `GFS`
- `NAM`
- `ICON`
- `ECMWF`

Each model is sampled near the selected site and converted into common `u/v` wind components. PondWind then computes a weighted vector blend that:

- stays close to the overall model mean
- gives a modest prior advantage to higher-resolution models
- penalizes models that are farther from the site grid point
- penalizes speed and direction outliers relative to the model cluster
- applies a gentle recent-skill adjustment from nearby surface observations when available

The resulting consensus wind is passed into WindNinja as the deterministic upstream boundary.

### Wind variability

Products 2 and 3 are not full local probabilistic forecasts. They are downscaled spread products based on `GEFS` mean and spread:

- product 2: wind-speed spread in `knots`
- product 3: wind-direction spread in `degrees`

These should be interpreted as relative uncertainty guidance, not as a perfectly calibrated pond-scale ensemble.

## How to read the report products

- `product_1_wind_speed_prediction_knots.png`
  - deterministic wind-speed forecast in `knots`
  - arrows show modeled flow direction
  - the footer and bottom table summarize the forecast time and upstream model values

- `product_2_wind_speed_variance_knots.png`
  - wind-speed spread in `knots`
  - higher values mean the upstream ensemble/downscaled solution is less certain there

- `product_3_wind_direction_variance_degrees.png`
  - directional spread in `degrees`
  - higher values mean the modeled wind direction is less stable or less agreed upon there

- `satellite_rgb_latest.png`
  - latest reasonably clear RGB scene over the report area

- `satellite_sst_latest.png`
  - latest usable surface-temperature product over water

- `satellite_chla_estimated.png`
  - estimated chlorophyll-a from Sentinel-2 reflectance

- `satellite_turbidity_estimated.png`
  - estimated turbidity from Sentinel-2 reflectance

All report images are rendered to the same report footprint for easier side-by-side comparison.

## Failure and fallback behavior

PondWind tries to complete a useful report even when some data sources are weak or unavailable.

### Forecast guidance

- If one deterministic model is missing, the weighted consensus continues with the remaining available models.
- Some models may provide sustained wind but not gust. In that case, the model still participates, and gust is shown as `n/a`.
- The app uses nearby-station skill weighting when recent observations are available. If not, the deterministic blend falls back to the static consensus weighting only.

### Water products

- If the selected area does not contain enough water pixels, the wind products still run.
- RGB still runs if imagery is available.
- `SST`, `chlorophyll-a`, and `turbidity` degrade gracefully to `N/A` panels instead of failing the whole report.

### Report structure

Each report folder contains:

- top-level final deliverables
- `report_manifest.json` with run metadata, source selections, and diagnostics
- `report_temp` with retained intermediates for inspection or reruns

## Run the GUI

```powershell
python scripts\predictweather_gui.py
```

The GUI lets the user choose the report save folder directly.

If no custom location is chosen, the packaged `.exe` writes reports to:

- `%LOCALAPPDATA%\PondWind\outputs\reports`

## Build the Windows app

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_exe.ps1
```

Optional explicit inputs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_exe.ps1 `
  -PythonExe C:\path\to\python.exe `
  -CondaPrefix C:\path\to\conda-env `
  -WindNinjaHome C:\path\to\WindNinjaApp
```

This produces:

- `dist\PondWind\PondWind.exe`

Keep the full `dist\PondWind` folder together when distributing the app.

## GitHub releases

This repository includes a Windows release workflow at:

- `.github/workflows/windows-release.yml`

It will:

- build `PondWind.exe` on `windows-latest`
- zip the `dist\PondWind` folder
- upload the zip as a workflow artifact
- attach the zip to GitHub releases for tags that start with `v`

By default, the workflow downloads the official WindNinja 3.12.2 Windows ZIP from the USDA Forest Service WindNinja download page, verifies the archive checksum, stages the runtime, and then packages PondWind.

Optional repository configuration:

- Secret `WINDNINJA_ARCHIVE_URL`: override the default upstream WindNinja archive URL.
- Variable `WINDNINJA_ARCHIVE_SHA256`: expected SHA256 for the override archive.

The archive may be either a portable runtime zip that contains `bin\WindNinja_cli.exe` somewhere inside it, or the official WindNinja installer zip. The workflow stages either shape without committing vendor binaries to this repository.

Typical release flow:

1. Push source changes to GitHub.
2. Create and push a tag like `v0.1.0`.
3. Let GitHub Actions build and attach `PondWind-windows.zip` to the release.

## GitHub distribution notes

- Do not commit `build/`, `dist/`, `outputs/`, or raw data caches.
- Publish the source repository separately from packaged binary releases.
- If you ship a binary release, zip the full `dist\PondWind` folder and keep the bundled license files with it.

## License and third-party components

- The repository license applies to the PondWind source code in this repo.
- Bundled third-party components keep their own licenses.
- See:
  - [`LICENSE`](LICENSE)
  - [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
  - [`SECURITY.md`](SECURITY.md)

## Security notes

- TLS verification is on by default.
- `Allow insecure SSL` is a troubleshooting escape hatch for intercepting networks and should not be used unless necessary.
- The app downloads external weather and satellite data and should be treated as network-connected software.
- No API secrets are required for the current supported model sources.
