# PondWind

PondWind builds race-day wind and satellite reports for a user-selected square area. It combines terrain-aware WindNinja downscaling with multi-model forecast consensus and nearby observation skill weighting.

## Latest report preview

The public dashboard shows the latest published PondWind report:

- [`adamrountrey.github.io/PondWind`](https://adamrountrey.github.io/PondWind/)

These preview images are pulled from the same GitHub Pages report folder.

<p align="center">
  <a href="https://adamrountrey.github.io/PondWind/">
    <img src="https://adamrountrey.github.io/PondWind/latest/thumbs/product_1_wind_speed_prediction_knots.jpg?v=readme-20260525" alt="Latest wind speed prediction" width="31%">
  </a>
  <a href="https://adamrountrey.github.io/PondWind/">
    <img src="https://adamrountrey.github.io/PondWind/latest/thumbs/product_2_wind_speed_variance_knots.jpg?v=readme-20260525" alt="Latest wind speed ensemble standard deviation" width="31%">
  </a>
  <a href="https://adamrountrey.github.io/PondWind/">
    <img src="https://adamrountrey.github.io/PondWind/latest/thumbs/product_3_wind_direction_variance_degrees.jpg?v=readme-20260525" alt="Latest wind direction ensemble standard deviation" width="31%">
  </a>
</p>

## Current capabilities

- User-selected site center and area size
- Buffered `1 m` USGS 3DEP terrain acquisition
- WindNinja solve on a larger buffered domain, cropped back to the report area
- Packaged Windows GUI build intended for non-technical users
- Optional experimental OpenFOAM CFD comparison product for local WSL/OpenFOAM experiments
- Deterministic wind boundary from a weighted consensus of:
  - `HRDPS`
  - `HRRR`
  - `GFS`
  - `NAM`
  - `ICON`
  - `ECMWF`
- GEFS real-member ensemble standard-deviation products
- Recent nearby observation skill adjustment for deterministic model weighting
- Satellite products on the same report footprint:
  - RGB
  - surface temperature over water
  - experimental chlorophyll-a index
  - experimental turbidity index
- GUI checkboxes to skip individual satellite products when you want a faster or smaller report
- Graceful no-water handling:
  - wind and RGB still run
  - water products become `N/A` panels if the selected area does not have enough water coverage

## Limitations

- This is not a full mesoscale weather model or CFD system.
- Deterministic winds are still only as good as the upstream forecast guidance.
- Observation weighting is based on nearby station observations, not on-pond observations.
- Satellite chlorophyll-a and turbidity products are experimental reflectance indices, not in situ measurements or locally validated aquatic retrievals.
- Some model gust products are unavailable or not trustworthy at all forecast hours.
- Experimental OpenFOAM products require a local WSL/OpenFOAM 13 installation. If it is missing, PondWind skips the CFD comparison and still builds the WindNinja report.

## Repository layout

- `src/predictweather`: core library
- `scripts/run_barton_weekly_report.py`: main report builder CLI
- `scripts/predictweather_gui.py`: desktop GUI
- `scripts/build_windows_exe.ps1`: Windows packaging script
- `PondWind.spec`: PyInstaller spec
- `tools/WindNinjaApp`: local untracked WindNinja runtime staging directory used for local builds and tests

## Environment

Most users should use a packaged Windows release zip from GitHub Releases. Unzip the full `PondWind` folder and launch:

- `PondWind.exe`

The packaged app bundles the Python runtime and WindNinja runtime needed for normal reports. Users do not need Conda, Python, WSL, or OpenFOAM unless they explicitly choose the experimental OpenFOAM CFD comparison.

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

Without `--report-output-dir`, source checkouts write outputs under:

- `outputs\reports\<report_name>`

If `--report-output-dir` is provided, the dated report folder is created there instead. Shared DEM and forecast downloads are retained in a visible `PondWind Working Data` folder beside the reports so later runs can reuse them.

Satellite products are enabled by default. To skip individual satellite products from the CLI:

```powershell
python scripts\run_barton_weekly_report.py `
  --no-satellite-rgb `
  --no-satellite-sst `
  --no-satellite-chla `
  --no-satellite-turbidity
```

Product 1 and the wind sensitivity products are built with WindNinja. To add a separate experimental OpenFOAM CFD comparison product:

```powershell
$env:PONDWIND_OPENFOAM_RUNNER = "C:\path\to\run_openfoam_case.ps1"
python scripts\run_barton_weekly_report.py `
  --wind-solver openfoam `
  --race-local-datetime 2026-05-10T14:00:00 `
  --center-lat 42.31295753946108 `
  --center-lon -83.75641581917375
```

The OpenFOAM runner is intentionally experimental. It must accept the command-line arguments PondWind passes, read `--request-json`, and write WindNinja-compatible wind speed, direction, `u`, and `v` ASCII grids. PondWind validates finite values, direction ranges, and speed sanity before the CFD product is allowed into the report.

For adapter testing without OpenFOAM, the repository includes a reference runner that writes uniform wind grids using the same output contract:

```powershell
$env:PONDWIND_OPENFOAM_RUNNER = ".\.conda-sitewind\python.exe scripts\openfoam_uniform_runner.py"
```

This confirms PondWind's OpenFOAM plumbing, but it is not a CFD solve and should not be used as a scientific wind product.

For an actual experimental CFD path, install OpenFOAM in WSL/Ubuntu and use the WSL runner:

```powershell
wsl --install -d Ubuntu-24.04
```

Inside Ubuntu, install OpenFOAM 13:

```bash
sudo apt update
sudo apt -y install wget software-properties-common
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt update
sudo apt -y install openfoam13
echo ". /opt/openfoam13/etc/bashrc" >> ~/.bashrc
. ~/.bashrc
foamRun -help
```

Then back in PowerShell:

```powershell
$env:PONDWIND_OPENFOAM_RUNNER = ".\.conda-sitewind\python.exe scripts\openfoam_wsl_terrain_runner.py"
$env:PONDWIND_OPENFOAM_MAX_HORIZONTAL_CELLS = "12000"
```

The WSL runner is intentionally experimental. It generates a neutral atmospheric boundary layer, terrain-following OpenFOAM 13 case, runs `blockMesh`, `checkMesh`, optional `potentialFoam` initialization, final `foamRun -solver incompressibleFluid`, and `postProcess -latestTime -func sample` at 10 m above terrain. It fails the OpenFOAM comparison product if mesh quality, convergence, sampling, or physical sanity checks fail; it does not substitute a fallback CFD wind map.

If OpenFOAM comparison is selected but WSL/OpenFOAM 13 is unavailable, PondWind writes a friendly skipped CFD panel and continues the report with the production WindNinja products.

OpenFOAM comparison defaults can be tuned with environment variables:

```powershell
$env:PONDWIND_OPENFOAM_VERTICAL_CELLS = "20"
$env:PONDWIND_OPENFOAM_DOMAIN_HEIGHT_M = "400"
$env:PONDWIND_OPENFOAM_MAX_HORIZONTAL_CELLS = "12000"
$env:PONDWIND_OPENFOAM_WATER_Z0_M = "0.0002"
$env:PONDWIND_OPENFOAM_GRASS_Z0_M = "0.03"
$env:PONDWIND_OPENFOAM_TREE_Z0_M = "0.3"
```

Top-level report deliverables are:

- `weekly_report.md`
- `product_1_wind_speed_prediction_knots.png`
- `product_2_wind_speed_variance_knots.png`
- `product_3_wind_direction_variance_degrees.png`
- `product_4_openfoam_experimental_cfd_knots.png` when OpenFOAM comparison is enabled
- `product_5_openfoam_turbulence_intensity_percent.png` when OpenFOAM turbulence sampling succeeds
- `product_6_sailing_polar_dem_overlay.png`
- `product_7_openfoam_sailing_polar_dem_overlay.png` when OpenFOAM comparison succeeds
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

The resulting consensus wind is passed into WindNinja as the production deterministic upstream boundary. If OpenFOAM comparison is enabled, the same boundary wind is also passed into a separate experimental WSL/OpenFOAM 13 neutral ABL solve and compared against WindNinja in `report_manifest.json`.

### Wind variability

Products 2 and 3 are not full local probabilistic forecasts. PondWind downloads real `GEFS` control/perturbation members near the site, selects a bounded set of weighted `u/v` cluster-medoid representatives for WindNinja solves, and computes weighted standard deviation from those downscaled member outputs:

- product 2: wind-speed ensemble standard deviation in `knots`
- product 3: wind-direction ensemble standard deviation in `degrees`

These should be interpreted as relative uncertainty guidance, not calibrated probabilities or a perfectly calibrated pond-scale ensemble. If real GEFS members are unavailable, PondWind falls back to the older GEFS mean/spread sensitivity method and says so in the report notes.

Runtime can be tuned with environment variables when testing:

```powershell
$env:PONDWIND_MODEL_DOWNLOAD_WORKERS = "4"
$env:PONDWIND_GEFS_MEMBER_DOWNLOAD_LIMIT = "31"
$env:PONDWIND_GEFS_MEMBER_DOWNLOAD_WORKERS = "3"
$env:PONDWIND_GEFS_MEMBER_SOLVE_LIMIT = "9"
$env:PONDWIND_WINDNINJA_MEMBER_WORKERS = "2"
```

Higher WindNinja member workers can speed ensemble products on larger CPUs, but each worker launches a separate terrain solve.

## How to read the report products

- `product_1_wind_speed_prediction_knots.png`
  - deterministic wind-speed forecast in `knots`
  - arrows show modeled flow direction
  - the footer and bottom table summarize the forecast time and upstream model values

- `product_2_wind_speed_variance_knots.png`
  - wind-speed ensemble standard deviation in `knots`
  - higher values mean weighted representative GEFS members create more local speed disagreement after WindNinja downscaling

- `product_3_wind_direction_variance_degrees.png`
  - directional ensemble standard deviation in `degrees`
  - higher values mean weighted representative GEFS members create more local direction disagreement after WindNinja downscaling

- `product_4_openfoam_experimental_cfd_knots.png`
  - optional experimental CFD comparison against product 1
  - uses the same boundary wind and bottom model/weather context as the production wind product
  - skipped automatically if WSL/OpenFOAM 13 is unavailable
  - custom contract runners are labeled as adapter output, not validated CFD

- `product_5_openfoam_turbulence_intensity_percent.png`
  - optional experimental neutral-ABL/OpenFOAM turbulence intensity in `%`
  - uses the same report color scale as the other wind diagnostic maps

- `product_6_sailing_polar_dem_overlay.png`
  - experimental ILCA 7 / Laser Standard relative point polar over the cropped DEM
  - centered on the sampled wind cell near the selected area center
  - green arrows show best upwind VMG headings, purple arrows show best downwind VMG headings
  - heuristic sailing aid only, not a calibrated VPP or routing model

- `product_7_openfoam_sailing_polar_dem_overlay.png`
  - optional experimental CFD version of the same sailing polar overlay
  - written only when the OpenFOAM comparison product completes

- `satellite_rgb_latest.png`
  - latest reasonably clear RGB scene over the report area

- `satellite_sst_latest.png`
  - latest usable Landsat/ECOSTRESS land-surface-temperature product masked to water pixels

- `satellite_chla_estimated.png`
  - experimental chlorophyll-a index from Sentinel-2 L2A reflectance

- `satellite_turbidity_estimated.png`
  - experimental turbidity index from Sentinel-2 L2A reflectance

All report images are rendered to the same report footprint for easier side-by-side comparison.

## Failure and fallback behavior

PondWind tries to complete a useful report even when some data sources are weak or unavailable.

### Forecast guidance

- If one deterministic model is missing, the weighted consensus continues with the remaining available models.
- Some models may provide sustained wind but not gust. In that case, the model still participates, and gust is shown as `n/a`.
- Remote providers can temporarily return `403`, `404`, `429`, or timeout responses. These are recorded in `report_manifest.json` under `wind.model_errors`.
- The app uses nearby-station skill weighting when recent observations are available. If not, the deterministic blend falls back to the static consensus weighting only.

### Experimental CFD

- OpenFOAM comparison is optional and never replaces product 1.
- If WSL or OpenFOAM 13 is missing, the report continues and marks the CFD comparison as skipped.
- If the OpenFOAM solve fails mesh, convergence, sampling, or physical sanity checks, the CFD product is marked failed instead of silently using fallback data.
- Passing the file contract only proves that a runner produced compatible grids; only the WSL terrain runner is labeled as an experimental OpenFOAM CFD candidate.

### Water products

- If the selected area does not contain enough water pixels, the wind products still run.
- RGB still runs if imagery is available.
- Surface temperature over water, chlorophyll-a, and turbidity products degrade gracefully to `N/A` panels instead of failing the whole report.
- If a satellite product is unchecked in the GUI, PondWind skips that final product and omits it from the markdown report.

### Report structure

Each report folder contains:

- top-level final deliverables
- `report_manifest.json` with run metadata, source selections, and diagnostics
- `report_temp` with retained intermediates for inspection or reruns

## Run the GUI

```powershell
python scripts\predictweather_gui.py
```

The GUI prompts for the report save location every time a build starts. The selected folder contains the dated report folder, its `report_temp` working files, and the shared `PondWind Working Data` download folder. Report files are not stored under `%LOCALAPPDATA%`.

The GUI also lets users:

- choose WindNinja-only output or optional experimental OpenFOAM comparison
- set wind grid size
- turn individual satellite products on or off
- force ECOSTRESS surface-temperature discovery
- allow insecure SSL only when a local network requires it

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

The build script copies `README.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `SECURITY.md` into the app folder.

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

## GitHub Pages report dashboard

PondWind can publish a lightweight static webpage showing the latest report images. The site is intended for GitHub Pages on a separate `gh-pages` branch so report images do not clutter the application source branch.

First-time setup:

```powershell
cd C:\Users\aroun\Documents\PredictWeather
powershell -ExecutionPolicy Bypass -File scripts\publish_latest_report_to_pages.ps1
```

The script:

- prompts for the folder containing PondWind reports, unless `-ReportsRoot` or `-ReportDir` is provided
- copies only whitelisted final PNG products
- writes a sanitized `latest\report.json`
- updates `index.html`, `styles.css`, and `.nojekyll` on the local `gh-pages` worktree
- commits and pushes `gh-pages`

The local Pages worktree is created under `.worktrees\gh-pages`, which is ignored by Git on the application branch.

To publish a specific report:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\publish_latest_report_to_pages.ps1 `
  -ReportDir "C:\Reports\PondWind\20260517_1400_barton_pond"
```

To test without pushing:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\publish_latest_report_to_pages.ps1 -NoPush
```

After the first push, enable GitHub Pages in repository settings:

- Source: `Deploy from a branch`
- Branch: `gh-pages`
- Folder: `/root`

The public page should then be available at:

- `https://adamrountrey.github.io/PondWind/`

## GitHub distribution notes

- Do not commit `build/`, `dist/`, `outputs/`, or raw data caches.
- Do not commit local Conda environments, WindNinja runtime staging folders, OpenFOAM cases, report outputs, or generated GIF/preview diagnostics.
- Publish the source repository separately from packaged binary releases.
- If you ship a binary release, zip the full `dist\PondWind` folder and keep the bundled license files with it.
- Normal users should download the release zip rather than installing Python dependencies manually.

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
