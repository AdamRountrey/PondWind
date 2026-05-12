# Third-Party Notices

PondWind source code in this repository is licensed under the MIT license in [`LICENSE`](LICENSE).

The packaged Windows application redistributes third-party software that keeps its own license terms.

## Bundled runtime and tools

- `WindNinja`
  - Staged locally for builds/tests under `tools/WindNinjaApp`
  - Upstream project: [firelab/windninja](https://github.com/firelab/windninja)
  - Do not assume the staged WindNinja runtime is covered by this repository's MIT license.

- `OpenFOAM`
  - Optional experimental external solver hook only
  - Not bundled in this repository or packaged Windows builds
  - Any local OpenFOAM installation, runner script, or generated case remains under its own license terms.

- `GDAL`, `PROJ`, `ecCodes`, `rasterio`, `pyproj`, `numpy`, `xarray`, `cfgrib`, `certifi`, and other packaged Python/runtime dependencies
  - These are redistributed in the packaged app folder under `dist/PondWind`
  - Their license files are included in the packaged runtime where available

## Practical policy

- Keep this repository license scoped to the PondWind source code.
- Preserve bundled third-party license files in distributed app builds.
- If you publish binary releases, include:
  - `LICENSE`
  - `THIRD_PARTY_NOTICES.md`
  - any upstream license files already bundled by the build output

## Data sources

PondWind downloads forecast, terrain, and satellite data from third-party public services at runtime. Those datasets are not relicensed by this repository. Users remain responsible for complying with the relevant source terms when redistributing downloaded data or derived products.
