# Security Policy

## Scope

This project is a local desktop/report-generation tool that downloads public weather, terrain, and satellite data at runtime.

## Supported release hygiene

- Keep `build/`, `dist/`, `outputs/`, and raw downloaded data out of version control.
- Do not commit local logs or crash dumps such as `startup_error.txt` or `report_build_error.txt`.
- Keep TLS verification enabled by default.
- Use the `Allow insecure SSL` option only as a last-resort workaround on intercepting networks.

## Reporting issues

If you find a security issue in the source code or packaging flow, report it privately to the project maintainer before opening a public issue.

Include:

- affected version or commit
- reproduction steps
- whether the issue affects source runs, packaged builds, or both
- whether the issue involves external downloads, local file writes, or bundled dependencies

## Known risk areas

- External data downloads from forecast, terrain, and satellite providers
- Optional insecure SSL override for difficult network environments
- Large third-party native runtime bundle in the packaged Windows app

## Current release checks

Before publishing a release, at minimum:

- run a secret scan on source files
- verify `.gitignore` excludes build/data/output artifacts
- rebuild the packaged app from a clean tree
- confirm `LICENSE` and `THIRD_PARTY_NOTICES.md` ship with the release
- confirm WSL/OpenFOAM is optional and missing CFD dependencies produce a skipped comparison, not a failed report
