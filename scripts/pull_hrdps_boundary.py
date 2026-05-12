from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictweather.config import DATA_RAW_DIR
from predictweather.hrdps import download_hrdps_boundary_conditions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ECCC HRDPS boundary-condition wind fields.")
    parser.add_argument("--lead-hours", type=float, default=2.0, help="Hours ahead from current UTC time.")
    parser.add_argument(
        "--allow-insecure-ssl",
        action="store_true",
        help="Allow insecure SSL for networks that inject a self-signed corporate certificate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allow_insecure_ssl:
        os.environ["PREDICTWEATHER_ALLOW_INSECURE_SSL"] = "1"
    selection = download_hrdps_boundary_conditions(
        destination_dir=DATA_RAW_DIR,
        lead_hours=args.lead_hours,
    )
    print(f"Requested at UTC: {selection.requested_at_utc}")
    print(f"Target time UTC: {selection.target_at_utc}")
    print(f"Selected valid UTC: {selection.selected_valid_at_utc}")
    print(f"Run UTC: {selection.run_at_utc}")
    print(f"Forecast hour: PT{selection.forecast_hour:03d}H")
    for variable, path in selection.files.items():
        print(f"{variable}: {path}")


if __name__ == "__main__":
    main()
