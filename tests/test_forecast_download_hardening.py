from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from predictweather.runtime import configure_geospatial_runtime

configure_geospatial_runtime()

from predictweather.boundary import load_cached_hrdps_manifest_for_valid_time
from predictweather.gfs import build_gfs_pds_url
from predictweather.hrrr import build_hrrr_pds_url
from predictweather.nam import build_nam_pds_url
from predictweather.noaa_pds import find_grib_message, parse_grib_index
from predictweather.nomads import nomads_filter_url


class ForecastDownloadHardeningTests(unittest.TestCase):
    def test_gfs_nomads_url_can_use_360_longitudes(self) -> None:
        url = nomads_filter_url(
            filter_script="filter_gfs_0p25.pl",
            file_name="gfs.t12z.pgrb2.0p25.f006",
            dir_path="/gfs.20260517/12/atmos",
            variable_name="UGRD",
            level_flag="lev_10_m_above_ground",
            lat=42.31,
            lon=-83.75,
            lon_360=True,
        )

        self.assertIn("leftlon=275.8000", url)
        self.assertIn("rightlon=276.7000", url)

    def test_hrdps_cache_rejects_wrong_valid_time_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "20260517T06Z" / "PT012H"
            data_dir.mkdir(parents=True)
            u_path = data_dir / "u.grib2"
            v_path = data_dir / "v.grib2"
            u_path.write_bytes(b"u")
            v_path.write_bytes(b"v")
            manifest = {
                "selected_valid_at_utc": "2026-05-17T18:00:00+00:00",
                "run_at_utc": "2026-05-17T06:00:00+00:00",
                "forecast_hour": 12,
                "files": {"UGRD": str(u_path), "VGRD": str(v_path)},
            }
            (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                load_cached_hrdps_manifest_for_valid_time(
                    root,
                    datetime(2026, 5, 17, 19, tzinfo=timezone.utc),
                )

    def test_noaa_grib_index_selects_message_range(self) -> None:
        entries = parse_grib_index(
            "\n".join(
                [
                    "1:0:d=2026051712:TMP:2 m above ground:6 hour fcst:",
                    "2:128:d=2026051712:UGRD:10 m above ground:6 hour fcst:",
                    "3:512:d=2026051712:VGRD:10 m above ground:6 hour fcst:",
                ]
            )
        )

        entry, next_offset = find_grib_message(entries, variable="UGRD", level_text="10 m above ground")

        self.assertEqual(entry.offset, 128)
        self.assertEqual(next_offset, 512)

    def test_noaa_public_bucket_urls(self) -> None:
        run_at = datetime(2026, 5, 17, 12, tzinfo=timezone.utc)

        self.assertEqual(
            build_gfs_pds_url(run_at, 6),
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260517/12/atmos/gfs.t12z.pgrb2.0p25.f006",
        )
        self.assertEqual(
            build_hrrr_pds_url(run_at, 6),
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260517/conus/hrrr.t12z.wrfsfcf06.grib2",
        )
        self.assertEqual(
            build_nam_pds_url(run_at, 6),
            "https://noaa-nam-pds.s3.amazonaws.com/nam.20260517/nam.t12z.conusnest.hiresf06.tm00.grib2",
        )


if __name__ == "__main__":
    unittest.main()
